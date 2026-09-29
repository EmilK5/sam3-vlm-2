from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from sam3_vlm.datasets.fscd147 import FSCD147, evaluate_counts
from sam3_vlm.experiments.fscd147 import run_dataset, evaluate_saved, select_variants, export_detections
from sam3_vlm.experiments.m8_smoke import load_m8_config
from sam3_vlm.models.sam3 import MockSAM3Adapter
from sam3_vlm.planning.qwen_planner import PlannerOutput


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / 'dataset'
    root.mkdir()
    (root / 'images_384_VarV2').mkdir()
    for name in ('a.jpg', 'b.jpg'):
        Image.new('RGB', (384, 384)).save(root / 'images_384_VarV2' / name)
    (root / 'Train_Test_Val_FSC_147.json').write_text(json.dumps({'val': ['a.jpg', 'b.jpg']}))
    (root / 'ImageClasses_FSC147.txt').write_text('a.jpg green apples\nb.jpg bottle caps\n')
    (root / 'annotation_FSC147_384.json').write_text(json.dumps({
        'a.jpg': {'points': [[2, 3]]*99}, 'b.jpg': {'points': [[2, 3]]*88}}))
    (root / 'instances_val.json').write_text(json.dumps({
        'images': [{'id': 1, 'file_name': 'a.jpg', 'width': 384, 'height': 384},
                   {'id': 2, 'file_name': 'b.jpg', 'width': 384, 'height': 384}],
        'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [10, 10, 40, 40]},
                        {'id': 2, 'image_id': 1, 'category_id': 1, 'bbox': [100, 100, 40, 40]}],
        'categories': [{'id': 1, 'name': 'object'}]}))
    return FSCD147(root)


@pytest.fixture
def deployment(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'configs/fscd147.json'
    return load_m8_config(SimpleNamespace(require_cuda=False, output_dir=str(tmp_path / 'run')), path)


class NoopPlanner:
    model = 'mock-qwen'

    def plan_scene(self, evidence, budget, config):
        assert evidence.user_prompt in {'green apples', 'bottle caps'}
        assert 'fruit on trees' not in config.planner.target_scope
        assert not evidence.discovery_diagnostics['search_region_locked']
        return PlannerOutput(proposed_actions=[])


class TraceSensor(MockSAM3Adapter):
    def __init__(self):
        super().__init__()
        self.actions = []

    def observe(self, image, action):
        self.actions.append(action)
        assert action.prompt in {'green apples', 'bottle caps'}
        assert action.search_region.bbox().as_tuple() == (0, 0, 384, 384)
        return super().observe(image, action)


def test_all_ae_arms_run_existing_pipeline_without_annotation_access(dataset, deployment, tmp_path, monkeypatch):
    original_read = Path.read_text
    def guarded_read(path, *args, **kwargs):
        if path.name in {'annotation_FSC147_384.json', 'instances_val.json'}:
            pytest.fail('Inference accessed ground truth')
        return original_read(path, *args, **kwargs)
    with monkeypatch.context() as guard:
        guard.setattr(Path, 'read_text', guarded_read)
        sensor = TraceSensor()
        path = run_dataset(dataset, deployment, sensor, NoopPlanner(), tmp_path / 'all_arms')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 10 and all(row['success'] for row in rows)
    assert {row['variant'][0] for row in rows} == set('ABCDE')
    assert all('gt_count' not in row for row in rows)
    for row in rows:
        assert row['budget']['qwen_calls'] == (0 if row['variant'][0] in 'AB' else 1 if row['variant'][0] == 'C' else 2)
        assert row['budget']['sam3_calls'] == (1 if row['variant'][0] == 'A' else 2)
        assert row['count_type'] == ('hard_candidate_count' if row['variant'][0] in 'AB' else 'soft_posterior_count')
    metrics = evaluate_saved(dataset, path)
    assert all(m['complete'] for m in metrics['variants'].values())
    # FSCD boxes define count: a=2, b=0, regardless of intentionally different point counts.
    assert metrics['variants']['A_SAM3_Global']['mae'] == 1
    assert all(m['rmse'] is not None for m in metrics['variants'].values())
    exported = json.loads(Path(metrics['variants']['A_SAM3_Global']['coco_output']).read_text())
    assert exported == [{'image_id': 1, 'category_id': 1, 'bbox': [10, 10, 40, 40], 'score': .88},
                        {'image_id': 2, 'category_id': 1, 'bbox': [10, 10, 40, 40], 'score': .88}]


def test_a_does_not_require_qwen_or_annotations_and_partial_split_has_no_metrics(dataset, deployment, tmp_path):
    (dataset.root / 'annotation_FSC147_384.json').unlink()
    path = run_dataset(dataset, deployment, TraceSensor(), None, tmp_path / 'partial', arm='A', max_images=1)
    metrics = evaluate_saved(dataset, path)['variants']['A_SAM3_Global']
    assert not metrics['complete'] and metrics['mae'] is None and metrics['rmse'] is None
    assert 'coco_output' not in metrics


def test_model_failure_is_saved_and_next_image_still_runs(dataset, deployment, tmp_path):
    class FailFirst(TraceSensor):
        failed = False
        def observe(self, image, action):
            if not self.failed:
                self.failed = True
                raise RuntimeError('sensor failed')
            return super().observe(image, action)
    path = run_dataset(dataset, deployment, FailFirst(), None, tmp_path / 'failed', arm='A')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]['success'] is False and rows[0]['predicted_count'] is None
    assert rows[1]['success'] is True
    metric = evaluate_saved(dataset, path)['variants']['A_SAM3_Global']
    assert not metric['complete'] and metric['mae'] is None
    assert metric['failures'] == {'a.jpg': 'sensor failed'}


def test_predictions_cannot_overwrite_existing_run(dataset, deployment, tmp_path):
    run_dataset(dataset, deployment, TraceSensor(), None, tmp_path / 'run', arm='A')
    with pytest.raises(FileExistsError):
        run_dataset(dataset, deployment, TraceSensor(), None, tmp_path / 'run', arm='A')


@pytest.mark.parametrize('change', ['duplicate', 'unknown_image', 'unknown_variant', 'count_type', 'split'])
def test_evaluator_rejects_corrupt_or_mismatched_predictions(dataset, deployment, tmp_path, change):
    path = run_dataset(dataset, deployment, TraceSensor(), None, tmp_path / 'run', arm='A')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if change == 'duplicate': rows.append(rows[0])
    elif change == 'unknown_image': rows[0]['image_id'] = 'missing.jpg'
    elif change == 'unknown_variant': rows[0]['variant'] = 'F'
    elif change == 'count_type': rows[0]['count_type'] = 'hard_belief_count'
    else:
        metadata_path = path.parent / 'metadata.json'
        metadata = json.loads(metadata_path.read_text())
        metadata['split'] = 'test'
        metadata_path.write_text(json.dumps(metadata))
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError): evaluate_saved(dataset, path)


def test_export_scales_predictions_to_coco_annotation_resolution(dataset):
    truth = dataset.ground_truth()
    rows = {'a.jpg': {'image_size': [192, 192], 'nodes': [{'box': [5, 10, 25, 30], 'score': .8}]}}
    assert export_detections(rows, truth) == [
        {'image_id': 1, 'category_id': 1, 'bbox': [10, 20, 40, 40], 'score': .8}]


def test_dataset_neutral_policy_retains_ae_caps_and_mask_dedup(deployment):
    variants = select_variants(deployment.v4_config)
    assert [v.config.budget.max_qwen_calls for v in variants] == [0, 0, 1, 2, 100]
    assert all(v.config.association.mask_only and v.config.association.enable_iom_dedup for v in variants)
    assert all(v.config.bootstrap.locked_context_prompt is None for v in variants)
    assert all(v.config.tiling.enable_adaptive for v in variants)
    assert all('all visible' in v.config.planner.target_scope for v in variants)
    assert all(v.config.sam3.qwen_discovery_threshold == .2 for v in variants)


@pytest.mark.parametrize('invalid', [float('nan'), -1, True, None])
def test_invalid_counts_never_produce_complete_split_metrics(dataset, invalid):
    metrics = evaluate_counts({'a.jpg': invalid, 'b.jpg': 0}, dataset.ground_truth())
    assert not metrics['complete'] and metrics['mae'] is None


def test_removed_f_suite_is_rejected(deployment):
    from sam3_vlm.experiments.m8_smoke import _pilot_variants
    with pytest.raises(ValueError, match='Unknown pilot suite'):
        _pilot_variants(deployment.v4_config, 'final-af')


def test_fscd_cli_dry_run_needs_no_models_or_annotations(dataset, tmp_path, monkeypatch, capsys):
    from sam3_vlm.experiments.fscd147 import main
    (dataset.root / 'instances_val.json').unlink()
    (dataset.root / 'annotation_FSC147_384.json').unlink()
    def forbidden(*args, **kwargs):
        pytest.fail('Dry run must not initialize models')
    monkeypatch.setattr('sam3_vlm.models.sam3.RealSAM3Sensor', forbidden)
    config_path = Path(__file__).resolve().parents[1] / 'configs/fscd147.json'
    assert main(['run', str(dataset.root), str(tmp_path / 'dry'), '--config', str(config_path),
                 '--dry-run', '--arm', 'D', '--max-images', '1']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == {'split_images': 2, 'run_images': 1, 'variants': ['D_Qwen_TwoRound']}
    assert not (tmp_path / 'dry').exists()


def test_optional_coco_ap_reports_perfect_and_empty_detections(dataset):
    pytest.importorskip('pycocotools')
    from sam3_vlm.experiments.fscd147 import coco_ap
    path = dataset.root / 'instances_val.json'
    annotations = json.loads(path.read_text())
    for box in annotations['annotations']:
        box.update(area=box['bbox'][2]*box['bbox'][3], iscrowd=0)
    path.write_text(json.dumps(annotations))
    detections = [{'image_id': item['image_id'], 'category_id': 1, 'bbox': item['bbox'], 'score': 1.0}
                  for item in annotations['annotations']]
    metrics = coco_ap(path, detections)
    assert metrics['AP'] == pytest.approx(1) and metrics['AP50'] == pytest.approx(1)
    assert metrics['max_detections'] == 1000
    empty = coco_ap(path, [])
    assert empty['AP'] == empty['AP50'] == 0
