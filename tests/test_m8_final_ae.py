"""Final A–E matrix, rectangle-only exports, and measurement tables."""
import csv
import json
from dataclasses import asdict, replace
from zipfile import ZipFile

import pytest
from PIL import Image, ImageDraw

from sam3_vlm.core.geometry import Box, BoxGeometry
from sam3_vlm.core.types import ActionSource, ClassBelief
from sam3_vlm.experiments.final_outputs import render_candidate_boxes, overlay_path, write_final_outputs, TABLE_FILES
from sam3_vlm.experiments.m8_smoke import _pilot_variants, m8_4_and_5_pilot
from sam3_vlm.models.sam3 import MockSAM3Adapter
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.node import Node
from test_m8_low_discovery import production
from test_m8_orchestration import DummyArgs
from test_m8_rejection_correction import SequencePlanner, proposal


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def test_final_ae_preserves_selected_d_and_lowers_only_ab_bootstrap():
    base = production()
    variants = _pilot_variants(base, 'final-ae')
    selected = _pilot_variants(base, 'final-ablation')[0].config
    a, b, c, d, e = [v.config for v in variants]
    assert [v.name[0] for v in variants] == list('ABCDE')
    assert [v.config.budget.max_qwen_calls for v in variants] == [0, 0, 1, 2, 100]
    assert [v.count_type for v in variants] == ['hard_candidate_count']*2 + ['soft_posterior_count']*3
    assert a.sam3.default_threshold == b.sam3.default_threshold == .2
    assert not a.bootstrap.enable_tiled_bootstrap and not a.bootstrap.enable_pseudoexemplar_refinement
    assert a.bootstrap.locked_context_prompt is None
    assert b.bootstrap == d.bootstrap == selected.bootstrap
    assert d == selected
    assert c == replace(d, budget=replace(d.budget, max_qwen_calls=1),
                        replanning=replace(d.replanning, max_replans=0))
    for cfg in (c, d, e):
        assert cfg.sam3 == selected.sam3
        assert cfg.sam3.default_threshold == .25
        assert cfg.sam3.qwen_discovery_threshold == .2
        assert cfg.planner == selected.planner
        assert cfg.belief == selected.belief
        assert not cfg.belief.neutral_confounder_misses
        assert cfg.association == selected.association
    assert e.budget.max_sam3_calls == 1000
    assert e.budget.max_sam3_tiles is None and e.budget.max_runtime_seconds is None
    assert e.stopping.max_iterations is None
    assert e.replanning.max_replans == 99 and e.replanning.continue_until_saturation
    assert all(v.config.budget.max_cleanup_calls == 0 for v in variants)


def test_renderer_draws_rectangles_only_without_filtering_or_mutation(tmp_path, monkeypatch):
    graph = SceneGraph()
    for i, (coords, probability) in enumerate([((10, 10, 30, 30), .1), ((40, 40, 60, 60), .9)]):
        graph.add_node(Node(str(i), BoxGeometry(Box(*coords)),
            class_belief=ClassBelief(probabilities={'target': probability, 'confounder1': 1-probability})))
    image = Image.new('RGB', (80, 80), 'white')
    before_graph, before_image = graph.to_dict(), image.tobytes()
    def forbidden_text(*args, **kwargs):
        pytest.fail('No text, score or ID may be drawn')
    monkeypatch.setattr(ImageDraw.ImageDraw, 'text', forbidden_text)
    path = tmp_path / 'boxes.png'
    result = render_candidate_boxes(image, graph, path)
    assert result['bbox_count'] == 2  # Includes low-probability nodes, like the soft-count graph.
    assert graph.to_dict() == before_graph and image.tobytes() == before_image
    with Image.open(path) as rendered:
        assert rendered.size == image.size
        assert rendered.getpixel((10, 10)) == (255, 48, 48)
        assert rendered.getpixel((20, 20)) == (255, 255, 255)  # No fill/mask.
        assert rendered.getpixel((0, 0)) == (255, 255, 255)
        expected = image.copy()
        pen = ImageDraw.Draw(expected)
        pen.rectangle((10, 10, 30, 30), outline=(255, 48, 48), width=2)
        pen.rectangle((40, 40, 60, 60), outline=(255, 48, 48), width=2)
        assert rendered.tobytes() == expected.tobytes()


def test_renderer_clips_boxes_and_handles_empty_graph(tmp_path):
    graph = SceneGraph()
    image = Image.new('RGB', (20, 20), 'white')
    empty = render_candidate_boxes(image, graph, tmp_path / 'empty.png')
    assert empty['bbox_count'] == 0
    with Image.open(tmp_path / 'empty.png') as output:
        assert output.tobytes() == image.tobytes()
    graph.add_node(Node('edge', BoxGeometry(Box(-5, -5, 8, 8))))
    graph.add_node(Node('outside', BoxGeometry(Box(40, 40, 50, 50))))
    out = render_candidate_boxes(image, graph, tmp_path / 'clipped.png')
    assert out['bbox_count'] == 1 and out['bbox_outside_image_count'] == 1
    with Image.open(tmp_path / 'clipped.png') as output:
        assert output.getpixel((0, 0)) == (255, 48, 48)


def test_overlay_names_cannot_escape_output_root(tmp_path):
    path = overlay_path(tmp_path, '../variant', '../../image')
    assert path.resolve().is_relative_to(tmp_path.resolve())
    assert path != overlay_path(tmp_path, '../variant', '..//image')
    assert overlay_path(tmp_path, 'D', 'image_1').name == 'image_1.png'


def test_tables_keep_zero_gt_and_failures_explicit(tmp_path):
    report = {'metadata': {'variants': ['A', 'D'], 'sample_count': 2},
              'aggregates': {'A': {'MAE': 1., 'MSE': 1., 'RMSE': 1., 'MRE': .2},
                             'D': {'MAE': 2., 'MSE': 4., 'RMSE': 2., 'MRE': None}},
              'samples': [
        {'variant': 'A', 'sample_id': 'zero', 'gt_count': 0, 'predicted_count': 1., 'absolute_error': 1., 'success': True},
        {'variant': 'A', 'sample_id': 'positive', 'gt_count': 5, 'predicted_count': 4., 'absolute_error': 1., 'success': True},
        {'variant': 'D', 'sample_id': 'zero', 'gt_count': 0, 'predicted_count': 2., 'absolute_error': 2., 'success': True},
        {'variant': 'D', 'sample_id': 'positive', 'gt_count': 5, 'success': False, 'failure_message': 'model failed'},
    ]}
    exports = write_final_outputs(report, tmp_path)
    rows = read_csv(tmp_path / 'aggregate_results.csv')
    assert rows[1]['n_failed_or_missing'] == '1'
    assert rows[1]['GT_total_all_images'] == '5' and rows[1]['GT_total_successful_images'] == '0'
    assert rows[1]['complete'] == 'False' and rows[1]['MRE_percent'] == ''
    assert rows[0]['n_common_success'] == '1'
    assert float(rows[0]['MAE_common_images']) == 1. and float(rows[1]['MAE_common_images']) == 2.
    assert read_csv(tmp_path / 'counts_by_image.csv')[1]['D'] == ''
    detail = read_csv(tmp_path / 'per_image_results.csv')
    assert detail[-1]['failure_message'] == 'model failed' and detail[-1]['predicted_count'] == ''
    assert exports['bbox_images_exported'] == 0
    with ZipFile(tmp_path / 'bbox_images.zip') as z:
        assert z.namelist() == []


def test_final_ae_pilot_exports_every_image_and_correct_tables(tmp_path, monkeypatch):
    image = tmp_path / 'input.jpg'
    Image.new('RGB', (64, 64), (30, 70, 20)).save(image)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps([{'sample_id': 'image_1', 'image_path': str(image),
                                    'target': 'green fruit', 'gt_count': 2}]))
    class CheckingSensor(MockSAM3Adapter):
        def observe(self, image, action):
            if action.source == ActionSource.QWEN:
                assert action.threshold == (.2 if action.family.value == 'DISCOVERY' else .5)
                assert not action.positive_exemplar_ids
            return super().observe(image, action)
    monkeypatch.setattr('sam3_vlm.experiments.m8_smoke._get_models',
        lambda args: (CheckingSensor(), SequencePlanner([proposal('dark green fruit', ['leaf'])])))
    output_dir = tmp_path / 'run'
    args = DummyArgs(manifest=str(manifest), max_samples=1, pilot_suite='final-ae', output_dir=str(output_dir))
    assert m8_4_and_5_pilot(args)  # Includes validation/replay for C/D/E.
    report = json.loads((output_dir / 'pilot_report.json').read_text())
    assert len(report['samples']) == 5
    assert all(r['success'] for r in report['samples'])
    assert report['metadata']['final_exports']['bbox_images_exported'] == 5
    assert report['metadata']['model_settings']['seed'] == 42
    aggregates, details, wide = (read_csv(output_dir / f) for f in TABLE_FILES[:3])
    assert len(aggregates) == 5 and len(details) == 5 and len(wide) == 1
    assert wide[0]['gt_count'] == '2'
    for result in report['samples']:
        aggregate = next(r for r in aggregates if r['variant'] == result['variant'])
        assert float(aggregate['MAE']) == pytest.approx(abs(result['predicted_count'] - 2))
        assert aggregate['n_success'] == '1'
        assert float(wide[0][result['variant']]) == result['predicted_count']
        assert result['bbox_count'] == result['candidate_count']
        with Image.open(result['bbox_image']) as overlay:
            assert overlay.size == (64, 64)
    with ZipFile(output_dir / 'bbox_images.zip') as z:
        assert len(z.namelist()) == 5
        assert all(n.endswith('.png') and n.startswith('bbox_images/') for n in z.namelist())
    with ZipFile(output_dir / 'compact_review.zip') as z:
        assert set(TABLE_FILES).issubset(z.namelist())
        assert not any(n.startswith('bbox_images/') for n in z.namelist())
        assert len(json.loads(z.read('counts.json'))) == 5


@pytest.mark.parametrize('box', [Box(0, 0, 5, 5, coordinate_space='tile'),
                                  Box(0, 0, float('nan'), 5), Box(1, 1, 1, 5)])
def test_renderer_rejects_invalid_geometry(tmp_path, box):
    graph = SceneGraph()
    graph.add_node(Node('invalid', BoxGeometry(box)))
    with pytest.raises(ValueError, match='Final boxes must'):
        render_candidate_boxes(Image.new('RGB', (20, 20)), graph, tmp_path / 'bad.png')
    assert not (tmp_path / 'bad.png').exists()


def test_inference_failure_still_exports_final_tables_for_other_variants(tmp_path, monkeypatch):
    image = tmp_path / 'input.jpg'
    Image.new('RGB', (64, 64)).save(image)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps([{'sample_id': 'image_1', 'image_path': str(image),
                                    'target': 'green fruit', 'gt_count': 0}]))
    class FailFirstSensor(MockSAM3Adapter):
        failed = False
        def observe(self, image, action):
            if not self.failed:
                self.failed = True
                raise RuntimeError('test sensor failure')
            return super().observe(image, action)
    monkeypatch.setattr('sam3_vlm.experiments.m8_smoke._get_models',
        lambda args: (FailFirstSensor(), SequencePlanner([proposal('dark green fruit')])))
    output_dir = tmp_path / 'run'
    args = DummyArgs(manifest=str(manifest), max_samples=1, pilot_suite='final-ae', output_dir=str(output_dir))
    assert not m8_4_and_5_pilot(args)
    aggregate = read_csv(output_dir / 'aggregate_results.csv')
    assert aggregate[0]['n_failed_or_missing'] == '1'
    assert aggregate[0]['MAE'] == '' and aggregate[0]['complete'] == 'False'
    assert aggregate[0]['count_type'] == 'hard_candidate_count'
    detail = read_csv(output_dir / 'per_image_results.csv')
    assert detail[0]['target'] == 'green fruit'
    assert detail[0]['failure_message'] == 'test sensor failure'
    assert all(r['relative_error_percent'] == '' for r in detail)
    with ZipFile(output_dir / 'bbox_images.zip') as z:
        assert len(z.namelist()) == 4
    with ZipFile(output_dir / 'compact_review.zip') as z:
        assert 'aggregate_results.csv' in z.namelist()
