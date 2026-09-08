import json
from zipfile import ZipFile

from PIL import Image

from sam3_vlm.core.config import BootstrapConfig, V4Config
from sam3_vlm.core.geometry import Box, BoxGeometry
from sam3_vlm.core.types import Detection, SpatialMode
from sam3_vlm.experiments.discovery_diagnostic import run_diagnostic
from sam3_vlm.sensing.observation import SAM3Observation
from sam3_vlm.models.sam3 import DummySAM3Sensor


class ProbeSensor:
    def __init__(self):
        self.actions = []

    def observe(self, image, action):
        self.actions.append(action)
        if action.spatial_mode == SpatialMode.GLOBAL or action.positive_exemplar_boxes:
            boxes = [(Box(10, 10, 30, 30), .9)]
        else:
            boxes = [(Box(12, 12, 16, 16), .9), (Box(60, 60, 70, 70), .35)]
        return SAM3Observation(
            call_id=f'sam3_{len(self.actions):06}', action_id=action.action_id, semantic_key='target',
            detections=[Detection(f'd_{i}', BoxGeometry(box), score) for i, (box, score) in enumerate(boxes) if score >= action.threshold],
            searched_regions=[action.search_region],
        )


def sample(tmp_path):
    image = tmp_path / 'image.png'
    Image.new('RGB', (100, 100)).save(image)
    return {'sample_id': 'test', 'target': 'green fruit', 'gt_count': 2, 'image_path': str(image)}


def test_fixed_bootstrap_factorial_probe_and_offline_association(tmp_path):
    sensor = ProbeSensor()
    cfg = V4Config(bootstrap=BootstrapConfig(enable_tiled_bootstrap=False))
    report = run_diagnostic([sample(tmp_path)], sensor, cfg, ['green fruit', 'dark green fruit'], tmp_path / 'out')
    assert report['complete'] and report['sam3_calls'] == 9 and report['qwen_calls'] == 0
    row, = report['samples']
    assert row['bootstrap_nodes'] == 1 and row['available_exemplars'] == 1
    assert len(row['probes']) == 8
    for probe in row['probes']:
        iou, dual = probe['association']['iou_only'], probe['association']['iou_iom']
        if probe['exemplars_requested']:
            assert iou['new_candidates'] == dual['new_candidates'] == 0
        elif probe['threshold'] == .25:
            assert iou['new_candidates'] == 2 and dual['new_candidates'] == 1
        else:
            assert iou['new_candidates'] == 1 and dual['new_candidates'] == 0
    assert all(a.search_region == sensor.actions[0].search_region for a in sensor.actions)
    assert all(a.positive_exemplar_boxes == ((10., 10., 30., 30.),) for a in sensor.actions if a.positive_exemplar_boxes)
    with ZipFile(tmp_path / 'out/discovery_review.zip') as z:
        assert len(z.namelist()) == 2
        assert json.loads(z.read('discovery_report.json'))['sam3_calls'] == 9


def test_diagnostic_call_cap_and_partial_report(tmp_path):
    sensor = ProbeSensor()
    report = run_diagnostic([sample(tmp_path)], sensor, V4Config(bootstrap=BootstrapConfig(enable_tiled_bootstrap=False)),
                            ['green fruit'], tmp_path / 'out', max_calls=1)
    assert len(sensor.actions) == report['sam3_calls'] == 1
    assert report['complete'] is False
    assert 'cap reached' in report['samples'][0]['error']
    assert (tmp_path / 'out/discovery_review.zip').is_file()


def test_no_seeds_is_explicitly_skipped_not_a_fake_exemplar_arm(tmp_path):
    report = run_diagnostic([sample(tmp_path)], DummySAM3Sensor(),
                            V4Config(bootstrap=BootstrapConfig(enable_tiled_bootstrap=False)),
                            ['green fruit'], tmp_path / 'out')
    assert report['complete'] and report['sam3_calls'] == 3
    assert sum('skipped' in p for p in report['samples'][0]['probes']) == 2
