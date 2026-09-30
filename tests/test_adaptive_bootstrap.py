from dataclasses import replace

import numpy as np
from PIL import Image
import pytest

from sam3_vlm.core.config import AssociationConfig, BootstrapConfig, BudgetConfig, TilingConfig, V4Config
from sam3_vlm.core.geometry import Box, GeometryRef
from sam3_vlm.core.types import Detection, SpatialMode
from sam3_vlm.experiments.m8_smoke import _run_sam3_baseline, _run_validator_and_replay, assemble_e2e_runner
from sam3_vlm.logging.artifacts import RunArtifactPaths
from sam3_vlm.models.sam3 import DummySAM3Sensor
from sam3_vlm.pipeline.bootstrap import BootstrapPipeline
from sam3_vlm.planning.qwen_planner import PlannerOutput


class DenseSensor(DummySAM3Sensor):
    def __init__(self, seed_score=.9):
        super().__init__()
        self.actions = []
        self.seed_score = seed_score

    def observe(self, image, action):
        observation = super().observe(image, action)
        self.actions.append(action)
        assert action.prompt == 'bottle caps'
        region = observation.searched_regions[0].bbox()
        for i in range(60):
            x, y = 100 + (i % 10)*8, 100 + (i // 10)*8
            x1, y1, x2, y2 = max(x, int(region.x1)), max(y, int(region.y1)), min(x+6, int(region.x2)), min(y+6, int(region.y2))
            if x1 >= x2 or y1 >= y2:
                continue
            observation.detections.append(Detection(
                f'det_{self.call_count}_{i}', GeometryRef(Box(x1, y1, x2, y2)), self.seed_score,
                raw_metadata={'mask': np.ones((y2-y1, x2-x1), dtype=bool),
                              'mask_offset_x': x1, 'mask_offset_y': y1,
                              'crop_boundary_clipped': (x1, y1, x2, y2) != (x, y, x+6, y+6)}))
        return observation


def config(tmp_path):
    return V4Config(
        bootstrap=BootstrapConfig(enable_pseudoexemplar_refinement=True, locked_context_prompt='tree canopy'),
        tiling=TilingConfig(enable_adaptive=True),
        association=AssociationConfig(mask_only=True, enable_iom_dedup=True),
        budget=BudgetConfig(max_sam3_calls=128, max_sam3_tiles=128, max_runtime_seconds=None),
        assets_dir=str(tmp_path / 'assets'),
    )


def test_dense_bootstrap_covers_full_image_and_budgets_each_actual_tile(tmp_path):
    sensor = DenseSensor()
    result = BootstrapPipeline(sensor, config=config(tmp_path)).execute_bootstrap(
        'caps', Image.new('RGB', (384, 384)), 'bottle caps')
    state = result.state
    assert sensor.actions[0].spatial_mode == SpatialMode.GLOBAL
    assert sensor.actions[0].positive_exemplar_boxes == ()
    assert sensor.actions[1].positive_exemplar_boxes
    assert len(sensor.actions) == state.budget.sam3_calls == 27  # global + refinement + 25 tiles
    assert state.budget.sam3_tiles == 25
    assert len(state.graph.active_nodes()) == 60  # offset fragments never add duplicates
    assert all(node.geometry.area() == 36 for node in state.graph.active_nodes())
    assert not state.search_region_locked
    assert state.search_region.bbox().as_tuple() == (0, 0, 384, 384)
    covered = np.zeros((384, 384), dtype=bool)
    for action in sensor.actions[2:]:
        assert action.tile_id and action.spatial_mode == SpatialMode.LOCAL
        x1, y1, x2, y2 = map(int, action.roi.bbox().as_tuple())
        covered[y1:y2, x1:x2] = True
    assert covered.all()
    plan = result.qwen_evidence_pack.discovery_diagnostics['adaptive_tiling']
    assert plan['object_count'] == 60 and plan['trigger']
    assert state.discovery_state.adaptive_tiling == plan


@pytest.mark.parametrize('limit', ['calls', 'tiles', 'runtime'])
def test_adaptive_bootstrap_checks_budgets_before_requests(tmp_path, limit):
    cfg = config(tmp_path)
    if limit == 'calls':
        cfg = replace(cfg, budget=replace(cfg.budget, max_sam3_calls=3))
        expected_calls = 3
    elif limit == 'tiles':
        cfg = replace(cfg, budget=replace(cfg.budget, max_sam3_tiles=1))
        expected_calls = 3
    else:
        cfg = replace(cfg, budget=replace(cfg.budget, max_runtime_seconds=.001))
        expected_calls = 1  # Dummy sensor reports 10ms for the first completed pass.
    sensor = DenseSensor()
    with pytest.raises(RuntimeError, match='budget exhausted'):
        BootstrapPipeline(sensor, config=cfg).execute_bootstrap('caps', (384, 384), 'bottle caps')
    assert sensor.call_count <= expected_calls


def test_low_score_seeds_do_not_trigger_density_tiles(tmp_path):
    sensor = DenseSensor(seed_score=.4)
    state = BootstrapPipeline(sensor, config=config(tmp_path)).execute_bootstrap('caps', (384, 384), 'bottle caps').state
    assert sensor.call_count == 1  # No strong exemplars or density seeds.
    assert state.discovery_state.adaptive_tiling['object_count'] == 0
    assert not state.discovery_state.adaptive_tiling['trigger']


def test_low_confidence_small_objects_trigger_fallback_without_gt(tmp_path):
    cfg = config(tmp_path)
    cfg = replace(cfg, tiling=replace(cfg.tiling, adaptive_enable_fallback=True))
    sensor = DenseSensor(seed_score=.4)
    state = BootstrapPipeline(sensor, config=cfg).execute_bootstrap('caps', (384, 384), 'bottle caps').state
    plan = state.discovery_state.adaptive_tiling
    assert plan['object_count'] == 0 and plan['candidate_count'] == 60
    assert plan['trigger_reason'] == 'small_objects'
    assert sensor.call_count == 26  # First pass + 25 tiles, no pseudoexemplars.
    assert state.budget.sam3_tiles == 25
    assert len(state.graph.active_nodes()) == 60


def test_dense_baseline_saves_masks_and_tiling_plan(tmp_path):
    paths = RunArtifactPaths(tmp_path / 'run')
    count, state = _run_sam3_baseline(
        paths=paths, config=config(tmp_path), sensor=DenseSensor(), run_id='dense',
        prompt='bottle caps', image_id='caps', image=Image.new('RGB', (384, 384)),
        seed=42, experiment_name='adaptive-test')
    assert count == 60
    import json
    summary = json.loads(paths.summary_json.read_text())
    assert summary['discovery_statistics']['adaptive_tiling']['trigger']
    assert len(list(paths.masks_dir.glob('*.npz'))) > 60
    for node in state.graph.active_nodes():
        with np.load(paths.base_dir / node.geometry.mask_artifact) as data:
            assert data['mask'].sum() == node.geometry.area()


def test_dense_adaptive_runner_passes_replay_with_mask_geometry(tmp_path):
    class Planner:
        model = 'none'
        def plan_scene(self, *args):
            return PlannerOutput(proposed_actions=[])
    paths = RunArtifactPaths(tmp_path / 'runner')
    runner, _ = assemble_e2e_runner(paths, config(tmp_path), DenseSensor(), Planner(),
                                   'dense', 'bottle caps', 'target', 'caps')
    runner.run(Image.new('RGB', (384, 384)), 'bottle caps', image_id='caps')
    assert _run_validator_and_replay(paths, runner.scene_state)


def test_recovered_children_retire_coarse_parent_and_replay(tmp_path):
    class GroupSensor(DummySAM3Sensor):
        def observe(self, image, action):
            obs = super().observe(image, action)
            masks = [(20, 20, 20)] if self.call_count == 1 else [(22, 22, 3), (32, 32, 3)]
            for i, (x, y, size) in enumerate(masks):
                obs.detections.append(Detection(f'd{self.call_count}_{i}', GeometryRef(Box(x, y, x+size, y+size)), .9,
                    raw_metadata={'mask': np.ones((size, size), dtype=bool), 'mask_offset_x': x, 'mask_offset_y': y}))
            return obs
    class Planner:
        model = 'none'
        def plan_scene(self, *args):
            return PlannerOutput(proposed_actions=[])
    paths = RunArtifactPaths(tmp_path / 'groups')
    runner, _ = assemble_e2e_runner(paths, config(tmp_path), GroupSensor(), Planner(),
                                   'groups', 'bottle caps', 'target', 'caps')
    runner.run(Image.new('RGB', (384, 384)), 'bottle caps', image_id='caps')
    assert len(runner.scene_state.graph.active_nodes()) == 2
    assert any(node.status.value == 'REJECTED' for node in runner.scene_state.graph.nodes.values())
    assert _run_validator_and_replay(paths, runner.scene_state)
