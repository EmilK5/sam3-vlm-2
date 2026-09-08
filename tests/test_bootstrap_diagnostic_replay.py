"""An unmatched bootstrap node can change diagnostics without new evidence."""

import json
from dataclasses import replace

import pytest

from sam3_vlm.core.config import BootstrapConfig, BudgetConfig, V4Config
from sam3_vlm.core.geometry import Box, BoxGeometry
from sam3_vlm.core.types import ActionFamily, Detection, SpatialMode
from sam3_vlm.experiments.m8_smoke import assemble_e2e_runner, _run_validator_and_replay
from sam3_vlm.logging.artifacts import RunArtifactPaths
from sam3_vlm.models.qwen import MockQwenPlanner
from sam3_vlm.pipeline.runner import Runner
from sam3_vlm.planning.qwen_planner import PlannerOutput, ProposedAction
from sam3_vlm.sensing.observation import SAM3Observation


class OverlapSensor:
    call_count = 0

    def observe(self, image, action):
        self.call_count += 1
        # The tiled pass adds a nearby node without retrieving either old node.
        boxes = [(17, 0, 37, 20)] if action.spatial_mode == SpatialMode.TILED else [
            (0, 0, 20, 20), (60, 60, 70, 70),
        ]
        return SAM3Observation(
            call_id=f'sam_{self.call_count}', action_id=action.action_id,
            semantic_key=action.semantic_key,
            detections=[Detection(f'det_{self.call_count}_{i}', BoxGeometry(Box(*box)), 0.8)
                        for i, box in enumerate(boxes)],
            searched_regions=[BoxGeometry(Box(0, 0, 100, 100))],
        )


def rejected_planner():
    return MockQwenPlanner(custom_output=PlannerOutput(proposed_actions=[ProposedAction(
        'target', 'green fruit hanging low', ActionFamily.DISCOVERY,
        semantic_prior={'target': 1.0},
    )]))


@pytest.mark.parametrize('iom', [False, True])
def test_unmatched_bootstrap_diagnostics_are_logged_and_replay_exactly(tmp_path, iom):
    config = V4Config(
        bootstrap=BootstrapConfig(enable_tiled_bootstrap=True, tiled_bootstrap_min_candidates=5),
        budget=BudgetConfig(max_qwen_calls=1, max_cleanup_calls=0),
        association=replace(V4Config().association, enable_iom_dedup=iom),
        tiling=replace(V4Config().tiling, tile_min_size=32),
        assets_dir=str(tmp_path / 'assets'),
    )
    paths = RunArtifactPaths(tmp_path / 'run')
    runner, _ = assemble_e2e_runner(paths, config, OverlapSensor(), rejected_planner(),
                                    'test', 'green fruit', 'target', 'img')
    runner.run((100, 100), 'green fruit', image_id='img')
    node = runner.scene_state.graph.get_node('node_000001')
    assert node.diagnostics.duplicate_risk > 0
    assert len(node.observations) == 1  # A diagnostic update must not invent an observation.
    events = [json.loads(line) for line in paths.events_jsonl.read_text().splitlines()]
    updates = [e for e in events if e['event_type'] == 'NODE_UPDATED']
    assert len(updates) == 1  # Do not log unchanged, unmatched node_000002 again.
    assert updates[0]['data']['node_id'] == 'node_000001'
    assert updates[0]['data']['provenance']['reason'] == 'ASSOCIATION_DIAGNOSTICS_CHANGED'
    assert _run_validator_and_replay(paths, runner.scene_state)

    unlogged = Runner(config, OverlapSensor(), rejected_planner())
    unlogged.run((100, 100), 'green fruit', image_id='img')
    assert unlogged.scene_state.graph.to_dict() == runner.scene_state.graph.to_dict()
