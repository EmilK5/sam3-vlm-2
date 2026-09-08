"""Qwen confounders provide negative evidence; counting uses posterior > 0.5."""

import json
from dataclasses import replace

import pytest

from sam3_vlm.core.config import BeliefConfig, BootstrapConfig, BudgetConfig, PlannerConfig, V4Config
from sam3_vlm.core.geometry import Box, BoxGeometry
from sam3_vlm.core.types import ActionFamily, ClassBelief, Detection, SpatialMode
from sam3_vlm.experiments.m8_smoke import _run_validator_and_replay, assemble_e2e_runner
from sam3_vlm.logging.artifacts import RunArtifactPaths
from sam3_vlm.models.qwen import MockQwenPlanner
from sam3_vlm.planning.qwen_planner import PlannerOutput, ProposedAction, QwenPlannerService
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.node import Node
from sam3_vlm.scene.state import CountEstimator
from sam3_vlm.sensing.evidence import ContactSheet, QwenEvidencePack
from sam3_vlm.sensing.observation import SAM3Observation
from sam3_vlm.core.types import BudgetState, StopReason


def config():
    return V4Config(
        planner=PlannerConfig(max_actions_per_prompt=1, execute_confounder_prompts=True),
        belief=BeliefConfig(target_count_hard_threshold=0.5),
        bootstrap=BootstrapConfig(enable_tiled_bootstrap=False),
        budget=BudgetConfig(max_qwen_calls=1, max_sam3_calls=10, max_cleanup_calls=0),
    )


def test_hard_count_boundary_keeps_soft_count_variance_and_beliefs():
    graph = SceneGraph()
    probabilities = [0, 0.49, 0.5, 0.51, 1]
    for i, p in enumerate(probabilities):
        graph.add_node(Node(str(i), BoxGeometry(Box(i*20, 0, i*20+10, 10)),
                            class_belief=ClassBelief(probabilities={'target': p, 'confounder1': 1-p})))
    before = graph.to_dict()
    count = CountEstimator.estimate(graph, 'target', target_hard_threshold=0.5)
    assert count.mean_count == count.committed_node_count == 2
    assert count.raw_soft_count == 2.5
    assert count.variance == pytest.approx(sum(p*(1-p) for p in probabilities))
    assert graph.to_dict() == before


@pytest.mark.parametrize('bad', [-0.1, 1.1, float('nan')])
def test_invalid_hard_threshold_rejected(bad):
    with pytest.raises(ValueError, match='target_count_hard_threshold'):
        BeliefConfig(target_count_hard_threshold=bad)
    with pytest.raises(ValueError, match='target_hard_threshold'):
        CountEstimator.estimate(SceneGraph(), 'target', target_hard_threshold=bad)


def test_hard_and_commit_rules_cannot_be_combined():
    with pytest.raises(ValueError, match='mutually exclusive'):
        BeliefConfig(target_count_hard_threshold=0.5, target_count_commit_threshold=0.8)


def test_labels_generate_negatives_with_frozen_slots_and_no_repeat():
    service = QwenPlannerService(MockQwenPlanner(custom_output=PlannerOutput(
        proposed_actions=[ProposedAction('target', 'shaded fruit', ActionFamily.DISCOVERY,
                                         semantic_prior={'target': 1.0})],
        likely_confounders=['leaf', 'branch', 'grass'],
    )))
    pack = QwenEvidencePack('img', 'green fruit', 'target', ContactSheet(),
                            belief_classes=['target', 'confounder1', 'confounder2'])
    output = service.plan_scene(pack, BudgetState(), config())
    assert [(a.prompt, a.family.value, a.semantic_prior) for a in output.proposed_actions] == [
        ('shaded fruit', 'DISCOVERY', {'target': 1.0}),
        ('leaf', 'CONFOUNDER', {'confounder1': 1.0}),
        ('branch', 'CONFOUNDER', {'confounder2': 1.0}),
    ]
    for a in output.proposed_actions[1:]:
        assert a.suggested_threshold == .5
        assert a.suggested_spatial_mode == SpatialMode.TILED
        assert a.positive_exemplar_ids == []
    pack.confounder_labels = {'confounder1': 'stone', 'confounder2': 'bark'}
    pack.discovery_diagnostics['tried_sam3_prompts'] = ['stone']
    output = service.plan_scene(pack, BudgetState(), config())
    assert [a.prompt for a in output.proposed_actions] == ['shaded fruit', 'bark']


class PrecisionSensor:
    def __init__(self):
        self.actions = []

    def observe(self, image, action):
        self.actions.append(action)
        k = len(self.actions)
        if action.family == ActionFamily.CONFOUNDER:
            assert action.semantic_key == 'confounder1'
            assert action.threshold == .5
            assert not action.positive_exemplar_ids and not action.positive_exemplar_boxes
            boxes = [(0, 0, 20, 20), (17, 0, 37, 20)]
            score = .99
        else:
            boxes = [(0, 0, 20, 20), (60, 60, 80, 80)]
            score = .9
        return SAM3Observation(
            f'sam{k}', action.action_id, action.semantic_key,
            detections=[Detection(f'd{k}_{i}', BoxGeometry(Box(*box)), score) for i,box in enumerate(boxes)],
            searched_regions=[BoxGeometry(Box(0, 0, 100, 100))],
        )


def run_precision(tmp_path, cfg, labels=('leaf',)):
    sensor = PrecisionSensor()
    planner = MockQwenPlanner(custom_output=PlannerOutput(
        proposed_actions=[ProposedAction('target', 'shaded fruit', ActionFamily.DISCOVERY,
                                        semantic_prior={'target': 1.0})], likely_confounders=list(labels)))
    cfg = replace(cfg, assets_dir=str(tmp_path / 'assets'))
    paths = RunArtifactPaths(tmp_path / 'run')
    runner, _ = assemble_e2e_runner(paths, cfg, sensor, planner, 'test', 'green fruit', 'target', 'img')
    runner.run((100, 100), 'green fruit', image_id='img')
    assert _run_validator_and_replay(paths, runner.scene_state)
    return runner, sensor, paths


@pytest.mark.parametrize('iom', [False, True])
def test_negative_detection_rejects_candidate_without_new_nodes_and_replays(tmp_path, iom):
    cfg = config()
    cfg = replace(cfg, association=replace(cfg.association, enable_iom_dedup=iom))
    runner, sensor, paths = run_precision(tmp_path, cfg)
    assert [a.family for a in sensor.actions] == [ActionFamily.DISCOVERY, ActionFamily.DISCOVERY, ActionFamily.CONFOUNDER]
    assert len(runner.scene_state.graph.nodes) == 2
    assert runner.scene_state.graph.get_node('node_000001').class_belief.probabilities['target'] < .5
    assert runner.scene_state.count_estimate.mean_count == 1
    assert runner.scene_state.budget.qwen_calls == 1
    assert runner.scene_state.graph.get_node('node_000001').diagnostics.duplicate_risk == 0
    summary = json.loads(paths.summary_json.read_text())
    assert summary['final_count'] == summary['final_soft_count'] == 1
    assert summary['discovery_statistics']['count_type'] == 'hard_posterior_count'
    assert summary['discovery_statistics']['raw_soft_count'] != 1
    assert summary['count_variance'] > 0


def test_pending_negative_respects_sam3_budget(tmp_path):
    cfg = config()
    cfg = replace(cfg, budget=replace(cfg.budget, max_sam3_calls=2))
    runner, sensor, _ = run_precision(tmp_path, cfg)
    assert len(sensor.actions) == 2
    assert runner.scene_state.stop_reason == StopReason.SAM3_BUDGET
    assert runner.scene_state.count_estimate.mean_count == 2


def test_numerical_saturation_waits_for_both_negative_queries(tmp_path):
    from sam3_vlm.models.sam3 import DummySAM3Sensor

    cfg = replace(config(), assets_dir=str(tmp_path / 'assets'))
    planner = MockQwenPlanner(custom_output=PlannerOutput(
        proposed_actions=[ProposedAction('target', 'shaded fruit', ActionFamily.DISCOVERY,
                                         semantic_prior={'target': 1.0})],
        likely_confounders=['leaf', 'branch'],
    ))
    paths = RunArtifactPaths(tmp_path / 'run')
    runner, _ = assemble_e2e_runner(paths, cfg, DummySAM3Sensor(), planner,
                                    'test', 'green fruit', 'target', 'img')
    runner.run((100, 100), 'green fruit', image_id='img')
    assert runner.scene_state.budget.qwen_calls == 1
    assert runner.scene_state.budget.sam3_calls == 4
    assert runner.scene_state.iteration == 3
    assert runner.scene_state.stop_reason == StopReason.DISCOVERY_AND_UNCERTAINTY_SATURATED
    assert not runner.scene_state.action_bank.unexecuted_entries()
    assert _run_validator_and_replay(paths, runner.scene_state)


def test_invalid_negative_label_is_rejected_and_not_frozen(tmp_path):
    runner, sensor, paths = run_precision(tmp_path, config(), labels=['green leaf in shade'])
    assert len(sensor.actions) == 2
    assert runner.scene_state.confounder_labels == {}
    artifact = json.loads(next(paths.qwen_dir.glob('*.json')).read_text())
    assert artifact['metadata']['rejections'][0]['reason'] == 'INVALID_GROUNDING_PROMPT'
    assert artifact['metadata']['fallback_used'] is False
