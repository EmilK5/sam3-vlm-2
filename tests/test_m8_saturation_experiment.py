"""Experiment E explores beyond short-run limits, with auditable hard caps."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from sam3_vlm.core.config import BootstrapConfig, SAM3Config, V4Config
from sam3_vlm.core.geometry import Box, BoxGeometry
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.core.types import ActionFamily, ActionSource, BudgetState, Detection, SpatialMode, StopReason
from sam3_vlm.experiments.m8_smoke import (
    _pilot_variants, _run_validator_and_replay, assemble_e2e_runner, load_m8_config,
)
from sam3_vlm.logging.artifacts import RunArtifactPaths
from sam3_vlm.models.qwen import MockQwenPlanner
from sam3_vlm.models.sam3 import DummySAM3Sensor
from sam3_vlm.planning.action_bank import ActionBank, ActionBankGenerator
from sam3_vlm.planning.qwen_planner import PlannerOutput, ProposedAction, QwenPlannerService, BudgetExceededError
from sam3_vlm.scene.belief import SemanticMemory
from sam3_vlm.sensing.evidence import ContactSheet, QwenEvidencePack


def experiment_e():
    return _pilot_variants(V4Config())[-1].config


def test_e_preserves_comparison_variants_and_removes_secondary_limits():
    base = load_m8_config(SimpleNamespace(require_cuda=False, output_dir='runs/test')).v4_config
    variants = {v.name: v.config for v in _pilot_variants(base)}
    e = variants['E_Qwen_UntilSaturation']
    assert variants['D_Qwen_TwoRound'] == base
    assert e.budget.max_qwen_calls == 100
    assert e.budget.max_sam3_calls == 1000
    assert e.budget.max_sam3_tiles is None
    assert e.budget.max_runtime_seconds is None
    assert e.stopping.max_iterations is None
    assert e.replanning.max_replans == 99
    assert e.replanning.continue_until_saturation
    assert e.budget.max_cleanup_calls == 0
    assert e.bootstrap == base.bootstrap
    assert all(v.sam3.qwen_prompt_threshold == 0.5 for v in variants.values())


@pytest.mark.parametrize('suggestion', [None, 0.1, 0.5, 0.95])
@pytest.mark.parametrize('config', [None, V4Config()])
def test_qwen_threshold_enforced_independently_of_suggestion(suggestion, config):
    proposal = ProposedAction('target', 'shaded fruit', ActionFamily.DISCOVERY,
                              semantic_prior={'target': 1.0}, suggested_threshold=suggestion)
    entries = ActionBankGenerator().generate_entries(
        PlannerOutput(proposed_actions=[proposal]), SemanticMemory(), ActionBank(),
        IDGenerator(), config=config, enforce_qwen_contract=True,
    )
    assert len(entries) == 1
    assert entries[0].action.threshold == 0.5
    assert proposal.suggested_threshold == suggestion  # Original suggestion stays auditable.


@pytest.mark.parametrize('threshold', [-0.01, 1.01, float('nan')])
def test_qwen_threshold_validation(threshold):
    with pytest.raises(ValueError, match='qwen_prompt_threshold'):
        SAM3Config(qwen_prompt_threshold=threshold)


class NovelPlanner:
    model = 'test-qwen'
    call_count = 0

    def plan_scene(self, evidence, budget, config):
        self.call_count += 1
        # Synthetic unique lexical phrases test controller mechanics, not model vocabulary.
        suffix = chr(97 + (self.call_count // 26)) + chr(97 + (self.call_count % 26))
        return PlannerOutput(proposed_actions=[ProposedAction(
            'target', f'{suffix} fruit', ActionFamily.DISCOVERY,
            semantic_prior={'target': 1.0}, priority=0.0, suggested_threshold=0.99,
            suggested_spatial_mode=SpatialMode.TILED,
        )])


class DiscoveringSensor(DummySAM3Sensor):
    def observe(self, image, action):
        observation = super().observe(image, action)
        if action.source == ActionSource.QWEN:
            assert action.threshold == 0.5
        else:
            assert action.threshold == 0.25  # Bootstrap retains its own threshold.
        x = self.call_count * 8
        observation.detections = [Detection(
            detection_id=f'd{self.call_count}',
            geometry=BoxGeometry(Box(x, 0, x + 4, 4)), score=0.6,
        )]
        # Simulated model time passes the old five-minute limit.
        observation.runtime_ms = 10000
        return observation


def run_e(tmp_path, sensor, config=None):
    config = config or experiment_e()
    config = replace(config, bootstrap=BootstrapConfig(enable_tiled_bootstrap=False),
                     assets_dir=str(tmp_path / 'assets'))
    planner = NovelPlanner()
    paths = RunArtifactPaths(tmp_path / 'run')
    runner, _ = assemble_e2e_runner(paths, config, sensor, planner,
                                    'test', 'green fruit', 'target', 'img')
    runner.run((1000, 1000), 'green fruit', image_id='img')
    assert _run_validator_and_replay(paths, runner.scene_state)
    return runner, planner


def test_e_runs_100_queries_and_replays_past_old_limits(tmp_path):
    runner, planner = run_e(tmp_path, DiscoveringSensor())
    state = runner.scene_state
    assert state.stop_reason == StopReason.QWEN_BUDGET
    assert planner.call_count == state.budget.qwen_calls == 100
    assert state.budget.sam3_calls == 101  # Bootstrap plus one action per Qwen round.
    assert state.iteration == 100
    assert state.replans_executed == 99
    assert state.budget.sam3_tiles > 64


def test_e_stops_on_numerical_saturation_before_caps(tmp_path):
    runner, planner = run_e(tmp_path, DummySAM3Sensor())
    assert runner.scene_state.stop_reason == StopReason.DISCOVERY_AND_UNCERTAINTY_SATURATED
    assert planner.call_count == 2  # A zero-gain first plan must not trip the old utility gate.


def test_e_stops_at_sam3_budget(tmp_path):
    config = experiment_e()
    config = replace(config, budget=replace(config.budget, max_sam3_calls=4))
    runner, planner = run_e(tmp_path, DiscoveringSensor(), config)
    assert runner.scene_state.stop_reason == StopReason.SAM3_BUDGET
    assert runner.scene_state.budget.sam3_calls == 4
    assert planner.call_count == 3


def test_e_empty_plan_during_discovery_only_plateau_is_visible_failure():
    planner = MockQwenPlanner(custom_output=PlannerOutput())
    service = QwenPlannerService(planner)
    pack = QwenEvidencePack('img', 'green fruit', 'target', ContactSheet(),
                            belief_classes=['target'], discovery_diagnostics={'discovery_saturated': True})
    budget = BudgetState()
    assert not service.plan_scene(pack, budget, experiment_e()).proposed_actions
    assert service.last_contract_diagnostic == 'EMPTY_UNSATURATED_PLAN'
    assert budget.qwen_calls == 1
    assert not service.last_repair_attempted and not service.last_fallback_used
    budget.qwen_calls = 100
    with pytest.raises(BudgetExceededError):
        service.plan_scene(pack, budget, experiment_e())
    assert planner.call_count == 1


def test_e_selects_novel_actions_even_with_negative_predicted_utility():
    from sam3_vlm.pipeline.runner import Runner
    from sam3_vlm.scene.state import SceneState
    from sam3_vlm.scene.graph import SceneGraph
    from sam3_vlm.sensing.action import SensingAction

    runner = Runner(experiment_e(), DummySAM3Sensor(), NovelPlanner())
    runner.scene_state = SceneState(image_id='img', user_prompt='green fruit', target_class='target',
                                    graph=SceneGraph(), semantic_memory=SemanticMemory(), action_bank=ActionBank())
    entry = runner.scene_state.action_bank.add_action(SensingAction(
        action_id='a1', semantic_key='target', prompt='shaded fruit', family=ActionFamily.DISCOVERY,
    ))
    runner.utility_evaluator = SimpleNamespace(
        evaluate_utility=lambda *args, **kwargs: SimpleNamespace(total_utility=-10.0))
    assert runner._choose_best_action() is entry
    runner.config = replace(runner.config, replanning=replace(
        runner.config.replanning, continue_until_saturation=False))
    assert runner._choose_best_action() is None
