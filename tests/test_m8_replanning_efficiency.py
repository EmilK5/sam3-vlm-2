"""Regression tests for bounded, evidence-driven M8 replanning."""

from types import SimpleNamespace

import pytest

from sam3_vlm.core.config import (
    BeliefConfig,
    BudgetConfig,
    ReplanningConfig,
    V4Config,
)
from sam3_vlm.core.geometry import Box, BoxGeometry
from sam3_vlm.core.types import ActionFamily, ClassBelief, SpatialMode, StopReason
from sam3_vlm.models.sam3 import MockSAM3Adapter
from sam3_vlm.pipeline.runner import Runner, RunnerState
from sam3_vlm.planning.action_bank import ActionBank, ActionBankEntry
from sam3_vlm.planning.qwen_planner import PlannerOutput, ProposedAction
from sam3_vlm.planning.replanning import ReplanEvidenceBuilder
from sam3_vlm.planning.utility import DefaultUtilityEvaluator
from sam3_vlm.scene.belief import SemanticMemory
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.node import Node
from sam3_vlm.scene.state import CountEstimator, SceneState
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.sensing.evidence import ContactSheet
from sam3_vlm.sensing.evidence import QwenEvidencePack
from sam3_vlm.sensing.observation import SAM3Observation


class _NoopPlanner:
    model = "noop-qwen"

    def __init__(self):
        self.call_count = 0

    def plan_scene(self, evidence, budget, config):
        self.call_count += 1
        return PlannerOutput(scene_summary="noop", proposed_actions=[])


def _canonical_state() -> SceneState:
    return SceneState(
        image_id="img",
        user_prompt="green fruit",
        target_class="target",
        graph=SceneGraph(),
        semantic_memory=SemanticMemory(),
        action_bank=ActionBank(),
        belief_classes=["target", "confounder1", "confounder2"],
    )


def _target_action(action_id: str, prompt: str, mode=SpatialMode.GLOBAL):
    return SensingAction(
        action_id=action_id,
        semantic_key="target",
        prompt=prompt,
        family=ActionFamily.DISCOVERY,
        spatial_mode=mode,
        tiling=V4Config().tiling if mode == SpatialMode.TILED else None,
        semantic_prior={"target": 1.0},
        correlation_group="target",
    )


def test_recent_zero_gain_discovery_overrides_early_success():
    state = _canonical_state()
    memory = state.semantic_memory
    memory.record_execution(_target_action("a1", "green fruit"), "s1", new_nodes=18)
    memory.record_execution(_target_action("a2", "small green fruit"), "s2", new_nodes=2)
    memory.record_execution(_target_action("a3", "shaded green fruit"), "s3", new_nodes=0)
    memory.record_execution(_target_action("a4", "round green fruit"), "s4", new_nodes=0)
    state.discovery_state.recent_new_node_counts = [18.0, 2.0, 0.0, 0.0]
    state.discovery_state.saturated = True

    candidate = _target_action("a5", "occluded green fruit")
    entry = ActionBankEntry(action=candidate, qwen_priority=0.9, redundancy=0.5)
    utility = DefaultUtilityEvaluator().evaluate_utility(entry, state, V4Config())

    assert utility.discovery_value == 0.0
    assert utility.discrimination_value == 0.0
    assert utility.total_utility < V4Config().stopping.utility_min_threshold


def test_discovery_plateau_keeps_target_only_uncertainty_value():
    state = _canonical_state()
    memory = state.semantic_memory
    memory.record_execution(_target_action("a1", "green fruit"), "s1", new_nodes=18)
    memory.record_execution(_target_action("a2", "shaded green fruit"), "s2", new_nodes=0)
    memory.record_execution(_target_action("a3", "round green fruit"), "s3", new_nodes=0)
    state.discovery_state.saturated = True
    node = Node(
        node_id="n1",
        geometry=BoxGeometry(Box(0, 0, 10, 10)),
        class_belief=ClassBelief(
            probabilities={
                "target": 1 / 3,
                "confounder1": 1 / 3,
                "confounder2": 1 / 3,
            },
            entropy=1.5,
        ),
    )
    state.graph.add_node(node)

    target_experiment = SensingAction(
        action_id="a4",
        semantic_key="target",
        prompt="dark green fruit",
        family=ActionFamily.DISCOVERY,
        semantic_prior={"target": 1.0},
        correlation_group="target",
    )
    entry = ActionBankEntry(
        action=target_experiment,
        qwen_priority=0.9,
        redundancy=0.5,
    )
    utility = DefaultUtilityEvaluator().evaluate_utility(entry, state, V4Config())

    assert utility.discovery_value == 0.0
    assert utility.discrimination_value > 0.0
    assert utility.total_utility >= V4Config().stopping.utility_min_threshold


def test_replan_evidence_lists_exact_execution_history_and_saturation():
    state = _canonical_state()
    state.semantic_memory.record_execution(
        _target_action("a1", "green fruit"),
        "s1",
        new_nodes=18,
        affected_nodes=18,
        entropy_change=-0.4,
        variance_change=-0.2,
    )
    state.semantic_memory.record_execution(
        _target_action("a2", "shadowed green fruit", SpatialMode.TILED),
        "s2",
        new_nodes=0,
        affected_nodes=20,
        entropy_change=-0.1,
        variance_change=-0.01,
    )
    state.discovery_state.recent_new_node_counts = [0.0, 0.0]
    state.discovery_state.spatial_coverage.coverage_ratio = 1.0

    contact_sheet_builder = SimpleNamespace(
        build_contact_sheet=lambda **kwargs: ContactSheet()
    )
    pack = ReplanEvidenceBuilder(contact_sheet_builder).build(
        state,
        config=V4Config(
            replanning=ReplanningConfig(discovery_plateau_steps=2)
        ),
    )

    assert "tried_prompts=['green fruit', 'shadowed green fruit']" in pack.scene_summary
    assert "sam3_prompt='shadowed green fruit'" in pack.scene_summary
    assert "family=DISCOVERY" in pack.scene_summary
    assert "spatial_mode=TILED" in pack.scene_summary
    assert "new_nodes=0" in pack.scene_summary
    assert "affected_nodes=20" in pack.scene_summary
    assert pack.discovery_diagnostics["discovery_saturated"] is True
    assert pack.discovery_diagnostics["tried_sam3_prompts"] == [
        "green fruit",
        "shadowed green fruit",
    ]


def test_replan_blacklist_includes_accepted_unexecuted_prompts():
    state = _canonical_state()
    state.action_bank.add_action(
        _target_action("a1", "occluded green fruit")
    )
    contact_sheet_builder = SimpleNamespace(
        build_contact_sheet=lambda **kwargs: ContactSheet()
    )

    pack = ReplanEvidenceBuilder(contact_sheet_builder).build(
        state,
        config=V4Config(),
    )

    assert pack.discovery_diagnostics["tried_sam3_prompts"] == [
        "occluded green fruit"
    ]


def test_unproductive_replan_stops_before_another_qwen_call():
    planner = _NoopPlanner()
    runner = Runner(V4Config(), MockSAM3Adapter(), planner)
    runner.scene_state = _canonical_state()
    runner.target_class = "target"
    action = _target_action("a7", "shaded green fruit")
    entry = runner.scene_state.action_bank.add_action(action)
    entry.executed = True
    runner.scene_state.semantic_memory.record_execution(
        action,
        "s7",
        new_nodes=0,
        affected_nodes=20,
        entropy_change=-0.01,
        variance_change=-0.01,
        realized_discrimination_proxy=0.01,
    )
    runner.scene_state.count_estimate.variance = 1.0
    runner.scene_state.last_plan_action_ids = ["a7"]
    runner.scene_state.last_plan_accepted_actions = 1
    runner.scene_state.actions_since_replan = 1
    runner.scene_state.replans_executed = 1

    runner._request_replan()

    assert planner.call_count == 0
    assert runner.scene_state.budget.qwen_calls == 0
    assert runner.scene_state.stop_reason == StopReason.LOW_MARGINAL_UTILITY
    assert runner.state == RunnerState.CLEANUP


@pytest.mark.parametrize(
    ("new_nodes", "variance_change"),
    [(1, 0.0), (0, -0.03)],
)
def test_productive_target_plan_allows_replanning(new_nodes, variance_change):
    runner = Runner(V4Config(), MockSAM3Adapter(), _NoopPlanner())
    runner.scene_state = _canonical_state()
    action = _target_action("a8", "occluded green fruit")
    entry = runner.scene_state.action_bank.add_action(action)
    entry.executed = True
    runner.scene_state.semantic_memory.record_execution(
        action,
        "s8",
        new_nodes=new_nodes,
        affected_nodes=20,
        variance_change=variance_change,
    )
    runner.scene_state.count_estimate.variance = 0.97
    runner.scene_state.last_plan_action_ids = ["a8"]

    assert runner._last_plan_had_marginal_value() is True


def test_end_to_end_unproductive_target_plan_stops_after_one_qwen_call():
    class EmptySensor:
        def __init__(self):
            self.call_count = 0

        def observe(self, image, action):
            self.call_count += 1
            return SAM3Observation(
                call_id=f"sam{self.call_count}",
                action_id=action.action_id,
                semantic_key=action.semantic_key,
                detections=[],
                searched_regions=[],
                runtime_ms=1.0,
            )

    class TwoRoundPlanner:
        model = "mock-qwen"

        def __init__(self):
            self.call_count = 0

        def plan_scene(self, evidence, budget, config):
            self.call_count += 1
            prompt = (
                "dark green fruit"
                if self.call_count == 1
                else "shaded green fruit"
            )
            return PlannerOutput(
                proposed_actions=[
                    ProposedAction(
                        semantic_key="target",
                        prompt=prompt,
                        family=ActionFamily.DISCOVERY,
                        priority=0.9,
                        semantic_prior={"target": 1.0},
                    )
                ]
            )

    config = V4Config(
        budget=BudgetConfig(max_qwen_calls=2, max_cleanup_calls=0),
        replanning=ReplanningConfig(max_replans=1, min_actions_between_replans=0),
    )
    sensor = EmptySensor()
    planner = TwoRoundPlanner()
    runner = Runner(config, sensor, planner)
    runner.state = RunnerState.PLAN
    runner.scene_state = _canonical_state()
    runner.evidence_pack = QwenEvidencePack(
        original_image_id="img",
        user_prompt="green fruit",
        target_class="target",
        contact_sheet=ContactSheet(),
        belief_classes=["target", "confounder1", "confounder2"],
    )

    runner.run("mock", "green fruit", image_id="img")

    assert planner.call_count == 1
    assert runner.scene_state.budget.qwen_calls == 1
    assert sensor.call_count == 1
    assert runner.scene_state.replans_executed == 0
    assert runner.scene_state.stop_reason == StopReason.LOW_MARGINAL_UTILITY


def test_replan_event_keeps_policy_reason(monkeypatch):
    recorded = []
    recorder = SimpleNamespace(
        record_replan_triggered=lambda reason: recorded.append(reason)
    )
    runner = Runner(V4Config(), MockSAM3Adapter(), _NoopPlanner(), recorder=recorder)
    runner.scene_state = _canonical_state()
    runner._pending_replan_reason = "DISCOVERY_PLATEAU_WITH_COUNT_VARIANCE"
    runner.state = RunnerState.REPLAN
    monkeypatch.setattr(runner, "_request_replan", lambda: None)

    runner._step()

    assert recorded == ["DISCOVERY_PLATEAU_WITH_COUNT_VARIANCE"]


def test_count_commit_threshold_changes_contribution_not_posterior():
    graph = SceneGraph()
    for node_id, probability in (("n1", 0.90), ("n2", 0.89)):
        node = Node(node_id=node_id, geometry=BoxGeometry(Box(0, 0, 10, 10)))
        node.class_belief = ClassBelief(
            probabilities={"target": probability, "confounder1": 1.0 - probability}
        )
        graph.add_node(node)

    estimate = CountEstimator.estimate(
        graph,
        "target",
        target_commit_threshold=0.90,
    )

    assert estimate.mean_count == pytest.approx(1.89)
    assert estimate.raw_soft_count == pytest.approx(1.79)
    assert estimate.committed_node_count == 1
    assert estimate.variance == pytest.approx(0.90 * 0.10 + 0.89 * 0.11)
    assert graph.get_node("n1").class_belief.probabilities["target"] == 0.90


def test_count_commit_threshold_is_validated_before_a_run():
    with pytest.raises(ValueError, match="target_count_commit_threshold"):
        BeliefConfig(target_count_commit_threshold=0.0)

    with pytest.raises(ValueError, match="target_count_commit_threshold"):
        BeliefConfig(target_count_commit_threshold=1.01)


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_relative_variance_reduction_threshold_is_validated(threshold):
    with pytest.raises(
        ValueError,
        match="min_relative_count_variance_reduction",
    ):
        ReplanningConfig(
            min_relative_count_variance_reduction=threshold,
        )
