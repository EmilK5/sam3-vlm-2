"""Regression tests for strict M8 controller semantics."""

from types import SimpleNamespace

import pytest

from sam3_vlm.core.config import V4Config
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.core.types import ActionFamily, SpatialMode, StopReason
from sam3_vlm.models.sam3 import MockSAM3Adapter
from sam3_vlm.models.qwen import MockQwenPlanner
from sam3_vlm.pipeline.runner import Runner, RunnerState
from sam3_vlm.planning.action_bank import (
    ActionBank,
    ActionBankGenerator,
    ActionRejectionReason,
)
from sam3_vlm.planning.qwen_planner import PlannerOutput, ProposedAction
from sam3_vlm.scene.belief import SemanticMemory
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.state import SceneState
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.sensing.evidence import ContactSheet, QwenEvidencePack


class _NoopPlanner:
    model = "noop-qwen"

    def plan_scene(self, evidence, budget, config):
        return PlannerOutput(scene_summary="noop", proposed_actions=[])


def _strict_m8_runner() -> Runner:
    runner = Runner(V4Config(), MockSAM3Adapter(), _NoopPlanner())
    runner.scene_state = SceneState(
        image_id="img",
        user_prompt="green fruit",
        target_class="target",
        graph=SceneGraph(),
        semantic_memory=SemanticMemory(),
        belief_classes=["target", "confounder1", "confounder2"],
    )
    runner.target_class = "target"
    return runner


def _target_proposal(prompt: str, priority: float = 0.8) -> ProposedAction:
    return ProposedAction(
        semantic_key="target",
        prompt=prompt,
        family=ActionFamily.DISCOVERY,
        priority=priority,
        semantic_prior={"target": 0.8},
        suggested_spatial_mode=SpatialMode.TILED,
    )


def test_mock_qwen_uses_the_target_only_m8_contract():
    output = MockQwenPlanner().plan_scene(
        QwenEvidencePack(
            original_image_id="img",
            user_prompt="green fruit",
            target_class="target",
            contact_sheet=ContactSheet(),
            belief_classes=["target", "confounder1", "confounder2"],
        )
    )

    assert len(output.proposed_actions) == 1
    action = output.proposed_actions[0]
    assert action.semantic_key == "target"
    assert action.family == ActionFamily.DISCOVERY
    assert action.semantic_prior == {"target": 1.0}


@pytest.mark.parametrize(
    "proposal",
    [
        ProposedAction(
            semantic_key="round_green_fruit",
            prompt="round green fruit",
            family=ActionFamily.DISCOVERY,
            semantic_prior={"target": 1.0},
        ),
        ProposedAction(
            semantic_key="target",
            prompt="flat green leaf",
            family=ActionFamily.CONFOUNDER,
            semantic_prior={"confounder1": 1.0},
        ),
        ProposedAction(
            semantic_key="target",
            prompt="round green fruit",
            family=ActionFamily.VERIFICATION,
            semantic_prior={"target": 1.0},
        ),
        ProposedAction(
            semantic_key="target",
            prompt="flat green leaf",
            family=ActionFamily.DISCOVERY,
            semantic_prior={"confounder1": 1.0},
        ),
    ],
)
def test_strict_m8_rejects_non_target_experiments(proposal):
    generator = ActionBankGenerator()
    entries = generator.generate_entries(
        PlannerOutput(proposed_actions=[proposal]),
        SemanticMemory(),
        ActionBank(),
        IDGenerator(),
        config=V4Config(),
        enforce_qwen_contract=True,
        allowed_belief_classes=["target", "confounder1", "confounder2"],
    )

    assert entries == []
    assert generator.last_rejections[0].reason == (
        ActionRejectionReason.NON_TARGET_ACTION.value
    )


def test_strict_m8_positive_discovery_zero_utility_is_unmeasured_history():
    memory = SemanticMemory()
    memory.records["target"] = SimpleNamespace(
        semantic_keys=["target"],
        prompts=["green fruit"],
        realized_utility_by_execution=[0.0, 0.0],
        new_nodes_by_execution=[21, 5],
    )
    generator = ActionBankGenerator()

    entries = generator.generate_entries(
        PlannerOutput(proposed_actions=[_target_proposal("occluded green fruit")]),
        memory,
        ActionBank(),
        IDGenerator(),
        config=V4Config(),
        enforce_qwen_contract=True,
        allowed_belief_classes=["target", "confounder1", "confounder2"],
    )

    assert len(entries) == 1
    assert entries[0].qwen_priority == pytest.approx(0.8)


def test_strict_m8_keeps_priority_for_a_novel_target_prompt():
    memory = SemanticMemory()
    memory.records["target"] = SimpleNamespace(
        semantic_keys=["target"],
        prompts=["green fruit", "shadowed fruit"],
        realized_utility_by_execution=[0.0, 0.0, 0.0],
        new_nodes_by_execution=[21, 5, 0],
    )
    generator = ActionBankGenerator()

    entries = generator.generate_entries(
        PlannerOutput(proposed_actions=[_target_proposal("occluded green fruit")]),
        memory,
        ActionBank(),
        IDGenerator(),
        config=V4Config(),
        enforce_qwen_contract=True,
        allowed_belief_classes=["target", "confounder1", "confounder2"],
    )

    assert len(entries) == 1
    assert entries[0].qwen_priority == pytest.approx(0.8)


def test_generic_history_keeps_frozen_zero_utility_behavior():
    memory = SemanticMemory()
    memory.records["target"] = SimpleNamespace(
        semantic_keys=["target"],
        prompts=["green fruit"],
        realized_utility_by_execution=[0.0],
        new_nodes_by_execution=[21],
    )
    proposal = ProposedAction(
        semantic_key="target_variant",
        prompt="occluded green fruit",
        family=ActionFamily.DISCOVERY,
        priority=0.8,
        suggested_spatial_mode=SpatialMode.GLOBAL,
        correlation_group="target",
    )

    entries = ActionBankGenerator().generate_entries(
        PlannerOutput(proposed_actions=[proposal]),
        memory,
        ActionBank(),
        IDGenerator(),
        config=V4Config(),
        enforce_qwen_contract=False,
    )

    assert len(entries) == 1
    assert entries[0].qwen_priority == pytest.approx(0.08)


def test_strict_m8_low_utility_plan_stops_without_evidence_free_replan(monkeypatch):
    runner = _strict_m8_runner()
    runner.scene_state.action_bank = ActionBank()
    runner.scene_state.last_plan_accepted_actions = 1
    runner.scene_state.actions_since_replan = 0
    runner.scene_state.budget.qwen_calls = 2
    runner.state = RunnerState.GLOBAL_SENSING

    monkeypatch.setattr(runner, "_choose_best_action", lambda: None)
    runner._step()

    assert runner.state == RunnerState.CLEANUP
    assert runner.scene_state.stop_reason == StopReason.ACTION_BANK_EXHAUSTED
    assert runner.scene_state.budget.qwen_calls == 2
