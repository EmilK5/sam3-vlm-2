"""Unit tests for defensive Qwen planner output validation, priority/threshold clamping, and action capping (V4 Design Spec §6.4)."""

import pytest
from sam3_vlm.core.config import V4Config
from sam3_vlm.core.types import ActionFamily, BudgetState, SpatialMode
from sam3_vlm.planning.qwen_planner import (
    PlannerOutput,
    ProposedAction,
    QwenPlannerService,
)
from sam3_vlm.sensing.evidence import ContactSheet, QwenEvidencePack


class FaultyBackend:
    def plan_scene(self, evidence, budget, config):
        raise RuntimeError("Model network error")


class OverflownBackend:
    def plan_scene(self, evidence, budget, config):
        # Returns 7 proposed actions with out-of-bound priorities & thresholds
        actions = [
            ProposedAction(
                semantic_key=f"act_{i}",
                prompt=f"prompt_{i}",
                family=ActionFamily.DISCOVERY,
                priority=1.5 if i == 0 else (0.1 * i),
                suggested_threshold=-0.5 if i == 0 else 1.2,
                semantic_prior={"target": 2.0},
            )
            for i in range(7)
        ]
        return PlannerOutput(scene_summary="Overflown", proposed_actions=actions)


def test_planner_defensive_clamping_and_action_capping():
    service = QwenPlannerService(planner_backend=OverflownBackend())
    cs = ContactSheet(crops=[], total_candidates=0)
    pack = QwenEvidencePack("img1", "citrus", "citrus", cs)
    budget = BudgetState(qwen_calls=0)

    output = service.plan_scene(pack, budget)

    # Actions must be capped at 5 max
    assert len(output.proposed_actions) == 5

    # Priority, threshold, and priors must be clamped to [0, 1]
    top_action = output.proposed_actions[0]
    assert top_action.priority == 1.0
    assert top_action.suggested_threshold >= 0.0 and top_action.suggested_threshold <= 1.0
    assert top_action.semantic_prior["target"] == 1.0


def test_planner_defensive_error_fallback():
    service = QwenPlannerService(planner_backend=FaultyBackend())
    cs = ContactSheet(crops=[], total_candidates=0)
    pack = QwenEvidencePack("img1", "citrus", "citrus", cs)
    budget = BudgetState(qwen_calls=0)

    output = service.plan_scene(pack, budget)

    assert "Model call failed" in output.scene_summary
    assert len(output.proposed_actions) == 0
