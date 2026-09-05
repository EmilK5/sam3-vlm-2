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


class RepairBackend:
    def __init__(self):
        self.call_count = 0

    def plan_scene(self, evidence, budget, config):
        self.call_count += 1
        if self.call_count == 1:
            return "invalid unparseable json [] {"
        # Second call expects the repair instruction
        assert "Your previous output was invalid" in evidence.user_prompt
        return PlannerOutput(
            scene_summary="Repaired",
            proposed_actions=[
                ProposedAction(
                    semantic_key="repaired",
                    prompt="repaired",
                    family=ActionFamily.DISCOVERY,
                )
            ]
        )


def test_planner_repair_pass():
    backend = RepairBackend()
    service = QwenPlannerService(planner_backend=backend)
    cs = ContactSheet(crops=[], total_candidates=0)
    pack = QwenEvidencePack("img1", "citrus", "citrus", cs)
    budget = BudgetState(qwen_calls=0)

    output = service.plan_scene(pack, budget)

    # Should have called backend twice (initial + 1 repair)
    assert backend.call_count == 2
    assert budget.qwen_calls == 2
    assert output.scene_summary == "Repaired"
    assert len(output.proposed_actions) == 1
    assert output.proposed_actions[0].semantic_key == "repaired"


class FencedJsonBackend:
    def __init__(self):
        self.call_count = 0

    def plan_scene(self, evidence, budget, config):
        self.call_count += 1
        return """```json
{
  "scene_summary": "Initial bootstrap candidates for green citrus target.",
  "missing_appearance_modes": ["green fruit"],
  "likely_confounders": [],
  "proposed_actions": [
    {
      "semantic_key": "target",
      "sam3_prompt": "green fruit",
      "family": "DISCOVERY",
      "priority": 0.8,
      "semantic_prior": {"target": 1.0},
      "suggested_threshold": 0.5,
      "suggested_spatial_mode": "GLOBAL",
      "rationale": "Search for the primary target concept."
    }
  ]
}
```yaml"""


def test_planner_accepts_fenced_json_without_spending_repair_call():
    backend = FencedJsonBackend()
    service = QwenPlannerService(planner_backend=backend)
    pack = QwenEvidencePack(
        "img1",
        "green citrus",
        "target",
        ContactSheet(crops=[], total_candidates=0),
    )
    budget = BudgetState(qwen_calls=0)

    output = service.plan_scene(pack, budget)

    assert backend.call_count == 1
    assert budget.qwen_calls == 1
    assert service.last_repair_attempted is False
    assert output.scene_summary.startswith("Initial bootstrap")
    assert [action.prompt for action in output.proposed_actions] == ["green fruit"]


class TotalFailureBackend:
    def __init__(self):
        self.call_count = 0

    def plan_scene(self, evidence, budget, config):
        self.call_count += 1
        return "invalid unparseable json [] {"


def test_planner_deterministic_fallback():
    backend = TotalFailureBackend()
    service = QwenPlannerService(planner_backend=backend)
    cs = ContactSheet(crops=[], total_candidates=0)
    pack = QwenEvidencePack("img1", "citrus_target", "citrus_target_class", cs)
    budget = BudgetState(qwen_calls=0)

    output = service.plan_scene(pack, budget)

    # Should have called backend twice (initial + 1 repair), both failed
    assert backend.call_count == 2
    assert budget.qwen_calls == 2
    assert "Deterministic fallback due to repeated model failure" in output.scene_summary
    assert len(output.proposed_actions) == 1
    assert output.proposed_actions[0].semantic_key == "target_fallback"
    assert output.proposed_actions[0].prompt == "citrus_target"
