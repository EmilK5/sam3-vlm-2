"""Unit tests for MockQwenPlanner, PlannerOutput parsing, and call budget enforcement (V4 Design Spec §6)."""

import pytest
from sam3_vlm.core.config import BudgetConfig, V4Config
from sam3_vlm.core.types import ActionFamily, BudgetState, SpatialMode
from sam3_vlm.models.qwen import MockQwenPlanner
from sam3_vlm.planning.qwen_planner import (
    BudgetExceededError,
    PlannerOutput,
    ProposedAction,
    QwenPlannerService,
)
from sam3_vlm.sensing.evidence import ContactSheet, QwenEvidencePack


def test_proposed_action_and_planner_output_serialization():
    pa = ProposedAction(
        semantic_key="small_citrus",
        prompt="small green citrus",
        family=ActionFamily.DISCOVERY,
        priority=0.9,
        semantic_prior={"target": 0.8, "leaf": 0.2},
    )

    po = PlannerOutput(
        scene_summary="Scene summary",
        proposed_actions=[pa],
        missing_appearance_modes=["small_citrus"],
        likely_confounders=["leaf"],
    )

    data = po.to_dict()
    assert data["scene_summary"] == "Scene summary"
    assert len(data["proposed_actions"]) == 1

    restored = PlannerOutput.from_dict(data)
    assert restored.proposed_actions[0].semantic_key == "small_citrus"
    assert restored.proposed_actions[0].family == ActionFamily.DISCOVERY

    json_str = po.to_json()
    assert "small_citrus" in json_str


def test_mock_qwen_planner_proposals():
    planner = MockQwenPlanner()
    cs = ContactSheet(crops=[], total_candidates=2)
    pack = QwenEvidencePack(
        original_image_id="img_001",
        user_prompt="green citrus",
        target_class="green_citrus",
        contact_sheet=cs,
    )

    output = planner.plan_scene(pack)
    assert planner.call_count == 1
    assert len(output.proposed_actions) == 3

    # Check proposed action families
    families = {a.family for a in output.proposed_actions}
    assert ActionFamily.DISCOVERY in families
    assert ActionFamily.CONFOUNDER in families
    assert ActionFamily.VERIFICATION in families


def test_qwen_planner_budget_enforcement():
    planner = MockQwenPlanner()
    service = QwenPlannerService(planner_backend=planner)
    cfg = V4Config(budget=BudgetConfig(max_qwen_calls=2))
    budget = BudgetState(qwen_calls=0)

    cs = ContactSheet(crops=[], total_candidates=2)
    pack = QwenEvidencePack(
        original_image_id="img_001",
        user_prompt="green citrus",
        target_class="green_citrus",
        contact_sheet=cs,
    )

    # Call 1 & 2 succeed
    service.plan_scene(pack, budget, cfg)
    assert budget.qwen_calls == 1

    service.plan_scene(pack, budget, cfg)
    assert budget.qwen_calls == 2

    # Call 3 exceeds max_qwen_calls -> raises BudgetExceededError
    with pytest.raises(BudgetExceededError, match="Qwen call budget exhausted"):
        service.plan_scene(pack, budget, cfg)


def test_valid_default_mock_produces_zero_invalid_entries():
    from sam3_vlm.planning.action_bank import ActionBank, ActionBankGenerator
    from sam3_vlm.scene.belief import SemanticMemory
    from sam3_vlm.core.id_generator import IDGenerator

    planner = MockQwenPlanner()
    cs = ContactSheet(crops=[], total_candidates=2)
    pack = QwenEvidencePack(
        original_image_id="img_001",
        user_prompt="green citrus",
        target_class="green_citrus",
        contact_sheet=cs,
    )

    output = planner.plan_scene(pack)
    generator = ActionBankGenerator()
    mem = SemanticMemory()
    bank = ActionBank()
    id_gen = IDGenerator()

    entries = generator.generate_entries(output, mem, bank, id_gen)
    
    assert len(entries) == 3
    assert len(bank.entries) == 3
    for entry in bank.entries:
        assert entry.invalid_reason is None
