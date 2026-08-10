"""Integration test for Qwen evidence pack -> Planner -> ActionBank pipeline (V4 Design Spec §6 / §7)."""

import pytest
from sam3_vlm.core.config import V4Config
from sam3_vlm.core.geometry import Box, GeometryRef
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.core.types import BudgetState, Detection
from sam3_vlm.models.qwen import MockQwenPlanner
from sam3_vlm.models.sam3 import MockSAM3Adapter
from sam3_vlm.pipeline.bootstrap import BootstrapPipeline
from sam3_vlm.planning.action_bank import ActionBank, ActionBankGenerator
from sam3_vlm.planning.qwen_planner import QwenPlannerService


def test_evidence_pack_to_action_bank_integration():
    """Verify end-to-end integration: Bootstrap -> EvidencePack -> MockQwenPlanner -> ActionBankGenerator -> ActionBank."""
    id_gen = IDGenerator()
    synth_dets = [
        Detection("d1", GeometryRef(Box(10.0, 10.0, 50.0, 50.0)), score=0.88),
    ]
    sensor = MockSAM3Adapter(id_gen=id_gen, synthetic_detections=synth_dets)
    bootstrap_pipeline = BootstrapPipeline(sensor=sensor, id_gen=id_gen)

    # 1. Bootstrap
    b_result = bootstrap_pipeline.execute_bootstrap(
        image_id="img_001",
        image=(1000, 1000),
        user_prompt="count green citrus",
        target_class="green_citrus",
    )
    evidence_pack = b_result.qwen_evidence_pack
    scene_state = b_result.state

    # 2. Qwen Planning Pass
    mock_qwen = MockQwenPlanner()
    planner_service = QwenPlannerService(planner_backend=mock_qwen)
    cfg = V4Config()

    planner_output = planner_service.plan_scene(evidence_pack, scene_state.budget, cfg)
    assert scene_state.budget.qwen_calls == 1
    assert len(planner_output.proposed_actions) == 3

    # 3. Action Bank Generation & Deduplication
    action_bank = ActionBank()
    generator = ActionBankGenerator()
    new_entries = generator.generate_entries(
        planner_output=planner_output,
        semantic_memory=scene_state.semantic_memory,
        action_bank=action_bank,
        id_gen=id_gen,
    )

    # 4. Verify results in ActionBank
    # Target class 'green_citrus' was executed in bootstrap, so actions matching 'green_citrus' are deduplicated
    assert len(action_bank.entries) >= 2
    for entry in action_bank.entries:
        assert entry.action.source.value == "QWEN"
        assert entry.qwen_priority is not None
