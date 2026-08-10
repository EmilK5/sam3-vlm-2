"""Unit tests for correlation group derivation and paraphrase redundancy marking (V4 Design Spec §7.1)."""

import pytest
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.core.types import ActionFamily, ActionSource
from sam3_vlm.planning.action_bank import (
    ActionBank,
    ActionBankGenerator,
    derive_correlation_group,
)
from sam3_vlm.planning.qwen_planner import PlannerOutput, ProposedAction
from sam3_vlm.scene.belief import SemanticMemory
from sam3_vlm.sensing.action import SensingAction


def test_derive_correlation_group():
    g1 = derive_correlation_group("round_green_citrus", "round green citrus fruit")
    g2 = derive_correlation_group("spherical_green_citrus", "spherical green fruit")
    assert g1 == "citrus_target"
    assert g2 == "citrus_target"

    g3 = derive_correlation_group("leaf_foliage", "shiny green leaf")
    assert g3 == "leaf_confounder"


def test_action_bank_generator_paraphrase_redundancy():
    generator = ActionBankGenerator()
    mem = SemanticMemory()
    bank = ActionBank()
    id_gen = IDGenerator()

    # Pre-populate memory with 'green_citrus'
    action_executed = SensingAction(
        action_id="a0",
        semantic_key="green_citrus",
        prompt="green citrus fruit",
        family=ActionFamily.DISCOVERY,
        source=ActionSource.USER_BOOTSTRAP,
    )
    mem.record_execution(action_executed, "sam3_1")

    # Qwen proposes a near-paraphrase action in the same correlation group ('spherical_green_citrus')
    p1 = ProposedAction(
        semantic_key="spherical_green_citrus",
        prompt="spherical green fruit",
        family=ActionFamily.DISCOVERY,
    )

    output = PlannerOutput(proposed_actions=[p1])
    entries = generator.generate_entries(output, mem, bank, id_gen)

    assert len(entries) == 1
    # Entry added, but flagged with redundancy=0.5 because it belongs to the same correlation group as memory!
    assert entries[0].redundancy == 0.5
