"""Unit tests for ActionBankGenerator deduplication and semantic key canonicalization (V4 Design Spec §7.1)."""

import pytest
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.core.types import ActionFamily, ActionSource
from sam3_vlm.planning.action_bank import (
    ActionBank,
    ActionBankGenerator,
    canonicalize_semantic_key,
)
from sam3_vlm.planning.qwen_planner import PlannerOutput, ProposedAction
from sam3_vlm.scene.belief import SemanticMemory
from sam3_vlm.sensing.action import SensingAction


def test_canonicalize_semantic_key():
    assert canonicalize_semantic_key("Green Citrus Fruit!") == "green_citrus_fruit"
    assert canonicalize_semantic_key("  leaf--foliage  ") == "leaf_foliage"
    assert canonicalize_semantic_key("") == ""


def test_action_bank_generator_deduplication():
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

    # Qwen proposes 3 actions, 1 of which is a duplicate of memory ('Green Citrus')
    p1 = ProposedAction(
        semantic_key="Green Citrus",
        prompt="green citrus",
        family=ActionFamily.DISCOVERY,
    )
    p2 = ProposedAction(
        semantic_key="leaf_foliage",
        prompt="shiny leaf",
        family=ActionFamily.CONFOUNDER,
    )
    p3 = ProposedAction(
        semantic_key="leaf_foliage",  # Duplicate within same proposal
        prompt="shiny leaf duplicate",
        family=ActionFamily.CONFOUNDER,
    )

    output = PlannerOutput(proposed_actions=[p1, p2, p3])
    entries = generator.generate_entries(output, mem, bank, id_gen)

    # Only p2 ('leaf_foliage') should be added
    assert len(entries) == 1
    assert entries[0].action.semantic_key == "leaf_foliage"
    assert len(bank.entries) == 1
