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


def test_action_bank_exact_prompt_deduplication():
    generator = ActionBankGenerator()
    mem = SemanticMemory()
    bank = ActionBank()
    id_gen = IDGenerator()

    # Pre-populate bank with a prompt
    action_in_bank = SensingAction(
        action_id="a0",
        semantic_key="some_key",
        prompt="exact duplicate prompt",
        family=ActionFamily.DISCOVERY,
        source=ActionSource.QWEN,
    )
    bank.add_action(action_in_bank)

    # Qwen proposes an action with a DIFFERENT semantic key but the SAME prompt
    p1 = ProposedAction(
        semantic_key="different_key",
        prompt="  Exact Duplicate Prompt  ",
        family=ActionFamily.DISCOVERY,
    )

    output = PlannerOutput(proposed_actions=[p1])
    entries = generator.generate_entries(output, mem, bank, id_gen)

    # Should be rejected because the exact prompt already exists
    assert len(entries) == 0
    assert len(bank.entries) == 1


def test_action_bank_invalid_exemplar_rejection():
    generator = ActionBankGenerator()
    mem = SemanticMemory()
    bank = ActionBank()
    id_gen = IDGenerator()

    # Qwen proposes an action with valid and invalid exemplars
    p1 = ProposedAction(
        semantic_key="valid_exemplar_action",
        prompt="valid",
        family=ActionFamily.VERIFICATION,
        positive_exemplar_ids=["node_1", "node_2"],
    )
    p2 = ProposedAction(
        semantic_key="invalid_exemplar_action",
        prompt="invalid",
        family=ActionFamily.VERIFICATION,
        positive_exemplar_ids=["node_9999"],
    )

    output = PlannerOutput(proposed_actions=[p1, p2])
    
    # Valid nodes set only contains node_1 and node_2
    valid_nodes = {"node_1", "node_2", "node_3"}
    entries = generator.generate_entries(output, mem, bank, id_gen, valid_node_ids=valid_nodes)

    # p2 should be rejected because node_9999 is not in valid_nodes
    assert len(entries) == 1
    assert entries[0].action.semantic_key == "valid_exemplar_action"
