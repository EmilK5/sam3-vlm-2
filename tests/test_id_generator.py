"""Test IDGenerator deterministic formatting, sequence incrementing, and resetting."""

import pytest
from sam3_vlm.core.id_generator import IDGenerator


def test_id_generator_formatting(id_gen: IDGenerator):
    node_id = id_gen.next_node_id()
    assert node_id == "node_000001"

    sam3_id = id_gen.next_sam3_call_id()
    assert sam3_id == "sam3_000001"

    action_id = id_gen.next_action_id()
    assert action_id == "action_000001"

    qwen_id = id_gen.next_qwen_call_id()
    assert qwen_id == "qwen_000001"


def test_id_generator_incrementing(id_gen: IDGenerator):
    n1 = id_gen.next_node_id()
    n2 = id_gen.next_node_id()
    n3 = id_gen.next_node_id()

    assert n1 == "node_000001"
    assert n2 == "node_000002"
    assert n3 == "node_000003"


def test_id_generator_reset(id_gen: IDGenerator):
    n1 = id_gen.next_node_id()
    assert n1 == "node_000001"

    id_gen.reset()
    n2 = id_gen.next_node_id()
    assert n2 == "node_000001"


def test_custom_prefix(id_gen: IDGenerator):
    custom_id = id_gen.next_id("custom")
    assert custom_id == "custom_000001"
