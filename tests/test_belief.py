"""Unit tests for Bayesian belief updates, presence/absence asymmetry, and entropy (V4 Design Spec §11)."""

import pytest
from sam3_vlm.core.geometry import Box, BoxGeometry
from sam3_vlm.core.types import ActionFamily, NodeObservationRef, ObservationRelation
from sam3_vlm.scene.belief import BeliefUpdater, SemanticMemory
from sam3_vlm.scene.node import Node
from sam3_vlm.sensing.action import SensingAction


def test_entropy_calculation():
    updater = BeliefUpdater()

    # Uniform distribution over 2 classes -> H = 1.0
    h_uniform = updater.calculate_entropy({"c1": 0.5, "c2": 0.5})
    assert pytest.approx(h_uniform, abs=1e-4) == 1.0

    # Deterministic distribution -> H = 0.0
    h_certain = updater.calculate_entropy({"c1": 1.0, "c2": 0.0})
    assert pytest.approx(h_certain, abs=1e-4) == 0.0


def test_target_discovery_belief_update():
    updater = BeliefUpdater()
    node = Node(
        node_id="n1",
        geometry=BoxGeometry(Box(0.0, 0.0, 10.0, 10.0)),
    )

    action = SensingAction(
        action_id="a1",
        semantic_key="green_citrus",
        prompt="green citrus fruit",
        family=ActionFamily.DISCOVERY,
    )

    obs_ref = NodeObservationRef(
        observation_id="o1",
        sam3_call_id="sam3_1",
        action_id="a1",
        semantic_key="green_citrus",
        relation=ObservationRelation.STRONG_MATCH,
        score=0.9,
    )

    updater.update_node_belief(node=node, action=action, obs_ref=obs_ref)

    assert node.class_belief.update_count == 1
    assert node.class_belief.probabilities["target"] > 0.5
    assert node.class_belief.probabilities["target"] > node.class_belief.probabilities["leaf"]


def test_confounder_prompt_belief_update():
    updater = BeliefUpdater()
    node = Node(
        node_id="n1",
        geometry=BoxGeometry(Box(0.0, 0.0, 10.0, 10.0)),
    )

    action = SensingAction(
        action_id="a2",
        semantic_key="green_leaf",
        prompt="green leaf foliage",
        family=ActionFamily.CONFOUNDER,
    )

    obs_ref = NodeObservationRef(
        observation_id="o2",
        sam3_call_id="sam3_2",
        action_id="a2",
        semantic_key="green_leaf",
        relation=ObservationRelation.STRONG_MATCH,
        score=0.95,
    )

    updater.update_node_belief(node=node, action=action, obs_ref=obs_ref)

    # Strong match on leaf prompt should increase leaf probability relative to target
    assert node.class_belief.probabilities["leaf"] > node.class_belief.probabilities["target"]


def test_semantic_memory_tracking():
    mem = SemanticMemory()
    action = SensingAction(
        action_id="a1",
        semantic_key="green_citrus",
        prompt="green citrus fruit",
        family=ActionFamily.DISCOVERY,
    )

    rec = mem.record_execution(action, sam3_call_id="sam3_1")
    assert rec.execution_count == 1
    assert "sam3_1" in rec.sam3_call_ids
    assert "green_citrus" in mem.records
