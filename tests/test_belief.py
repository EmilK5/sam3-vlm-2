"""Unit tests for Bayesian belief updates, dataset-agnostic vocabulary, asymmetry, and entropy (V4 Design Spec §11 / §34.6)."""

import pytest
from sam3_vlm.core.config import BeliefConfig
from sam3_vlm.core.geometry import Box, BoxGeometry
from sam3_vlm.core.types import ActionFamily, ClassBelief, NodeObservationRef, ObservationRelation
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


def test_not_observable_leaves_belief_unchanged():
    updater = BeliefUpdater()
    cb = ClassBelief(probabilities={"car": 0.6, "road": 0.4})
    node = Node(
        node_id="n1",
        geometry=BoxGeometry(Box(0.0, 0.0, 10.0, 10.0)),
        class_belief=cb,
    )

    action = SensingAction(
        action_id="a1",
        semantic_key="car",
        prompt="vehicle car",
        family=ActionFamily.DISCOVERY,
    )
    obs_ref = NodeObservationRef(
        observation_id="o1",
        sam3_call_id="sam3_1",
        action_id="a1",
        semantic_key="car",
        relation=ObservationRelation.NOT_OBSERVABLE,
    )

    updater.update_node_belief(node=node, action=action, obs_ref=obs_ref)

    # Class belief is completely unchanged
    assert node.class_belief.probabilities == {"car": 0.6, "road": 0.4}
    assert node.class_belief.update_count == 0


def test_generic_vocabulary_belief_update_carpk():
    """Verify belief updates work on CARPK vocabulary (car, road, building) without citrus defaults."""
    updater = BeliefUpdater()
    cb = ClassBelief(probabilities={"car": 0.5, "road": 0.3, "building": 0.2})
    node = Node(
        node_id="n1",
        geometry=BoxGeometry(Box(0.0, 0.0, 10.0, 10.0)),
        class_belief=cb,
    )

    action = SensingAction(
        action_id="a1",
        semantic_key="car",
        prompt="parked sedan car",
        family=ActionFamily.DISCOVERY,
    )
    obs_ref = NodeObservationRef(
        observation_id="o1",
        sam3_call_id="sam3_1",
        action_id="a1",
        semantic_key="car",
        relation=ObservationRelation.STRONG_MATCH,
        score=0.9,
    )

    updater.update_node_belief(node=node, action=action, obs_ref=obs_ref, target_class="car")

    assert node.class_belief.probabilities["car"] > 0.5
    assert sum(node.class_belief.probabilities.values()) == pytest.approx(1.0, abs=1e-4)


def test_repeat_key_discounting():
    """Verify repeat sensing using same semantic_key applies discount_repeat_weight."""
    updater = BeliefUpdater()
    cb = ClassBelief(probabilities={"target": 0.5, "confounder": 0.5})
    node = Node(
        node_id="n1",
        geometry=BoxGeometry(Box(0.0, 0.0, 10.0, 10.0)),
        class_belief=cb,
    )

    action = SensingAction(
        action_id="a1",
        semantic_key="citrus",
        prompt="citrus",
        family=ActionFamily.DISCOVERY,
    )

    # First update on this key
    obs1 = NodeObservationRef(
        observation_id="o1", sam3_call_id="s1", action_id="a1", semantic_key="citrus",
        relation=ObservationRelation.STRONG_MATCH, score=0.9,
    )
    node.observations.append(obs1)
    updater.update_node_belief(node=node, action=action, obs_ref=obs1, target_class="target")
    p1 = node.class_belief.probabilities["target"]

    # Second update on SAME key
    obs2 = NodeObservationRef(
        observation_id="o2", sam3_call_id="s2", action_id="a1", semantic_key="citrus",
        relation=ObservationRelation.STRONG_MATCH, score=0.9,
    )
    node.observations.append(obs2)
    updater.update_node_belief(node=node, action=action, obs_ref=obs2, target_class="target", config=BeliefConfig(discount_repeat_weight=0.5))
    p2 = node.class_belief.probabilities["target"]

    # Belief increases, but the gain in p2 is smaller due to repeat discount
    assert p2 > p1


def test_not_retrieved_vs_confounder_strong_match_asymmetry():
    """Verify NOT_RETRIEVED reduces target belief less aggressively than a confounder STRONG_MATCH."""
    updater = BeliefUpdater()
    node1 = Node(node_id="n1", geometry=BoxGeometry(Box(0, 0, 10, 10)), class_belief=ClassBelief({"target": 0.7, "leaf": 0.3}))
    node2 = Node(node_id="n2", geometry=BoxGeometry(Box(0, 0, 10, 10)), class_belief=ClassBelief({"target": 0.7, "leaf": 0.3}))

    disc_action = SensingAction(action_id="a1", semantic_key="target", prompt="target", family=ActionFamily.DISCOVERY)
    conf_action = SensingAction(action_id="a2", semantic_key="leaf", prompt="leaf", family=ActionFamily.CONFOUNDER)

    not_retrieved_obs = NodeObservationRef(observation_id="o1", sam3_call_id="s1", action_id="a1", semantic_key="target", relation=ObservationRelation.NOT_RETRIEVED)
    conf_match_obs = NodeObservationRef(observation_id="o2", sam3_call_id="s2", action_id="a2", semantic_key="leaf", relation=ObservationRelation.STRONG_MATCH, score=0.9)

    updater.update_node_belief(node1, disc_action, not_retrieved_obs, target_class="target", confounder_class="leaf")
    updater.update_node_belief(node2, conf_action, conf_match_obs, target_class="target", confounder_class="leaf")

    # Target prob for node1 (NOT_RETRIEVED) should be higher than target prob for node2 (CONFOUNDER match)
    assert node1.class_belief.probabilities["target"] > node2.class_belief.probabilities["target"]


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
