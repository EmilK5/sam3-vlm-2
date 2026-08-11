"""Integration test simulating deterministic multi-pass synthetic sensing scenario (V4 Exit Criteria M1)."""

import pytest
from sam3_vlm.core.config import AssociationConfig, BeliefConfig
from sam3_vlm.core.geometry import Box, BoxGeometry, GeometryRef
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.core.types import ActionFamily, ActionSource, NodeObservationRef, ObservationRelation, NodeStatus, Detection
from sam3_vlm.scene.association import IoUAssociationPolicy
from sam3_vlm.scene.belief import BeliefUpdater, SemanticMemory
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.state import SceneState
from sam3_vlm.sensing.action import SensingAction


def test_synthetic_multipass_scene_graph_evolution():
    """Simulate a 3-pass synthetic scenario:

    Pass 1: User Bootstrap (Target prompt) -> Creates initial candidates (n1, n2)
    Pass 2: Global Target Sensing -> Matches n1 & n2, strengthens target belief
    Pass 3: Global Confounder Sensing (Leaf prompt) -> Matches n2 strongly (identifying leaf), n1 weakly
    """
    id_gen = IDGenerator()
    graph = SceneGraph()
    assoc_policy = IoUAssociationPolicy()
    belief_updater = BeliefUpdater()
    semantic_memory = SemanticMemory()

    target_cls = "target"
    confounder_cls = "leaf"

    # Pass 1: Bootstrap sensing (detections at d1=(10,10,50,50), d2=(100,100,140,140))
    bootstrap_action = SensingAction(
        action_id=id_gen.next_action_id(),
        semantic_key="green_citrus",
        prompt="green citrus fruit",
        family=ActionFamily.DISCOVERY,
        source=ActionSource.USER_BOOTSTRAP,
    )
    semantic_memory.record_execution(bootstrap_action, "sam3_000001")

    p1_detections = [
        Detection("det_1", GeometryRef(Box(10.0, 10.0, 50.0, 50.0)), score=0.85),
        Detection("det_2", GeometryRef(Box(100.0, 100.0, 140.0, 140.0)), score=0.75),
    ]

    res1 = assoc_policy.associate(
        graph=graph,
        detections=p1_detections,
        sam3_call_id="sam3_000001",
        action_id=bootstrap_action.action_id,
        semantic_key=bootstrap_action.semantic_key,
        id_gen=id_gen,
    )

    assert len(res1.new_nodes) == 2
    assert len(graph.active_nodes()) == 2
    n1_id = res1.new_nodes[0].node_id
    n2_id = res1.new_nodes[1].node_id

    # Update beliefs for Pass 1
    for new_node in res1.new_nodes:
        belief_updater.update_node_belief(
            new_node, bootstrap_action, new_node.observations[0], target_class=target_cls, confounder_class=confounder_cls
        )

    # Pass 2: Follow-up target discovery pass
    pass2_action = SensingAction(
        action_id=id_gen.next_action_id(),
        semantic_key="green_citrus",
        prompt="citrus fruit",
        family=ActionFamily.DISCOVERY,
        source=ActionSource.QWEN,
    )
    semantic_memory.record_execution(pass2_action, "sam3_000002")

    p2_detections = [
        Detection("det_3", GeometryRef(Box(11.0, 10.0, 51.0, 50.0)), score=0.90),
    ]

    res2 = assoc_policy.associate(
        graph=graph,
        detections=p2_detections,
        sam3_call_id="sam3_000002",
        action_id=pass2_action.action_id,
        semantic_key=pass2_action.semantic_key,
        id_gen=id_gen,
    )

    assert len(res2.matched_observations) == 1
    assert len(res2.new_nodes) == 0

    matched_node_ids = {nid for nid, _ in res2.matched_observations}
    for node_id, obs_ref in res2.matched_observations:
        node = graph.get_node(node_id)
        if node:
            belief_updater.update_node_belief(
                node, pass2_action, obs_ref, target_class=target_cls, confounder_class=confounder_cls
            )
            
    # Apply NOT_RETRIEVED for n2 manually since we bypassed Runner
    n2 = graph.get_node(n2_id)
    if n2.node_id not in matched_node_ids:
        obs_ref = NodeObservationRef("o_nr", "call2", pass2_action.action_id, pass2_action.semantic_key, relation=ObservationRelation.NOT_RETRIEVED, score=0.0)
        n2.observations.append(obs_ref)
        belief_updater.update_node_belief(n2, pass2_action, obs_ref, target_class=target_cls, confounder_class=confounder_cls)

    # n1 should have high target belief after Pass 2
    n1 = graph.get_node(n1_id)
    assert n1 is not None and n2 is not None
    assert n1.class_belief.probabilities["target"] > 0.6
    # n2 should have decreased target belief (or around 0.5)
    assert n2.class_belief.probabilities["target"] < 0.7

    # Pass 3: Global Confounder Pass (leaf prompt matches n2 strongly, n1 NOT retrieved)
    pass3_action = SensingAction(
        action_id=id_gen.next_action_id(),
        semantic_key="leaf_foliage",
        prompt="shiny green leaf",
        family=ActionFamily.CONFOUNDER,
        source=ActionSource.QWEN,
        semantic_prior={"leaf": 0.9, "target": 0.1}
    )
    semantic_memory.record_execution(pass3_action, "sam3_000003")

    p3_detections = [
        # Match only n2 (100, 100, 140, 140) strongly
        Detection("det_5", GeometryRef(Box(99.0, 100.0, 139.0, 140.0)), score=0.95),
    ]

    res3 = assoc_policy.associate(
        graph=graph,
        detections=p3_detections,
        sam3_call_id="sam3_000003",
        action_id=pass3_action.action_id,
        semantic_key=pass3_action.semantic_key,
        id_gen=id_gen,
    )

    for node_id, obs_ref in res3.matched_observations:
        node = graph.get_node(node_id)
        if node:
            belief_updater.update_node_belief(
                node, pass3_action, obs_ref, target_class=target_cls, confounder_class=confounder_cls
            )

    # Result: n2 should now be identified as leaf confounder (leaf > target)
    # n1 remains high target probability
    assert n1.class_belief.probabilities["target"] > n1.class_belief.probabilities["leaf"]
    assert n2.class_belief.probabilities["leaf"] > n2.class_belief.probabilities["target"]

    # Diagnostics check
    assert n1.diagnostics.support_count == 2
    assert n2.diagnostics.support_count == 2
    assert n1.diagnostics.independent_semantic_support_count == 1
    assert n2.diagnostics.independent_semantic_support_count == 2

    # Graph JSON serialization test
    json_str = graph.to_json()
    restored_graph = SceneGraph.from_json(json_str)
    assert len(restored_graph.nodes) == 2
    restored_n2 = restored_graph.get_node(n2_id)
    assert restored_n2 is not None
    assert restored_n2.class_belief.probabilities["leaf"] > restored_n2.class_belief.probabilities["target"]
