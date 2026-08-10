"""Unit tests for cross-pass detection association and registration (V4 Design Spec §10)."""

import pytest
from sam3_vlm.core.config import AssociationConfig
from sam3_vlm.core.geometry import Box, BoxGeometry, GeometryRef
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.core.types import Detection, ObservationRelation
from sam3_vlm.scene.association import IoUAssociationPolicy
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.node import Node


def test_iou_association_strong_and_weak_match():
    policy = IoUAssociationPolicy()
    graph = SceneGraph()
    id_gen = IDGenerator()

    # Create initial node at (0, 0, 10, 10)
    node1 = Node(
        node_id="node_000001",
        geometry=BoxGeometry(Box(0.0, 0.0, 10.0, 10.0)),
    )
    graph.add_node(node1)

    # Detection 1: High overlap (IoU close to 1.0) -> STRONG_MATCH
    det1 = Detection(
        detection_id="det_001",
        geometry=GeometryRef(Box(0.0, 0.0, 10.0, 10.0)),
        score=0.95,
    )

    # Detection 2: Moderate overlap (IoU = 0.35) -> WEAK_MATCH
    det2 = Detection(
        detection_id="det_002",
        geometry=GeometryRef(Box(4.0, 0.0, 14.0, 10.0)),
        score=0.60,
    )

    cfg = AssociationConfig(iou_match_threshold=0.5, new_node_iou_threshold=0.3)
    res = policy.associate(
        graph=graph,
        detections=[det1, det2],
        sam3_call_id="sam3_000001",
        action_id="act_000001",
        semantic_key="green_citrus",
        id_gen=id_gen,
        config=cfg,
    )

    assert len(res.matched_observations) == 2
    assert len(res.new_nodes) == 0

    obs1_relation = node1.observations[0].relation
    assert obs1_relation == ObservationRelation.STRONG_MATCH

    obs2_relation = node1.observations[1].relation
    assert obs2_relation == ObservationRelation.WEAK_MATCH


def test_iou_association_unmatched_creates_new_node():
    policy = IoUAssociationPolicy()
    graph = SceneGraph()
    id_gen = IDGenerator()

    # Initial node at (0, 0, 10, 10)
    node1 = Node(
        node_id=id_gen.next_node_id(),
        geometry=BoxGeometry(Box(0.0, 0.0, 10.0, 10.0)),
    )
    graph.add_node(node1)

    # Unmatched detection far away at (50, 50, 60, 60)
    unmatched_det = Detection(
        detection_id="det_999",
        geometry=GeometryRef(Box(50.0, 50.0, 60.0, 60.0)),
        score=0.88,
    )

    res = policy.associate(
        graph=graph,
        detections=[unmatched_det],
        sam3_call_id="sam3_000002",
        action_id="act_000002",
        semantic_key="green_citrus",
        id_gen=id_gen,
    )

    assert len(res.matched_observations) == 0
    assert len(res.new_nodes) == 1
    assert len(graph.active_nodes()) == 2

    new_node = res.new_nodes[0]
    assert new_node.observations[0].relation == ObservationRelation.NEW_DETECTION


def test_ambiguous_association_and_diagnostics():
    policy = IoUAssociationPolicy()
    graph = SceneGraph()
    id_gen = IDGenerator()

    # Two overlapping active nodes
    n1 = Node(node_id="n1", geometry=BoxGeometry(Box(0.0, 0.0, 10.0, 10.0)))
    n2 = Node(node_id="n2", geometry=BoxGeometry(Box(2.0, 0.0, 12.0, 10.0)))
    graph.add_node(n1)
    graph.add_node(n2)

    # Detection overlapping both
    amb_det = Detection(
        detection_id="det_amb",
        geometry=GeometryRef(Box(1.0, 0.0, 11.0, 10.0)),
        score=0.90,
    )

    res = policy.associate(
        graph=graph,
        detections=[amb_det],
        sam3_call_id="sam3_000003",
        action_id="act_000003",
        semantic_key="green_citrus",
        id_gen=id_gen,
    )

    # Should flag ambiguous association and record merge_risk
    assert len(res.matched_observations) == 1
    matched_node_id, obs_ref = res.matched_observations[0]
    assert obs_ref.relation == ObservationRelation.AMBIGUOUS_ASSOCIATION

    matched_node = graph.get_node(matched_node_id)
    assert matched_node is not None
    assert matched_node.diagnostics.merge_risk > 0.0
    assert len(matched_node.diagnostics.ambiguous_with) > 0
