"""Unit tests for SceneGraph operations and serialization (V4 Design Spec §3.2 / §16.1)."""

import pytest
from sam3_vlm.core.geometry import Box, BoxGeometry
from sam3_vlm.core.types import ClassBelief, NodeStatus
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.node import Node


def test_node_creation_and_properties():
    geom = BoxGeometry(Box(10.0, 20.0, 50.0, 80.0))
    cb = ClassBelief(probabilities={"target": 0.8, "leaf": 0.2})
    node = Node(node_id="node_000001", geometry=geom, class_belief=cb)

    assert node.node_id == "node_000001"
    assert node.status == NodeStatus.ACTIVE
    assert node.class_belief.probabilities["target"] == 0.8


def test_scene_graph_operations():
    graph = SceneGraph()
    geom1 = BoxGeometry(Box(0.0, 0.0, 10.0, 10.0))
    geom2 = BoxGeometry(Box(20.0, 20.0, 30.0, 30.0))

    n1 = Node(node_id="n1", geometry=geom1)
    n2 = Node(node_id="n2", geometry=geom2)

    graph.add_node(n1)
    graph.add_node(n2)

    assert len(graph.nodes) == 2
    assert len(graph.active_nodes()) == 2

    # Resolve n1
    graph.resolve_node("n1")
    assert n1.status == NodeStatus.RESOLVED
    assert len(graph.active_nodes()) == 1

    # Reject n2
    graph.reject_node("n2", reason="low confidence")
    assert n2.status == NodeStatus.REJECTED
    assert len(graph.active_nodes()) == 0


def test_scene_graph_merge_nodes():
    graph = SceneGraph()
    geom1 = BoxGeometry(Box(0.0, 0.0, 10.0, 10.0))
    geom2 = BoxGeometry(Box(2.0, 2.0, 10.0, 10.0))

    n1 = Node(node_id="n1", geometry=geom1)
    n2 = Node(node_id="n2", geometry=geom2)

    graph.add_node(n1)
    graph.add_node(n2)

    merged_primary = graph.merge_nodes("n1", ["n2"])

    assert merged_primary.node_id == "n1"
    assert n2.status == NodeStatus.REJECTED
    assert len(graph.active_nodes()) == 1


def test_graph_serialization_dict_roundtrip():
    graph = SceneGraph()
    geom = BoxGeometry(Box(10.0, 10.0, 50.0, 50.0))
    cb = ClassBelief(probabilities={"target": 0.7, "leaf": 0.3})

    n1 = Node(node_id="node_000001", geometry=geom, class_belief=cb)
    graph.add_node(n1)

    data = graph.to_dict()
    assert "nodes" in data
    assert "node_000001" in data["nodes"]

    restored_graph = SceneGraph.from_dict(data)
    restored_node = restored_graph.get_node("node_000001")

    assert restored_node is not None
    assert restored_node.node_id == "node_000001"
    assert restored_node.geometry.bbox() == Box(10.0, 10.0, 50.0, 50.0)
    assert restored_node.class_belief.probabilities == {"target": 0.7, "leaf": 0.3}


def test_graph_serialization_json_roundtrip():
    graph = SceneGraph()
    geom = BoxGeometry(Box(5.0, 5.0, 25.0, 25.0))
    n1 = Node(node_id="n1", geometry=geom)
    graph.add_node(n1)

    json_str = graph.to_json()
    assert "n1" in json_str

    restored = SceneGraph.from_json(json_str)
    assert restored.get_node("n1") is not None
