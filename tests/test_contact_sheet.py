"""Unit tests for ContactSheetBuilder candidate sampling and crop annotations (V4 Design Spec §5.3)."""

import pytest
from sam3_vlm.core.geometry import Box, BoxGeometry
from sam3_vlm.core.types import (
    ActionFamily,
    ClassBelief,
    NodeObservationRef,
    ObservationRelation,
)
from sam3_vlm.scene.belief import SemanticMemory
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.node import Node
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.sensing.evidence import ContactSheetBuilder


def test_contact_sheet_builder_sampling():
    graph = SceneGraph()
    builder = ContactSheetBuilder()

    # Empty graph produces empty contact sheet
    cs_empty = builder.build_contact_sheet(graph)
    assert cs_empty.total_candidates == 0
    assert len(cs_empty.crops) == 0

    # Add 5 nodes across different confidence/diagnostics strata
    n1 = Node(node_id="n1", geometry=BoxGeometry(Box(0, 0, 10, 10)), class_belief=ClassBelief({"target": 0.9, "leaf": 0.1}))
    n1.observations.append(NodeObservationRef("o1", "sam3_1", "a1", "key", detection_id="d1", relation=ObservationRelation.STRONG_MATCH, score=0.9))

    n2 = Node(node_id="n2", geometry=BoxGeometry(Box(20, 20, 30, 30)), class_belief=ClassBelief({"target": 0.5, "leaf": 0.5}))
    n2.observations.append(NodeObservationRef("o2", "sam3_1", "a1", "key", detection_id="d2", relation=ObservationRelation.WEAK_MATCH, score=0.5))

    n3 = Node(node_id="n3", geometry=BoxGeometry(Box(40, 40, 50, 50)), class_belief=ClassBelief({"target": 0.2, "leaf": 0.8}))
    n3.observations.append(NodeObservationRef("o3", "sam3_1", "a1", "key", detection_id="d3", relation=ObservationRelation.WEAK_MATCH, score=0.2))

    n4 = Node(node_id="n4", geometry=BoxGeometry(Box(60, 60, 70, 70)))
    n4.diagnostics.duplicate_risk = 0.8  # Outlier stratum

    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    graph.add_node(n4)

    cs = builder.build_contact_sheet(graph, max_crops=16)

    assert cs.total_candidates == 4
    assert len(cs.crops) == 4
    assert cs.strata_counts["high"] == 1
    assert cs.strata_counts["medium"] == 1
    assert cs.strata_counts["low"] == 1
    assert cs.strata_counts["outlier"] == 1

    # Verify annotation structure for n1
    crop1 = next(c for c in cs.crops if c.node_id == "n1")
    assert crop1.sam3_score == 0.9
    assert crop1.provenance == "sam3_1"
    assert crop1.target_support_semantic_key == "key"
    assert crop1.latest_observation_score == 0.9
    assert crop1.latest_observation_relation == "STRONG_MATCH"
    assert crop1.target_posterior == 0.9
    assert crop1.class_belief["target"] == 0.9


def _semantic_memory_with_target_and_confounder_calls():
    memory = SemanticMemory()
    memory.record_execution(
        SensingAction(
            action_id="target_action",
            semantic_key="target",
            prompt="green fruit",
            family=ActionFamily.DISCOVERY,
        ),
        "sam3_target",
    )
    memory.record_execution(
        SensingAction(
            action_id="leaf_action",
            semantic_key="confounder1",
            prompt="green leaf",
            family=ActionFamily.CONFOUNDER,
        ),
        "sam3_leaf",
    )
    return memory


def test_latest_confounder_non_retrieval_does_not_erase_target_support():
    graph = SceneGraph()
    node = Node(
        node_id="fruit_1",
        geometry=BoxGeometry(Box(0, 0, 20, 20)),
        class_belief=ClassBelief({"target": 0.81, "confounder1": 0.19}),
    )
    node.observations.extend(
        [
            NodeObservationRef(
                "obs_target",
                "sam3_target",
                "target_action",
                "target",
                detection_id="det_target",
                relation=ObservationRelation.STRONG_MATCH,
                score=0.82,
            ),
            NodeObservationRef(
                "obs_leaf_absent",
                "sam3_leaf",
                "leaf_action",
                "confounder1",
                relation=ObservationRelation.NOT_RETRIEVED,
                score=0.0,
            ),
        ]
    )
    graph.add_node(node)

    sheet = ContactSheetBuilder().build_contact_sheet(
        graph,
        semantic_memory=_semantic_memory_with_target_and_confounder_calls(),
    )
    crop = sheet.crops[0]

    assert crop.target_support_score == 0.82
    assert crop.target_support_semantic_key == "target"
    assert crop.target_support_call_id == "sam3_target"
    assert crop.latest_observation_score == 0.0
    assert crop.latest_observation_semantic_key == "confounder1"
    assert crop.latest_observation_relation == "NOT_RETRIEVED"
    assert crop.latest_observation_call_id == "sam3_leaf"
    assert crop.target_posterior == 0.81


def test_strong_confounder_detection_is_not_serialized_as_target_support():
    graph = SceneGraph()
    node = Node(
        node_id="leaf_like_1",
        geometry=BoxGeometry(Box(0, 0, 20, 20)),
        class_belief=ClassBelief({"target": 0.2, "confounder1": 0.8}),
    )
    node.observations.extend(
        [
            NodeObservationRef(
                "obs_target",
                "sam3_target",
                "target_action",
                "target",
                detection_id="det_target",
                relation=ObservationRelation.WEAK_MATCH,
                score=0.61,
            ),
            NodeObservationRef(
                "obs_leaf",
                "sam3_leaf",
                "leaf_action",
                "confounder1",
                detection_id="det_leaf",
                relation=ObservationRelation.STRONG_MATCH,
                score=0.96,
            ),
        ]
    )
    graph.add_node(node)

    crop = ContactSheetBuilder().build_contact_sheet(
        graph,
        semantic_memory=_semantic_memory_with_target_and_confounder_calls(),
    ).crops[0]

    assert crop.target_support_score == 0.61
    assert crop.latest_observation_score == 0.96
    assert crop.latest_observation_semantic_key == "confounder1"
    assert crop.target_posterior == 0.2
