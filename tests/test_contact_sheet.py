"""Unit tests for ContactSheetBuilder candidate sampling and crop annotations (V4 Design Spec §5.3)."""

import pytest
from sam3_vlm.core.geometry import Box, BoxGeometry
from sam3_vlm.core.types import ClassBelief, NodeObservationRef, ObservationRelation
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.node import Node
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
    n1.observations.append(NodeObservationRef("o1", "sam3_1", "a1", "key", relation=ObservationRelation.STRONG_MATCH, score=0.9))

    n2 = Node(node_id="n2", geometry=BoxGeometry(Box(20, 20, 30, 30)), class_belief=ClassBelief({"target": 0.5, "leaf": 0.5}))
    n2.observations.append(NodeObservationRef("o2", "sam3_1", "a1", "key", relation=ObservationRelation.WEAK_MATCH, score=0.5))

    n3 = Node(node_id="n3", geometry=BoxGeometry(Box(40, 40, 50, 50)), class_belief=ClassBelief({"target": 0.2, "leaf": 0.8}))
    n3.observations.append(NodeObservationRef("o3", "sam3_1", "a1", "key", relation=ObservationRelation.WEAK_MATCH, score=0.2))

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
    assert crop1.class_belief["target"] == 0.9
