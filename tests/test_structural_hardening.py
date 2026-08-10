"""Unit tests for foundational structural hardening (V4 Design Spec §3 / §4 / §11 / §21)."""

import pytest
from sam3_vlm.core.config import TilingConfig
from sam3_vlm.core.geometry import Box, BoxGeometry
from sam3_vlm.core.types import ActionFamily, ActionSource, ClassBelief, Detection, SpatialMode
from sam3_vlm.models.sam3 import MockSAM3Adapter
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.node import Node
from sam3_vlm.scene.state import SceneState
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.sensing.evidence import ContactSheetBuilder


def test_class_belief_entropy_auto_computation():
    """Verify ClassBelief computes entropy dynamically on init (Spec §11)."""
    cb_uniform = ClassBelief(probabilities={"target": 0.5, "leaf": 0.5})
    assert pytest.approx(cb_uniform.entropy, abs=1e-4) == 1.0

    cb_certain = ClassBelief(probabilities={"target": 1.0, "leaf": 0.0})
    assert pytest.approx(cb_certain.entropy, abs=1e-4) == 0.0


def test_scene_graph_duplicate_node_id_exception():
    """Verify SceneGraph.add_node() raises ValueError on duplicate persistent node ID (Spec §3.2)."""
    graph = SceneGraph()
    n1 = Node(node_id="n1", geometry=BoxGeometry(Box(0, 0, 10, 10)))
    graph.add_node(n1)

    n1_duplicate = Node(node_id="n1", geometry=BoxGeometry(Box(5, 5, 15, 15)))
    with pytest.raises(ValueError, match="Duplicate persistent node ID"):
        graph.add_node(n1_duplicate)


def test_coordinate_space_mismatch_exception():
    """Verify Box.intersection() raises ValueError on coordinate space mismatch (Spec §21.2)."""
    box_image = Box(0, 0, 100, 100, coordinate_space="image")
    box_tile = Box(0, 0, 100, 100, coordinate_space="tile")

    with pytest.raises(ValueError, match="Cannot compute intersection between different coordinate spaces"):
        box_image.intersection(box_tile)


def test_mock_sam3_tiled_mode_threshold_consistency():
    """Verify MockSAM3Adapter filters synthetic detections by action.threshold in tiled mode (Spec §4.2)."""
    low_score_det = Detection("d1", BoxGeometry(Box(10, 10, 50, 50)), score=0.20)
    high_score_det = Detection("d2", BoxGeometry(Box(200, 200, 250, 250)), score=0.90)

    adapter = MockSAM3Adapter(synthetic_detections=[low_score_det, high_score_det])

    action = SensingAction(
        action_id="a1",
        semantic_key="target",
        prompt="target",
        family=ActionFamily.DISCOVERY,
        threshold=0.50,  # Should filter out low_score_det (0.20)
        spatial_mode=SpatialMode.TILED,
        tiling=TilingConfig(grid_rows=2, grid_cols=2),
        source=ActionSource.USER_BOOTSTRAP,
    )

    obs = adapter.observe(image=(1000, 1000), action=action)
    for det in obs.detections:
        assert det.score >= 0.50


def test_spatial_contact_sheet_sampling():
    """Verify ContactSheetBuilder includes spatial distribution across nodes."""
    graph = SceneGraph()

    # Add 4 nodes across 4 spatial regions
    n1 = Node(node_id="n1", geometry=BoxGeometry(Box(10, 10, 50, 50)))
    n2 = Node(node_id="n2", geometry=BoxGeometry(Box(500, 10, 550, 50)))
    n3 = Node(node_id="n3", geometry=BoxGeometry(Box(10, 500, 50, 550)))
    n4 = Node(node_id="n4", geometry=BoxGeometry(Box(500, 500, 550, 550)))

    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    graph.add_node(n4)

    builder = ContactSheetBuilder()
    cs = builder.build_contact_sheet(graph, max_crops=4)

    assert cs.total_candidates == 4
    assert len(cs.crops) == 4
    assert cs.contact_sheet_image_path is None
