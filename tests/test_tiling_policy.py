"""Unit tests for DefaultTilingPolicy spatial decomposition rules (V4 Design Spec §5.2 / §22.3)."""

import pytest
from sam3_vlm.core.config import TilingConfig
from sam3_vlm.core.geometry import Box, BoxGeometry
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.node import Node
from sam3_vlm.sensing.tiling import DefaultTilingPolicy


def test_tiling_policy_low_resolution_image():
    policy = DefaultTilingPolicy()
    cfg = TilingConfig(tile_min_size=512, grid_rows=2, grid_cols=2)

    # Low resolution image (400x400) below minimum threshold
    decision = policy.evaluate_tiling(image_width=400, image_height=400, config=cfg)

    assert decision.should_tile is False
    assert "below minimum" in decision.reason
    assert len(decision.tiles) == 0


def test_tiling_policy_high_resolution_image():
    policy = DefaultTilingPolicy()
    cfg = TilingConfig(tile_min_size=512, grid_rows=2, grid_cols=2)

    # High resolution image (2000x2000)
    decision = policy.evaluate_tiling(image_width=2000, image_height=2000, config=cfg)

    assert decision.should_tile is True
    assert len(decision.tiles) == 4


def test_tiling_policy_small_candidate_trigger():
    policy = DefaultTilingPolicy()
    cfg = TilingConfig(tile_min_size=512, grid_rows=2, grid_cols=2)
    graph = SceneGraph()

    # Add small candidate node (10x10 area relative to 2000x2000 image)
    small_node = Node(node_id="n1", geometry=BoxGeometry(Box(100.0, 100.0, 110.0, 110.0)))
    graph.add_node(small_node)

    decision = policy.evaluate_tiling(image_width=2000, image_height=2000, config=cfg, graph=graph)

    assert decision.should_tile is True
    assert "small candidate" in decision.reason
