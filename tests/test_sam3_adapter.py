"""Unit tests for MockSAM3Adapter, coordinate space transformations, and budget accounting (V4 Design Spec §4)."""

import pytest
from sam3_vlm.core.config import TilingConfig
from sam3_vlm.core.geometry import Box, GeometryRef
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.core.types import ActionFamily, ActionSource, Detection, SpatialMode
from sam3_vlm.models.sam3 import MockSAM3Adapter
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.sensing.tiling import image_box_to_tile_box, tile_box_to_image_box


def test_mock_sam3_adapter_global_sensing():
    adapter = MockSAM3Adapter()
    action = SensingAction(
        action_id="act_001",
        semantic_key="green_citrus",
        prompt="green citrus fruit",
        family=ActionFamily.DISCOVERY,
        spatial_mode=SpatialMode.GLOBAL,
        source=ActionSource.USER_BOOTSTRAP,
    )

    obs = adapter.observe(image=(1000, 1000), action=action)

    assert obs.action_id == "act_001"
    assert obs.semantic_key == "green_citrus"
    assert len(obs.searched_regions) == 1
    assert obs.runtime_ms > 0.0
    assert len(obs.detections) >= 1
    assert obs.detections[0].geometry.box.coordinate_space == "image"


def test_mock_sam3_adapter_tiled_sensing():
    adapter = MockSAM3Adapter()
    tiling_cfg = TilingConfig(grid_rows=2, grid_cols=2)

    action = SensingAction(
        action_id="act_002",
        semantic_key="green_citrus",
        prompt="green citrus fruit",
        family=ActionFamily.DISCOVERY,
        spatial_mode=SpatialMode.TILED,
        tiling=tiling_cfg,
        source=ActionSource.USER_BOOTSTRAP,
    )

    obs = adapter.observe(image=(1000, 1000), action=action)

    assert len(obs.searched_regions) == 4  # 2x2 grid
    assert len(obs.detections) >= 1
    # All detections must be converted to original image coordinates
    for det in obs.detections:
        assert det.geometry.box.coordinate_space == "image"


def test_coordinate_transformation_helpers():
    tile_region = Box(x1=500.0, y1=500.0, x2=1000.0, y2=1000.0)
    local_box = Box(x1=10.0, y1=20.0, x2=100.0, y2=150.0, coordinate_space="tile")

    # Local tile -> Image global
    global_box = tile_box_to_image_box(local_box, tile_region)
    assert global_box.x1 == 510.0
    assert global_box.y1 == 520.0
    assert global_box.x2 == 600.0
    assert global_box.y2 == 650.0
    assert global_box.coordinate_space == "image"

    # Image global -> Local tile
    restored_local = image_box_to_tile_box(global_box, tile_region)
    assert restored_local.x1 == 10.0
    assert restored_local.y1 == 20.0
    assert restored_local.coordinate_space == "tile"
