"""Tiling helper interfaces, decision schemas, and spatial decomposition (V4 Design Spec §22.3)."""

from dataclasses import dataclass, field
from typing import List, Protocol
from sam3_vlm.core.config import TilingConfig
from sam3_vlm.core.geometry import Box, BoxGeometry
from sam3_vlm.scene.graph import SceneGraph


@dataclass
class TilingDecision:
    """Decision output evaluating whether spatial tiling is required."""

    should_tile: bool
    reason: str
    grid_rows: int
    grid_cols: int
    tiles: List[BoxGeometry] = field(default_factory=list)


class TilingPolicy(Protocol):
    """Protocol for determining when and how an image should be spatially tiled."""

    def evaluate_tiling(
        self,
        image_width: int,
        image_height: int,
        config: TilingConfig,
        graph: SceneGraph | None = None,
    ) -> TilingDecision:
        ...


class DefaultTilingPolicy:
    """Configurable tiling policy evaluating image resolution and candidate size distribution (V4 Design Spec §5.2)."""

    def evaluate_tiling(
        self,
        image_width: int,
        image_height: int,
        config: TilingConfig,
        graph: SceneGraph | None = None,
    ) -> TilingDecision:
        # Check image resolution
        min_res = config.tile_min_size * max(config.grid_rows, config.grid_cols)
        if image_width < min_res or image_height < min_res:
            return TilingDecision(
                should_tile=False,
                reason=f"Image dimensions ({image_width}x{image_height}) below minimum tiling threshold ({min_res}x{min_res}).",
                grid_rows=config.grid_rows,
                grid_cols=config.grid_cols,
                tiles=[],
            )

        tiles = compute_tiles(image_width, image_height, config)

        # If graph provided, check candidate sizes relative to image size
        if graph and len(graph.active_nodes()) > 0:
            active_nodes = graph.active_nodes()
            img_area = float(image_width * image_height)
            small_candidates = [
                n for n in active_nodes if (n.geometry.area() / img_area) < 0.05
            ]
            if len(small_candidates) > 0:
                return TilingDecision(
                    should_tile=True,
                    reason=f"Found {len(small_candidates)} small candidate objects requiring tiled high-resolution sensing.",
                    grid_rows=config.grid_rows,
                    grid_cols=config.grid_cols,
                    tiles=tiles,
                )

        return TilingDecision(
            should_tile=True,
            reason=f"High resolution image ({image_width}x{image_height}) triggers default tiling pass.",
            grid_rows=config.grid_rows,
            grid_cols=config.grid_cols,
            tiles=tiles,
        )


def compute_tiles(image_width: int, image_height: int, config: TilingConfig) -> List[BoxGeometry]:
    """Compute overlapping tile geometries over image dimensions."""
    tiles = []
    tile_w = image_width // config.grid_cols
    tile_h = image_height // config.grid_rows

    for r in range(config.grid_rows):
        for c in range(config.grid_cols):
            x1 = max(0, int(c * tile_w - config.overlap_ratio * tile_w))
            y1 = max(0, int(r * tile_h - config.overlap_ratio * tile_h))
            x2 = min(image_width, int((c + 1) * tile_w + config.overlap_ratio * tile_w))
            y2 = min(image_height, int((r + 1) * tile_h + config.overlap_ratio * tile_h))
            tiles.append(BoxGeometry(Box(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))))

    return tiles


def tile_box_to_image_box(tile_box: Box, tile_region: Box) -> Box:
    """Translate local tile coordinates to original image coordinates."""
    global_x1 = tile_region.x1 + tile_box.x1
    global_y1 = tile_region.y1 + tile_box.y1
    global_x2 = tile_region.x1 + tile_box.x2
    global_y2 = tile_region.y1 + tile_box.y2
    return Box(x1=global_x1, y1=global_y1, x2=global_x2, y2=global_y2, coordinate_space="image")


def image_box_to_tile_box(image_box: Box, tile_region: Box) -> Box:
    """Translate original image coordinates to local tile coordinates."""
    local_x1 = max(0.0, image_box.x1 - tile_region.x1)
    local_y1 = max(0.0, image_box.y1 - tile_region.y1)
    local_x2 = max(local_x1, min(tile_region.width, image_box.x2 - tile_region.x1))
    local_y2 = max(local_y1, min(tile_region.height, image_box.y2 - tile_region.y1))
    return Box(x1=local_x1, y1=local_y1, x2=local_x2, y2=local_y2, coordinate_space="tile")
