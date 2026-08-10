"""Tiling helper interfaces and spatial decomposition for SAM3-VLM V4."""

from typing import List
from sam3_vlm.core.config import TilingConfig
from sam3_vlm.core.geometry import Box, BoxGeometry


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
