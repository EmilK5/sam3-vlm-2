"""Paper-inspired density decision and ROI-guided overlapping image tiles.

All boxes are integer [x1, y1, x2, y2] coordinates in the original image.
The published image method uses a full-image pass before this decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

Region = tuple[int, int, int, int]


@dataclass(frozen=True)
class TilingPlan:
    density_score: float
    coverage: float
    object_count: int
    mean_box_area_ratio: float
    size_score: float
    trigger: bool
    roi: Region | None
    tile_size: int | None
    overlap: int | None
    tiles: tuple[Region, ...]


def _box_area(box: Region) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection_area(a: Region, b: Region) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _positions(length: int, tile_size: int, stride: int) -> tuple[int, ...]:
    count = max(1, math.ceil(length / stride))
    return tuple(dict.fromkeys(min(i * stride, max(0, length - tile_size)) for i in range(count)))


def plan_adaptive_tiles(
    boxes: list[Region], width: int, height: int, *,
    density_threshold: float = 0.69,
    min_tile_size: int = 97,
    max_tile_size: int = 1024,
) -> TilingPlan:
    """Use the paper's weighted density score and its small-object tile regime.

    The official image implementation activates the SMALL (6 by 4, 25% overlap)
    regime above 0.69. The ROI is the padded union of Stage-1 boxes; tiles are
    generated on the full image and kept on any positive ROI intersection.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not boxes:
        return TilingPlan(0.0, 0.0, 0, 0.0, 0.0, False, None, None, None, ())
    image_area = width * height
    total_box_area = sum(_box_area(box) for box in boxes)
    count = len(boxes)
    coverage = total_box_area / image_area
    mean_area_ratio = total_box_area / count / image_area
    size_score = 1.0 - min(mean_area_ratio / 0.1, 1.0)
    density_score = 0.3 * coverage + 0.5 * min(count / 50.0, 1.0) + 0.2 * size_score
    trigger = density_score > density_threshold
    x1, y1 = min(box[0] for box in boxes), min(box[1] for box in boxes)
    x2, y2 = max(box[2] for box in boxes), max(box[3] for box in boxes)
    pad_x = max(16, 0.04 * (x2 - x1))
    pad_y = max(16, 0.04 * (y2 - y1))
    roi = (max(0, math.floor(x1 - pad_x)), max(0, math.floor(y1 - pad_y)),
           min(width, math.ceil(x2 + pad_x)), min(height, math.ceil(y2 + pad_y)))
    if not trigger:
        return TilingPlan(density_score, coverage, count, mean_area_ratio,
                          size_score, False, roi, None, None, ())
    tile_size = max(min_tile_size, min(int(max(width / 6, height / 4)), max_tile_size))
    overlap = int(tile_size * 0.25)
    stride = max(1, tile_size - overlap)
    tiles = tuple(
        (x, y, min(width, x + tile_size), min(height, y + tile_size))
        for y in _positions(height, tile_size, stride)
        for x in _positions(width, tile_size, stride)
    )
    selected = tuple(tile for tile in tiles if _intersection_area(tile, roi) > 0)
    if not selected:
        selected = tiles
    return TilingPlan(density_score, coverage, count, mean_area_ratio,
                      size_score, True, roi, tile_size, overlap, selected)
