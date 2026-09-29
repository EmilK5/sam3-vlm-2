"""Paper-inspired density decision and full-image overlapping tiles.

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
    image_region: Region
    tile_size: int | None
    overlap: int | None
    tiles: tuple[Region, ...]


def _box_area(box: Region) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _positions(length: int, tile_size: int, stride: int) -> tuple[int, ...]:
    count = max(1, math.ceil(length / stride))
    return tuple(dict.fromkeys(min(i * stride, max(0, length - tile_size)) for i in range(count)))


def plan_adaptive_tiles(
    boxes: list[Region], width: int, height: int, *,
    density_threshold: float = 0.69,
    min_tile_size: int = 97,
    max_tile_size: int = 1024,
) -> TilingPlan:
    """Use the density score and SMALL (6 by 4, 25% overlap) tile regime.

    Unlike the former F workflow, no ROI is selected: all tiles cover the
    original image so areas with missed Stage-1 detections remain searchable.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0 <= density_threshold <= 1 or not 1 <= min_tile_size <= max_tile_size:
        raise ValueError("invalid adaptive tiling parameters")
    image_region = (0, 0, width, height)
    if any(not (0 <= b[0] < b[2] <= width and 0 <= b[1] < b[3] <= height) for b in boxes):
        raise ValueError("seed boxes must be positive and inside the image")
    if not boxes:
        return TilingPlan(0.0, 0.0, 0, 0.0, 0.0, False, image_region, None, None, ())
    image_area = width * height
    total_box_area = sum(_box_area(box) for box in boxes)
    count = len(boxes)
    coverage = total_box_area / image_area
    mean_area_ratio = total_box_area / count / image_area
    size_score = 1.0 - min(mean_area_ratio / 0.1, 1.0)
    density_score = 0.3 * coverage + 0.5 * min(count / 50.0, 1.0) + 0.2 * size_score
    trigger = density_score > density_threshold
    if not trigger:
        return TilingPlan(density_score, coverage, count, mean_area_ratio,
                          size_score, False, image_region, None, None, ())
    tile_size = max(min_tile_size, min(int(max(width / 6, height / 4)), max_tile_size))
    overlap = int(tile_size * 0.25)
    stride = max(1, tile_size - overlap)
    tiles = tuple(
        (x, y, min(width, x + tile_size), min(height, y + tile_size))
        for y in _positions(height, tile_size, stride)
        for x in _positions(width, tile_size, stride))
    return TilingPlan(density_score, coverage, count, mean_area_ratio,
                      size_score, True, image_region, tile_size, overlap, tiles)
