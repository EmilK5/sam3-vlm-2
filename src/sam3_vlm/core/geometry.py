"""Geometry contract and bounding box utilities for SAM3-VLM V4.

Invariants (V4 Design Spec §21.2):
- All persistent boxes and geometries refer to original-image coordinates
  unless explicitly marked local/tile coordinates.
- Geometry exposes at least bbox(), area(), and iou(other).
- Dense masks are stored as external artifacts referenced by URI/path, not inlined in graph state.
"""

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence, Tuple, runtime_checkable
import numpy as np


@dataclass(frozen=True)
class Box:
    """Bounding box in original image pixel coordinates (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float
    coordinate_space: Literal["image", "tile", "local"] = "image"

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(
                f"Invalid box coordinates: ({self.x1}, {self.y1}, {self.x2}, {self.y2}). "
                "x2 >= x1 and y2 >= y1 are required."
            )

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def intersection(self, other: "Box") -> float:
        """Compute intersection area with another box."""
        if self.coordinate_space != other.coordinate_space:
            raise ValueError(
                f"Cannot compute intersection between different coordinate spaces: "
                f"'{self.coordinate_space}' vs '{other.coordinate_space}'."
            )
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        return (ix2 - ix1) * (iy2 - iy1)

    def union(self, other: "Box") -> float:
        """Compute union area with another box."""
        return self.area + other.area - self.intersection(other)

    def iou(self, other: "Box") -> float:
        """Compute Intersection-over-Union with another box."""
        u = self.union(other)
        if u <= 0.0:
            return 0.0
        return self.intersection(other) / u

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@runtime_checkable
class Geometry(Protocol):
    """Abstract spatial geometry interface."""

    def bbox(self) -> Box:
        ...

    def area(self) -> float:
        ...

    def iou(self, other: "Geometry") -> float:
        ...


@dataclass(frozen=True)
class BoxGeometry:
    """Concrete Geometry implementation wrapping a Box."""

    box: Box

    def bbox(self) -> Box:
        return self.box

    def area(self) -> float:
        return self.box.area

    def iou(self, other: Geometry) -> float:
        return self.box.iou(other.bbox())


@dataclass(frozen=True)
class PolygonGeometry:
    """Concrete Geometry implementation using polygon boundary points."""

    points: Tuple[Tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError("PolygonGeometry requires at least 3 points.")

    def bbox(self) -> Box:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return Box(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))

    def area(self) -> float:
        # Shoelace formula for polygon area
        n = len(self.points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.points[i][0] * self.points[j][1]
            area -= self.points[j][0] * self.points[i][1]
        return abs(area) / 2.0

    def iou(self, other: Geometry) -> float:
        # Approximation via bbox IoU unless detailed polygon intersection is added
        return self.bbox().iou(other.bbox())


@dataclass(frozen=True)
class GeometryRef:
    """Reference to geometry data with optional external mask artifact pointer."""

    box: Box
    mask_artifact: str | None = None

    def bbox(self) -> Box:
        return self.box
        
    def area(self) -> float:
        return self.box.area
        
    def iou(self, other: Geometry) -> float:
        return self.box.iou(other.bbox())


@dataclass(frozen=True, eq=False)
class MaskGeometry:
    """Compact binary mask with its top-left offset in original-image pixels.

    Replay keeps the artifact pointer and area without inlining pixel arrays.
    Live association always requires the binary pixels of both geometries.
    """

    box: Box
    mask: np.ndarray | None
    offset: Tuple[int, int]
    pixel_area: int
    mask_artifact: str | None = None
    crop_boundary_clipped: bool = False

    def bbox(self) -> Box:
        return self.box

    def area(self) -> float:
        return float(self.pixel_area)

    def iou(self, other: Geometry) -> float:
        return mask_overlap(self, other)[0]


def mask_overlap(a: Geometry, b: Geometry) -> Tuple[float, float]:
    """Mask IoU and intersection/minimum mask area, with no box fallback."""
    if not isinstance(a, MaskGeometry) or not isinstance(b, MaskGeometry):
        raise ValueError("Mask association requires masks on both objects")
    if a.mask is None or b.mask is None:
        raise ValueError("Load mask artifacts before resuming live association")
    ax, ay = a.offset
    bx, by = b.offset
    x1, y1 = max(ax, bx), max(ay, by)
    x2 = min(ax + a.mask.shape[1], bx + b.mask.shape[1])
    y2 = min(ay + a.mask.shape[0], by + b.mask.shape[0])
    intersection = 0
    if x2 > x1 and y2 > y1:
        intersection = int(np.count_nonzero(
            a.mask[y1-ay:y2-ay, x1-ax:x2-ax] & b.mask[y1-by:y2-by, x1-bx:x2-bx]
        ))
    union = a.pixel_area + b.pixel_area - intersection
    minimum = min(a.pixel_area, b.pixel_area)
    return (intersection / union if union else 0.0,
            intersection / minimum if minimum else 0.0)


class EmptyMaskError(ValueError):
    """A valid binary sensor proposal with no foreground pixels."""


def detection_mask_geometry(detection) -> MaskGeometry:
    """Read, validate and trim a sensor mask while preserving global offsets."""
    raw = detection.raw_metadata
    if "mask" not in raw:
        raise ValueError(f"Detection {detection.detection_id} has no mask; box fallback is disabled")
    mask = np.asarray(raw["mask"])
    if mask.ndim != 2 or min(mask.shape, default=0) == 0 or not np.all(np.isfinite(mask)) or not np.all((mask == 0) | (mask == 1)):
        raise ValueError("SAM3 must supply a finite 2D binary mask")
    ox, oy = raw.get("mask_offset_x", 0), raw.get("mask_offset_y", 0)
    if any(not np.isfinite(v) or int(v) != v or v < 0 for v in (ox, oy)):
        raise ValueError("Mask offsets must be nonnegative integer image coordinates")
    clipped = raw.get("crop_boundary_clipped", False)
    if type(clipped) is not bool:
        raise ValueError("Mask crop-boundary provenance must be boolean")
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise EmptyMaskError("SAM3 supplied an empty mask")
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    ox, oy = int(ox) + left, int(oy) + top
    pixels = np.array(mask[top:bottom, left:right], dtype=bool, copy=True)
    return MaskGeometry(Box(ox, oy, ox + pixels.shape[1], oy + pixels.shape[0]),
                        pixels, (ox, oy), int(pixels.sum()), detection.mask_artifact, clipped)

def deserialize_geometry(data: dict) -> Geometry:
    if "box" in data:
        coords = data["box"]
        space = data.get("coordinate_space", "image")
        b = Box(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3], coordinate_space=space)
        return BoxGeometry(b)
    elif "points" in data:
        pts = tuple(tuple(p) for p in data["points"])
        return PolygonGeometry(pts)
    raise ValueError(f"Unknown geometry format: {data}")
