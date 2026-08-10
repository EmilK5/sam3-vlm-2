"""Geometry contract and bounding box utilities for SAM3-VLM V4.

Invariants (V4 Design Spec §21.2):
- All persistent boxes and geometries refer to original-image coordinates
  unless explicitly marked local/tile coordinates.
- Geometry exposes at least bbox(), area(), and iou(other).
- Dense masks are stored as external artifacts referenced by URI/path, not inlined in graph state.
"""

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence, Tuple, runtime_checkable


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
