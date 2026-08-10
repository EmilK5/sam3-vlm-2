"""Test Box and Geometry spatial math, area, intersection, and IoU."""

import pytest
from sam3_vlm.core.geometry import Box, BoxGeometry, PolygonGeometry, Geometry


def test_box_properties():
    box = Box(x1=10.0, y1=20.0, x2=50.0, y2=80.0)
    assert box.width == 40.0
    assert box.height == 60.0
    assert box.area == 2400.0
    assert box.as_tuple() == (10.0, 20.0, 50.0, 80.0)


def test_invalid_box():
    with pytest.raises(ValueError):
        Box(x1=50.0, y1=20.0, x2=10.0, y2=80.0)


def test_box_intersection_and_iou():
    box1 = Box(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
    box2 = Box(x1=5.0, y1=0.0, x2=15.0, y2=10.0)

    # Intersection width = 5, height = 10 -> area = 50
    assert box1.intersection(box2) == 50.0

    # Union = 100 + 100 - 50 = 150
    assert box1.union(box2) == 150.0

    # IoU = 50 / 150 = 1/3
    assert pytest.approx(box1.iou(box2), abs=1e-5) == 1.0 / 3.0


def test_disjoint_boxes():
    box1 = Box(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
    box2 = Box(x1=20.0, y1=20.0, x2=30.0, y2=30.0)

    assert box1.intersection(box2) == 0.0
    assert box1.iou(box2) == 0.0


def test_box_geometry_protocol(sample_box: Box):
    geom = BoxGeometry(box=sample_box)
    assert isinstance(geom, Geometry)
    assert geom.bbox() == sample_box
    assert geom.area() == sample_box.area


def test_polygon_geometry():
    poly = PolygonGeometry(points=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))
    assert isinstance(poly, Geometry)
    assert poly.area() == 100.0
    assert poly.bbox() == Box(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
