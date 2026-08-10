"""Shared pytest fixtures for SAM3-VLM V4 unit tests."""

import pytest
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.core.geometry import Box, BoxGeometry


@pytest.fixture
def id_gen() -> IDGenerator:
    """Fixture providing a fresh IDGenerator instance."""
    return IDGenerator()


@pytest.fixture
def sample_box() -> Box:
    """Fixture providing a valid Box object."""
    return Box(x1=10.0, y1=20.0, x2=50.0, y2=80.0, coordinate_space="image")


@pytest.fixture
def sample_geometry(sample_box: Box) -> BoxGeometry:
    """Fixture providing a BoxGeometry object."""
    return BoxGeometry(box=sample_box)
