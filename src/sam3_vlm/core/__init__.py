"""Core foundational primitives, types, IDs, geometry, and config."""

from sam3_vlm.core.types import (
    NodeStatus,
    ObservationRelation,
    ActionFamily,
    SpatialMode,
    ActionSource,
    BudgetState,
    ClassBelief,
    RegistrationDiagnostics,
    NodeObservationRef,
    Detection,
)
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.core.geometry import Box, Geometry, BoxGeometry, PolygonGeometry, GeometryRef
from sam3_vlm.core.config import V4Config, TilingConfig, BudgetConfig, StoppingConfig

__all__ = [
    "NodeStatus",
    "ObservationRelation",
    "ActionFamily",
    "SpatialMode",
    "ActionSource",
    "BudgetState",
    "ClassBelief",
    "RegistrationDiagnostics",
    "NodeObservationRef",
    "Detection",
    "IDGenerator",
    "Box",
    "Geometry",
    "BoxGeometry",
    "PolygonGeometry",
    "GeometryRef",
    "V4Config",
    "TilingConfig",
    "BudgetConfig",
    "StoppingConfig",
]
