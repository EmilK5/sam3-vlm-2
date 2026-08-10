"""Graph Node schema representing a sensor-grounded object hypothesis (V4 Design Spec §3.3)."""

from dataclasses import dataclass, field
from typing import List
from sam3_vlm.core.geometry import Geometry
from sam3_vlm.core.types import ClassBelief, NodeObservationRef, NodeStatus


@dataclass
class Node:
    """Scene graph node hypothesis."""

    node_id: str
    geometry: Geometry
    class_belief: ClassBelief = field(default_factory=ClassBelief)
    existence_score: float = 1.0
    duplicate_risk: float = 0.0
    merge_risk: float = 0.0
    observations: List[NodeObservationRef] = field(default_factory=list)
    created_by_call_id: str = ""
    status: NodeStatus = NodeStatus.ACTIVE
