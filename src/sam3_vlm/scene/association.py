"""Cross-pass detection association and node registration interfaces (V4 Design Spec §10)."""

from typing import List, Protocol
from sam3_vlm.core.types import Detection, NodeObservationRef
from sam3_vlm.scene.graph import SceneGraph


class AssociationPolicy(Protocol):
    """Protocol for associating new SAM3 detections with existing graph nodes."""

    def associate(
        self, graph: SceneGraph, detections: List[Detection]
    ) -> List[NodeObservationRef]:
        ...
