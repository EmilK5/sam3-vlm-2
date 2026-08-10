"""Graph Node schema representing a sensor-grounded object hypothesis (V4 Design Spec §3.3)."""

from dataclasses import dataclass, field
from typing import Any, Dict, List
from sam3_vlm.core.geometry import Box, BoxGeometry, Geometry, GeometryRef
from sam3_vlm.core.types import (
    ClassBelief,
    NodeObservationRef,
    NodeStatus,
    ObservationRelation,
    RegistrationDiagnostics,
)


@dataclass
class Node:
    """Scene graph node hypothesis (V4 Design Spec §3.3)."""

    node_id: str
    geometry: Geometry
    class_belief: ClassBelief = field(default_factory=ClassBelief)
    diagnostics: RegistrationDiagnostics = field(default_factory=RegistrationDiagnostics)
    observations: List[NodeObservationRef] = field(default_factory=list)
    created_by_call_id: str = ""
    status: NodeStatus = NodeStatus.ACTIVE

    @property
    def existence_score(self) -> float:
        return self.diagnostics.existence_score

    @existence_score.setter
    def existence_score(self, value: float) -> None:
        self.diagnostics.existence_score = value

    @property
    def duplicate_risk(self) -> float:
        return self.diagnostics.duplicate_risk

    @duplicate_risk.setter
    def duplicate_risk(self, value: float) -> None:
        self.diagnostics.duplicate_risk = value

    @property
    def merge_risk(self) -> float:
        return self.diagnostics.merge_risk

    @merge_risk.setter
    def merge_risk(self, value: float) -> None:
        self.diagnostics.merge_risk = value

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node state to a serializable dictionary."""
        box = self.geometry.bbox()
        return {
            "node_id": self.node_id,
            "box": box.as_tuple(),
            "coordinate_space": box.coordinate_space,
            "class_belief": {
                "probabilities": dict(self.class_belief.probabilities),
                "update_count": self.class_belief.update_count,
                "entropy": self.class_belief.entropy,
                "last_update_event_id": self.class_belief.last_update_event_id,
            },
            "diagnostics": {
                "existence_score": self.diagnostics.existence_score,
                "duplicate_risk": self.diagnostics.duplicate_risk,
                "merge_risk": self.diagnostics.merge_risk,
                "ambiguous_with": list(self.diagnostics.ambiguous_with),
                "support_count": self.diagnostics.support_count,
                "independent_semantic_support_count": self.diagnostics.independent_semantic_support_count,
            },
            "observations": [
                {
                    "observation_id": obs.observation_id,
                    "sam3_call_id": obs.sam3_call_id,
                    "action_id": obs.action_id,
                    "semantic_key": obs.semantic_key,
                    "detection_id": obs.detection_id,
                    "relation": obs.relation.value,
                    "score": obs.score,
                    "association_score": obs.association_score,
                }
                for obs in self.observations
            ],
            "created_by_call_id": self.created_by_call_id,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        """Deserialize node state from a dictionary."""
        box_coords = data["box"]
        coord_space = data.get("coordinate_space", "image")
        box = Box(x1=box_coords[0], y1=box_coords[1], x2=box_coords[2], y2=box_coords[3], coordinate_space=coord_space)
        geometry = BoxGeometry(box=box)

        cb_data = data.get("class_belief", {})
        class_belief = ClassBelief(
            probabilities=dict(cb_data.get("probabilities", {})),
            update_count=cb_data.get("update_count", 0),
            entropy=cb_data.get("entropy", 0.0),
            last_update_event_id=cb_data.get("last_update_event_id"),
        )

        diag_data = data.get("diagnostics", {})
        diagnostics = RegistrationDiagnostics(
            existence_score=diag_data.get("existence_score", 1.0),
            duplicate_risk=diag_data.get("duplicate_risk", 0.0),
            merge_risk=diag_data.get("merge_risk", 0.0),
            ambiguous_with=list(diag_data.get("ambiguous_with", [])),
            support_count=diag_data.get("support_count", 1),
            independent_semantic_support_count=diag_data.get("independent_semantic_support_count", 1),
        )

        obs_list = []
        for o_data in data.get("observations", []):
            obs_list.append(
                NodeObservationRef(
                    observation_id=o_data["observation_id"],
                    sam3_call_id=o_data["sam3_call_id"],
                    action_id=o_data["action_id"],
                    semantic_key=o_data["semantic_key"],
                    detection_id=o_data.get("detection_id"),
                    relation=ObservationRelation(o_data["relation"]),
                    score=o_data.get("score"),
                    association_score=o_data.get("association_score"),
                )
            )

        return cls(
            node_id=data["node_id"],
            geometry=geometry,
            class_belief=class_belief,
            diagnostics=diagnostics,
            observations=obs_list,
            created_by_call_id=data.get("created_by_call_id", ""),
            status=NodeStatus(data.get("status", "ACTIVE")),
        )
