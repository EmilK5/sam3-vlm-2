"""Evidence collection, contact sheet construction, and Qwen evidence pack assembly (V4 Design Spec §5.3 / §6.1)."""

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional
from sam3_vlm.core.geometry import Box
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.node import Node
from sam3_vlm.sensing.observation import SAM3Observation


@dataclass
class CropCandidateAnnotation:
    """Structured crop metadata annotation for Qwen evidence representation (V4 Design Spec §5.3)."""

    node_id: str
    box: Box
    sam3_score: float
    support_count: int
    provenance: str
    class_belief: Dict[str, float] = field(default_factory=dict)
    crop_image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "box": self.box.as_tuple(),
            "sam3_score": self.sam3_score,
            "support_count": self.support_count,
            "provenance": self.provenance,
            "class_belief": dict(self.class_belief),
            "crop_image_path": self.crop_image_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CropCandidateAnnotation":
        box_coords = data["box"]
        box = Box(x1=box_coords[0], y1=box_coords[1], x2=box_coords[2], y2=box_coords[3])
        return cls(
            node_id=data["node_id"],
            box=box,
            sam3_score=data["sam3_score"],
            support_count=data["support_count"],
            provenance=data["provenance"],
            class_belief=dict(data.get("class_belief", {})),
            crop_image_path=data.get("crop_image_path"),
        )


@dataclass
class ContactSheet:
    """Compact representative candidate contact sheet container (V4 Design Spec §5.3)."""

    crops: List[CropCandidateAnnotation] = field(default_factory=list)
    total_candidates: int = 0
    strata_counts: Dict[str, int] = field(default_factory=dict)
    contact_sheet_image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "crops": [c.to_dict() for c in self.crops],
            "total_candidates": self.total_candidates,
            "strata_counts": dict(self.strata_counts),
            "contact_sheet_image_path": self.contact_sheet_image_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContactSheet":
        crops = [CropCandidateAnnotation.from_dict(c) for c in data.get("crops", [])]
        return cls(
            crops=crops,
            total_candidates=data.get("total_candidates", len(crops)),
            strata_counts=dict(data.get("strata_counts", {})),
            contact_sheet_image_path=data.get("contact_sheet_image_path"),
        )


@dataclass
class QwenEvidencePack:
    """Structured evidence pack presented to Qwen scene planner (V4 Design Spec §6.1)."""

    original_image_id: str
    user_prompt: str
    target_class: str
    contact_sheet: ContactSheet
    image_path: Optional[str] = None
    scene_summary: str = ""
    discovery_diagnostics: Dict[str, Any] = field(default_factory=dict)
    belief_classes: List[str] = field(default_factory=list)
    confounder_labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_image_id": self.original_image_id,
            "user_prompt": self.user_prompt,
            "target_class": self.target_class,
            "contact_sheet": self.contact_sheet.to_dict(),
            "image_path": self.image_path,
            "scene_summary": self.scene_summary,
            "discovery_diagnostics": dict(self.discovery_diagnostics),
            "belief_classes": list(self.belief_classes),
            "confounder_labels": dict(self.confounder_labels),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QwenEvidencePack":
        cs_data = data.get("contact_sheet", {})
        contact_sheet = ContactSheet.from_dict(cs_data)
        return cls(
            original_image_id=data["original_image_id"],
            user_prompt=data["user_prompt"],
            target_class=data["target_class"],
            contact_sheet=contact_sheet,
            image_path=data.get("image_path"),
            scene_summary=data.get("scene_summary", ""),
            discovery_diagnostics=dict(data.get("discovery_diagnostics", {})),
            belief_classes=list(data.get("belief_classes", [])),
            confounder_labels=dict(data.get("confounder_labels", {})),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "QwenEvidencePack":
        return cls.from_dict(json.loads(json_str))

    def to_prompt_text(self) -> str:
        """Format evidence pack into a compact token-efficient text prompt for Qwen (V4 Design Spec §6.1)."""
        lines = [
            "=== IMPORTANT QWEN INSTRUCTIONS ===",
            "These crop panels are UNVERIFIED visual sensor candidates from SAM3.",
            "Do NOT label them as ground truth or final positive detections.",
            "Your role is to analyze candidate appearances (e.g. shadow, leaf, occluded fruit) and propose scene-level sensing actions.",
            "Do NOT attempt to output final object counts or raw bounding boxes directly.",
            "",
            "=== SCENE EVIDENCE PACK ===",
            f"Image ID: {self.original_image_id}",
            f"Image Path: {self.image_path or 'Not provided'}",
            f"Contact Sheet Image: {self.contact_sheet.contact_sheet_image_path or 'Not rendered'}",
            f"User Target Concept: '{self.user_prompt}' (posterior class: target)",
            f"Frozen Belief Classes: {self.belief_classes or ['target']}",
            f"Frozen Confounder Slot Labels: {self.confounder_labels or 'not assigned yet'}",
            f"Discovery Diagnostics: {self.discovery_diagnostics}",
            f"Summary: {self.scene_summary or 'Initial bootstrap candidates.'}",
            f"Total Candidates Found: {self.contact_sheet.total_candidates}",
            f"Sampled Contact Sheet Crops ({len(self.contact_sheet.crops)} crops):",
        ]
        for c in self.contact_sheet.crops:
            top_class = (
                max(c.class_belief.items(), key=lambda x: x[1])[0]
                if c.class_belief
                else "unclassified"
            )
            crop_path_str = f" | crop_img={c.crop_image_path}" if c.crop_image_path else ""
            lines.append(
                f"  - [{c.node_id}] box=({c.box.x1:.1f}, {c.box.y1:.1f}, {c.box.x2:.1f}, {c.box.y2:.1f}) | "
                f"score={c.sam3_score:.2f} | support={c.support_count} | prov={c.provenance} | system_belief={top_class}{crop_path_str}"
            )
        lines.append("==========================")
        return "\n".join(lines)


@dataclass
class EvidencePack:
    """Pack of sensory evidence prepared for graph updates or Qwen planning."""

    observations: List[SAM3Observation] = field(default_factory=list)

def _anchor_support_observation(node):
    """Return stable sensor support for a graph candidate.

    A contact-sheet score should describe the semantic sensing experiment that
    grounded/created the candidate, not whichever unrelated experiment happened
    to run most recently.

    The anchor semantic key is taken from the first sensor-grounded detection
    observation. Repeated detections under that same semantic key may strengthen
    the score. NOT_RETRIEVED observations and later confounder experiments do
    not overwrite it.
    """
    observations = list(getattr(node, "observations", ()) or ())
    if not observations:
        return None

    # Find the first actual sensor detection. This is the experiment that
    # grounded the node in the graph.
    anchor = next(
        (
            obs
            for obs in observations
            if getattr(obs, "detection_id", None) is not None
            and getattr(obs, "score", None) is not None
        ),
        None,
    )

    if anchor is None:
        return None

    anchor_key = anchor.semantic_key

    # A repeated execution of the same semantic experiment may strengthen the
    # candidate. Use its strongest real detection, not a later NOT_RETRIEVED=0.
    same_anchor_detections = [
        obs
        for obs in observations
        if obs.semantic_key == anchor_key
        and getattr(obs, "detection_id", None) is not None
        and getattr(obs, "score", None) is not None
    ]

    if not same_anchor_detections:
        return anchor

    return max(
        same_anchor_detections,
        key=lambda obs: float(obs.score),
    )

class ContactSheetBuilder:
    """Stratified and spatially distributed candidate crop sampler (V4 Design Spec §5.3)."""

    def build_contact_sheet(
        self,
        graph: SceneGraph,
        max_crops: int = 24,
        image: Any = None,
        assets_dir: str = "assets",
        image_id: str = "bootstrap",
    ) -> ContactSheet:
        """Sample candidate crops across confidence/support strata and spatial quadrants."""
        active_nodes = graph.active_nodes()
        total_candidates = len(active_nodes)

        if total_candidates == 0:
            return ContactSheet(crops=[], total_candidates=0, strata_counts={})

        high_stratum: List[Node] = []
        medium_stratum: List[Node] = []
        low_stratum: List[Node] = []
        outlier_stratum: List[Node] = []

        for node in active_nodes:
            support_obs = _anchor_support_observation(node)
            score = (
                float(support_obs.score)
                if support_obs is not None and support_obs.score is not None
                else 0.5
            )

            if node.diagnostics.duplicate_risk > 0.6 or node.diagnostics.merge_risk > 0.6:
                outlier_stratum.append(node)
            elif score >= 0.7:
                high_stratum.append(node)
            elif score >= 0.4:
                medium_stratum.append(node)
            else:
                low_stratum.append(node)

        per_stratum_cap = max(1, max_crops // 4)
        sampled_nodes: List[Node] = []

        sampled_nodes.extend(high_stratum[:per_stratum_cap])
        sampled_nodes.extend(medium_stratum[:per_stratum_cap])
        sampled_nodes.extend(low_stratum[:per_stratum_cap])
        sampled_nodes.extend(outlier_stratum[:per_stratum_cap])

        # Fill remaining budget ensuring spatial distribution
        remaining_budget = max_crops - len(sampled_nodes)
        if remaining_budget > 0:
            already_sampled_ids = {n.node_id for n in sampled_nodes}
            remaining_nodes = [n for n in active_nodes if n.node_id not in already_sampled_ids]
            
            # Sort remaining nodes by spatial centroid distance from origin for spatial diversity
            remaining_nodes.sort(key=lambda n: (n.geometry.bbox().x1 + n.geometry.bbox().y1))
            sampled_nodes.extend(remaining_nodes[:remaining_budget])

        from sam3_vlm.sensing.visuals import crop_image_region, render_contact_sheet
        
        crops: List[CropCandidateAnnotation] = []
        crop_arrays = []
        
        from pathlib import Path
        assets_path = Path(assets_dir)
        crops_dir = assets_path / "crops"

        for node in sampled_nodes:
            support_obs = _anchor_support_observation(node)

            score = (
                float(support_obs.score)
                if support_obs is not None and support_obs.score is not None
                else 0.5
            )

            prov = (
                support_obs.sam3_call_id
                if support_obs is not None
                else node.created_by_call_id
            )

            crop_path_str = None
            if image is not None:
                crop_arr = crop_image_region(image, node.geometry.bbox())
                if crop_arr is not None:
                    crop_path = crops_dir / f"{node.node_id}.jpg"
                    from sam3_vlm.sensing.visuals import save_image
                    if save_image(crop_arr, str(crop_path)):
                        crop_path_str = str(crop_path)
                    crop_arrays.append(crop_arr)

            crops.append(
                CropCandidateAnnotation(
                    node_id=node.node_id,
                    box=node.geometry.bbox(),
                    sam3_score=score,
                    support_count=node.diagnostics.support_count,
                    provenance=prov,
                    class_belief=dict(node.class_belief.probabilities),
                    crop_image_path=crop_path_str,
                )
            )

        strata_counts = {
            "high": len(high_stratum),
            "medium": len(medium_stratum),
            "low": len(low_stratum),
            "outlier": len(outlier_stratum),
            "sampled": len(crops),
        }
        
        contact_sheet_path_str = None
        if crop_arrays:
            cs_path = assets_path / "contact_sheets" / f"sheet_{image_id}.jpg"
            if render_contact_sheet(crop_arrays, str(cs_path)):
                contact_sheet_path_str = str(cs_path)

        return ContactSheet(
            crops=crops,
            total_candidates=total_candidates,
            strata_counts=strata_counts,
            contact_sheet_image_path=contact_sheet_path_str,
        )

