"""Evidence collection, contact sheet construction, and Qwen evidence pack assembly (V4 Design Spec §5.3 / §6.1)."""

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional, Set
from sam3_vlm.core.geometry import Box
from sam3_vlm.core.types import ActionFamily
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.node import Node
from sam3_vlm.sensing.observation import SAM3Observation


@dataclass
class CropCandidateAnnotation:
    """Structured crop metadata annotation for Qwen evidence representation (V4 Design Spec §5.3)."""

    node_id: str
    box: Box
    target_support_score: float
    support_count: int
    target_support_semantic_key: Optional[str] = None
    target_support_call_id: Optional[str] = None
    target_support_action_id: Optional[str] = None
    latest_observation_score: Optional[float] = None
    latest_observation_semantic_key: Optional[str] = None
    latest_observation_relation: Optional[str] = None
    latest_observation_call_id: Optional[str] = None
    target_posterior: Optional[float] = None
    class_belief: Dict[str, float] = field(default_factory=dict)
    crop_image_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.target_posterior is None and "target" in self.class_belief:
            self.target_posterior = float(self.class_belief["target"])

    @property
    def sam3_score(self) -> float:
        """Backward-compatible alias with explicit target-support semantics."""
        return self.target_support_score

    @property
    def provenance(self) -> Optional[str]:
        """Backward-compatible alias for target-support call provenance."""
        return self.target_support_call_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "box": self.box.as_tuple(),
            "target_support_score": self.target_support_score,
            "target_support_semantic_key": self.target_support_semantic_key,
            "target_support_call_id": self.target_support_call_id,
            "target_support_action_id": self.target_support_action_id,
            "latest_observation_score": self.latest_observation_score,
            "latest_observation_semantic_key": self.latest_observation_semantic_key,
            "latest_observation_relation": self.latest_observation_relation,
            "latest_observation_call_id": self.latest_observation_call_id,
            "target_posterior": self.target_posterior,
            # Deprecated aliases retained so older replay/artifact consumers can
            # read the new schema without interpreting a latest-action score as
            # target support.
            "sam3_score": self.target_support_score,
            "support_count": self.support_count,
            "provenance": self.target_support_call_id,
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
            target_support_score=float(
                data.get("target_support_score", data.get("sam3_score", 0.5))
            ),
            support_count=data["support_count"],
            target_support_semantic_key=data.get("target_support_semantic_key"),
            target_support_call_id=data.get(
                "target_support_call_id", data.get("provenance")
            ),
            target_support_action_id=data.get("target_support_action_id"),
            latest_observation_score=data.get("latest_observation_score"),
            latest_observation_semantic_key=data.get(
                "latest_observation_semantic_key"
            ),
            latest_observation_relation=data.get("latest_observation_relation"),
            latest_observation_call_id=data.get("latest_observation_call_id"),
            target_posterior=data.get("target_posterior"),
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

    @property
    def uses_canonical_m8_policy(self) -> bool:
        classes = list(self.belief_classes or [])
        expected = ["target"] + [f"confounder{i}" for i in range(1, len(classes))]
        return bool(classes) and self.target_class == "target" and classes == expected

    def to_prompt_text(self, *, enforce_qwen_contract: bool = False) -> str:
        """Format evidence pack into a compact token-efficient text prompt for Qwen (V4 Design Spec §6.1)."""
        lines = [
            "=== IMPORTANT QWEN INSTRUCTIONS ===",
            "These crop panels are UNVERIFIED visual sensor candidates from SAM3.",
            "Do NOT label them as ground truth or final positive detections.",
            "target_support_score is target-family SAM3 sensor support, not a posterior probability.",
            "latest_observation fields describe the most recent target experiment or non-retrieval.",
            "Use possible confounders only to formulate a more specific target description.",
            "Every executable action must be a novel scene-level prompt for the target.",
            "On replanning, never repeat an exact SAM3 prompt listed in tried_sam3_prompts or semantic history.",
            (
                "Qwen never decides whether the pipeline should stop. Unless discovery_saturated is explicitly true, "
                "proposed_actions must contain exactly one novel target DISCOVERY experiment, even when current "
                "candidates look convincing. An empty proposed_actions list is permitted only when discovery is "
                "explicitly saturated. Only the controller may stop after evaluating sensor evidence and budget."
                if enforce_qwen_contract or self.uses_canonical_m8_policy
                else "If no useful new target prompt remains, return no actions."
            ),
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
            crop_path_str = f" | crop_img={c.crop_image_path}" if c.crop_image_path else ""
            belief_str = (
                "{" + ", ".join(
                    f"{name}:{probability:.3f}"
                    for name, probability in sorted(c.class_belief.items())
                ) + "}"
                if c.class_belief
                else "{}"
            )
            latest_score = (
                f"{c.latest_observation_score:.2f}"
                if c.latest_observation_score is not None
                else "none"
            )
            target_posterior = (
                f"{c.target_posterior:.3f}"
                if c.target_posterior is not None
                else "none"
            )
            lines.append(
                f"  - [{c.node_id}] box=({c.box.x1:.1f}, {c.box.y1:.1f}, {c.box.x2:.1f}, {c.box.y2:.1f}) | "
                f"target_support_score={c.target_support_score:.2f} | "
                f"target_support_key={c.target_support_semantic_key or 'unknown'} | "
                f"target_support_call={c.target_support_call_id or 'unknown'} | "
                f"latest_observation_score={latest_score} | "
                f"latest_semantic_key={c.latest_observation_semantic_key or 'none'} | "
                f"latest_relation={c.latest_observation_relation or 'none'} | "
                f"target_posterior={target_posterior} | class_belief={belief_str} | "
                f"support={c.support_count}{crop_path_str}"
            )
        lines.append("==========================")
        return "\n".join(lines)


@dataclass
class EvidencePack:
    """Pack of sensory evidence prepared for graph updates or Qwen planning."""

    observations: List[SAM3Observation] = field(default_factory=list)

def _target_family_call_ids(semantic_memory: Any) -> Set[str]:
    """Return call IDs belonging to target-oriented semantic experiments."""
    target_call_ids: Set[str] = set()
    records = getattr(semantic_memory, "records", {}) if semantic_memory else {}
    for record in records.values():
        family = getattr(record, "family", None)
        family_value = getattr(family, "value", family)
        if family_value in {
            ActionFamily.DISCOVERY.value,
            ActionFamily.VERIFICATION.value,
        }:
            target_call_ids.update(getattr(record, "sam3_call_ids", ()) or ())
    return target_call_ids


def _target_support_observation(node: Node, semantic_memory: Any = None):
    """Return stable target-family sensor support for a graph candidate.

    When semantic history is available, target-oriented DISCOVERY and
    VERIFICATION calls define eligible evidence. The first grounded detection
    and its semantic key provide a backward-compatible fallback for bootstrap
    callers and older replay states.

    The anchor semantic key is taken from the first sensor-grounded detection
    observation. Repeated detections under that same semantic key may strengthen
    the score. NOT_RETRIEVED observations and later confounder experiments do
    not overwrite it.
    """
    observations = list(getattr(node, "observations", ()) or ())
    if not observations:
        return None

    grounded = [
        obs
        for obs in observations
        if getattr(obs, "detection_id", None) is not None
        and getattr(obs, "score", None) is not None
    ]
    anchor = grounded[0] if grounded else None

    if anchor is None:
        return None

    target_call_ids = _target_family_call_ids(semantic_memory)
    target_detections = [
        obs for obs in grounded if obs.sam3_call_id in target_call_ids
    ]
    if not target_detections:
        target_detections = [
            obs for obs in grounded if obs.semantic_key == anchor.semantic_key
        ]

    return max(target_detections, key=lambda obs: float(obs.score))

class ContactSheetBuilder:
    """Stratified and spatially distributed candidate crop sampler (V4 Design Spec §5.3)."""

    def build_contact_sheet(
        self,
        graph: SceneGraph,
        max_crops: int = 24,
        image: Any = None,
        assets_dir: str = "assets",
        image_id: str = "bootstrap",
        semantic_memory: Any = None,
        target_class: str = "target",
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
            support_obs = _target_support_observation(node, semantic_memory)
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
            support_obs = _target_support_observation(node, semantic_memory)
            latest_obs = node.observations[-1] if node.observations else None

            score = (
                float(support_obs.score)
                if support_obs is not None and support_obs.score is not None
                else 0.5
            )

            target_support_call_id = (
                support_obs.sam3_call_id if support_obs is not None else node.created_by_call_id
            )
            latest_relation = getattr(latest_obs, "relation", None)
            latest_relation_value = getattr(latest_relation, "value", latest_relation)
            target_posterior = node.class_belief.probabilities.get(target_class)
            if target_posterior is None and target_class != "target":
                target_posterior = node.class_belief.probabilities.get("target")

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
                    target_support_score=score,
                    support_count=node.diagnostics.support_count,
                    target_support_semantic_key=(
                        support_obs.semantic_key if support_obs is not None else None
                    ),
                    target_support_call_id=target_support_call_id,
                    target_support_action_id=(
                        support_obs.action_id if support_obs is not None else None
                    ),
                    latest_observation_score=(
                        float(latest_obs.score)
                        if latest_obs is not None and latest_obs.score is not None
                        else None
                    ),
                    latest_observation_semantic_key=(
                        latest_obs.semantic_key if latest_obs is not None else None
                    ),
                    latest_observation_relation=(
                        str(latest_relation_value)
                        if latest_relation_value is not None
                        else None
                    ),
                    latest_observation_call_id=(
                        latest_obs.sam3_call_id if latest_obs is not None else None
                    ),
                    target_posterior=(
                        float(target_posterior)
                        if target_posterior is not None
                        else None
                    ),
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
