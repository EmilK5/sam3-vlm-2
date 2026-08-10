"""Evidence collection, contact sheet construction, and Qwen evidence pack assembly (V4 Design Spec §5.3 / §6.1)."""

from dataclasses import dataclass, field
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


@dataclass
class ContactSheet:
    """Compact representative candidate contact sheet container (V4 Design Spec §5.3)."""

    crops: List[CropCandidateAnnotation] = field(default_factory=list)
    total_candidates: int = 0
    strata_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class QwenEvidencePack:
    """Structured evidence pack presented to Qwen scene planner (V4 Design Spec §6.1)."""

    original_image_id: str
    user_prompt: str
    target_class: str
    contact_sheet: ContactSheet
    scene_summary: str = ""
    discovery_diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidencePack:
    """Pack of sensory evidence prepared for graph updates or Qwen planning."""

    observations: List[SAM3Observation] = field(default_factory=list)


class ContactSheetBuilder:
    """Stratified candidate crop sampler for compact Qwen evidence representation (V4 Design Spec §5.3)."""

    def build_contact_sheet(
        self, graph: SceneGraph, max_crops: int = 24
    ) -> ContactSheet:
        """Sample candidate crops across confidence/support strata."""
        active_nodes = graph.active_nodes()
        total_candidates = len(active_nodes)

        if total_candidates == 0:
            return ContactSheet(crops=[], total_candidates=0, strata_counts={})

        # Categorize nodes into strata
        high_stratum: List[Node] = []
        medium_stratum: List[Node] = []
        low_stratum: List[Node] = []
        outlier_stratum: List[Node] = []

        for node in active_nodes:
            # Score from latest observation or default 0.5
            score = node.observations[-1].score if node.observations and node.observations[-1].score is not None else 0.5

            if node.diagnostics.duplicate_risk > 0.6 or node.diagnostics.merge_risk > 0.6:
                outlier_stratum.append(node)
            elif score >= 0.7:
                high_stratum.append(node)
            elif score >= 0.4:
                medium_stratum.append(node)
            else:
                low_stratum.append(node)

        # Budget per stratum
        per_stratum_cap = max(1, max_crops // 4)
        sampled_nodes: List[Node] = []

        sampled_nodes.extend(high_stratum[:per_stratum_cap])
        sampled_nodes.extend(medium_stratum[:per_stratum_cap])
        sampled_nodes.extend(low_stratum[:per_stratum_cap])
        sampled_nodes.extend(outlier_stratum[:per_stratum_cap])

        # Fill remaining budget from remaining nodes
        remaining_budget = max_crops - len(sampled_nodes)
        if remaining_budget > 0:
            already_sampled_ids = {n.node_id for n in sampled_nodes}
            remaining_nodes = [n for n in active_nodes if n.node_id not in already_sampled_ids]
            sampled_nodes.extend(remaining_nodes[:remaining_budget])

        crops: List[CropCandidateAnnotation] = []
        for node in sampled_nodes:
            latest_obs = node.observations[-1] if node.observations else None
            score = latest_obs.score if latest_obs and latest_obs.score is not None else 0.5
            prov = latest_obs.sam3_call_id if latest_obs else node.created_by_call_id

            crops.append(
                CropCandidateAnnotation(
                    node_id=node.node_id,
                    box=node.geometry.bbox(),
                    sam3_score=score,
                    support_count=node.diagnostics.support_count,
                    provenance=prov,
                    class_belief=dict(node.class_belief.probabilities),
                )
            )

        strata_counts = {
            "high": len(high_stratum),
            "medium": len(medium_stratum),
            "low": len(low_stratum),
            "outlier": len(outlier_stratum),
            "sampled": len(crops),
        }

        return ContactSheet(
            crops=crops,
            total_candidates=total_candidates,
            strata_counts=strata_counts,
        )
