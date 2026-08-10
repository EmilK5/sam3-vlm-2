"""Cross-pass detection association and node registration policy (V4 Design Spec §10)."""

from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Set, Tuple
from sam3_vlm.core.config import AssociationConfig
from sam3_vlm.core.geometry import BoxGeometry
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.core.types import (
    Detection,
    NodeObservationRef,
    ObservationRelation,
)
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.node import Node


@dataclass
class AssociationResult:
    """Output container for association pass execution."""

    matched_observations: List[Tuple[str, NodeObservationRef]] = field(default_factory=list)
    new_nodes: List[Node] = field(default_factory=list)
    unmatched_detections: List[Detection] = field(default_factory=list)


class AssociationPolicy(Protocol):
    """Protocol for associating new SAM3 detections with existing graph nodes."""

    def associate(
        self,
        graph: SceneGraph,
        detections: List[Detection],
        sam3_call_id: str,
        action_id: str,
        semantic_key: str,
        id_gen: IDGenerator,
        config: AssociationConfig = AssociationConfig(),
    ) -> AssociationResult:
        ...


class IoUAssociationPolicy:
    """Deterministic IoU-based data association policy (V4 Design Spec §10.3)."""

    def associate(
        self,
        graph: SceneGraph,
        detections: List[Detection],
        sam3_call_id: str,
        action_id: str,
        semantic_key: str,
        id_gen: IDGenerator,
        config: AssociationConfig = AssociationConfig(),
    ) -> AssociationResult:
        result = AssociationResult()
        active_nodes = graph.active_nodes()

        # Track which detections are matched
        matched_detection_ids: Set[str] = set()

        for det in detections:
            det_box = det.geometry.box
            best_node: Node | None = None
            best_iou = 0.0
            overlapping_nodes: List[Tuple[Node, float]] = []

            for node in active_nodes:
                iou = det_box.iou(node.geometry.bbox())
                if iou >= config.new_node_iou_threshold:
                    overlapping_nodes.append((node, iou))
                    if iou > best_iou:
                        best_iou = iou
                        best_node = node

            if best_node is not None:
                matched_detection_ids.add(det.detection_id)

                # Determine observation relation
                if len(overlapping_nodes) > 1:
                    relation = ObservationRelation.AMBIGUOUS_ASSOCIATION
                elif best_iou >= config.iou_match_threshold:
                    relation = ObservationRelation.STRONG_MATCH
                else:
                    relation = ObservationRelation.WEAK_MATCH

                obs_id = id_gen.next_observation_id()
                obs_ref = NodeObservationRef(
                    observation_id=obs_id,
                    sam3_call_id=sam3_call_id,
                    action_id=action_id,
                    semantic_key=semantic_key,
                    detection_id=det.detection_id,
                    relation=relation,
                    score=det.score,
                    association_score=best_iou,
                )

                best_node.observations.append(obs_ref)
                best_node.diagnostics.support_count += 1

                # Update independent semantic support count if new semantic_key
                distinct_keys = {o.semantic_key for o in best_node.observations}
                best_node.diagnostics.independent_semantic_support_count = len(distinct_keys)

                # Diagnostic merge risk & ambiguous_with
                if len(overlapping_nodes) > 1:
                    best_node.diagnostics.merge_risk = max(best_node.diagnostics.merge_risk, 0.7)
                    best_node.diagnostics.ambiguous_with = [
                        n.node_id for n, _ in overlapping_nodes if n.node_id != best_node.node_id
                    ]

                result.matched_observations.append((best_node.node_id, obs_ref))
            else:
                result.unmatched_detections.append(det)

        # Unmatched detections create NEW nodes (V4 Invariant §10.4)
        for det in result.unmatched_detections:
            new_node_id = id_gen.next_node_id()
            obs_id = id_gen.next_observation_id()

            obs_ref = NodeObservationRef(
                observation_id=obs_id,
                sam3_call_id=sam3_call_id,
                action_id=action_id,
                semantic_key=semantic_key,
                detection_id=det.detection_id,
                relation=ObservationRelation.NEW_DETECTION,
                score=det.score,
                association_score=None,
            )

            new_node = Node(
                node_id=new_node_id,
                geometry=BoxGeometry(det.geometry.box),
                created_by_call_id=sam3_call_id,
                observations=[obs_ref],
            )
            graph.add_node(new_node)
            result.new_nodes.append(new_node)

        # Update duplicate_risk across all active nodes in graph
        for node in graph.active_nodes():
            other_nodes = [n for n in graph.active_nodes() if n.node_id != node.node_id]
            max_overlap = 0.0
            for other in other_nodes:
                iou = node.geometry.bbox().iou(other.geometry.bbox())
                if iou > max_overlap:
                    max_overlap = iou
            node.diagnostics.duplicate_risk = max_overlap

        return result
