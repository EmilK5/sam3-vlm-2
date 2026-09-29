"""Mask IoU/IoM association for A–E, with a historical box-fixture mode.

Supported runs compare binary mask pixels in original-image coordinates.
IoM associates crop fragments without letting box overlap merge disjoint masks.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from sam3_vlm.core.config import AssociationConfig
from sam3_vlm.core.geometry import Box, BoxGeometry, MaskGeometry, mask_overlap, detection_mask_geometry
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.core.types import Detection, NodeObservationRef, ObservationRelation
from sam3_vlm.scene.association import AssociationResult
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.node import Node


def box_iom(box_a: Box, box_b: Box) -> float:
    """Intersection over minimum box area, in [0, 1]."""
    min_area = min(float(box_a.area), float(box_b.area))
    if min_area <= 0.0:
        return 0.0
    return float(box_a.intersection(box_b)) / min_area


def dual_overlap(box_a: Box, box_b: Box) -> Tuple[float, float]:
    return float(box_a.iou(box_b)), box_iom(box_a, box_b)


def _same_detection(
    box_a: Box,
    box_b: Box,
    *,
    iou_threshold: float,
    iom_threshold: float,
) -> bool:
    iou, iom = dual_overlap(box_a, box_b)
    return iou >= iou_threshold or iom >= iom_threshold


def deduplicate_observation_detections(
    detections: List[Detection],
    config: AssociationConfig,
) -> List[Detection]:
    """Score-greedy intra-call suppression using the IoU OR IoM dual gate.

    Association used to compare a call only with nodes that existed before the
    call.  Consequently two detections of one fruit from overlapping tiles
    could both be unmatched and both become new nodes.  Suppressing duplicates
    inside the observation first closes that hole.
    """
    ranked = sorted(
        enumerate(detections),
        key=lambda item: (-float(item[1].score), str(item[1].detection_id)),
    )
    kept_ranked: List[Detection] = []
    kept_indices = set()
    geometries = {id(det): detection_mask_geometry(det) for det in detections} if config.mask_only else {}
    for index, detection in ranked:
        candidate = geometries.get(id(detection), detection.geometry.bbox())
        duplicate = False
        for survivor in kept_ranked:
            if config.mask_only:
                iou, iom = mask_overlap(candidate, geometries[id(survivor)])
                duplicate = (iou >= config.tiled_nms_threshold or
                             (config.enable_iom_dedup and iom >= config.tiled_nms_iom_threshold))
            else:
                duplicate = _same_detection(candidate, survivor.geometry.bbox(),
                    iou_threshold=config.tiled_nms_threshold,
                    iom_threshold=config.tiled_nms_iom_threshold)
            if duplicate:
                break
        if duplicate:
            continue
        kept_ranked.append(detection)
        kept_indices.add(index)
    # Keep surviving detections in sensor order so node-ID assignment remains
    # backward-compatible; score ordering is used only to choose the survivor.
    return [det for index, det in enumerate(detections) if index in kept_indices]


class IoUIoMAssociationPolicy:
    """Deterministic dual-gate association and graph registration policy."""

    def associate(
        self,
        graph: SceneGraph,
        detections: List[Detection],
        sam3_call_id: str,
        action_id: str,
        semantic_key: str,
        id_gen: IDGenerator,
        config: AssociationConfig = AssociationConfig(),
        correlation_group: Optional[str] = None,
    ) -> AssociationResult:
        del correlation_group  # semantic correlation is handled by belief fusion
        result = AssociationResult()
        active_nodes = graph.active_nodes()

        for det in deduplicate_observation_detections(detections, config):
            det_geometry = detection_mask_geometry(det) if config.mask_only else BoxGeometry(det.geometry.bbox())
            det_box = det_geometry.bbox()
            best_node: Optional[Node] = None
            best_iou = 0.0
            best_iom = 0.0
            best_score = -1.0
            overlapping_nodes: List[Tuple[Node, float, float]] = []

            for node in active_nodes:
                node_box = node.geometry.bbox()
                iou, iom = mask_overlap(det_geometry, node.geometry) if config.mask_only else dual_overlap(det_box, node_box)
                if config.mask_only and not config.enable_iom_dedup:
                    iom = 0.0
                iom_candidate = iom >= config.new_node_iom_threshold
                if iou >= config.new_node_iou_threshold or iom_candidate:
                    overlapping_nodes.append((node, iou, iom))
                    overlap_score = max(iou, iom)
                    if overlap_score > best_score:
                        best_score = overlap_score
                        best_iou = iou
                        best_iom = iom
                        best_node = node

            if best_node is None:
                result.unmatched_detections.append(det)
                continue

            if len(overlapping_nodes) > 1:
                relation = ObservationRelation.AMBIGUOUS_ASSOCIATION
            elif (
                best_iou >= config.iou_match_threshold
                or best_iom >= config.iom_match_threshold
            ):
                relation = ObservationRelation.STRONG_MATCH
            else:
                relation = ObservationRelation.WEAK_MATCH

            obs_ref = NodeObservationRef(
                observation_id=id_gen.next_observation_id(),
                sam3_call_id=sam3_call_id,
                action_id=action_id,
                semantic_key=semantic_key,
                detection_id=det.detection_id,
                relation=relation,
                score=det.score,
                association_score=max(best_iou, best_iom),
            )
            # Keep the complete mask when later tiles return clipped fragments.
            if config.mask_only and det_geometry.area() > best_node.geometry.area() and len(overlapping_nodes) == 1:
                best_node.geometry = det_geometry
            best_node.observations.append(obs_ref)
            best_node.diagnostics.support_count += 1
            best_node.diagnostics.independent_semantic_support_count = len(
                {obs.semantic_key for obs in best_node.observations}
            )

            if len(overlapping_nodes) > 1:
                best_node.diagnostics.merge_risk = max(
                    best_node.diagnostics.merge_risk, 0.7
                )
                best_node.diagnostics.ambiguous_with = [
                    node.node_id
                    for node, _, _ in overlapping_nodes
                    if node.node_id != best_node.node_id
                ]

            result.matched_observations.append((best_node.node_id, obs_ref))

        for det in result.unmatched_detections:
            obs_ref = NodeObservationRef(
                observation_id=id_gen.next_observation_id(),
                sam3_call_id=sam3_call_id,
                action_id=action_id,
                semantic_key=semantic_key,
                detection_id=det.detection_id,
                relation=ObservationRelation.NEW_DETECTION,
                score=det.score,
                association_score=None,
            )
            new_node = Node(
                node_id=id_gen.next_node_id(),
                geometry=detection_mask_geometry(det) if config.mask_only else BoxGeometry(det.geometry.bbox()),
                created_by_call_id=sam3_call_id,
                observations=[obs_ref],
            )
            graph.add_node(new_node)
            result.new_nodes.append(new_node)

        active_after = graph.active_nodes()
        for node in active_after:
            max_overlap = 0.0
            node_box = node.geometry.bbox()
            for other in active_after:
                if other.node_id == node.node_id:
                    continue
                iou, iom = (mask_overlap(node.geometry, other.geometry) if config.mask_only
                            else dual_overlap(node_box, other.geometry.bbox()))
                if config.mask_only and not config.enable_iom_dedup:
                    iom = 0.0
                max_overlap = max(max_overlap, iou, iom)
            node.diagnostics.duplicate_risk = max_overlap

        return result


__all__ = [
    "IoUIoMAssociationPolicy",
    "box_iom",
    "deduplicate_observation_detections",
    "dual_overlap",
]
