"""Observational mask overlap diagnostics, without changing associations or counts."""

from itertools import combinations
from statistics import median

from sam3_vlm.core.geometry import MaskGeometry, mask_overlap


def audit_masks(nodes, *, iou_threshold=.3, iom_threshold=.9, top_k=20):
    nodes = list(nodes)
    valid, missing = [], []
    for node in nodes:
        geometry = node.geometry
        if not isinstance(geometry, MaskGeometry) or geometry.mask is None or geometry.pixel_area <= 0:
            missing.append(node.node_id)
        else:
            valid.append(node)
    iou_pairs = iom_pairs = large_ratio_pairs = 0
    examples = []
    for first, second in combinations(valid, 2):
        iou, iom = mask_overlap(first.geometry, second.geometry)
        high_iou, high_iom = iou >= iou_threshold, iom >= iom_threshold
        iou_pairs += high_iou
        iom_pairs += high_iom
        if not (high_iou or high_iom):
            continue
        areas = [first.geometry.pixel_area, second.geometry.pixel_area]
        ratio = max(areas) / min(areas)
        large_ratio_pairs += high_iom and ratio > 4
        examples.append({"node_ids": [first.node_id, second.node_id], "iou": iou, "iom": iom,
            "pixel_areas": areas, "area_ratio": ratio,
            "crop_boundary_clipped": [first.geometry.crop_boundary_clipped, second.geometry.crop_boundary_clipped],
            "target_probabilities": [node.class_belief.probabilities.get("target") for node in (first, second)],
            "created_by_call_ids": [first.created_by_call_id, second.created_by_call_id]})
        examples.sort(key=lambda pair: (-pair["iou"], -pair["iom"], pair["node_ids"]))
        del examples[top_k:]
    areas = [node.geometry.pixel_area for node in valid]
    return {"complete": not missing, "active_nodes": len(nodes), "audited_mask_nodes": len(valid),
        "missing_mask_nodes": missing[:top_k], "missing_mask_count": len(missing),
        "iou_threshold": iou_threshold, "iom_threshold": iom_threshold,
        "high_iou_pairs": iou_pairs, "high_iom_pairs": iom_pairs,
        "large_area_ratio_containment_pairs": large_ratio_pairs,
        "median_mask_area": median(areas) if areas else None,
        "crop_boundary_clipped_nodes": sum(node.geometry.crop_boundary_clipped for node in valid),
        "top_overlap_pairs": examples,
        "note": "Overlap flags are candidates for visual review, not proof of duplicate objects. Masks only; no box fallback."}
