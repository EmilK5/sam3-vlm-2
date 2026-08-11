"""Bipartite matching for evaluation (V4 Design Spec §17.3)."""

from typing import List, Tuple, Any
from sam3_vlm.core.geometry import BoxGeometry

def _iou(box1: BoxGeometry, box2: BoxGeometry) -> float:
    return box1.iou(box2)

def compute_matching(
    predicted_boxes: List[BoxGeometry], 
    gt_boxes: List[BoxGeometry], 
    iou_threshold: float = 0.5
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Computes bipartite matching between predicted and ground truth boxes based on IoU.
    Returns:
        matches: list of (pred_idx, gt_idx)
        unmatched_preds: list of pred_idx
        unmatched_gts: list of gt_idx
    """
    if not predicted_boxes or not gt_boxes:
        return [], list(range(len(predicted_boxes))), list(range(len(gt_boxes)))
        
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    
    cost_matrix = np.zeros((len(predicted_boxes), len(gt_boxes)))
    for i, p in enumerate(predicted_boxes):
        for j, g in enumerate(gt_boxes):
            iou = _iou(p, g)
            if iou >= iou_threshold:
                cost_matrix[i, j] = 1.0 - iou
            else:
                cost_matrix[i, j] = 1.0  # Max cost
                
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches = []
    unmatched_preds = set(range(len(predicted_boxes)))
    unmatched_gts = set(range(len(gt_boxes)))
    
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 1.0:
            matches.append((int(r), int(c)))
            unmatched_preds.discard(int(r))
            unmatched_gts.discard(int(c))
            
    return matches, list(unmatched_preds), list(unmatched_gts)
