"""Evaluation metrics for counting and discovery (V4 Design Spec §17.3)."""

from typing import Dict, Any, Optional

def compute_count_metrics(pred_count: float, gt_count: int) -> Dict[str, Any]:
    """Compute count error metrics."""
    abs_err = abs(pred_count - gt_count)
    metrics = {
        "absolute_error": abs_err,
        "signed_error": pred_count - gt_count,
        "squared_error": (pred_count - gt_count) ** 2,
    }
    
    if gt_count > 0:
        metrics["relative_error"] = abs_err / gt_count
    else:
        metrics["relative_error"] = None
        
    return metrics

def compute_discovery_metrics(matches: list, unmatched_preds: list, unmatched_gts: list) -> Dict[str, Any]:
    """Compute precision, recall, F1 based on bipartite matching."""
    tp = len(matches)
    fp = len(unmatched_preds)
    fn = len(unmatched_gts)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn
    }

from dataclasses import dataclass

@dataclass
class CountingMetrics:
    """Standard evaluation metrics summary for a sample or dataset."""

    absolute_error: float
    squared_error: float
    relative_error: float
    true_count: float
    predicted_count: float
    sam3_calls: int = 0
    qwen_calls: int = 0
    total_runtime_ms: float = 0.0
