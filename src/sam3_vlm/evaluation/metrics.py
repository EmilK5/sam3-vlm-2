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
    signed_error: float
    squared_error: float
    relative_error: float
    true_count: float
    predicted_count: float
    sam3_calls: int = 0
    qwen_calls: int = 0
    sam3_tiles: int = 0
    cleanup_calls: int = 0
    replans: int = 0
    iterations: int = 0
    total_runtime_ms: float = 0.0
    storage_bytes: int = 0
    events_json_bytes: int = 0
    mask_bytes: int = 0
    contact_sheet_bytes: int = 0
    qwen_bytes: int = 0

def aggregate_count_metrics(metrics_list: list[CountingMetrics]) -> Dict[str, Any]:
    """Aggregate a list of CountingMetrics into dataset-level statistics."""
    if not metrics_list:
        return {}
        
    n = len(metrics_list)
    mae = sum(m.absolute_error for m in metrics_list) / n
    mse = sum(m.squared_error for m in metrics_list) / n
    rmse = mse ** 0.5
    
    valid_rel = [m.relative_error for m in metrics_list if m.relative_error is not None]
    mre = sum(valid_rel) / len(valid_rel) if valid_rel else None
    
    mean_signed_error = sum(m.signed_error for m in metrics_list) / n
    
    avg_sam3 = sum(m.sam3_calls for m in metrics_list) / n
    avg_sam3_tiles = sum(m.sam3_tiles for m in metrics_list) / n
    avg_cleanup = sum(m.cleanup_calls for m in metrics_list) / n
    avg_replans = sum(m.replans for m in metrics_list) / n
    avg_iterations = sum(m.iterations for m in metrics_list) / n
    avg_qwen = sum(m.qwen_calls for m in metrics_list) / n
    avg_runtime = sum(m.total_runtime_ms for m in metrics_list) / n
    avg_storage = sum(m.storage_bytes for m in metrics_list) / n
    avg_events_json_bytes = sum(m.events_json_bytes for m in metrics_list) / n
    avg_mask_bytes = sum(m.mask_bytes for m in metrics_list) / n
    avg_contact_sheet_bytes = sum(m.contact_sheet_bytes for m in metrics_list) / n
    avg_qwen_bytes = sum(m.qwen_bytes for m in metrics_list) / n
    
    zero_gt_metrics = [m for m in metrics_list if m.true_count == 0]
    avg_zero_gt_pred = (sum(m.predicted_count for m in zero_gt_metrics) / len(zero_gt_metrics)) if zero_gt_metrics else 0.0
    
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MRE": mre,
        "mean_signed_error": mean_signed_error,
        "avg_zero_gt_pred": avg_zero_gt_pred,
        "avg_sam3_calls": avg_sam3,
        "avg_sam3_tiles": avg_sam3_tiles,
        "avg_cleanup_calls": avg_cleanup,
        "avg_replans": avg_replans,
        "avg_iterations": avg_iterations,
        "avg_qwen_calls": avg_qwen,
        "avg_runtime_ms": avg_runtime,
        "avg_storage_bytes": avg_storage,
        "avg_events_json_bytes": avg_events_json_bytes,
        "avg_mask_bytes": avg_mask_bytes,
        "avg_contact_sheet_bytes": avg_contact_sheet_bytes,
        "avg_qwen_bytes": avg_qwen_bytes,
        "n_samples": n
    }
