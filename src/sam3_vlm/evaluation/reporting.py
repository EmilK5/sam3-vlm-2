"""Reporting aggregation for M7 evaluation (V4 Design Spec §17.3)."""

from typing import Dict, Any, List
import json
from pathlib import Path

from sam3_vlm.logging.schema import RunSummary

def generate_report(summaries: List[RunSummary]) -> Dict[str, Any]:
    """Aggregates multiple run summaries into a single dataset report."""
    if not summaries:
        return {}
        
    total_sam3 = sum(s.sam3_calls for s in summaries)
    total_qwen = sum(s.qwen_calls for s in summaries)
    total_runtime = sum(s.runtime_ms for s in summaries)
    
    total_sam3_tiles = sum(s.sam3_tiles for s in summaries)
    total_cleanup = sum(s.cleanup_calls for s in summaries)
    total_replans = sum(s.number_of_replans for s in summaries)
    total_storage = sum(s.evaluation_fields.get("total_run_bytes", 0) for s in summaries)
    
    runs_with_gt = sum(1 for s in summaries if "absolute_error" in s.evaluation_fields)
    avg_mae = sum(s.evaluation_fields.get("absolute_error", 0.0) for s in summaries if "absolute_error" in s.evaluation_fields) / runs_with_gt if runs_with_gt > 0 else 0.0
    avg_mse = sum(s.evaluation_fields.get("squared_error", 0.0) for s in summaries if "squared_error" in s.evaluation_fields) / runs_with_gt if runs_with_gt > 0 else 0.0
    rmse = avg_mse ** 0.5
    bias = sum(s.evaluation_fields.get("signed_error", 0.0) for s in summaries if "signed_error" in s.evaluation_fields) / runs_with_gt if runs_with_gt > 0 else 0.0
    
    return {
        "num_runs": len(summaries),
        "total_sam3_calls": total_sam3,
        "total_sam3_tiles": total_sam3_tiles,
        "total_cleanup_calls": total_cleanup,
        "total_qwen_calls": total_qwen,
        "total_replans": total_replans,
        "total_runtime_ms": total_runtime,
        "total_storage_bytes": total_storage,
        "average_runtime_ms": total_runtime / len(summaries),
        "average_mae": avg_mae,
        "rmse": rmse,
        "bias": bias
    }
