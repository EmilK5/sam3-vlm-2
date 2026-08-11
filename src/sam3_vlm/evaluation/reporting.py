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
    
    return {
        "num_runs": len(summaries),
        "total_sam3_calls": total_sam3,
        "total_qwen_calls": total_qwen,
        "total_runtime_ms": total_runtime,
        "average_runtime_ms": total_runtime / len(summaries),
    }
