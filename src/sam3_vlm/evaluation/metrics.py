"""Evaluation metrics containers (V4 Design Spec §17)."""

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
