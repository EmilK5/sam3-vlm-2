"""Post-inference count evaluation; annotations never enter the controller."""

from __future__ import annotations

from .core import Result


def evaluate_count(result: Result, ground_truth_count: int) -> dict[str, float | int | None]:
    if type(ground_truth_count) is not int or ground_truth_count < 0:
        raise ValueError("ground_truth_count must be a non-negative integer")
    soft = result.counts["soft"]
    hard = result.counts["hard"]
    return {"ground_truth_count": ground_truth_count,
            "soft_absolute_error": None if soft is None else abs(soft - ground_truth_count),
            "hard_absolute_error": None if hard is None else abs(hard - ground_truth_count)}
