"""Scene-level action utility function interface and breakdown (V4 Design Spec §8 / §24.2)."""

from dataclasses import dataclass
from typing import Protocol
from sam3_vlm.sensing.action import SensingAction


@dataclass
class UtilityBreakdown:
    """Explicit component breakdown of action utility calculation U_t(x)."""

    discovery_value: float = 0.0
    discrimination_value: float = 0.0
    redundancy_cost: float = 0.0
    compute_cost: float = 0.0
    qwen_priority: float = 0.0
    total_utility: float = 0.0


class UtilityEvaluator(Protocol):
    """Evaluates expected scene-level utility U_t(x) for candidate actions."""

    def evaluate_utility(self, action: SensingAction) -> UtilityBreakdown:
        ...
