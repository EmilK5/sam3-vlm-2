"""Scene-level action utility function interface (V4 Design Spec §8)."""

from typing import Protocol
from sam3_vlm.sensing.action import SensingAction


class UtilityEvaluator(Protocol):
    """Evaluates expected scene-level utility U_t(x) for candidate actions."""

    def evaluate_utility(self, action: SensingAction) -> float:
        ...
