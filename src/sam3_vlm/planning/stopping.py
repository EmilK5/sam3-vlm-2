"""Pipeline stopping criteria protocol (V4 Design Spec §14)."""

from typing import Protocol
from sam3_vlm.core.config import StoppingConfig
from sam3_vlm.core.types import BudgetState


class StoppingCondition(Protocol):
    """Evaluates whether loop termination criteria have been met."""

    def should_stop(self, budget: BudgetState, config: StoppingConfig) -> bool:
        ...
