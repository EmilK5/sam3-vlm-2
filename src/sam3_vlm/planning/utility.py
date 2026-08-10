"""Scene-level action utility function interface and breakdown (V4 Design Spec §8 / §24.2)."""

from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING
from sam3_vlm.sensing.action import SensingAction

if TYPE_CHECKING:
    from sam3_vlm.planning.action_bank import ActionBankEntry


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

    def evaluate_utility(self, entry: "ActionBankEntry", iteration: int = 0) -> UtilityBreakdown:
        ...


class DefaultUtilityEvaluator:
    """Default numerical active perception controller utility (V4 Design Spec §8)."""
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0, lambda_: float = 0.5, eta: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_ = lambda_
        self.eta = eta

    def evaluate_utility(self, entry: "ActionBankEntry", iteration: int = 0) -> UtilityBreakdown:
        from sam3_vlm.core.types import ActionFamily, SpatialMode
        
        # 1. Discovery value D_t(x) decays slightly over iterations
        discovery_value = 0.0
        if entry.action.family == ActionFamily.DISCOVERY:
            # high initial discovery value, decaying
            discovery_value = max(0.2, 1.0 - (iteration * 0.1))
        elif entry.action.family == ActionFamily.CONTEXT:
            discovery_value = 0.5
            
        # 2. Discrimination value I_t(x) increases slightly over iterations
        discrimination_value = 0.0
        if entry.action.family in (ActionFamily.VERIFICATION, ActionFamily.CONFOUNDER):
            discrimination_value = min(1.0, 0.5 + (iteration * 0.1))

        # 3. Redundancy cost R_t(x)
        redundancy_cost = entry.redundancy

        # 4. Compute cost C(x)
        compute_cost = 1.0
        if entry.action.spatial_mode == SpatialMode.TILED:
            # Tiled costs much more
            compute_cost = 4.0
            
        # 5. Qwen priority Q_t(x)
        qwen_priority = entry.qwen_priority if entry.qwen_priority is not None else 0.5

        total_utility = (
            self.alpha * discovery_value
            + self.beta * discrimination_value
            - self.gamma * redundancy_cost
            - self.lambda_ * compute_cost
            + self.eta * qwen_priority
        )

        return UtilityBreakdown(
            discovery_value=discovery_value,
            discrimination_value=discrimination_value,
            redundancy_cost=redundancy_cost,
            compute_cost=compute_cost,
            qwen_priority=qwen_priority,
            total_utility=total_utility,
        )
