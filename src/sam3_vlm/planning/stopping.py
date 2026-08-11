"""Pipeline stopping criteria protocol (V4 Design Spec §14)."""

from typing import Protocol, TYPE_CHECKING, Optional
from sam3_vlm.core.config import StoppingConfig
from sam3_vlm.core.types import BudgetState, StopReason

if TYPE_CHECKING:
    from sam3_vlm.scene.state import SceneState
    from sam3_vlm.core.config import V4Config


class StoppingCondition(Protocol):
    """Evaluates whether loop termination criteria have been met."""

    def should_stop(self, state: "SceneState", config: "V4Config") -> Optional[StopReason]:
        ...


class BudgetStoppingCondition:
    """Stops when hard SAM3 budget is exhausted."""
    
    def should_stop(self, state: "SceneState", config: "V4Config") -> Optional[StopReason]:
        if state.budget.sam3_calls >= config.budget.max_sam3_calls:
            return StopReason.SAM3_BUDGET
        if state.budget.qwen_calls >= config.budget.max_qwen_calls:
            # We don't always stop entirely on Qwen budget (we can still sense if bank is valid), 
            # but if we run out of valid actions and Qwen budget is empty, this handles it. 
            pass 
        if config.budget.max_runtime_seconds and (state.budget.total_runtime_ms / 1000.0) >= config.budget.max_runtime_seconds:
            return StopReason.RUNTIME_BUDGET
        return None


class IterationStoppingCondition:
    """Stops when the maximum number of iterations is reached."""
    
    def should_stop(self, state: "SceneState", config: "V4Config") -> Optional[StopReason]:
        if state.iteration >= config.stopping.max_iterations:
            return StopReason.MAX_ITERATIONS
        return None


class DiscoveryAndUncertaintySaturatedStoppingCondition:
    """Stops when discovery hits a plateau AND count variance/entropy is low."""
    
    def should_stop(self, state: "SceneState", config: "V4Config") -> Optional[StopReason]:
        # Check if the target mass increment over the last `discovery_plateau_steps` is below saturation threshold
        plateau_steps = config.replanning.discovery_plateau_steps
        if state.iteration < plateau_steps:
            return None
            
        recent_counts = state.discovery_state.recent_new_node_counts
        if len(recent_counts) >= plateau_steps:
            rolling_sum = sum(recent_counts[-plateau_steps:])
            if rolling_sum <= config.stopping.discovery_saturation_threshold:
                # We have reached a discovery plateau.
                # Now check if uncertainty is also saturated
                entropy = sum(n.class_belief.entropy for n in state.graph.active_nodes())
                variance = state.count_estimate.variance
                
                if (entropy <= config.replanning.unresolved_entropy_threshold and 
                    variance <= config.stopping.count_variance_threshold):
                    return StopReason.DISCOVERY_AND_UNCERTAINTY_SATURATED
                    
        return None


class CompositeStoppingCondition:
    """Checks multiple stopping conditions, returning the first StopReason if any are met."""
    
    def __init__(self, conditions: list[StoppingCondition]):
        self.conditions = conditions
        
    def should_stop(self, state: "SceneState", config: "V4Config") -> Optional[StopReason]:
        for condition in self.conditions:
            reason = condition.should_stop(state, config)
            if reason is not None:
                return reason
        return None
