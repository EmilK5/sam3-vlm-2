"""Pipeline stopping criteria protocol (V4 Design Spec §14)."""

from typing import Protocol, TYPE_CHECKING
from sam3_vlm.core.config import StoppingConfig
from sam3_vlm.core.types import BudgetState

if TYPE_CHECKING:
    from sam3_vlm.scene.state import SceneState
    from sam3_vlm.core.config import V4Config


class StoppingCondition(Protocol):
    """Evaluates whether loop termination criteria have been met."""

    def should_stop(self, state: "SceneState", config: "V4Config") -> bool:
        ...


class BudgetStoppingCondition:
    """Stops when hard SAM3 budget is exhausted."""
    
    def should_stop(self, state: "SceneState", config: "V4Config") -> bool:
        return state.budget.sam3_calls >= config.budget.max_sam3_calls


class IterationStoppingCondition:
    """Stops when the maximum number of iterations is reached."""
    
    def should_stop(self, state: "SceneState", config: "V4Config") -> bool:
        return state.iteration >= config.stopping.max_iterations


class MarginalUtilityStoppingCondition:
    """Stops when all available actions in the bank have low utility."""
    
    def should_stop(self, state: "SceneState", config: "V4Config") -> bool:
        if state.action_bank is None or not state.action_bank.unexecuted_entries():
            return False # handled by empty bank replanning
        
        # This requires utility evaluation, which might be better handled directly in the Runner
        return False


class DiscoveryPlateauStoppingCondition:
    """Stops when discovery hits a plateau (no new target mass found recently)."""
    
    def should_stop(self, state: "SceneState", config: "V4Config") -> bool:
        # Check if the target mass increment over the last `discovery_plateau_steps` is below saturation threshold
        plateau_steps = config.replanning.discovery_plateau_steps
        if state.iteration < plateau_steps:
            return False
            
        recent_counts = state.discovery_state.recent_new_node_counts
        if len(recent_counts) >= plateau_steps:
            rolling_sum = sum(recent_counts[-plateau_steps:])
            if rolling_sum <= config.stopping.discovery_saturation_threshold:
                # We have reached a discovery plateau.
                # Do NOT stop if there are useful CONFOUNDER or VERIFICATION actions unexecuted.
                if state.action_bank is not None:
                    from sam3_vlm.core.types import ActionFamily
                    for entry in state.action_bank.unexecuted_entries():
                        if entry.action.family in (ActionFamily.CONFOUNDER, ActionFamily.VERIFICATION):
                            return False # let discrimination continue
                return True
        return False


class CompositeStoppingCondition:
    """Checks multiple stopping conditions, returning True if any are met."""
    
    def __init__(self, conditions: list[StoppingCondition]):
        self.conditions = conditions
        self.stop_reason = None
        
    def should_stop(self, state: "SceneState", config: "V4Config") -> bool:
        for condition in self.conditions:
            if condition.should_stop(state, config):
                self.stop_reason = condition.__class__.__name__
                return True
        self.stop_reason = None
        return False
