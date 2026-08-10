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

    def evaluate_utility(self, entry: "ActionBankEntry", state: "SceneState", config: "V4Config") -> UtilityBreakdown:
        ...


class DefaultUtilityEvaluator:
    """State-aware active perception controller utility (V4 Design Spec §8)."""
    
    def evaluate_utility(self, entry: "ActionBankEntry", state: "SceneState", config: "V4Config") -> UtilityBreakdown:
        from sam3_vlm.core.types import ActionFamily, SpatialMode
        
        cfg = config.action_selection
        iteration = state.iteration
        
        # Determine empirical penalty from semantic history
        history_penalty = 0.0
        rec = state.semantic_memory.records.get(entry.action.semantic_key)
        if rec and rec.execution_count > 0:
            if sum(rec.new_nodes_by_execution) == 0:
                history_penalty = 0.5 * rec.execution_count # Heavy penalty for repeatedly failing to find anything
                
        # 1. Discovery value D_t(x) based on novelty and history
        discovery_value = 0.0
        if entry.action.family == ActionFamily.DISCOVERY:
            # decays slightly over iterations, but penalized if past executions of this key failed
            discovery_value = max(0.0, 1.0 - (iteration * 0.1) - history_penalty)
        elif entry.action.family == ActionFamily.CONTEXT:
            discovery_value = 0.2
            
        # 2. Discrimination value I_t(x) based on uncertainty and iteration
        discrimination_value = 0.0
        if entry.action.family in (ActionFamily.VERIFICATION, ActionFamily.CONFOUNDER):
            # scales with iteration
            discrimination_value = min(1.0, 0.5 + (iteration * 0.1) - history_penalty)

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
            cfg.alpha_discovery * discovery_value
            + cfg.beta_discrimination * discrimination_value
            - cfg.gamma_redundancy * redundancy_cost
            - cfg.lambda_cost * compute_cost
            + cfg.eta_qwen_priority * qwen_priority
        )

        return UtilityBreakdown(
            discovery_value=discovery_value,
            discrimination_value=discrimination_value,
            redundancy_cost=redundancy_cost,
            compute_cost=compute_cost,
            qwen_priority=qwen_priority,
            total_utility=total_utility,
        )
