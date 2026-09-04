"""Scene-level action utility function interface and breakdown (V4 Design Spec §8 / §24.2)."""

from dataclasses import dataclass
import math
from typing import Any, Optional, Protocol, TYPE_CHECKING

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

    @staticmethod
    def _history_record(entry: "ActionBankEntry", state: "SceneState") -> Optional[Any]:
        """Resolve history by correlation group, not by posterior coordinate alias."""
        from sam3_vlm.planning.action_bank import canonicalize_semantic_key

        group = entry.action.correlation_group or entry.action.semantic_key
        if group in state.semantic_memory.records:
            return state.semantic_memory.records[group]
        normalized = canonicalize_semantic_key(group)
        for key, record in state.semantic_memory.records.items():
            if canonicalize_semantic_key(key) == normalized:
                return record
        return None

    def evaluate_utility(self, entry: "ActionBankEntry", state: "SceneState", config: "V4Config") -> UtilityBreakdown:
        from sam3_vlm.core.types import ActionFamily, SpatialMode
        
        cfg = config.action_selection
        strict_m8 = state.uses_canonical_m8_policy

        # Context is never a global-loop experiment. Strict M8 additionally
        # permits only target discovery prompts; those prompts serve both
        # discovery and uncertainty reduction.
        if entry.action.family == ActionFamily.CONTEXT:
            return UtilityBreakdown(total_utility=-1.0)
        if strict_m8 and (
            entry.action.family != ActionFamily.DISCOVERY
            or entry.action.semantic_key != "target"
        ):
            return UtilityBreakdown(total_utility=-1.0)

        # Preserve the generic M4-M7 history penalty. M8 uses exact prompt
        # deduplication and a scene-level discovery plateau instead.
        history_penalty = 0.0
        rec = self._history_record(entry, state)
        if rec and rec.execution_count > 0 and not strict_m8:
            if sum(rec.new_nodes_by_execution) == 0:
                history_penalty = 0.5 * rec.execution_count
                
        # Calculate active node stats
        active_nodes = state.graph.active_nodes()
        total_entropy = sum(n.class_belief.entropy for n in active_nodes)
        
        # 1. Discovery value D_t(x) based on novelty and history
        discovery_value = 0.0
        if entry.action.family == ActionFamily.DISCOVERY:
            base_discovery = max(0.0, 1.0 - (len(active_nodes) * 0.02))
            if strict_m8:
                recent_window = max(1, config.replanning.discovery_plateau_steps)
                recent_gains = state.discovery_state.recent_new_node_counts[
                    -recent_window:
                ]
                plateaued = (
                    state.discovery_state.saturated
                    or (
                        len(recent_gains) >= recent_window
                        and sum(recent_gains)
                        <= config.stopping.discovery_saturation_threshold
                    )
                )
                discovery_value = 0.0 if plateaued else base_discovery
            else:
                discovery_value = max(0.0, base_discovery - history_penalty)
            
        # 2. Uncertainty-reduction proxy. M8 computes it only for a target
        # prompt. Generic runs retain verification/confounder behavior.
        discrimination_value = 0.0
        if strict_m8 and active_nodes:
            max_entropy = math.log2(max(2, len(state.belief_classes)))
            discrimination_value = min(
                1.0,
                total_entropy / (len(active_nodes) * max_entropy),
            )
        elif entry.action.family in (ActionFamily.VERIFICATION, ActionFamily.CONFOUNDER):
            discrimination_value = min(1.0, 0.2 + (total_entropy * 0.1) - history_penalty)

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
