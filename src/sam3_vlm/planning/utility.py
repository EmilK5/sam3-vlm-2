"""Scene-level action utility function interface and breakdown (V4 Design Spec §8 / §24.2)."""

from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, TYPE_CHECKING

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
    def _uses_canonical_m8_policy(state: "SceneState") -> bool:
        classes = list(getattr(state, "belief_classes", []) or [])
        expected = ["target"] + [f"confounder{i}" for i in range(1, len(classes))]
        return bool(classes) and classes == expected and state.target_class == "target"

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

    @staticmethod
    def _values_for_family(record: Any, values: List[float], family: Any) -> List[float]:
        """Select execution-aligned values, with old-artifact compatibility."""
        family_value = getattr(family, "value", family)
        recorded_families = list(
            getattr(record, "families_by_execution", []) or []
        )
        if len(recorded_families) == len(values):
            return [
                float(value)
                for value, recorded_family in zip(values, recorded_families)
                if getattr(recorded_family, "value", recorded_family) == family_value
            ]

        record_family = getattr(record, "family", None)
        if getattr(record_family, "value", record_family) == family_value:
            return [float(value) for value in values]
        return []

    def evaluate_utility(self, entry: "ActionBankEntry", state: "SceneState", config: "V4Config") -> UtilityBreakdown:
        from sam3_vlm.core.types import ActionFamily, SpatialMode
        
        cfg = config.action_selection
        iteration = state.iteration
        
        # If action is CONTEXT, return useless utility for M4
        if entry.action.family == ActionFamily.CONTEXT:
            return UtilityBreakdown(total_utility=-1.0)
            
        # Determine empirical penalty from semantic history.  Generic M4-M7
        # behavior remains frozen; strict M8 uses execution-aligned family data.
        history_penalty = 0.0
        rec = self._history_record(entry, state)
        strict_m8 = self._uses_canonical_m8_policy(state)
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
            if strict_m8 and rec:
                discovery_gains = self._values_for_family(
                    rec,
                    list(getattr(rec, "new_nodes_by_execution", []) or []),
                    ActionFamily.DISCOVERY,
                )
                recent_window = max(1, config.replanning.discovery_plateau_steps)
                recent_gains = discovery_gains[-recent_window:]
                zero_streak = 0
                for gain in reversed(discovery_gains):
                    if gain > config.stopping.discovery_saturation_threshold:
                        break
                    zero_streak += 1

                plateaued = (
                    len(recent_gains) >= recent_window
                    and sum(recent_gains)
                    <= config.stopping.discovery_saturation_threshold
                )
                if plateaued or state.discovery_state.saturated:
                    discovery_value = 0.0
                else:
                    discovery_value = max(
                        0.0,
                        base_discovery
                        - cfg.recent_zero_gain_penalty * zero_streak,
                    )
            else:
                discovery_value = max(0.0, base_discovery - history_penalty)
            
        # 2. Discrimination value I_t(x) based on uncertainty and iteration
        discrimination_value = 0.0
        if entry.action.family in (ActionFamily.VERIFICATION, ActionFamily.CONFOUNDER):
            # scales with total entropy in the scene
            discrimination_value = min(1.0, 0.2 + (total_entropy * 0.1) - history_penalty)
            if strict_m8 and rec:
                effects = self._values_for_family(
                    rec,
                    list(
                        getattr(
                            rec,
                            "realized_discrimination_proxy_by_execution",
                            [],
                        )
                        or []
                    ),
                    entry.action.family,
                )
                effect_window = max(1, config.replanning.discovery_plateau_steps)
                if effects and max(effects[-effect_window:]) <= (
                    config.stopping.utility_min_threshold
                ):
                    discrimination_value *= cfg.ineffective_discrimination_scale

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
