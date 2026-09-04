"""Event-driven replanning logic and trigger evaluation (V4 Design Spec §12)."""

from typing import Optional, Tuple, Any
from sam3_vlm.core.config import V4Config
from sam3_vlm.scene.state import SceneState
from sam3_vlm.sensing.evidence import ContactSheetBuilder, QwenEvidencePack


def discovery_is_plateaued(state: SceneState, config: V4Config) -> bool:
    """Return the controller's rolling zero-gain discovery decision."""
    plateau_steps = max(1, config.replanning.discovery_plateau_steps)
    recent = state.discovery_state.recent_new_node_counts
    return (
        len(recent) >= plateau_steps
        and sum(recent[-plateau_steps:])
        <= config.stopping.discovery_saturation_threshold
    )


class ReplanningPolicy:
    """Determines when Qwen should be called based on scene state."""

    def should_replan(self, state: SceneState, config: V4Config) -> Tuple[bool, Optional[str]]:

        # Cooldown check
        if state.actions_since_replan < config.replanning.min_actions_between_replans:
            if state.action_bank and list(state.action_bank.unexecuted_entries()):
                # Give the current bank a chance unless it's exhausted
                return False, None

        # 1. Action bank exhaustion
        if state.action_bank is None:
            return True, "INITIAL_PLANNING"
        
        unexecuted = list(state.action_bank.unexecuted_entries())
        if not unexecuted:
            return True, "ACTION_BANK_EXHAUSTED"

        # 2. Low utility
        # Check if the best action has utility below threshold
        best_entry = max(unexecuted, key=lambda e: e.total_utility or -9999.0)
        best_utility = best_entry.total_utility or 0.0
        if best_utility < config.stopping.utility_min_threshold:
            return True, "LOW_MARGINAL_UTILITY"

        # 3. Discovery plateau with high uncertainty
        # Plateau: recent_new_node_counts sum is low
        if discovery_is_plateaued(state, config):
            # We have a plateau. Is there still high uncertainty?
            entropy = sum(n.class_belief.entropy for n in state.graph.active_nodes())
            if entropy > config.replanning.unresolved_entropy_threshold:
                return True, "DISCOVERY_PLATEAU_WITH_UNCERTAINTY"
            if state.count_estimate.variance > config.replanning.count_variance_threshold:
                return True, "DISCOVERY_PLATEAU_WITH_COUNT_VARIANCE"

        return False, None


class ReplanEvidenceBuilder:
    """Constructs the current evidence pack for a Qwen replanning call (V4 Design Spec §12.1)."""

    def __init__(self, contact_sheet_builder: ContactSheetBuilder):
        self.contact_sheet_builder = contact_sheet_builder

    @staticmethod
    def _value_at(values, index, default):
        return values[index] if index < len(values) else default

    def build(
        self,
        state: SceneState,
        image: Any = None,
        assets_dir: str = "assets",
        config: Optional[V4Config] = None,
    ) -> QwenEvidencePack:
        # Build contact sheet from current graph
        contact_sheet = self.contact_sheet_builder.build_contact_sheet(
            graph=state.graph,
            max_crops=24,
            image=image,
            assets_dir=assets_dir,
            image_id=f"{state.image_id}_replan_{state.qwen_round}",
            semantic_memory=state.semantic_memory,
            target_class=state.target_class,
        )

        # Build compact semantic history summary
        history_lines = ["=== SEMANTIC HISTORY ==="]
        tried_prompts = []
        for key, record in state.semantic_memory.records.items():
            if record.execution_count > 0:
                tried_prompts.extend(
                    prompt
                    for prompt in record.prompts
                    if prompt not in tried_prompts
                )
                avg_ent = (
                    sum(record.entropy_change_by_execution)
                    / record.execution_count
                )
                avg_var = (
                    sum(record.variance_change_by_execution)
                    / record.execution_count
                )
                avg_disc = (
                    sum(record.realized_discrimination_proxy_by_execution)
                    / record.execution_count
                )
                avg_util = (
                    sum(record.realized_utility_by_execution)
                    / record.execution_count
                )
                history_lines.append(
                    f"- group='{key}': executions={record.execution_count}; "
                    f"tried_prompts={record.prompts!r}; "
                    f"nodes_found={sum(record.new_nodes_by_execution)}; "
                    f"affected={sum(record.affected_nodes_by_execution)}; "
                    f"avg_ent_delta={avg_ent:.2f}; "
                    f"avg_var_delta={avg_var:.2f}; "
                    f"avg_disc={avg_disc:.2f}; avg_util={avg_util:.2f}"
                )
                prompts = list(
                    getattr(record, "prompts_by_execution", []) or []
                )
                semantic_keys = list(
                    getattr(record, "semantic_keys_by_execution", []) or []
                )
                families = list(
                    getattr(record, "families_by_execution", []) or []
                )
                spatial_modes = list(
                    getattr(record, "spatial_modes_by_execution", []) or []
                )
                for index in range(record.execution_count):
                    prompt = self._value_at(
                        prompts,
                        index,
                        record.prompts[index]
                        if index < len(record.prompts)
                        else "unknown",
                    )
                    semantic_key = self._value_at(
                        semantic_keys,
                        index,
                        record.semantic_keys[0]
                        if record.semantic_keys
                        else key,
                    )
                    family = self._value_at(
                        families,
                        index,
                        getattr(record.family, "value", record.family),
                    )
                    spatial_mode = self._value_at(
                        spatial_modes,
                        index,
                        "UNKNOWN",
                    )
                    new_nodes = self._value_at(
                        record.new_nodes_by_execution, index, 0
                    )
                    affected = self._value_at(
                        record.affected_nodes_by_execution, index, 0
                    )
                    entropy_delta = self._value_at(
                        record.entropy_change_by_execution, index, 0.0
                    )
                    variance_delta = self._value_at(
                        record.variance_change_by_execution, index, 0.0
                    )
                    discrimination = self._value_at(
                        record.realized_discrimination_proxy_by_execution,
                        index,
                        0.0,
                    )
                    history_lines.append(
                        f"  execution[{index + 1}]: semantic_key={semantic_key!r}; "
                        f"sam3_prompt={prompt!r}; family={getattr(family, 'value', family)}; "
                        f"spatial_mode={getattr(spatial_mode, 'value', spatial_mode)}; "
                        f"new_nodes={new_nodes}; affected_nodes={affected}; "
                        f"entropy_delta={float(entropy_delta):.6f}; "
                        f"variance_delta={float(variance_delta):.6f}; "
                        f"discrimination_proxy={float(discrimination):.6f}"
                    )

        # Include accepted-but-unexecuted prompts. They are still known
        # experiments and must not consume another Qwen proposal slot.
        if state.action_bank is not None:
            for entry in state.action_bank.entries:
                prompt = entry.action.prompt
                if prompt not in tried_prompts:
                    tried_prompts.append(prompt)
        
        scene_summary = "\n".join(history_lines) if len(history_lines) > 1 else "No semantic history yet."

        saturated = (
            discovery_is_plateaued(state, config)
            if config is not None
            else bool(state.discovery_state.saturated)
        )
        pack = QwenEvidencePack(
            original_image_id=state.image_id,
            user_prompt=state.user_prompt,
            target_class=state.target_class,
            contact_sheet=contact_sheet,
            image_path=state.image_path,
            scene_summary=scene_summary,
            discovery_diagnostics={
                "recent_new_nodes_count": len(state.discovery_state.recent_new_nodes),
                "recent_new_node_counts": list(state.discovery_state.recent_new_node_counts),
                "coverage_ratio": state.discovery_state.spatial_coverage.coverage_ratio,
                "discovery_saturated": saturated,
                "plateau_score": state.discovery_state.plateau_score,
                "tried_sam3_prompts": tried_prompts,
                "unresolved_entropy": sum(n.class_belief.entropy for n in state.graph.active_nodes()),
                "count_variance": state.count_estimate.variance,
                "search_region": state.search_region.bbox().as_tuple() if state.search_region else None,
                "search_region_source": state.search_region_source,
            },
            belief_classes=list(state.belief_classes),
            confounder_labels=dict(state.confounder_labels),
        )
        return pack
