"""Event-driven replanning logic and trigger evaluation (V4 Design Spec §12)."""

from typing import Optional, Tuple, Any
from sam3_vlm.core.config import V4Config
from sam3_vlm.scene.state import SceneState
from sam3_vlm.sensing.evidence import ContactSheetBuilder, QwenEvidencePack


class ReplanningPolicy:
    """Determines when Qwen should be called based on scene state."""

    def should_replan(self, state: SceneState, config: V4Config) -> Tuple[bool, Optional[str]]:
        # 0. Max replans check
        if state.replans_executed >= config.replanning.max_replans:
            return False, None

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
        plateau_steps = config.replanning.discovery_plateau_steps
        if len(state.discovery_state.recent_new_node_counts) >= plateau_steps:
            rolling_sum = sum(state.discovery_state.recent_new_node_counts[-plateau_steps:])
            if rolling_sum <= config.stopping.discovery_saturation_threshold:
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

    def build(self, state: SceneState, image: Any = None, assets_dir: str = "assets") -> QwenEvidencePack:
        # Build contact sheet from current graph
        contact_sheet = self.contact_sheet_builder.build_contact_sheet(
            graph=state.graph,
            max_crops=24,
            image=image,
            assets_dir=assets_dir,
            image_id=f"{state.image_id}_replan_{state.qwen_round}"
        )

        # Build compact semantic history summary
        history_lines = ["=== SEMANTIC HISTORY ==="]
        for key, record in state.semantic_memory.records.items():
            if record.execution_count > 0:
                history_lines.append(
                    f"- key='{key}': execs={record.execution_count}, nodes_found={sum(record.new_nodes_by_execution)}, "
                    f"avg_utility={sum(record.realized_utility_by_execution)/record.execution_count:.2f}"
                )
        
        scene_summary = "\n".join(history_lines) if len(history_lines) > 1 else "No semantic history yet."

        pack = QwenEvidencePack(
            original_image_id=state.image_id,
            user_prompt=state.user_prompt,
            target_class=state.target_class,
            contact_sheet=contact_sheet,
            image_path=state.image_path,
            scene_summary=scene_summary,
            discovery_diagnostics={
                "recent_new_nodes_count": len(state.discovery_state.recent_new_nodes),
                "unresolved_entropy": sum(n.class_belief.entropy for n in state.graph.active_nodes()),
                "count_variance": state.count_estimate.variance
            }
        )
        return pack
