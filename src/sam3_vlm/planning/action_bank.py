"""Action bank container, entry lifecycle management, and action bank generation with deduplication (V4 Design Spec §7 / §24.1)."""

from dataclasses import dataclass, field
import re
from typing import List, Optional, Set
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.core.types import ActionSource
from sam3_vlm.planning.qwen_planner import PlannerOutput
from sam3_vlm.scene.belief import SemanticMemory
from sam3_vlm.sensing.action import SensingAction


def canonicalize_semantic_key(key: str) -> str:
    """Canonicalize a semantic key string for deduplication (V4 Design Spec §7.1)."""
    if not key:
        return ""
    clean = re.sub(r"[^\w]+", "_", key.lower()).strip("_")
    return clean


def derive_correlation_group(semantic_key: str, prompt: str) -> str:
    """Derive correlation group for near-paraphrase grouping (V4 Design Spec §7.1)."""
    return canonicalize_semantic_key(semantic_key)



@dataclass
class ActionBankEntry:
    """Tracked unit within the action bank holding action state and metadata."""

    action: SensingAction
    qwen_priority: Optional[float] = None
    predicted_discovery_value: Optional[float] = None
    predicted_discrimination_value: Optional[float] = None
    redundancy: float = 0.0
    estimated_cost: float = 1.0
    executed: bool = False
    invalid_reason: Optional[str] = None


@dataclass
class ActionBank:
    """Active bank of proposed sensing action entries."""

    entries: List[ActionBankEntry] = field(default_factory=list)

    def add_action(
        self, action: SensingAction, qwen_priority: Optional[float] = None
    ) -> Optional[ActionBankEntry]:
        """Validate action and append entry to the bank.

        Invalid actions are recorded with an invalid_reason rather than crashing.
        """
        entry = ActionBankEntry(
            action=action,
            qwen_priority=qwen_priority if qwen_priority is not None else action.qwen_priority,
        )

        try:
            action.validate()
        except ValueError as e:
            entry.invalid_reason = str(e)
            self.entries.append(entry)
            return None

        self.entries.append(entry)
        return entry

    def pop_next(self) -> Optional[ActionBankEntry]:
        """Pop the next valid, unexecuted entry and mark it executed."""
        for entry in self.entries:
            if not entry.executed and entry.invalid_reason is None:
                entry.executed = True
                return entry
        return None

    def unexecuted_entries(self) -> List[ActionBankEntry]:
        return [e for e in self.entries if not e.executed and e.invalid_reason is None]

    def executed_entries(self) -> List[ActionBankEntry]:
        return [e for e in self.entries if e.executed]


class ActionBankGenerator:
    """Converts structured Qwen planner outputs into deduplicated ActionBank entries (V4 Design Spec §7.1)."""

    def generate_entries(
        self,
        planner_output: PlannerOutput,
        semantic_memory: SemanticMemory,
        action_bank: ActionBank,
        id_gen: IDGenerator,
        valid_node_ids: Optional[Set[str]] = None,
    ) -> List[ActionBankEntry]:
        """Convert proposed actions into ActionBankEntry objects with deduplication and correlation grouping."""
        added_entries: List[ActionBankEntry] = []

        existing_keys: Set[str] = set()
        existing_correlation_groups: Set[str] = set()
        existing_prompts: Set[str] = set()

        for mem_key, record in semantic_memory.records.items():
            ckey = canonicalize_semantic_key(mem_key)
            existing_keys.add(ckey)
            group = derive_correlation_group(mem_key, " ".join(record.prompts))
            existing_correlation_groups.add(group)
            for prompt in record.prompts:
                existing_prompts.add(prompt.strip().lower())

        for entry in action_bank.entries:
            ckey = canonicalize_semantic_key(entry.action.semantic_key)
            existing_keys.add(ckey)
            group = entry.action.correlation_group or derive_correlation_group(
                entry.action.semantic_key, entry.action.prompt
            )
            existing_correlation_groups.add(group)
            existing_prompts.add(entry.action.prompt.strip().lower())

        for p_action in planner_output.proposed_actions:
            canonical_key = canonicalize_semantic_key(p_action.semantic_key)
            if not canonical_key:
                canonical_key = "unnamed_action"

            # Check exact key deduplication
            if canonical_key in existing_keys:
                continue

            # Check exact prompt deduplication (Spec M3.5 Phase 3)
            clean_prompt = p_action.prompt.strip().lower()
            if clean_prompt in existing_prompts:
                continue

            # Validate exemplars if valid_node_ids is provided (Spec M3.5 Phase 3)
            if valid_node_ids is not None:
                invalid_exemplars = False
                for ex_id in p_action.positive_exemplar_ids:
                    if ex_id not in valid_node_ids:
                        invalid_exemplars = True
                        break
                for ex_id in p_action.negative_exemplar_ids:
                    if ex_id not in valid_node_ids:
                        invalid_exemplars = True
                        break
                if invalid_exemplars:
                    continue

            corr_group = p_action.correlation_group or derive_correlation_group(
                canonical_key, p_action.prompt
            )

            roi_geom = None
            if p_action.roi and len(p_action.roi) == 4:
                from sam3_vlm.core.geometry import Box
                roi_geom = Box(x1=p_action.roi[0], y1=p_action.roi[1], x2=p_action.roi[2], y2=p_action.roi[3])

            tiling_cfg = None
            if p_action.tiling:
                from sam3_vlm.core.config import TilingConfig
                tiling_cfg = TilingConfig(**p_action.tiling)

            # Check if near-paraphrase in same correlation group already exists
            is_correlated = corr_group in existing_correlation_groups

            action = SensingAction(
                action_id=id_gen.next_action_id(),
                semantic_key=canonical_key,
                prompt=p_action.prompt,
                family=p_action.family,
                threshold=p_action.suggested_threshold if p_action.suggested_threshold is not None else 0.25,
                spatial_mode=p_action.suggested_spatial_mode,
                source=ActionSource.QWEN,
                qwen_priority=p_action.priority,
                semantic_prior=p_action.semantic_prior,
                correlation_group=corr_group,
                roi=roi_geom,
                positive_exemplar_ids=tuple(p_action.positive_exemplar_ids),
                negative_exemplar_ids=tuple(p_action.negative_exemplar_ids),
                tiling=tiling_cfg,
            )

            entry = action_bank.add_action(action, qwen_priority=p_action.priority)
            existing_keys.add(canonical_key)
            existing_correlation_groups.add(corr_group)
            existing_prompts.add(clean_prompt)

            if entry:
                if is_correlated:
                    entry.redundancy = 0.5  # Mark correlated paraphrase redundancy
                added_entries.append(entry)

        return added_entries
