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
    ) -> List[ActionBankEntry]:
        """Convert proposed actions into ActionBankEntry objects with deduplication against past memory and active bank."""
        added_entries: List[ActionBankEntry] = []

        # Collect existing keys in memory and bank for deduplication
        existing_keys: Set[str] = set()
        for mem_key in semantic_memory.records:
            existing_keys.add(canonicalize_semantic_key(mem_key))

        for entry in action_bank.entries:
            existing_keys.add(canonicalize_semantic_key(entry.action.semantic_key))

        for p_action in planner_output.proposed_actions:
            canonical_key = canonicalize_semantic_key(p_action.semantic_key)
            if not canonical_key:
                canonical_key = "unnamed_action"

            # Check deduplication
            if canonical_key in existing_keys:
                # Deduplicated duplicate key -> skip adding duplicate action
                continue

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
            )

            entry = action_bank.add_action(action, qwen_priority=p_action.priority)
            existing_keys.add(canonical_key)
            if entry:
                added_entries.append(entry)

        return added_entries
