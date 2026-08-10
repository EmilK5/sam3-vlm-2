"""Action bank container and entry lifecycle management (V4 Design Spec §7 / §24.1)."""

from dataclasses import dataclass, field
from typing import List, Optional
from sam3_vlm.sensing.action import SensingAction


@dataclass
class ActionBankEntry:
    """Tracked unit within the action bank holding action state and metadata."""

    action: SensingAction
    qwen_priority: Optional[float] = None
    predicted_discovery_value: Optional[float] = None
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
