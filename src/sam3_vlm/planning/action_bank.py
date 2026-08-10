"""Action bank container and lifecycle management (V4 Design Spec §7)."""

from dataclasses import dataclass, field
from typing import List, Optional
from sam3_vlm.sensing.action import SensingAction


@dataclass
class ActionBank:
    """Active bank of proposed sensing actions."""

    actions: List[SensingAction] = field(default_factory=list)

    def add_action(self, action: SensingAction) -> None:
        action.validate()
        self.actions.append(action)

    def pop_next(self) -> Optional[SensingAction]:
        if not self.actions:
            return None
        return self.actions.pop(0)
