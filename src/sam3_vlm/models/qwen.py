"""Qwen scene planner interface protocol and adapter stubs (V4 Design Spec §6)."""

from typing import Any, List, Protocol, runtime_checkable
from sam3_vlm.sensing.action import SensingAction


@runtime_checkable
class QwenPlanner(Protocol):
    """Clean Qwen scene planner contract."""

    def plan_actions(self, evidence_pack: Any) -> List[SensingAction]:
        """Propose structured sensing action bank from scene evidence."""
        ...


class DummyQwenPlanner:
    """Mock Qwen planner for testing and foundation verification."""

    def __init__(self) -> None:
        self.call_count = 0

    def plan_actions(self, evidence_pack: Any) -> List[SensingAction]:
        self.call_count += 1
        return []
