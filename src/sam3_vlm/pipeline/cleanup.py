"""Residual targeted cleanup phase (V4 Design Spec §13)."""

from typing import Any, Protocol


class ResidualCleanupStage(Protocol):
    """Protocol for batched residual cleanup on unresolved candidate clusters."""

    def execute_cleanup(self, scene_state: Any) -> Any:
        ...
