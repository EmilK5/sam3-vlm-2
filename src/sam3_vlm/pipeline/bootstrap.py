"""Bootstrap stage definitions (V4 Design Spec §5)."""

from typing import Any, Protocol


class BootstrapStage(Protocol):
    """Protocol for initial global/tiled bootstrap pass."""

    def execute_bootstrap(self, image: Any, prompt: str) -> Any:
        ...
