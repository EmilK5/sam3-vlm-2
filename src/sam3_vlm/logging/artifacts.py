"""Run artifact path management and serialization utilities (V4 Design Spec §16.1)."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunArtifactPaths:
    """Artifact output directory layout manager."""

    base_dir: Path

    @property
    def run_json(self) -> Path:
        return self.base_dir / "run.json"

    @property
    def summary_json(self) -> Path:
        return self.base_dir / "summary.json"

    @property
    def events_jsonl(self) -> Path:
        return self.base_dir / "events.jsonl"

    @property
    def masks_dir(self) -> Path:
        return self.base_dir / "artifacts" / "masks"

    @property
    def contact_sheets_dir(self) -> Path:
        return self.base_dir / "artifacts" / "contact_sheets"

    @property
    def qwen_dir(self) -> Path:
        return self.base_dir / "artifacts" / "qwen"

    def ensure_directories(self) -> None:
        """Create all required artifact directories if missing."""
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.contact_sheets_dir.mkdir(parents=True, exist_ok=True)
        self.qwen_dir.mkdir(parents=True, exist_ok=True)
