"""Configuration strategy and dataclasses for SAM3-VLM V4."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TilingConfig:
    """Tiling parameters for high-resolution images."""

    grid_rows: int = 2
    grid_cols: int = 2
    overlap_ratio: float = 0.15
    tile_min_size: int = 512


@dataclass
class BudgetConfig:
    """Hard compute limits for a single pipeline execution (V4 Design Spec §15)."""

    max_qwen_calls: int = 4
    max_sam3_calls: int = 15
    max_sam3_tiles: int = 30
    max_runtime_seconds: Optional[float] = 300.0


@dataclass
class StoppingConfig:
    """Stopping threshold configuration (V4 Design Spec §14)."""

    discovery_saturation_threshold: float = 0.05
    utility_min_threshold: float = 0.02
    max_iterations: int = 20


@dataclass
class V4Config:
    """Global configuration object for SAM3-VLM V4."""

    tiling: TilingConfig = field(default_factory=TilingConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    stopping: StoppingConfig = field(default_factory=StoppingConfig)
    device: str = "cuda"
    output_dir: str = "out"
