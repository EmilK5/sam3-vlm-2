"""Experiment configuration mapping for M7 sweeps and evaluation."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from sam3_vlm.core.config import V4Config

@dataclass
class ExperimentConfig:
    """Top-level configuration for an evaluation run across a dataset."""
    
    experiment_name: str
    dataset_name: str
    split: str = "val"
    seed: int = 42
    
    # Overrides applied to the base V4Config
    v4_config_overrides: Dict[str, Any] = field(default_factory=dict)
    
    # E.g., max images to process
    max_samples: Optional[int] = None
    
    # Ablation Flags
    tiled_bootstrap_enabled: bool = True
    qwen_replanning_enabled: bool = True
    cleanup_enabled: bool = True
    
    def apply_to(self, base_config: V4Config) -> V4Config:
        """Apply overrides to a base V4Config."""
        import dataclasses
        return dataclasses.replace(base_config, **self.v4_config_overrides)
