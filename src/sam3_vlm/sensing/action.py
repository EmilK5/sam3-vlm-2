"""Sensing action schema and validation for SAM3-VLM V4 (V4 Design Spec §4.1 / §21.7)."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from sam3_vlm.core.config import TilingConfig
from sam3_vlm.core.geometry import Geometry
from sam3_vlm.core.types import ActionFamily, ActionSource, SpatialMode


@dataclass(frozen=True)
class SensingAction:
    """Immutable sensing action sent to the SAM3 visual sensor."""

    action_id: str
    semantic_key: str
    prompt: str
    family: ActionFamily
    threshold: float = 0.25
    spatial_mode: SpatialMode = SpatialMode.GLOBAL
    source: ActionSource = ActionSource.QWEN
    roi: Optional[Geometry] = None
    positive_exemplar_ids: Tuple[str, ...] = field(default_factory=tuple)
    negative_exemplar_ids: Tuple[str, ...] = field(default_factory=tuple)
    tiling: Optional[TilingConfig] = None
    qwen_priority: Optional[float] = None
    semantic_prior: Optional[Dict[str, float]] = None
    correlation_group: Optional[str] = None

    def validate(self) -> None:
        """Validate action schema invariants (V4 Design Spec §21.7).

        Raises:
            ValueError: If any validation check fails.
        """
        if not self.prompt or not self.prompt.strip():
            raise ValueError("SensingAction prompt cannot be empty.")
        if not self.semantic_key or not self.semantic_key.strip():
            raise ValueError("SensingAction semantic_key cannot be empty.")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"SensingAction threshold {self.threshold} outside [0, 1].")

        # Check exemplar disjointness
        pos_set = set(self.positive_exemplar_ids)
        neg_set = set(self.negative_exemplar_ids)
        if pos_set.intersection(neg_set):
            raise ValueError("Positive and negative exemplar node IDs must be disjoint.")

        # Check tiling & spatial mode compatibility
        if self.spatial_mode == SpatialMode.TILED and self.tiling is None:
            raise ValueError("SensingAction with TILED spatial_mode must specify tiling configuration.")
        if self.tiling is not None and self.spatial_mode != SpatialMode.TILED:
            raise ValueError("SensingAction with tiling configuration specified must use TILED spatial_mode.")

        # Check semantic prior range
        if self.semantic_prior:
            for cls_name, prob in self.semantic_prior.items():
                if not (0.0 <= prob <= 1.0):
                    raise ValueError(f"semantic_prior for '{cls_name}' must be in [0, 1], got {prob}")
