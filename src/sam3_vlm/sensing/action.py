"""Sensing action schema and validation for SAM3-VLM V4 (V4 Design Spec §4.1 / §21.7)."""

from dataclasses import dataclass, field
import re
from typing import Dict, Optional, Tuple
from sam3_vlm.core.config import TilingConfig
from sam3_vlm.core.geometry import Geometry
from sam3_vlm.core.types import ActionFamily, ActionSource, SpatialMode


_PROMPT_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9-]*$")
_FORBIDDEN_PROMPT_TOKENS = {
    "analyze", "analyse", "analysis", "using", "use", "perform", "scan", "search",
    "find", "verify", "check", "detect", "detection", "segment", "segmentation",
    "spectral", "multispectral", "clustering", "cluster", "channel", "channels",
    "enhance", "enhancement", "edge", "imaging", "contrast", "with", "under",
    "inside", "within", "of", "for", "in", "on", "and", "or", "to",
}


def validate_sam3_prompt_contract(prompt: str) -> None:
    """Enforce the executable SAM3 language contract.

    Real sensing prompts are deliberately tiny: one or two visual modifiers
    followed by a head noun (2--3 lexical tokens total).  POS tagging is not
    introduced into the controller; the Qwen system prompt enforces adjective
    semantics while this deterministic guard rejects prose, clauses, and
    image-processing instructions.
    """
    text = (prompt or "").strip()
    parts = text.split()
    if len(parts) not in (2, 3):
        raise ValueError(
            "SAM3 prompt must contain exactly one or two visual modifiers plus a noun (2-3 words)."
        )
    for token in parts:
        if not _PROMPT_TOKEN_RE.match(token):
            raise ValueError(f"SAM3 prompt contains non-lexical token: {token!r}")
        if token.lower() in _FORBIDDEN_PROMPT_TOKENS:
            raise ValueError(f"SAM3 prompt contains forbidden method/prose token: {token!r}")


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
        """Validate executable action invariants before SAM3."""
        if not self.prompt or not self.prompt.strip():
            raise ValueError("SensingAction prompt cannot be empty.")
        validate_sam3_prompt_contract(self.prompt)
        if not self.semantic_key or not self.semantic_key.strip():
            raise ValueError("SensingAction semantic_key cannot be empty.")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"SensingAction threshold {self.threshold} outside [0, 1].")

        pos_set = set(self.positive_exemplar_ids)
        neg_set = set(self.negative_exemplar_ids)
        if pos_set.intersection(neg_set):
            raise ValueError("Positive and negative exemplar node IDs must be disjoint.")

        if self.spatial_mode == SpatialMode.TILED and self.tiling is None:
            raise ValueError("SensingAction with TILED spatial_mode must specify tiling configuration.")
        if self.tiling is not None and self.spatial_mode != SpatialMode.TILED:
            raise ValueError("SensingAction with tiling configuration specified must use TILED spatial_mode.")
        if self.spatial_mode in (SpatialMode.LOCAL, SpatialMode.ROI_BATCH) and self.roi is None:
            raise ValueError(f"SensingAction requires ROI for spatial_mode={self.spatial_mode.value}.")
        if self.roi is not None:
            roi_box = self.roi.bbox() if hasattr(self.roi, "bbox") else self.roi
            if not hasattr(roi_box, "area") or roi_box.area <= 0.0:
                raise ValueError("SensingAction ROI must have positive area.")

        if self.semantic_prior:
            for cls_name, prob in self.semantic_prior.items():
                if not (0.0 <= prob <= 1.0):
                    raise ValueError(f"semantic_prior for '{cls_name}' must be in [0, 1], got {prob}")

        if self.spatial_mode in (SpatialMode.LOCAL, SpatialMode.ROI_BATCH) and self.roi is None:
            raise ValueError(
                f"SensingAction with {self.spatial_mode.value} spatial_mode must specify ROI."
            )

        if self.roi is not None and self.spatial_mode not in (
            SpatialMode.LOCAL,
            SpatialMode.ROI_BATCH,
        ):
            raise ValueError(
                "SensingAction with ROI must use LOCAL or ROI_BATCH spatial_mode."
            )
