"""Sensing package: actions, observations, evidence, and tiling."""

from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.sensing.observation import SAM3Observation
from sam3_vlm.sensing.evidence import (
    EvidencePack,
    CropCandidateAnnotation,
    ContactSheet,
    QwenEvidencePack,
    ContactSheetBuilder,
)
from sam3_vlm.sensing.tiling import (
    TilingDecision,
    TilingPolicy,
    DefaultTilingPolicy,
    compute_tiles,
    tile_box_to_image_box,
    image_box_to_tile_box,
)

__all__ = [
    "SensingAction",
    "SAM3Observation",
    "EvidencePack",
    "CropCandidateAnnotation",
    "ContactSheet",
    "QwenEvidencePack",
    "ContactSheetBuilder",
    "TilingDecision",
    "TilingPolicy",
    "DefaultTilingPolicy",
    "compute_tiles",
    "tile_box_to_image_box",
    "image_box_to_tile_box",
]
