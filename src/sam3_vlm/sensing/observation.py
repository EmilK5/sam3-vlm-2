"""SAM3 Observation output schema for SAM3-VLM V4 (V4 Design Spec §4.2)."""

from dataclasses import dataclass, field
from typing import Any, Dict, List
from sam3_vlm.core.geometry import Geometry
from sam3_vlm.core.types import Detection


@dataclass
class SAM3Observation:
    """Sensor observation returned from a SAM3 execution."""

    call_id: str
    action_id: str
    semantic_key: str
    detections: List[Detection] = field(default_factory=list)
    searched_regions: List[Geometry] = field(default_factory=list)
    runtime_ms: float = 0.0
    model_metadata: Dict[str, Any] = field(default_factory=dict)
