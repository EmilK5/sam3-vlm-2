"""Evidence collection and processing for SAM3-VLM V4."""

from dataclasses import dataclass, field
from typing import List
from sam3_vlm.sensing.observation import SAM3Observation


@dataclass
class EvidencePack:
    """Pack of sensory evidence prepared for graph updates or Qwen planning."""

    observations: List[SAM3Observation] = field(default_factory=list)
