"""Qwen Planner high-level interfaces (V4 Design Spec §6)."""

from typing import List, Protocol
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.sensing.evidence import EvidencePack


class PlannerService(Protocol):
    """High level planning service contract."""

    def propose_actions(self, evidence: EvidencePack) -> List[SensingAction]:
        ...
