"""Semantic memory and belief updating primitives (V4 Design Spec §3.5 / §11)."""

from dataclasses import dataclass, field
from typing import List
from sam3_vlm.core.types import ActionFamily


@dataclass
class SemanticRecord:
    """Tracking structure for attempted semantic experiments (V4 Design Spec §3.5)."""

    semantic_key: str
    prompts: List[str] = field(default_factory=list)
    family: ActionFamily = ActionFamily.DISCOVERY
    execution_count: int = 0
    sam3_call_ids: List[str] = field(default_factory=list)
    total_cost: float = 0.0
    new_nodes_by_execution: List[int] = field(default_factory=list)
    realized_utility_by_execution: List[float] = field(default_factory=list)


@dataclass
class SemanticMemory:
    """Persistent history of semantic keys and queries tested."""

    records: dict[str, SemanticRecord] = field(default_factory=dict)
