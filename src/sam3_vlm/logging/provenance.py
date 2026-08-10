"""Sensor provenance tracker for graph node updates (V4 Design Spec §16.1 / invariant 11)."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ProvenanceRecord:
    """Explicit sensor provenance link for every node or belief state change."""

    entity_id: str
    sam3_call_id: str
    action_id: str
    semantic_key: str
    event_id: str


@dataclass
class ProvenanceTracker:
    """Container for tracking full historical provenance across run execution."""

    records: List[ProvenanceRecord] = field(default_factory=list)

    def record_update(self, rec: ProvenanceRecord) -> None:
        self.records.append(rec)
