"""Structured event log schemas (V4 Design Spec §16.2)."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Event:
    """Immutable audit trail log event."""

    event_id: str
    event_type: str
    timestamp_ms: float
    run_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    parent_event_id: Optional[str] = None
