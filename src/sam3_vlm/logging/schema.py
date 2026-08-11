"""Schema definitions for M7 logging (V4 Design Spec §16.2)."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

RUN_SCHEMA_VERSION = "1.0"
EVENT_SCHEMA_VERSION = "1.0"
SUMMARY_SCHEMA_VERSION = "1.0"
GRAPH_SCHEMA_VERSION = "1.0"

@dataclass
class RunManifest:
    """Immutable run metadata."""
    run_id: str
    schema_version: str = RUN_SCHEMA_VERSION
    image_id: Optional[str] = None
    user_prompt: str = ""
    target_class: str = ""
    experiment_config: Dict[str, Any] = field(default_factory=dict)
    model_identifiers: Dict[str, str] = field(default_factory=dict)
    seed: Optional[int] = None
    creation_timestamp_ms: float = field(default_factory=lambda: time.time() * 1000)

@dataclass
class RunSummary:
    """Final run summary metrics."""
    run_id: str
    schema_version: str = SUMMARY_SCHEMA_VERSION
    final_soft_count: float = 0.0
    count_variance: float = 0.0
    final_stop_reason: Optional[str] = None
    node_count: int = 0
    qwen_calls: int = 0
    sam3_calls: int = 0
    sam3_tiles: int = 0
    cleanup_calls: int = 0
    runtime_ms: float = 0.0
    number_of_replans: int = 0
    discovery_statistics: Dict[str, Any] = field(default_factory=dict)
    evaluation_fields: Dict[str, Any] = field(default_factory=dict)
