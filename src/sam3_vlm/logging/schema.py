"""Schema definitions for M7 logging (V4 Design Spec §16.2)."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

RUN_SCHEMA_VERSION = "1.0"
EVENT_SCHEMA_VERSION = "1.0"
SUMMARY_SCHEMA_VERSION = "1.0"
GRAPH_SCHEMA_VERSION = "1.0"

from enum import Enum

class EventKind(str, Enum):
    """Enumeration of persistent state-changing events."""
    RUN_STARTED = "RUN_STARTED"
    BOOTSTRAP_STARTED = "BOOTSTRAP_STARTED"
    BOOTSTRAP_COMPLETED = "BOOTSTRAP_COMPLETED"
    SAM3_ACTION_SELECTED = "SAM3_ACTION_SELECTED"
    SAM3_ACTION_STARTED = "SAM3_ACTION_STARTED"
    SAM3_ACTION_COMPLETED = "SAM3_ACTION_COMPLETED"
    ASSOCIATION_COMPLETED = "ASSOCIATION_COMPLETED"
    NODE_CREATED = "NODE_CREATED"
    NODE_UPDATED = "NODE_UPDATED"
    BELIEF_UPDATE_COMPLETED = "BELIEF_UPDATE_COMPLETED"
    SEMANTIC_MEMORY_UPDATED = "SEMANTIC_MEMORY_UPDATED"
    DISCOVERY_STATE_UPDATED = "DISCOVERY_STATE_UPDATED"
    CONTROLLER_STATE_UPDATED = "CONTROLLER_STATE_UPDATED"
    BUDGET_UPDATED = "BUDGET_UPDATED"
    STOP_DECIDED = "STOP_DECIDED"
    QWEN_PLAN_STARTED = "QWEN_PLAN_STARTED"
    QWEN_PLAN_COMPLETED = "QWEN_PLAN_COMPLETED"
    REPLAN_TRIGGERED = "REPLAN_TRIGGERED"
    ACTION_BANK_REFRESHED = "ACTION_BANK_REFRESHED"
    CLEANUP_STARTED = "CLEANUP_STARTED"
    CLEANUP_ACTION_COMPLETED = "CLEANUP_ACTION_COMPLETED"
    FINAL_COUNT = "FINAL_COUNT"
    RUN_COMPLETED = "RUN_COMPLETED"
    RUN_FAILED = "RUN_FAILED"

@dataclass
class ArtifactRef:
    """A compact, hash-verified reference to an external artifact."""
    relative_path: str
    artifact_type: str
    sha256: str
    size_bytes: int
    shape: Optional[List[int]] = None
    dtype: Optional[str] = None

@dataclass
class RunManifest:
    """Immutable run metadata."""
    run_id: str
    schema_version: str = RUN_SCHEMA_VERSION
    image_id: Optional[str] = None
    user_prompt: str = ""
    target_class: str = ""
    v4_config: Dict[str, Any] = field(default_factory=dict)
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
    wall_runtime_ms: float = 0.0
    sam3_runtime_ms: float = 0.0
    qwen_runtime_ms: float = 0.0
    controller_runtime_ms: float = 0.0
    number_of_replans: int = 0
    discovery_statistics: Dict[str, Any] = field(default_factory=dict)
    evaluation_fields: Dict[str, Any] = field(default_factory=dict)
