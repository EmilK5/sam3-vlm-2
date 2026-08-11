"""Core enums, dataclasses, and domain schemas for SAM3-VLM V4.

Invariants (V4 Design Spec §21 / §23):
- Structured schemas for all state, actions, observations, and belief primitives.
- No model instances or GPU tensors allowed in core serializable types.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional
from sam3_vlm.core.geometry import GeometryRef


class NodeStatus(str, Enum):
    ACTIVE = "ACTIVE"
    RESOLVED = "RESOLVED"
    REJECTED = "REJECTED"
    AMBIGUOUS = "AMBIGUOUS"


class ObservationRelation(str, Enum):
    STRONG_MATCH = "STRONG_MATCH"
    WEAK_MATCH = "WEAK_MATCH"
    NOT_RETRIEVED = "NOT_RETRIEVED"
    NOT_OBSERVABLE = "NOT_OBSERVABLE"
    NEW_DETECTION = "NEW_DETECTION"
    AMBIGUOUS_ASSOCIATION = "AMBIGUOUS_ASSOCIATION"


class ActionFamily(str, Enum):
    DISCOVERY = "DISCOVERY"
    CONFOUNDER = "CONFOUNDER"
    CONTEXT = "CONTEXT"
    VERIFICATION = "VERIFICATION"


class SpatialMode(str, Enum):
    GLOBAL = "GLOBAL"
    TILED = "TILED"
    ROI_BATCH = "ROI_BATCH"
    LOCAL = "LOCAL"


class ActionSource(str, Enum):
    """Action source taxonomy (V4 Design Spec §7.2)."""

    USER_BOOTSTRAP = "USER_BOOTSTRAP"
    QWEN = "QWEN"
    CONTROLLER = "CONTROLLER"
    CLEANUP = "CLEANUP"


class StopReason(str, Enum):
    """M6 structured stopping reasons."""
    SAM3_BUDGET = "SAM3_BUDGET"
    QWEN_BUDGET = "QWEN_BUDGET"
    TILE_BUDGET = "TILE_BUDGET"
    RUNTIME_BUDGET = "RUNTIME_BUDGET"
    MAX_ITERATIONS = "MAX_ITERATIONS"
    DISCOVERY_AND_UNCERTAINTY_SATURATED = "DISCOVERY_AND_UNCERTAINTY_SATURATED"
    LOW_MARGINAL_UTILITY = "LOW_MARGINAL_UTILITY"
    ACTION_BANK_EXHAUSTED = "ACTION_BANK_EXHAUSTED"
    CLEANUP_COMPLETE = "CLEANUP_COMPLETE"
    CLEANUP_BUDGET = "CLEANUP_BUDGET"
    NO_VALID_ACTIONS = "NO_VALID_ACTIONS"


@dataclass
class BudgetState:
    """Computational resource accounting (V4 Design Spec §15)."""

    qwen_calls: int = 0
    sam3_calls: int = 0
    sam3_tiles: int = 0
    cleanup_calls: int = 0
    model_runtime_ms: float = 0.0
    total_runtime_ms: float = 0.0


@dataclass
class ClassBelief:
    """Generic class belief state container (V4 Design Spec §21.5)."""

    probabilities: Dict[str, float] = field(default_factory=dict)
    update_count: int = 0
    entropy: float = 0.0
    last_update_event_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.probabilities:
            total = 0.0
            for k, v in self.probabilities.items():
                if not math.isfinite(v):
                    raise ValueError(f"Probability for class '{k}' must be finite, got: {v}")
                if v < 0.0:
                    raise ValueError(f"Probability for class '{k}' cannot be negative, got: {v}")
                total += v
            if abs(total - 1.0) > 1e-4:
                raise ValueError(
                    f"Class probabilities must sum to 1.0 within numerical tolerance, got sum={total}"
                )
            
            # Recompute entropy dynamically
            h = 0.0
            for p in self.probabilities.values():
                if p > 0.0:
                    h -= p * math.log2(p)
            self.entropy = max(0.0, h)


@dataclass
class RegistrationDiagnostics:
    """Graph node registration & association state (V4 Design Spec §21.6 / §25.4)."""

    existence_score: float = 1.0
    duplicate_risk: float = 0.0
    merge_risk: float = 0.0
    split_risk: float = 0.0
    ambiguous_with: List[str] = field(default_factory=list)
    support_count: int = 1
    independent_semantic_support_count: int = 1


@dataclass
class NodeObservationRef:
    """Pointer from a graph node to the observation update (V4 Design Spec §21.4)."""

    observation_id: str
    sam3_call_id: str
    action_id: str
    semantic_key: str
    correlation_group: Optional[str] = None
    detection_id: Optional[str] = None
    relation: ObservationRelation = ObservationRelation.STRONG_MATCH
    score: Optional[float] = None
    association_score: Optional[float] = None


@dataclass
class Detection:
    """Raw sensor detection output from SAM3 (V4 Design Spec §21.3)."""

    detection_id: str
    geometry: GeometryRef
    score: float
    source_tile_id: Optional[str] = None
    local_geometry: Optional[GeometryRef] = None
    mask_artifact: Optional[str] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)
