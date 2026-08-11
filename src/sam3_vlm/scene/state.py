"""Scene state and discovery state containers (V4 Design Spec §3.2 / §3.4)."""

from dataclasses import dataclass, field
from typing import Any, List, Optional, TYPE_CHECKING
from sam3_vlm.core.geometry import Geometry
from sam3_vlm.core.types import BudgetState, StopReason
from sam3_vlm.scene.belief import SemanticMemory
from sam3_vlm.scene.graph import SceneGraph

if TYPE_CHECKING:
    from sam3_vlm.planning.action_bank import ActionBank


@dataclass
class CoverageSummary:
    """Summary of spatial sensing coverage across the scene."""

    total_area_searched: float = 0.0
    coverage_ratio: float = 0.0


@dataclass
class DiscoveryState:
    """Discovery state diagnostics and missed target mass approximations (V4 Design Spec §3.4)."""

    recent_new_nodes: List[str] = field(default_factory=list)
    recent_new_node_counts: List[float] = field(default_factory=list)
    spatial_coverage: CoverageSummary = field(default_factory=CoverageSummary)
    tiled_bootstrap_gain: Optional[float] = None
    plateau_score: float = 0.0
    unresolved_regions: List[Geometry] = field(default_factory=list)
    qwen_missing_modes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        import dataclasses
        d = dataclasses.asdict(self)
        # Handle custom serialization for unresolved_regions if they are Geometry objects
        # asdict usually handles nested dataclasses automatically.
        if "unresolved_regions" in d and hasattr(self, "unresolved_regions") and self.unresolved_regions:
            # If Geometry has a to_dict, use it, else asdict is fine.
            if hasattr(self.unresolved_regions[0], 'to_dict'):
                d["unresolved_regions"] = [geom.to_dict() for geom in self.unresolved_regions]
        return d


@dataclass
class CountEstimate:
    """Soft count and uncertainty estimation (V4 Design Spec §14.2)."""
    mean_count: float = 0.0
    variance: float = 0.0
    std_dev: float = 0.0


class CountEstimator:
    """Computes soft counts and variances from node belief probabilities (M5)."""

    @staticmethod
    def estimate(graph: SceneGraph, target_class: str) -> CountEstimate:
        mean_count = 0.0
        variance = 0.0
        for node in graph.active_nodes():
            p = node.class_belief.probabilities.get(target_class, 0.0)
            mean_count += p
            variance += p * (1.0 - p)
        return CountEstimate(
            mean_count=mean_count,
            variance=variance,
            std_dev=variance ** 0.5
        )


@dataclass
class SceneState:
    """Primary operational state object B_t = (G_t, rho_t, U_t, S_t, A_t, C_t) (V4 Design Spec §3.2)."""

    image_id: str
    user_prompt: str
    target_class: str
    graph: SceneGraph
    semantic_memory: SemanticMemory
    image_path: Optional[str] = None
    discovery_state: DiscoveryState = field(default_factory=DiscoveryState)
    count_estimate: CountEstimate = field(default_factory=CountEstimate)
    action_bank: Optional["ActionBank"] = None
    budget: BudgetState = field(default_factory=BudgetState)
    iteration: int = 0
    qwen_round: int = 0
    stop_reason: Optional[StopReason] = None
    actions_since_replan: int = 0
    replans_executed: int = 0

    def set_stop_reason(self, candidate: StopReason) -> None:
        """Sets the stop reason enforcing deterministic precedence."""
        if self.stop_reason is None:
            self.stop_reason = candidate
            return
            
        precedence = {
            StopReason.SAM3_BUDGET: 100,
            StopReason.TILE_BUDGET: 90,
            StopReason.RUNTIME_BUDGET: 80,
            StopReason.MAX_ITERATIONS: 70,
            StopReason.CLEANUP_BUDGET: 60,
            StopReason.NO_VALID_ACTIONS: 50,
            StopReason.LOW_MARGINAL_UTILITY: 40,
            StopReason.CLEANUP_COMPLETE: 30,
            StopReason.DISCOVERY_AND_UNCERTAINTY_SATURATED: 20,
            StopReason.ACTION_BANK_EXHAUSTED: 10,
            StopReason.QWEN_BUDGET: 0,
        }
        
        current_prio = precedence.get(self.stop_reason, -1)
        candidate_prio = precedence.get(candidate, -1)
        
        if candidate_prio > current_prio:
            self.stop_reason = candidate

