"""Scene state and discovery state containers (V4 Design Spec §3.2 / §3.4)."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple
from sam3_vlm.core.geometry import Box, Geometry
from sam3_vlm.core.types import BudgetState, StopReason
from sam3_vlm.scene.belief import SemanticMemory
from sam3_vlm.scene.graph import SceneGraph

if TYPE_CHECKING:
    from sam3_vlm.planning.action_bank import ActionBank


def _rectangle_union_area(rectangles: List[Tuple[float, float, float, float]]) -> float:
    """Exact union area for a small set of axis-aligned image-space boxes."""
    if not rectangles:
        return 0.0
    xs = sorted({x for r in rectangles for x in (r[0], r[2])})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (y1, y2) for x1, y1, x2, y2 in rectangles
            if x1 < right and x2 > left and y2 > y1
        )
        if not intervals:
            continue
        covered_y = 0.0
        cur_start, cur_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= cur_end:
                cur_end = max(cur_end, end)
            else:
                covered_y += cur_end - cur_start
                cur_start, cur_end = start, end
        covered_y += cur_end - cur_start
        area += (right - left) * covered_y
    return area


@dataclass
class CoverageSummary:
    """Spatial sensing coverage relative to the active search domain."""

    total_area_searched: float = 0.0
    coverage_ratio: float = 0.0
    searched_boxes: List[Tuple[float, float, float, float]] = field(default_factory=list)

    def update(self, regions: List[Geometry], search_domain: Optional[Geometry]) -> None:
        if not search_domain:
            return
        domain = search_domain.bbox()
        if domain.area <= 0.0:
            return

        for region in regions or []:
            box = region.bbox()
            x1 = max(domain.x1, box.x1)
            y1 = max(domain.y1, box.y1)
            x2 = min(domain.x2, box.x2)
            y2 = min(domain.y2, box.y2)
            if x2 > x1 and y2 > y1:
                clipped = (float(x1), float(y1), float(x2), float(y2))
                if clipped not in self.searched_boxes:
                    self.searched_boxes.append(clipped)

        union_area = _rectangle_union_area(self.searched_boxes)
        self.total_area_searched = min(domain.area, union_area)
        self.coverage_ratio = min(1.0, self.total_area_searched / domain.area)


@dataclass
class DiscoveryState:
    """Discovery diagnostics and missed-target approximations (V4 Design Spec §3.4)."""

    recent_new_nodes: List[str] = field(default_factory=list)
    recent_new_node_counts: List[float] = field(default_factory=list)
    spatial_coverage: CoverageSummary = field(default_factory=CoverageSummary)
    tiled_bootstrap_gain: Optional[float] = None
    plateau_score: float = 0.0
    saturated: bool = False
    unresolved_regions: List[Geometry] = field(default_factory=list)
    qwen_missing_modes: List[str] = field(default_factory=list)

    def record_search(self, regions: List[Geometry], search_domain: Optional[Geometry]) -> None:
        self.spatial_coverage.update(regions, search_domain)

    def record_discovery_gain(
        self,
        new_node_count: int,
        new_node_ids: Optional[List[str]] = None,
        plateau_window: int = 3,
    ) -> None:
        """Record every discovery execution, including an explicit zero gain."""
        self.recent_new_node_counts.append(float(new_node_count))
        if new_node_ids:
            self.recent_new_nodes.extend(new_node_ids)
        window = self.recent_new_node_counts[-max(1, plateau_window):]
        zero_count = sum(1 for value in window if value <= 0.0)
        self.plateau_score = zero_count / float(len(window)) if window else 0.0
        self.saturated = len(window) >= max(1, plateau_window) and zero_count == len(window)

    def to_dict(self) -> dict:
        import dataclasses
        d = dataclasses.asdict(self)
        if self.unresolved_regions:
            d["unresolved_regions"] = [
                {"box": geom.bbox().as_tuple(), "coordinate_space": geom.bbox().coordinate_space}
                for geom in self.unresolved_regions
            ]
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
            std_dev=variance ** 0.5,
        )


@dataclass
class SceneState:
    """Primary operational state object B_t = (G_t, rho_t, U_t, S_t, A_t, C_t)."""

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
    belief_classes: List[str] = field(default_factory=lambda: ["target", "confounder1", "confounder2"])
    confounder_labels: Dict[str, str] = field(default_factory=dict)
    search_region: Optional[Geometry] = None
    search_region_locked: bool = False
    search_region_source: Optional[str] = None
    search_region_fallback_used: bool = False
    search_region_call_id: Optional[str] = None
    iteration: int = 0
    qwen_round: int = 0
    stop_reason: Optional[StopReason] = None
    actions_since_replan: int = 0
    replans_executed: int = 0
    last_plan_accepted_actions: int = 0

    def set_stop_reason(self, candidate: Optional[StopReason]) -> None:
        """Set stop reason with deterministic frozen M6 precedence."""
        if candidate is None:
            return
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
        if precedence.get(candidate, -1) > precedence.get(self.stop_reason, -1):
            self.stop_reason = candidate
