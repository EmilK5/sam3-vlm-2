"""Scene state and discovery state containers (V4 Design Spec §3.2 / §3.4)."""

from dataclasses import dataclass, field
from typing import Any, List, Optional, TYPE_CHECKING
from sam3_vlm.core.geometry import Geometry
from sam3_vlm.core.types import BudgetState
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

    recent_new_nodes: List[int] = field(default_factory=list)
    recent_new_target_mass: List[float] = field(default_factory=list)
    spatial_coverage: CoverageSummary = field(default_factory=CoverageSummary)
    tiled_bootstrap_gain: Optional[float] = None
    plateau_score: float = 0.0
    unresolved_regions: List[Geometry] = field(default_factory=list)
    qwen_missing_modes: List[str] = field(default_factory=list)


@dataclass
class SceneState:
    """Primary operational state object B_t = (G_t, rho_t, U_t, S_t, A_t, C_t) (V4 Design Spec §3.2)."""

    image_id: str
    user_prompt: str
    target_class: str
    graph: SceneGraph
    semantic_memory: SemanticMemory
    discovery_state: DiscoveryState = field(default_factory=DiscoveryState)
    action_bank: Optional["ActionBank"] = None
    budget: BudgetState = field(default_factory=BudgetState)
    iteration: int = 0
    qwen_round: int = 0
