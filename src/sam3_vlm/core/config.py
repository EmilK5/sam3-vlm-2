"""Configuration strategy and dataclasses for SAM3-VLM V4 (V4 Design Spec §31)."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class TilingConfig:
    """Tiling parameters for high-resolution images."""

    grid_rows: int = 2
    grid_cols: int = 2
    overlap_ratio: float = 0.15
    tile_min_size: int = 512


@dataclass(frozen=True)
class BudgetConfig:
    """Hard compute limits for a single pipeline execution (V4 Design Spec §15)."""

    max_qwen_calls: int = 4
    max_sam3_calls: int = 15
    max_sam3_tiles: int = 30
    max_cleanup_calls: int = 5
    max_runtime_seconds: Optional[float] = 300.0


@dataclass(frozen=True)
class StoppingConfig:
    """Stopping threshold configuration (V4 Design Spec §14)."""

    discovery_saturation_threshold: float = 0.05
    utility_min_threshold: float = 0.02
    max_iterations: int = 20
    count_variance_threshold: float = 0.5


@dataclass(frozen=True)
class BootstrapConfig:
    """Bootstrap pipeline configuration (V4 Design Spec §5).

    ``locked_context_prompt`` is deliberately generic.  Dataset/deployment
    configuration may use it to establish one SAM3-grounded search domain
    before target bootstrap.  For the green-citrus deployment this is
    ``"tree canopy"``.  The core controller never hard-codes that concept.
    """

    enable_tiled_bootstrap: bool = True
    tiled_bootstrap_min_candidates: int = 5
    locked_context_prompt: Optional[str] = None
    locked_context_threshold: float = 0.40
    locked_context_fallback_full_image: bool = True
    # M8-only by configuration. Generic/M4-M7 runs keep this disabled unless
    # explicitly requested so historical call-count semantics remain frozen.
    enable_pseudoexemplar_refinement: bool = False
    pseudoexemplar_max_count: int = 5
    pseudoexemplar_min_score: float = 0.60


@dataclass(frozen=True)
class PlannerConfig:
    """Qwen scene planner configuration (V4 Design Spec §6)."""

    max_actions_per_prompt: int = 5
    temperature: float = 0.2


@dataclass(frozen=True)
class SAM3Config:
    """SAM3 visual sensor configuration (V4 Design Spec §4)."""

    default_threshold: float = 0.25
    box_nms_iou_threshold: float = 0.7


@dataclass(frozen=True)
class ActionSelectionConfig:
    """Action selection & utility weighting configuration (V4 Design Spec §8)."""

    alpha_discovery: float = 1.0
    beta_discrimination: float = 1.0
    gamma_redundancy: float = 0.5
    lambda_cost: float = 0.1
    eta_qwen_priority: float = 0.2
    # Empirical controller terms.  Discovery is judged by newly registered
    # nodes, while verification/confounder actions are judged by belief change.
    recent_zero_gain_penalty: float = 0.5
    ineffective_discrimination_scale: float = 0.25


@dataclass(frozen=True)
class AssociationConfig:
    """Detection association & graph registration configuration (V4 Design Spec §10)."""

    iou_match_threshold: float = 0.5
    new_node_iou_threshold: float = 0.3
    tiled_nms_threshold: float = 0.7
    # IoM dual-gate is opt-in so legacy/M4-M7 association semantics stay frozen.
    enable_iom_dedup: bool = False
    iom_match_threshold: float = 0.90
    new_node_iom_threshold: float = 0.90
    tiled_nms_iom_threshold: float = 0.90


@dataclass(frozen=True)
class BeliefConfig:
    """Belief & evidence update configuration (V4 Design Spec §11).

    The posterior ontology is intentionally anonymous and frozen.  The only
    legal dimensions are ``target`` plus ``confounder1..N``.  Human-readable
    Qwen labels live in planner metadata, never in the probability vector.
    """

    prior_pseudocount: float = 1.0
    discount_repeat_weight: float = 0.8
    num_confounders: int = 2
    # Optional reporting-only commitment rule.  A target posterior at or above
    # this threshold contributes 1.0 to the count without mutating the node's
    # posterior.  ``None`` preserves a purely soft posterior sum.
    target_count_commit_threshold: Optional[float] = None

    def __post_init__(self) -> None:
        threshold = self.target_count_commit_threshold
        if threshold is not None and not (0.0 < threshold <= 1.0):
            raise ValueError(
                "target_count_commit_threshold must be in (0, 1] or None"
            )


@dataclass(frozen=True)
class ReplanningConfig:
    """Qwen replanning trigger configuration (V4 Design Spec §12)."""

    max_replans: int = 2
    discovery_plateau_steps: int = 2
    unresolved_entropy_threshold: float = 1.0
    count_variance_threshold: float = 0.5
    min_actions_between_replans: int = 2


@dataclass(frozen=True)
class CleanupConfig:
    """Residual cleanup configuration (V4 Design Spec §13)."""

    cleanup_residual_max_nodes: int = 10
    cleanup_ambiguity_threshold: float = 0.8
    cleanup_min_utility: float = 0.05
    roi_batch_size: int = 4


@dataclass(frozen=True)
class LoggingConfig:
    """Logging and provenance configuration (V4 Design Spec §16)."""

    log_events: bool = True
    save_masks: bool = True
    save_contact_sheets: bool = True


@dataclass(frozen=True)
class V4Config:
    """Global immutable configuration object for SAM3-VLM V4 (V4 Design Spec §31)."""

    tiling: TilingConfig = field(default_factory=TilingConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    stopping: StoppingConfig = field(default_factory=StoppingConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    sam3: SAM3Config = field(default_factory=SAM3Config)
    action_selection: ActionSelectionConfig = field(default_factory=ActionSelectionConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    belief: BeliefConfig = field(default_factory=BeliefConfig)
    replanning: ReplanningConfig = field(default_factory=ReplanningConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    device: str = "cuda"
    output_dir: str = "out"
    assets_dir: str = "assets"
