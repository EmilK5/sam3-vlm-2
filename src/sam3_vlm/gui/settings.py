"""Explicit UI controls and validated conversion into the existing V4 config.

Qwen's instruction version, target scope and one-action contract are intentionally
absent from the editable fields. The target text is a separate run input.
"""

from dataclasses import dataclass, replace
import math

from sam3_vlm.core.config import V4Config
from sam3_vlm.experiments.m8_smoke import _pilot_variants


@dataclass(frozen=True)
class Control:
    path: str
    label: str
    kind: str = "float"
    minimum: float = 0.0
    maximum: float | None = None
    nullable: bool = False
    info: str = ""

    def parse(self, value):
        if value is None or value == "":
            if self.nullable:
                return None
            raise ValueError(f"{self.label} is required.")
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{self.label} must be a checkbox value.")
            return value
        if self.kind == "text":
            if not isinstance(value, str):
                raise ValueError(f"{self.label} must be text.")
            return value.strip() or None
        if isinstance(value, bool):
            raise ValueError(f"{self.label} must be a number.")
        try:
            number = float(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{self.label} must be a number.") from exc
        if not math.isfinite(number) or number < self.minimum:
            raise ValueError(f"{self.label} must be finite and at least {self.minimum}.")
        if self.maximum is not None and number > self.maximum:
            raise ValueError(f"{self.label} must be at most {self.maximum}.")
        if self.kind == "int":
            if not number.is_integer():
                raise ValueError(f"{self.label} must be a whole number.")
            return int(number)
        return number


# Order here is also the input/output order used by Gradio's preset callback.
CONTROL_GROUPS = {
    "Detection and Qwen": (
        Control("sam3.default_threshold", "Bootstrap target threshold", maximum=1),
        Control("sam3.qwen_discovery_threshold", "Qwen positive threshold", maximum=1),
        Control("sam3.qwen_confounder_threshold", "Negative threshold", maximum=1),
        Control("planner.execute_confounder_prompts", "Run negative prompts", "bool"),
        Control("budget.max_qwen_calls", "Maximum Qwen calls", "int", 0, 100,
                info="A cap, not a guaranteed number. Repairs also consume calls."),
        Control("budget.max_sam3_calls", "Maximum SAM3 actions", "int", 1, 1000),
        Control("replanning.continue_until_saturation", "Continue exploration (E policy)", "bool"),
    ),
    "Bootstrap and tiles": (
        Control("bootstrap.locked_context_prompt", "SAM3 context prompt (blank = full image)", "text", nullable=True,
                info="Localizes a search rectangle; Qwen's tree-fruit scope stays unchanged."),
        Control("bootstrap.locked_context_threshold", "Context threshold", maximum=1),
        Control("bootstrap.locked_context_fallback_full_image", "Use full image if context is absent", "bool"),
        Control("bootstrap.enable_tiled_bootstrap", "Tiled bootstrap", "bool"),
        Control("bootstrap.enable_pseudoexemplar_refinement", "Bootstrap exemplar refinement", "bool"),
        Control("bootstrap.pseudoexemplar_max_count", "Maximum exemplars", "int", 0),
        Control("bootstrap.pseudoexemplar_min_score", "Minimum exemplar SAM3 score", maximum=1),
        Control("sam3.qwen_discovery_use_exemplars", "Use exemplars for Qwen positives", "bool"),
        Control("tiling.grid_rows", "Tile rows", "int", 1, 16),
        Control("tiling.grid_cols", "Tile columns", "int", 1, 16),
        Control("tiling.overlap_ratio", "Tile overlap", maximum=.95),
        Control("tiling.tile_min_size", "Minimum tile size (pixels)", "int", 1),
    ),
    "Confidence and counting": (
        Control("belief.neutral_confounder_misses", "Neutral negative non-retrieval", "bool",
                info="Off preserves the selected D policy; a negative miss can raise target probability."),
        Control("belief.discount_repeat_weight", "Repeated-evidence discount", maximum=1),
        Control("belief.num_confounders", "Confounder slots", "int", 1, 10),
        Control("belief.target_count_hard_threshold", "Hard count cutoff (blank = off)", maximum=1, nullable=True,
                info="Qwen runs only: count 1 if posterior is strictly above this cutoff."),
        Control("belief.target_count_commit_threshold", "Soft commitment cutoff (blank = off)", minimum=.000001,
                maximum=1, nullable=True, info="Qwen runs only. Leave BOTH cutoffs blank for pure soft counting."),
    ),
    "Association and duplicate suppression": (
        Control("association.enable_iom_dedup", "Use IoU + containment (IoM)", "bool"),
        Control("association.iou_match_threshold", "Strong-match IoU", maximum=1),
        Control("association.new_node_iou_threshold", "Existing-node overlap IoU", maximum=1),
        Control("association.iom_match_threshold", "Strong-match IoM", maximum=1),
        Control("association.new_node_iom_threshold", "Existing-node overlap IoM", maximum=1),
        Control("association.tiled_nms_threshold", "Within-observation suppression IoU", maximum=1),
        Control("association.tiled_nms_iom_threshold", "Within-observation suppression IoM", maximum=1),
    ),
    "Budgets and stopping": (
        Control("budget.max_sam3_tiles", "Maximum SAM3 tiles (blank = no separate cap)", "int", nullable=True),
        Control("budget.max_runtime_seconds", "Runtime limit in seconds (blank = no limit)", minimum=.001, nullable=True),
        Control("stopping.max_iterations", "Maximum sensing iterations (blank = no limit)", "int", 1, nullable=True),
        Control("replanning.discovery_plateau_steps", "Discovery plateau window", "int", 1),
        Control("replanning.min_actions_between_replans", "Minimum actions between replans", "int", 1),
        Control("replanning.min_relative_count_variance_reduction", "Useful relative variance reduction", maximum=1),
        Control("replanning.unresolved_entropy_threshold", "Unresolved entropy threshold"),
        Control("replanning.count_variance_threshold", "Variance threshold for replanning"),
        Control("stopping.count_variance_threshold", "Variance threshold for saturation stop"),
        Control("stopping.discovery_saturation_threshold", "Discovery gain threshold"),
        Control("stopping.utility_min_threshold", "Minimum action utility"),
    ),
    "Qwen request settings (instructions stay fixed)": (
        Control("planner.temperature", "Qwen temperature", maximum=2),
        Control("planner.max_output_tokens", "Qwen output token limit", "int", 1),
        Control("planner.request_timeout_seconds", "Qwen request timeout (seconds)", minimum=.001),
        Control("planner.enable_rejection_correction", "Allow rejected-plan correction", "bool"),
        Control("planner.correct_missing_target", "Correct missing target even with accepted negatives", "bool"),
    ),
    "Utility and optional cleanup": (
        Control("action_selection.alpha_discovery", "Discovery weight"),
        Control("action_selection.beta_discrimination", "Discrimination weight"),
        Control("action_selection.gamma_redundancy", "Redundancy penalty"),
        Control("action_selection.lambda_cost", "Compute-cost penalty"),
        Control("action_selection.eta_qwen_priority", "Qwen-priority weight"),
        Control("budget.max_cleanup_calls", "Maximum cleanup calls (0 = disabled)", "int", 0, 1000),
        Control("cleanup.cleanup_residual_max_nodes", "Maximum residual nodes for cleanup", "int", 0),
        Control("cleanup.cleanup_ambiguity_threshold", "Cleanup ambiguity threshold", maximum=1),
        Control("cleanup.cleanup_min_utility", "Minimum cleanup utility"),
        Control("cleanup.roi_batch_size", "Cleanup ROI batch size", "int", 1),
    ),
}
CONTROLS = tuple(c for group in CONTROL_GROUPS.values() for c in group)


def presets(base: V4Config):
    return {v.name: v for v in _pilot_variants(base, "final-ae")}


def control_values(config: V4Config):
    values = []
    for control in CONTROLS:
        section, name = control.path.split(".")
        value = getattr(getattr(config, section), name)
        if control.path.startswith("sam3.qwen_") and control.path.endswith("threshold") and value is None:
            value = config.sam3.qwen_prompt_threshold
        values.append(value)
    return values


def build_config(base: V4Config, enable_qwen: bool, values: list) -> V4Config:
    if len(values) != len(CONTROLS):
        raise ValueError("The form and configuration fields do not match.")
    if not isinstance(enable_qwen, bool):
        raise ValueError("Enable Qwen must be a checkbox value.")
    sections = {}
    for control, value in zip(CONTROLS, values):
        section, name = control.path.split(".")
        sections.setdefault(section, {})[name] = control.parse(value)
    config = replace(base, **{
        section: replace(getattr(base, section), **overrides)
        for section, overrides in sections.items()
    })
    if enable_qwen and config.budget.max_qwen_calls < 1:
        raise ValueError("Enable Qwen requires at least one Qwen call.")
    # Changing the call cap must also enable the corresponding ordinary replans.
    # Corrections still share that cap; they do not create extra calls.
    config = replace(config, replanning=replace(config.replanning,
        max_replans=max(0, config.budget.max_qwen_calls - 1) if enable_qwen else 0))
    if not enable_qwen:
        config = replace(config,
            budget=replace(config.budget, max_qwen_calls=0, max_cleanup_calls=0),
            planner=replace(config.planner, execute_confounder_prompts=False),
            belief=replace(config.belief, target_count_hard_threshold=None, target_count_commit_threshold=None))
    if config.association.new_node_iou_threshold > config.association.iou_match_threshold:
        raise ValueError("Existing-node IoU must not exceed strong-match IoU.")
    if config.association.new_node_iom_threshold > config.association.iom_match_threshold:
        raise ValueError("Existing-node IoM must not exceed strong-match IoM.")
    return config
