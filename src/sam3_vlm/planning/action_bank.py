"""Action bank lifecycle, validation, deduplication, and rejection telemetry."""

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, List, Optional, Set
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.core.types import ActionSource, SpatialMode
from sam3_vlm.planning.qwen_planner import PlannerOutput, ProposedAction
from sam3_vlm.scene.belief import SemanticMemory, canonical_belief_classes
from sam3_vlm.sensing.action import SensingAction, validate_sam3_prompt_contract


def canonicalize_semantic_key(key: str) -> str:
    if not key:
        return ""
    return re.sub(r"[^\w]+", "_", key.lower()).strip("_")


def derive_correlation_group(semantic_key: str, prompt: str) -> str:
    return canonicalize_semantic_key(semantic_key)


class ActionRejectionReason(str, Enum):
    INVALID_SPATIAL_MODE = "INVALID_SPATIAL_MODE"
    MISSING_ROI = "MISSING_ROI"
    DUPLICATE_SEMANTIC_KEY = "DUPLICATE_SEMANTIC_KEY"
    CORRELATED_DUPLICATE = "CORRELATED_DUPLICATE"
    SCHEMA_INVALID = "SCHEMA_INVALID"
    EMPTY_PROMPT = "EMPTY_PROMPT"
    INVALID_GROUNDING_PROMPT = "INVALID_GROUNDING_PROMPT"
    UNKNOWN_CLASS_PRIOR = "UNKNOWN_CLASS_PRIOR"


@dataclass
class ActionRejection:
    semantic_key: str
    sam3_prompt: str
    reason: str
    detail: str
    suggested_spatial_mode: str

    def to_dict(self) -> dict:
        return {
            "semantic_key": self.semantic_key,
            "sam3_prompt": self.sam3_prompt,
            "reason": self.reason,
            "detail": self.detail,
            "suggested_spatial_mode": self.suggested_spatial_mode,
        }


@dataclass
class ActionBankEntry:
    action: SensingAction
    qwen_priority: Optional[float] = None
    predicted_discovery_value: Optional[float] = None
    predicted_discrimination_value: Optional[float] = None
    redundancy: float = 0.0
    estimated_cost: float = 1.0
    executed: bool = False
    invalid_reason: Optional[str] = None
    invalid_detail: Optional[str] = None
    total_utility: Optional[float] = None


@dataclass
class ActionBank:
    entries: List[ActionBankEntry] = field(default_factory=list)

    def add_action(self, action: SensingAction, qwen_priority: Optional[float] = None) -> Optional[ActionBankEntry]:
        entry = ActionBankEntry(
            action=action,
            qwen_priority=qwen_priority if qwen_priority is not None else action.qwen_priority,
        )
        try:
            action.validate()
        except ValueError as exc:
            entry.invalid_reason = ActionRejectionReason.SCHEMA_INVALID.value
            entry.invalid_detail = str(exc)
            self.entries.append(entry)
            return None
        self.entries.append(entry)
        return entry

    def pop_next(self) -> Optional[ActionBankEntry]:
        for entry in self.entries:
            if not entry.executed and entry.invalid_reason is None:
                entry.executed = True
                return entry
        return None

    def unexecuted_entries(self) -> List[ActionBankEntry]:
        return [e for e in self.entries if not e.executed and e.invalid_reason is None]

    def executed_entries(self) -> List[ActionBankEntry]:
        return [e for e in self.entries if e.executed]

    def purge_stale_actions(self, min_utility: float):
        for entry in self.entries:
            if entry.executed or entry.invalid_reason is not None:
                continue
            if entry.total_utility is not None and entry.total_utility < min_utility:
                entry.invalid_reason = "LOW_UTILITY"
                entry.invalid_detail = f"utility={entry.total_utility:.6f} < {min_utility:.6f}"


class ActionBankGenerator:
    """Turn Qwen proposals into executable scene-level actions with auditable rejections."""

    def __init__(self) -> None:
        self.last_rejections: List[ActionRejection] = []

    def _reject(self, proposal: ProposedAction, reason: ActionRejectionReason, detail: str) -> None:
        self.last_rejections.append(
            ActionRejection(
                semantic_key=proposal.semantic_key,
                sam3_prompt=proposal.prompt,
                reason=reason.value,
                detail=detail,
                suggested_spatial_mode=proposal.suggested_spatial_mode.value,
            )
        )

    def generate_entries(
        self,
        planner_output: PlannerOutput,
        semantic_memory: SemanticMemory,
        action_bank: ActionBank,
        id_gen: IDGenerator,
        valid_node_ids: Optional[Set[str]] = None,
        config: Optional[Any] = None,
        search_region: Optional[Any] = None,
    ) -> List[ActionBankEntry]:
        self.last_rejections = []
        added_entries: List[ActionBankEntry] = []
        existing_keys: Set[str] = set()
        existing_groups: Set[str] = set()
        existing_prompts: Set[str] = set()
        group_avg_utilities = {}

        for group, record in semantic_memory.records.items():
            existing_groups.add(group)
            existing_keys.update(canonicalize_semantic_key(k) for k in record.semantic_keys)
            existing_prompts.update(p.strip().lower() for p in record.prompts)
            if record.execution_count > 0:
                group_avg_utilities[group] = (
                    sum(record.realized_utility_by_execution) / record.execution_count
                )

        for entry in action_bank.entries:
            existing_keys.add(canonicalize_semantic_key(entry.action.semantic_key))
            existing_groups.add(
                entry.action.correlation_group
                or derive_correlation_group(entry.action.semantic_key, entry.action.prompt)
            )
            existing_prompts.add(entry.action.prompt.strip().lower())

        allowed_classes = set(
            canonical_belief_classes(config.belief.num_confounders if config else 2)
        )

        for proposal in planner_output.proposed_actions:
            canonical_key = canonicalize_semantic_key(proposal.semantic_key)
            if not canonical_key:
                self._reject(proposal, ActionRejectionReason.SCHEMA_INVALID, "empty semantic_key")
                continue

            if not proposal.prompt or not proposal.prompt.strip():
                self._reject(proposal, ActionRejectionReason.EMPTY_PROMPT, "sam3_prompt is empty")
                continue
            try:
                validate_sam3_prompt_contract(proposal.prompt)
            except ValueError as exc:
                self._reject(proposal, ActionRejectionReason.INVALID_GROUNDING_PROMPT, str(exc))
                continue

            unknown_prior = set(proposal.semantic_prior or {}) - allowed_classes
            if unknown_prior:
                self._reject(
                    proposal,
                    ActionRejectionReason.UNKNOWN_CLASS_PRIOR,
                    f"only {sorted(allowed_classes)} are legal; got {sorted(unknown_prior)}",
                )
                continue

            # Scene-level Qwen does not own geometry.  LOCAL/ROI_BATCH belong to
            # controller cleanup; the persistent run search region is injected below.
            if proposal.suggested_spatial_mode not in (SpatialMode.GLOBAL, SpatialMode.TILED):
                reason = (
                    ActionRejectionReason.MISSING_ROI
                    if proposal.suggested_spatial_mode in (SpatialMode.LOCAL, SpatialMode.ROI_BATCH)
                    and not proposal.roi
                    else ActionRejectionReason.INVALID_SPATIAL_MODE
                )
                self._reject(
                    proposal,
                    reason,
                    "Qwen scene actions may use only GLOBAL or TILED; controller owns local ROIs.",
                )
                continue
            if proposal.roi is not None:
                self._reject(
                    proposal,
                    ActionRejectionReason.INVALID_SPATIAL_MODE,
                    "Qwen may not provide geometry; the controller injects the locked search region.",
                )
                continue

            if canonical_key in existing_keys:
                self._reject(
                    proposal,
                    ActionRejectionReason.DUPLICATE_SEMANTIC_KEY,
                    f"semantic key {canonical_key!r} already exists",
                )
                continue
            clean_prompt = proposal.prompt.strip().lower()
            if clean_prompt in existing_prompts:
                self._reject(
                    proposal,
                    ActionRejectionReason.DUPLICATE_SEMANTIC_KEY,
                    "exact SAM3 prompt already exists",
                )
                continue

            corr_group = proposal.correlation_group or derive_correlation_group(
                canonical_key, proposal.prompt
            )
            if corr_group in existing_groups:
                self._reject(
                    proposal,
                    ActionRejectionReason.CORRELATED_DUPLICATE,
                    f"correlation group {corr_group!r} already searched/proposed",
                )
                continue

            if valid_node_ids is not None:
                referenced = set(proposal.positive_exemplar_ids) | set(proposal.negative_exemplar_ids)
                missing = referenced - valid_node_ids
                if missing:
                    self._reject(
                        proposal,
                        ActionRejectionReason.SCHEMA_INVALID,
                        f"unknown exemplar node ids: {sorted(missing)}",
                    )
                    continue

            tiling_cfg = None
            if proposal.suggested_spatial_mode == SpatialMode.TILED:
                if proposal.tiling:
                    from sam3_vlm.core.config import TilingConfig
                    try:
                        tiling_cfg = TilingConfig(**proposal.tiling)
                    except Exception as exc:
                        self._reject(proposal, ActionRejectionReason.SCHEMA_INVALID, str(exc))
                        continue
                elif config is not None:
                    tiling_cfg = config.tiling

            adjusted_priority = proposal.priority
            if corr_group in group_avg_utilities:
                avg_utility = group_avg_utilities[corr_group]
                threshold = config.stopping.utility_min_threshold if config else 0.05
                adjusted_priority *= 0.1 if avg_utility < threshold else 1.2
                adjusted_priority = min(1.0, adjusted_priority)

            action = SensingAction(
                action_id=id_gen.next_action_id(),
                semantic_key=canonical_key,
                prompt=proposal.prompt,
                family=proposal.family,
                threshold=proposal.suggested_threshold if proposal.suggested_threshold is not None else 0.25,
                spatial_mode=proposal.suggested_spatial_mode,
                source=ActionSource.QWEN,
                qwen_priority=adjusted_priority,
                semantic_prior=proposal.semantic_prior,
                correlation_group=corr_group,
                roi=search_region,
                positive_exemplar_ids=tuple(proposal.positive_exemplar_ids),
                negative_exemplar_ids=tuple(proposal.negative_exemplar_ids),
                tiling=tiling_cfg,
            )
            try:
                action.validate()
            except ValueError as exc:
                reason = (
                    ActionRejectionReason.MISSING_ROI
                    if "requires ROI" in str(exc)
                    else ActionRejectionReason.SCHEMA_INVALID
                )
                self._reject(proposal, reason, str(exc))
                continue

            entry = action_bank.add_action(action, qwen_priority=adjusted_priority)
            if entry is None:
                self._reject(
                    proposal,
                    ActionRejectionReason.SCHEMA_INVALID,
                    "action failed ActionBank validation",
                )
                continue

            existing_keys.add(canonical_key)
            existing_groups.add(corr_group)
            existing_prompts.add(clean_prompt)
            added_entries.append(entry)

        return added_entries
