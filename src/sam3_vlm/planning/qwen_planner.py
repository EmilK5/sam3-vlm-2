"""Qwen Planner schemas, output contracts, and service wrapper (V4 Design Spec §6 / §21.7)."""

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional, Protocol
from sam3_vlm.core.config import V4Config
from sam3_vlm.core.types import ActionFamily, BudgetState, SpatialMode
from sam3_vlm.sensing.evidence import QwenEvidencePack


class BudgetExceededError(RuntimeError):
    """Raised when a Qwen call attempt exceeds hard budget limits (V4 Design Spec §6.4 / §15)."""

    pass


@dataclass
class ProposedAction:
    """Individual action proposal returned by Qwen scene planner (V4 Design Spec §6.2)."""

    semantic_key: str
    prompt: str
    family: ActionFamily
    priority: float = 1.0
    semantic_prior: Dict[str, float] = field(default_factory=dict)
    suggested_threshold: Optional[float] = 0.25
    suggested_spatial_mode: SpatialMode = SpatialMode.GLOBAL
    exemplar_policy: Optional[str] = None
    rationale: str = ""
    correlation_group: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "semantic_key": self.semantic_key,
            "prompt": self.prompt,
            "family": self.family.value,
            "priority": self.priority,
            "semantic_prior": dict(self.semantic_prior),
            "suggested_threshold": self.suggested_threshold,
            "suggested_spatial_mode": self.suggested_spatial_mode.value,
            "exemplar_policy": self.exemplar_policy,
            "rationale": self.rationale,
            "correlation_group": self.correlation_group,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProposedAction":
        return cls(
            semantic_key=data["semantic_key"],
            prompt=data["prompt"],
            family=ActionFamily(data["family"]),
            priority=float(data.get("priority", 1.0)),
            semantic_prior=dict(data.get("semantic_prior", {})),
            suggested_threshold=data.get("suggested_threshold", 0.25),
            suggested_spatial_mode=SpatialMode(data.get("suggested_spatial_mode", "GLOBAL")),
            exemplar_policy=data.get("exemplar_policy"),
            rationale=data.get("rationale", ""),
            correlation_group=data.get("correlation_group"),
        )


@dataclass
class PlannerOutput:
    """Constrained structured object returned by Qwen scene planner (V4 Design Spec §6.2)."""

    scene_summary: str = ""
    proposed_actions: List[ProposedAction] = field(default_factory=list)
    missing_appearance_modes: List[str] = field(default_factory=list)
    likely_confounders: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_summary": self.scene_summary,
            "proposed_actions": [a.to_dict() for a in self.proposed_actions],
            "missing_appearance_modes": list(self.missing_appearance_modes),
            "likely_confounders": list(self.likely_confounders),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlannerOutput":
        actions = [ProposedAction.from_dict(a) for a in data.get("proposed_actions", [])]
        return cls(
            scene_summary=data.get("scene_summary", ""),
            proposed_actions=actions,
            missing_appearance_modes=list(data.get("missing_appearance_modes", [])),
            likely_confounders=list(data.get("likely_confounders", [])),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "PlannerOutput":
        return cls.from_dict(json.loads(json_str))


class PlannerService(Protocol):
    """High-level planning service contract."""

    def plan_scene(
        self,
        evidence: QwenEvidencePack,
        budget: BudgetState,
        config: V4Config = V4Config(),
    ) -> PlannerOutput:
        ...


class QwenPlannerService:
    """Planner service wrapper enforcing Qwen call budget, defensive validation, and error repair (V4 Design Spec §6.4)."""

    def __init__(self, planner_backend: Any) -> None:
        self.planner_backend = planner_backend

    def plan_scene(
        self,
        evidence: QwenEvidencePack,
        budget: BudgetState,
        config: V4Config = V4Config(),
    ) -> PlannerOutput:
        """Execute Qwen planning pass if within budget, with defensive validation and deterministic capping."""
        if budget.qwen_calls >= config.budget.max_qwen_calls:
            raise BudgetExceededError(
                f"Qwen call budget exhausted ({budget.qwen_calls}/{config.budget.max_qwen_calls})."
            )

        budget.qwen_calls += 1

        raw_output = None
        try:
            if hasattr(self.planner_backend, "plan_scene"):
                raw_output = self.planner_backend.plan_scene(evidence, budget, config)
            elif hasattr(self.planner_backend, "plan_actions"):
                raw_output = self.planner_backend.plan_actions(evidence)
        except Exception as e:
            # Defensive fallback on model call failure
            return PlannerOutput(
                scene_summary=f"Model call failed ({type(e).__name__}: {e}). Using deterministic fallback.",
                proposed_actions=[],
            )

        output = self._coerce_to_planner_output(raw_output)
        return self._validate_and_normalize_output(output, max_actions=5)

    def _coerce_to_planner_output(self, raw_output: Any) -> PlannerOutput:
        """Coerce raw backend response (dict, json string, or PlannerOutput) into a typed PlannerOutput."""
        if isinstance(raw_output, PlannerOutput):
            return raw_output
        if isinstance(raw_output, dict):
            try:
                return PlannerOutput.from_dict(raw_output)
            except Exception:
                pass
        if isinstance(raw_output, str):
            try:
                return PlannerOutput.from_json(raw_output)
            except Exception:
                pass
        return PlannerOutput(scene_summary="Raw output unparseable.", proposed_actions=[])

    def _validate_and_normalize_output(
        self, output: PlannerOutput, max_actions: int = 5
    ) -> PlannerOutput:
        """Defensively clamp numerical values and cap action count to max_actions."""
        normalized_actions: List[ProposedAction] = []
        for action in output.proposed_actions:
            clamped_priority = max(0.0, min(1.0, float(action.priority)))
            clamped_threshold = (
                max(0.0, min(1.0, float(action.suggested_threshold)))
                if action.suggested_threshold is not None
                else 0.25
            )
            clamped_priors = {
                k: max(0.0, min(1.0, float(v))) for k, v in action.semantic_prior.items()
            }

            normalized_actions.append(
                ProposedAction(
                    semantic_key=action.semantic_key,
                    prompt=action.prompt,
                    family=action.family,
                    priority=clamped_priority,
                    semantic_prior=clamped_priors,
                    suggested_threshold=clamped_threshold,
                    suggested_spatial_mode=action.suggested_spatial_mode,
                    exemplar_policy=action.exemplar_policy,
                    rationale=action.rationale,
                    correlation_group=action.correlation_group,
                )
            )

        # Sort actions by priority descending and cap at max_actions
        normalized_actions.sort(key=lambda a: a.priority, reverse=True)
        capped_actions = normalized_actions[:max_actions]

        return PlannerOutput(
            scene_summary=output.scene_summary,
            proposed_actions=capped_actions,
            missing_appearance_modes=list(output.missing_appearance_modes),
            likely_confounders=list(output.likely_confounders),
        )
