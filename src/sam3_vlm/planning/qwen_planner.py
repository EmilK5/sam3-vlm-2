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
    """Planner service wrapper enforcing Qwen call budget and output schema parsing (V4 Design Spec §6.4)."""

    def __init__(self, planner_backend: Any) -> None:
        self.planner_backend = planner_backend

    def plan_scene(
        self,
        evidence: QwenEvidencePack,
        budget: BudgetState,
        config: V4Config = V4Config(),
    ) -> PlannerOutput:
        """Execute Qwen planning pass if within budget."""
        if budget.qwen_calls >= config.budget.max_qwen_calls:
            raise BudgetExceededError(
                f"Qwen call budget exhausted ({budget.qwen_calls}/{config.budget.max_qwen_calls})."
            )

        budget.qwen_calls += 1

        if hasattr(self.planner_backend, "plan_scene"):
            return self.planner_backend.plan_scene(evidence, budget, config)
        elif hasattr(self.planner_backend, "plan_actions"):
            actions = self.planner_backend.plan_actions(evidence)
            if isinstance(actions, PlannerOutput):
                return actions
            return PlannerOutput(scene_summary="Planner execution complete.", proposed_actions=[])

        return PlannerOutput(scene_summary="No planner backend available.", proposed_actions=[])
