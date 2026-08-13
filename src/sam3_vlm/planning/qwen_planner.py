"""Qwen planner schemas, output contracts, and budgeted service wrapper."""

from dataclasses import dataclass, field
import json
import time
from typing import Any, Dict, List, Optional, Protocol
from sam3_vlm.core.config import V4Config
from sam3_vlm.core.types import ActionFamily, BudgetState, SpatialMode
from sam3_vlm.sensing.action import validate_sam3_prompt_contract
from sam3_vlm.sensing.evidence import QwenEvidencePack


class BudgetExceededError(RuntimeError):
    """Raised when a Qwen call attempt exceeds its hard call budget."""


@dataclass
class ProposedAction:
    """Individual scene-level experiment proposed by Qwen.

    ``prompt`` remains the internal compatibility name.  Persistent Qwen JSON
    uses ``sam3_prompt`` to make the planner/reasoner separation explicit.
    """

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
    roi: Optional[List[float]] = None
    positive_exemplar_ids: List[str] = field(default_factory=list)
    negative_exemplar_ids: List[str] = field(default_factory=list)
    tiling: Optional[Dict[str, Any]] = None

    @property
    def sam3_prompt(self) -> str:
        return self.prompt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "semantic_key": self.semantic_key,
            "sam3_prompt": self.prompt,
            "family": self.family.value,
            "priority": self.priority,
            "semantic_prior": dict(self.semantic_prior),
            "suggested_threshold": self.suggested_threshold,
            "suggested_spatial_mode": self.suggested_spatial_mode.value,
            "exemplar_policy": self.exemplar_policy,
            "rationale": self.rationale,
            "correlation_group": self.correlation_group,
            "roi": self.roi,
            "positive_exemplar_ids": self.positive_exemplar_ids,
            "negative_exemplar_ids": self.negative_exemplar_ids,
            "tiling": self.tiling,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProposedAction":
        prompt = data.get("sam3_prompt", data.get("prompt"))
        if prompt is None:
            raise KeyError("sam3_prompt")
        return cls(
            semantic_key=data["semantic_key"],
            prompt=prompt,
            family=ActionFamily(data["family"]),
            priority=float(data.get("priority", 1.0)),
            semantic_prior=dict(data.get("semantic_prior", {})),
            suggested_threshold=data.get("suggested_threshold", 0.25),
            suggested_spatial_mode=SpatialMode(data.get("suggested_spatial_mode", "GLOBAL")),
            exemplar_policy=data.get("exemplar_policy"),
            rationale=data.get("rationale", ""),
            correlation_group=data.get("correlation_group"),
            roi=data.get("roi"),
            positive_exemplar_ids=data.get("positive_exemplar_ids", []),
            negative_exemplar_ids=data.get("negative_exemplar_ids", []),
            tiling=data.get("tiling"),
        )


@dataclass
class PlannerOutput:
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
        return cls(
            scene_summary=data.get("scene_summary", ""),
            proposed_actions=[ProposedAction.from_dict(a) for a in data.get("proposed_actions", [])],
            missing_appearance_modes=list(data.get("missing_appearance_modes", [])),
            likely_confounders=list(data.get("likely_confounders", [])),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "PlannerOutput":
        return cls.from_dict(json.loads(json_str))


class PlannerService(Protocol):
    def plan_scene(
        self,
        evidence: QwenEvidencePack,
        budget: BudgetState,
        config: V4Config = V4Config(),
    ) -> PlannerOutput:
        ...


class QwenPlannerService:
    """Budgeted Qwen service with measured latency and one repair attempt."""

    def __init__(self, planner_backend: Any) -> None:
        self.planner_backend = planner_backend
        self.last_repair_attempted = False
        self.last_fallback_used = False
        self.last_call_runtime_ms = 0.0

    def _invoke_backend(self, evidence: QwenEvidencePack, budget: BudgetState, config: V4Config) -> Any:
        start = time.perf_counter()
        try:
            if hasattr(self.planner_backend, "plan_scene"):
                return self.planner_backend.plan_scene(evidence, budget, config)
            if hasattr(self.planner_backend, "plan_actions"):
                return self.planner_backend.plan_actions(evidence)
            return None
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.last_call_runtime_ms += elapsed_ms
            budget.qwen_runtime_ms += elapsed_ms
            budget.model_runtime_ms += elapsed_ms
            budget.total_runtime_ms += elapsed_ms

    def plan_scene(
        self,
        evidence: QwenEvidencePack,
        budget: BudgetState,
        config: V4Config = V4Config(),
    ) -> PlannerOutput:
        if budget.qwen_calls >= config.budget.max_qwen_calls:
            raise BudgetExceededError(
                f"Qwen call budget exhausted ({budget.qwen_calls}/{config.budget.max_qwen_calls})."
            )

        self.last_repair_attempted = False
        self.last_fallback_used = False
        self.last_call_runtime_ms = 0.0
        budget.qwen_calls += 1
        raw_output = None
        try:
            raw_output = self._invoke_backend(evidence, budget, config)
        except Exception as exc:
            if getattr(self.planner_backend, "strict_model_errors", False):
                raise
            return PlannerOutput(
                scene_summary=f"Model call failed ({type(exc).__name__}: {exc}).",
                proposed_actions=[],
            )

        output = self._coerce_to_planner_output(raw_output)
        if not output.proposed_actions and output.scene_summary == "Raw output unparseable.":
            if budget.qwen_calls < config.budget.max_qwen_calls:
                self.last_repair_attempted = True
                budget.qwen_calls += 1
                try:
                    import copy
                    repair_evidence = copy.deepcopy(evidence)
                    repair_evidence.scene_summary += (
                        "\nPrevious output was malformed. Return only the required JSON schema."
                    )
                    raw_output = self._invoke_backend(repair_evidence, budget, config)
                    output = self._coerce_to_planner_output(raw_output)
                except Exception:
                    pass

        if not output.proposed_actions and output.scene_summary == "Raw output unparseable.":
            if getattr(self.planner_backend, "strict_model_errors", False):
                raise ValueError(f"Strict Qwen execution failed: malformed response: {raw_output}")
            self.last_fallback_used = True
            fallback_prompt = evidence.user_prompt or "visible object"
            try:
                validate_sam3_prompt_contract(fallback_prompt)
            except ValueError:
                fallback_prompt = "visible object"
            output = PlannerOutput(
                scene_summary="Deterministic fallback due to repeated planner parse failure.",
                proposed_actions=[
                    ProposedAction(
                        semantic_key="target_fallback",
                        prompt=fallback_prompt,
                        family=ActionFamily.DISCOVERY,
                        priority=1.0,
                        semantic_prior={"target": 1.0},
                        suggested_threshold=0.25,
                        suggested_spatial_mode=SpatialMode.GLOBAL,
                    )
                ],
            )

        return self._validate_and_normalize_output(
            output, max_actions=config.planner.max_actions_per_prompt
        )

    def _coerce_to_planner_output(self, raw_output: Any) -> PlannerOutput:
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

    def _validate_and_normalize_output(self, output: PlannerOutput, max_actions: int = 5) -> PlannerOutput:
        # Do not silently drop semantically invalid actions here.  ActionBankGenerator
        # records explicit rejection codes so real runs explain why a bank became empty.
        normalized_actions: List[ProposedAction] = []
        for action in output.proposed_actions:
            priority = max(0.0, min(1.0, float(action.priority if action.priority is not None else 0.5)))
            threshold = (
                max(0.0, min(1.0, float(action.suggested_threshold)))
                if action.suggested_threshold is not None else 0.25
            )
            priors = {
                k: max(0.0, min(1.0, float(v))) for k, v in (action.semantic_prior or {}).items()
            }
            normalized_actions.append(
                ProposedAction(
                    semantic_key=action.semantic_key,
                    prompt=action.prompt,
                    family=action.family,
                    priority=priority,
                    semantic_prior=priors,
                    suggested_threshold=threshold,
                    suggested_spatial_mode=action.suggested_spatial_mode,
                    exemplar_policy=action.exemplar_policy,
                    rationale=action.rationale,
                    correlation_group=action.correlation_group,
                    roi=action.roi,
                    positive_exemplar_ids=list(action.positive_exemplar_ids),
                    negative_exemplar_ids=list(action.negative_exemplar_ids),
                    tiling=action.tiling,
                )
            )
        normalized_actions.sort(key=lambda a: a.priority, reverse=True)
        return PlannerOutput(
            scene_summary=output.scene_summary,
            proposed_actions=normalized_actions[:max_actions],
            missing_appearance_modes=list(output.missing_appearance_modes),
            likely_confounders=list(output.likely_confounders),
        )
