"""Qwen scene planner interface protocol and adapter implementations (V4 Design Spec §6)."""

from typing import Any, List, Protocol, runtime_checkable
from sam3_vlm.core.types import ActionFamily, SpatialMode
from sam3_vlm.planning.qwen_planner import PlannerOutput, ProposedAction
from sam3_vlm.sensing.evidence import QwenEvidencePack


@runtime_checkable
class QwenPlanner(Protocol):
    """Clean Qwen scene planner contract."""

    def plan_scene(self, evidence_pack: QwenEvidencePack, *args: Any, **kwargs: Any) -> PlannerOutput:
        """Propose structured sensing action bank from scene evidence."""
        ...


class DummyQwenPlanner:
    """Basic mock Qwen planner for testing and foundation verification."""

    def __init__(self) -> None:
        self.call_count = 0

    def plan_scene(self, evidence_pack: QwenEvidencePack, *args: Any, **kwargs: Any) -> PlannerOutput:
        self.call_count += 1
        return PlannerOutput(scene_summary="Dummy planner pass.", proposed_actions=[])

    def plan_actions(self, evidence_pack: Any) -> List[Any]:
        self.call_count += 1
        return []


class MockQwenPlanner:
    """Configurable mock Qwen planner returning structured action proposals (V4 Design Spec §6.2)."""

    def __init__(self, custom_output: PlannerOutput | None = None) -> None:
        self.call_count = 0
        self.custom_output = custom_output

    def plan_scene(self, evidence_pack: QwenEvidencePack, *args: Any, **kwargs: Any) -> PlannerOutput:
        self.call_count += 1
        if self.custom_output is not None:
            return self.custom_output

        target_cls = evidence_pack.target_class or "target"
        prompt_concept = evidence_pack.user_prompt or "target object"

        p1 = ProposedAction(
            semantic_key=f"small_{target_cls}",
            prompt=f"small {prompt_concept}",
            family=ActionFamily.DISCOVERY,
            priority=0.90,
            semantic_prior={target_cls: 0.85, "confounder": 0.15},
            suggested_threshold=0.25,
            suggested_spatial_mode=SpatialMode.GLOBAL,
            rationale="Search for smaller missed target appearance modes.",
        )

        p2 = ProposedAction(
            semantic_key="leaf_foliage",
            prompt="shiny green leaf foliage",
            family=ActionFamily.CONFOUNDER,
            priority=0.85,
            semantic_prior={target_cls: 0.10, "leaf": 0.90},
            suggested_threshold=0.25,
            suggested_spatial_mode=SpatialMode.GLOBAL,
            rationale="Disambiguate leaf foliage confounders.",
        )

        p3 = ProposedAction(
            semantic_key=f"cluster_{target_cls}",
            prompt=f"dense cluster of {prompt_concept}",
            family=ActionFamily.VERIFICATION,
            priority=0.75,
            semantic_prior={target_cls: 0.70, "leaf": 0.30},
            suggested_threshold=0.30,
            suggested_spatial_mode=SpatialMode.TILED,
            rationale="High-resolution tiled pass over dense clusters.",
        )

        return PlannerOutput(
            scene_summary=f"Mock Qwen planning call {self.call_count}: proposed 3 actions.",
            proposed_actions=[p1, p2, p3],
            missing_appearance_modes=[f"small_{target_cls}", f"shadowed_{target_cls}"],
            likely_confounders=["leaf_foliage", "branch_shadow"],
        )
