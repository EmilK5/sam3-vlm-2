"""Qwen scene planner interface protocol and adapter implementations (V4 Design Spec §6)."""

from typing import Any, List, Protocol, runtime_checkable
from sam3_vlm.core.types import ActionFamily, SpatialMode
from sam3_vlm.planning.qwen_planner import PlannerOutput, ProposedAction
from sam3_vlm.sensing.evidence import QwenEvidencePack
import os
import logging
from sam3_vlm.core.config import V4Config
from sam3_vlm.core.types import BudgetState

logger = logging.getLogger(__name__)


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
            tiling={"grid_rows": 2, "grid_cols": 2, "overlap_ratio": 0.15, "tile_min_size": 512},
            rationale="High-resolution tiled pass over dense clusters.",
        )

        return PlannerOutput(
            scene_summary=f"Mock Qwen planning call {self.call_count}: proposed 3 actions.",
            proposed_actions=[p1, p2, p3],
            missing_appearance_modes=[f"small_{target_cls}", f"shadowed_{target_cls}"],
            likely_confounders=["leaf_foliage", "branch_shadow"],
        )


class RealQwenPlanner:
    """Real Qwen scene planner using an OpenAI-compatible endpoint (V4 Design Spec §6)."""

    SYSTEM_PROMPT = (
        "You are a rigorous semantic planner for a visual perception system. "
        "Your task is to analyze the provided scene and propose a set of semantic sensing actions. "
        "Always respond with ONLY a valid JSON object matching the requested schema."
    )

    def __init__(
        self, 
        base_url: str = None, 
        model: str = None, 
        api_key: str = None,
        strict_model_errors: bool = False,
    ) -> None:
        self.call_count = 0
        self.base_url = base_url or os.environ.get("QWEN_BASE_URL")
        self.model = model or os.environ.get("QWEN_MODEL")
        self.api_key = api_key or os.environ.get("QWEN_API_KEY", "EMPTY")
        self.strict_model_errors = strict_model_errors

        if not self.base_url or not self.model:
            raise ValueError("RealQwenPlanner requires QWEN_BASE_URL and QWEN_MODEL environment variables.")

        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai package is required for RealQwenPlanner. Please install it.")

        logger.info(f"Loading real Qwen planner: {self.model} at {self.base_url}")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def plan_scene(self, evidence_pack: QwenEvidencePack, budget: BudgetState, config: V4Config) -> str:
        self.call_count += 1

        text = evidence_pack.to_prompt_text()
        text += (
            "\n\nProvide a valid JSON response with the following schema:\n"
            "{\n"
            "  \"scene_summary\": \"<string>\",\n"
            "  \"missing_appearance_modes\": [\"<string>\"],\n"
            "  \"likely_confounders\": [\"<string>\"],\n"
            "  \"proposed_actions\": [\n"
            "    {\n"
            "      \"semantic_key\": \"<string>\",\n"
            "      \"prompt\": \"<string>\",\n"
            "      \"family\": \"DISCOVERY | CONFOUNDER | CONTEXT | VERIFICATION\",\n"
            "      \"priority\": <float 0.0-1.0>,\n"
            "      \"semantic_prior\": {\"<class>\": <float>},\n"
            "      \"suggested_threshold\": <float 0.0-1.0>,\n"
            "      \"suggested_spatial_mode\": \"GLOBAL | TILED | ROI_BATCH | LOCAL\",\n"
            "      \"rationale\": \"<string>\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "Ensure the output is ONLY the JSON object, with no markdown formatting."
        )

        content = [{"type": "text", "text": text}]

        def get_mime_type(path: str) -> str:
            ext = os.path.splitext(path)[1].lower()
            if ext in (".jpg", ".jpeg"):
                return "image/jpeg"
            elif ext == ".png":
                return "image/png"
            elif ext == ".webp":
                return "image/webp"
            return "image/jpeg"

        import base64
        
        orig_path = evidence_pack.image_path
        if not orig_path:
            if self.strict_model_errors:
                raise ValueError("Original image is strictly required for M8 real Qwen planning.")
        elif not os.path.exists(orig_path):
            if self.strict_model_errors:
                raise ValueError(f"Original image not found at {orig_path}")
        else:
            with open(orig_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            mime = get_mime_type(orig_path)
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        cs_path = evidence_pack.contact_sheet.contact_sheet_image_path
        if cs_path and os.path.exists(cs_path):
            with open(cs_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            mime = get_mime_type(cs_path)
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            if self.strict_model_errors:
                raise RuntimeError(f"Strict Qwen execution failed: {e}") from e
            raise e

