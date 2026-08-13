"""Qwen scene planner interface protocol and adapter implementations (V4 Design Spec §6)."""

from typing import Any, List, Protocol, runtime_checkable
from sam3_vlm.core.types import ActionFamily, SpatialMode
from sam3_vlm.planning.qwen_planner import PlannerOutput, ProposedAction
from sam3_vlm.sensing.evidence import QwenEvidencePack
import os
import logging
from sam3_vlm.core.config import V4Config
from sam3_vlm.core.types import BudgetState
from sam3_vlm.scene.belief import canonical_belief_classes

logger = logging.getLogger(__name__)


@runtime_checkable
class QwenPlanner(Protocol):
    def plan_scene(self, evidence_pack: QwenEvidencePack, *args: Any, **kwargs: Any) -> PlannerOutput:
        ...


class DummyQwenPlanner:
    def __init__(self) -> None:
        self.call_count = 0

    def plan_scene(self, evidence_pack: QwenEvidencePack, *args: Any, **kwargs: Any) -> PlannerOutput:
        self.call_count += 1
        return PlannerOutput(scene_summary="Dummy planner pass.", proposed_actions=[])

    def plan_actions(self, evidence_pack: Any) -> List[Any]:
        self.call_count += 1
        return []


class MockQwenPlanner:
    """Deterministic planner whose outputs obey the real executable contract."""

    def __init__(self, custom_output: PlannerOutput | None = None) -> None:
        self.call_count = 0
        self.custom_output = custom_output
        self.model = "mock-qwen"

    def plan_scene(self, evidence_pack: QwenEvidencePack, *args: Any, **kwargs: Any) -> PlannerOutput:
        self.call_count += 1
        if self.custom_output is not None:
            return self.custom_output

        target_noun = (evidence_pack.user_prompt or "target object").strip().split()[-1]
        actions = [
            ProposedAction(
                semantic_key="small_target",
                prompt=f"small {target_noun}",
                family=ActionFamily.DISCOVERY,
                priority=0.90,
                semantic_prior={"target": 0.85, "confounder1": 0.15},
                suggested_threshold=0.25,
                suggested_spatial_mode=SpatialMode.GLOBAL,
                rationale="Search a smaller target appearance mode.",
            ),
            ProposedAction(
                semantic_key="flat_foliage",
                prompt="flat foliage",
                family=ActionFamily.CONFOUNDER,
                priority=0.85,
                semantic_prior={"target": 0.10, "confounder1": 0.90},
                suggested_threshold=0.25,
                suggested_spatial_mode=SpatialMode.GLOBAL,
                rationale="Test a visually plausible flat confounder.",
            ),
            ProposedAction(
                semantic_key="round_green_target",
                prompt=f"round green {target_noun}",
                family=ActionFamily.VERIFICATION,
                priority=0.75,
                semantic_prior={"target": 0.80, "confounder1": 0.20},
                suggested_threshold=0.30,
                suggested_spatial_mode=SpatialMode.TILED,
                rationale="Test a more discriminative target appearance at higher spatial resolution.",
            ),
        ]
        return PlannerOutput(
            scene_summary=f"Mock Qwen planning call {self.call_count}: proposed 3 actions.",
            proposed_actions=actions,
            missing_appearance_modes=["small target", "shadowed target"],
            likely_confounders=["foliage", "background"],
        )


class RealQwenPlanner:
    """Real Qwen scene planner using an OpenAI-compatible endpoint."""

    SYSTEM_PROMPT = (
        "You propose semantic experiments for SAM3. You are NOT a detector and must not count objects. "
        "Every executable sam3_prompt MUST be only 2 or 3 words: exactly one or two directly visible "
        "visual modifiers followed by one visible object noun. No verbs, clauses, locations, prepositions, "
        "analysis methods, imaging methods, edge detection, spectral/multispectral language, clustering, "
        "channel analysis, or contrast enhancement may appear in sam3_prompt. Put all reasoning in rationale. "
        "Scene-level actions may use only GLOBAL or TILED spatial modes. Never output boxes/ROIs. "
        "Belief classes are frozen generic slots: target, confounder1, confounder2, etc.; semantic object names "
        "must never become class keys. Return ONLY valid JSON matching the requested schema."
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
        self.api_key = api_key or os.environ.get("QWEN_API_KEY") or "EMPTY"
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
        belief_classes = canonical_belief_classes(config.belief.num_confounders)
        confounder_slots = [c for c in belief_classes if c != "target"]
        existing_mapping = evidence_pack.confounder_labels or {}

        text = evidence_pack.to_prompt_text()
        text += (
            "\n\nEXECUTABLE ACTION CONTRACT:\n"
            "- sam3_prompt: exactly 2 or 3 words: one/two visible modifiers + one object noun.\n"
            "- Examples of valid shape: 'green fruit', 'round green fruit', 'flat leaf'.\n"
            "- rationale: unrestricted short reasoning; reasoning NEVER goes into sam3_prompt.\n"
            "- suggested_spatial_mode: GLOBAL or TILED only. The controller owns the locked search ROI.\n"
            f"- semantic_prior keys may ONLY be: {belief_classes}.\n"
            f"- likely_confounders has at most {len(confounder_slots)} entries; entry i names confounder{i+1}.\n"
        )
        if existing_mapping:
            text += (
                f"- Existing confounder slot mapping is FROZEN: {existing_mapping}. "
                "Do not rename/reorder those slots on replanning.\n"
            )
        text += (
            "\nReturn JSON:\n"
            "{\n"
            '  "scene_summary": "<string>",\n'
            '  "missing_appearance_modes": ["<string>"],\n'
            '  "likely_confounders": ["<semantic label aligned to confounder slots>"],\n'
            '  "proposed_actions": [\n'
            "    {\n"
            '      "semantic_key": "<string>",\n'
            '      "sam3_prompt": "<2 or 3 words only>",\n'
            '      "family": "DISCOVERY | CONFOUNDER | CONTEXT | VERIFICATION",\n'
            '      "priority": <float 0.0-1.0>,\n'
            '      "semantic_prior": {"target/confounderN": <float>},\n'
            '      "suggested_threshold": <float 0.0-1.0>,\n'
            '      "suggested_spatial_mode": "GLOBAL | TILED",\n'
            '      "rationale": "<short reasoning>"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Output JSON only."
        )

        content = [{"type": "text", "text": text}]

        def get_mime_type(path: str) -> str:
            ext = os.path.splitext(path)[1].lower()
            if ext in (".jpg", ".jpeg"):
                return "image/jpeg"
            if ext == ".png":
                return "image/png"
            if ext == ".webp":
                return "image/webp"
            raise ValueError(f"Unsupported image extension for Qwen: {ext}")

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
            content.append({"type": "image_url", "image_url": {"url": f"data:{get_mime_type(orig_path)};base64,{b64}"}})

        cs_path = evidence_pack.contact_sheet.contact_sheet_image_path
        if cs_path and os.path.exists(cs_path):
            with open(cs_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{get_mime_type(cs_path)};base64,{b64}"}})

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=config.planner.temperature,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as exc:
            if self.strict_model_errors:
                raise RuntimeError(f"Strict Qwen execution failed: {exc}") from exc
            raise
