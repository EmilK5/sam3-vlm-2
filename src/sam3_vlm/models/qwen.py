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
                tiling={
                    "grid_rows": 2,
                    "grid_cols": 2,
                    "overlap_ratio": 0.15,
                    "tile_min_size": 512,
                },
                rationale="Test a more discriminative target appearance at higher spatial resolution.",
            ),
        ]
        canonical_m8 = bool(evidence_pack.belief_classes) and (
            evidence_pack.belief_classes[0] == "target"
        )
        selected_actions = (
            [
                ProposedAction(
                    semantic_key="target",
                    prompt=f"small {target_noun}",
                    family=ActionFamily.DISCOVERY,
                    priority=0.90,
                    semantic_prior={"target": 1.0},
                    suggested_threshold=0.25,
                    suggested_spatial_mode=SpatialMode.GLOBAL,
                    rationale="Search a smaller target appearance mode.",
                )
            ]
            if canonical_m8
            else actions
        )
        return PlannerOutput(
            scene_summary=(
                f"Mock Qwen planning call {self.call_count}: "
                f"proposed {len(selected_actions)} actions."
            ),
            proposed_actions=selected_actions,
            missing_appearance_modes=["small target", "shadowed target"],
            likely_confounders=["foliage", "background"],
        )


class RealQwenPlanner:
    """Real Qwen scene planner using an OpenAI-compatible endpoint."""

    SYSTEM_PROMPT = (
        "You propose target-search experiments for SAM3. You are NOT a detector and must not count objects. "
        "Qwen never decides whether the pipeline should stop. If discovery is not explicitly saturated, "
        "proposed_actions must contain exactly one novel target DISCOVERY experiment, even when current "
        "candidates look convincing. An empty proposed_actions list is permitted only when discovery is "
        "explicitly saturated. Never propose more than one action. Only the controller may stop after "
        "evaluating sensor evidence and budget. "
        "Every executable sam3_prompt MUST be only 2 or 3 words: exactly one or two directly visible "
        "visual modifiers followed by one visible object noun. No verbs, clauses, locations, prepositions, "
        "analysis methods, imaging methods, edge detection, spectral/multispectral language, clustering, "
        "channel analysis, or contrast enhancement may appear in sam3_prompt. Put all reasoning in rationale. "
        "Scene-level actions may use only GLOBAL or TILED spatial modes. Never output boxes/ROIs. "
        "Every executable action must search for the user's target: semantic_key must be 'target', family must "
        "be 'DISCOVERY', and semantic_prior must be {'target': 1.0}. Confounders may be described in "
        "likely_confounders or rationale, but never proposed as separate SAM3 actions. Return ONLY valid JSON "
        "matching the requested schema."
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
        # QwenPlannerService owns the single explicit repair attempt. Disable
        # SDK-level retries so one timed-out request cannot silently multiply
        # the configured inference deadline.
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            max_retries=0,
        )

    def plan_scene(self, evidence_pack: QwenEvidencePack, budget: BudgetState, config: V4Config) -> str:
        self.call_count += 1
        belief_classes = canonical_belief_classes(config.belief.num_confounders)
        confounder_slots = [c for c in belief_classes if c != "target"]
        existing_mapping = evidence_pack.confounder_labels or {}

        text = evidence_pack.to_prompt_text(enforce_qwen_contract=True)
        text += (
            "\n\nEXECUTABLE ACTION CONTRACT:\n"
            "- sam3_prompt: exactly 2 or 3 words: one/two visible modifiers + one object noun.\n"
            "- Examples of valid shape: 'green fruit', 'round green fruit', 'small red car'.\n"
            "- rationale: unrestricted short reasoning; reasoning NEVER goes into sam3_prompt.\n"
            "- suggested_spatial_mode: GLOBAL or TILED only. The controller owns the locked search ROI.\n"
            "- Every action must use semantic_key='target', family='DISCOVERY', "
            "and semantic_prior={'target': 1.0}.\n"
            f"- Belief state remains internal and may contain: {belief_classes}.\n"
            f"- likely_confounders has at most {len(confounder_slots)} entries; "
            "use it only as non-executable scene context.\n"
        )
        if existing_mapping:
            text += (
                f"- Existing confounder slot mapping is FROZEN: {existing_mapping}. "
                "Do not rename/reorder those slots on replanning.\n"
            )
        tried_prompts = list(
            evidence_pack.discovery_diagnostics.get(
                "tried_sam3_prompts", []
            )
            or []
        )
        if tried_prompts:
            text += (
                f"- EXACT PROMPT BLACKLIST: {tried_prompts}. Never propose any "
                "of these SAM3 prompts again, even with a different spatial mode.\n"
            )
        if evidence_pack.discovery_diagnostics.get("discovery_saturated") is True:
            text += (
                "- DISCOVERY IS SATURATED: propose one novel target description "
                "only if it can reduce remaining target uncertainty; otherwise "
                "return an empty proposed_actions list.\n"
            )
        else:
            text += (
                "- DISCOVERY IS NOT SATURATED (false or absent): proposed_actions MUST contain exactly one "
                "novel target DISCOVERY experiment, even when current candidates look convincing. "
                "Do not return an empty list.\n"
            )
        text += (
            "\nReturn JSON:\n"
            "{\n"
            '  "scene_summary": "<string>",\n'
            '  "missing_appearance_modes": ["<string>"],\n'
            '  "likely_confounders": ["<semantic label aligned to confounder slots>"],\n'
            '  "proposed_actions": [\n'
            "    {\n"
            '      "semantic_key": "target",\n'
            '      "sam3_prompt": "<2 or 3 words only>",\n'
            '      "family": "DISCOVERY",\n'
            '      "priority": <float 0.0-1.0>,\n'
            '      "semantic_prior": {"target": 1.0},\n'
            '      "suggested_threshold": <float 0.0-1.0>,\n'
            '      "suggested_spatial_mode": "GLOBAL | TILED",\n'
            '      "rationale": "<short reasoning>"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Output JSON only."
        )

        text_bytes = len((self.SYSTEM_PROMPT + text).encode("utf-8"))
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

        def append_image(path: str) -> None:
            mime = get_mime_type(path)
            with open(path, "rb") as file:
                data = file.read()
            b64 = base64.b64encode(data).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        orig_path = evidence_pack.image_path
        if not orig_path:
            if self.strict_model_errors:
                raise ValueError("Original image is strictly required for M8 real Qwen planning.")
        elif not os.path.exists(orig_path):
            if self.strict_model_errors:
                raise ValueError(f"Original image not found at {orig_path}")
        else:
            append_image(orig_path)

        cs_path = evidence_pack.contact_sheet.contact_sheet_image_path
        if cs_path and os.path.exists(cs_path):
            append_image(cs_path)

        logger.info("Qwen payload: text_bytes=%d, images=%d", text_bytes, len(content) - 1)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        try:
            extra_body = {}
            if config.planner.reasoning_effort is not None:
                # Both current Ollama and vLLM OpenAI-compatible endpoints map
                # reasoning_effort="none" to Qwen's enable_thinking=False.
                extra_body["reasoning_effort"] = config.planner.reasoning_effort
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=config.planner.temperature,
                messages=messages,
                max_tokens=config.planner.max_output_tokens,
                response_format={"type": "json_object"},
                timeout=config.planner.request_timeout_seconds,
                extra_body=extra_body or None,
            )
            return response.choices[0].message.content
        except Exception as exc:
            if self.strict_model_errors:
                raise RuntimeError(f"Strict Qwen execution failed: {exc}") from exc
            raise
