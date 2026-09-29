"""Thin real-model adapters for the MVP controller."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import os
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from .core import Detection, Region


class RealSAM3:
    """One text and one crop per request; masks are returned in image coordinates."""

    def __init__(self, model_id: str = "facebook/sam3", device: str | None = None,
                 sensor: Any | None = None):
        if sensor is None:
            from sam3_vlm.models.sam3 import RealSAM3Sensor
            sensor = RealSAM3Sensor(model_id=model_id, device=device)
        if not callable(getattr(sensor, "_run_inference", None)):
            raise TypeError("SAM3 sensor must provide _run_inference")
        self.sensor = sensor

    def search(self, image: Image.Image, prompt: str, region: Region,
               exemplar_boxes: tuple[Region, ...] = ()) -> list[Detection]:
        x1, y1, x2, y2 = region
        crop = image.crop(region)
        local_boxes = []
        for bx1, by1, bx2, by2 in exemplar_boxes:
            if not (x1 <= bx1 < bx2 <= x2 and y1 <= by1 < by2 <= y2):
                raise ValueError("exemplar box must be fully inside the search region")
            local_boxes.append([bx1 - x1, by1 - y1, bx2 - x1, by2 - y1])
        _, scores, masks = self.sensor._run_inference(crop, prompt, 0.0,
                                                       positive_boxes=local_boxes)
        if len(scores) != len(masks):
            raise ValueError("SAM3 response lacks one mask per score")
        result = []
        for i, (score, raw) in enumerate(zip(scores, masks)):
            local = np.asarray(raw)
            if local.shape != (y2 - y1, x2 - x1):
                local = np.asarray(Image.fromarray((local > 0.5).astype("uint8") * 255).resize(
                    (x2 - x1, y2 - y1), Image.Resampling.NEAREST)) > 0
            else:
                local = local > 0.5
            global_mask = np.zeros((image.height, image.width), dtype=bool)
            global_mask[y1:y2, x1:x2] = local
            result.append(Detection(global_mask, float(score), f"sam3_d{i:03d}"))
        return result


class RealVLM:
    """OpenAI-compatible multimodal Qwen endpoint with SDK retries disabled."""

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, scope: str | None = None,
                 max_output_tokens: int = 512, timeout_seconds: float = 45.0,
                 reasoning_effort: str | None = "none"):
        self.model = model or os.environ.get("QWEN_MODEL")
        endpoint = base_url or os.environ.get("QWEN_BASE_URL")
        if not self.model or not endpoint:
            raise ValueError("QWEN_MODEL and QWEN_BASE_URL are required")
        self.api_key = api_key or os.environ.get("QWEN_API_KEY") or "EMPTY"
        parsed = urlsplit(endpoint)
        # Ollama's native API reliably disables Qwen thinking. Some Ollama
        # versions leave OpenAI-compatible message.content empty instead.
        self.ollama_chat_url = (urlunsplit((parsed.scheme, parsed.netloc, "/api/chat", "", ""))
                                if parsed.port == 11434 and parsed.path.rstrip("/") == "/v1"
                                else None)
        if self.ollama_chat_url is None:
            from openai import OpenAI
            self.client = OpenAI(base_url=endpoint, api_key=self.api_key, max_retries=0)
        else:
            self.client = None
        self.scope = scope.strip() if scope else None
        self.request_log: list[dict[str, str]] = []
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.reasoning_effort = reasoning_effort

    @staticmethod
    def _image_part(image: Image.Image) -> dict[str, Any]:
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=85)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}

    def _complete(self, instructions: str, content: list[dict[str, Any]]) -> Any:
        if getattr(self, "ollama_chat_url", None):
            return self._complete_ollama(instructions, content)
        options = {"model": self.model,
                   "messages": [{"role": "system", "content": instructions},
                                {"role": "user", "content": content}],
                   "temperature": 0,
                   "response_format": {"type": "json_object"},
                   "max_tokens": getattr(self, "max_output_tokens", 512),
                   "timeout": getattr(self, "timeout_seconds", 45.0)}
        reasoning_effort = getattr(self, "reasoning_effort", "none")
        if reasoning_effort is not None:
            options["extra_body"] = {"reasoning_effort": reasoning_effort}
        response = self.client.chat.completions.create(**options)
        choice = response.choices[0]
        answer = choice.message.content
        if not isinstance(answer, str) or not answer.strip():
            usage = getattr(response, "usage", None)
            raise ValueError("Qwen returned empty message.content "
                             f"(finish_reason={getattr(choice, 'finish_reason', None)}, "
                             f"completion_tokens={getattr(usage, 'completion_tokens', None)}); "
                             "check whether the server spent its output budget on thinking")
        return answer

    def _complete_ollama(self, instructions: str, content: list[dict[str, Any]]) -> str:
        images = []
        user_text = []
        for part in content:
            if part["type"] == "text":
                user_text.append(part["text"])
            elif part["type"] == "image_url":
                url = part["image_url"]["url"]
                if not url.startswith("data:image/") or "," not in url:
                    raise ValueError("Ollama requires embedded image data")
                images.append(url.split(",", 1)[1])
            else:
                raise ValueError(f"Unsupported Qwen content part: {part['type']}")
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": instructions},
                         {"role": "user", "content": "\n".join(user_text), "images": images}],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": self.max_output_tokens},
        }
        if self.reasoning_effort is not None:
            payload["think"] = (False if self.reasoning_effort == "none"
                                else self.reasoning_effort)
        request = Request(self.ollama_chat_url, data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json",
                                   "Authorization": f"Bearer {self.api_key}"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.load(response)
        except HTTPError as exc:
            detail = exc.read(500).decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama native Qwen request failed: HTTP {exc.code}: {detail}") from exc
        answer = result.get("message", {}).get("content")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Ollama returned empty message.content "
                             f"(done_reason={result.get('done_reason')}); "
                             "no VLM plan was available")
        return answer

    def propose(self, image: Image.Image, target: str, state: dict[str, Any]) -> Any:
        limit = state["max_actions"]
        roi_guidance = (f"The permitted ROI is {state['roi']}; null means this ROI. "
                        "Every explicit region must stay within it. " if "roi" in state else "")
        scope_guidance = (f"Target scope: {self.scope} Preserve this scope in every search. "
                          if getattr(self, "scope", None) else "")
        instructions = (
            "You plan positive text searches for SAM3 in this fixed image. Preserve the user's target meaning. "
            + scope_guidance + roi_guidance +
            "Propose up to the stated maximum actions to find missed targets or check uncertain candidates. "
            "Return only JSON of the form {\"actions\":[{\"prompt\":\"short target phrase\","
            "\"region\":null}]}. A region may be null for the whole image or [x1,y1,x2,y2] "
            "with exclusive right/bottom image coordinates. Do not count or declare detections. "
            "Do not propose confounder searches."
        )
        compact = {k: v for k, v in state.items() if k != "candidate_boxes"}
        content: list[dict[str, Any]] = [
            {"type": "text", "text": f"Target: {target}. Maximum actions: {limit}. State: {json.dumps(compact)}"},
            self._image_part(image),
        ]
        for box in state["candidate_boxes"]:
            content.append(self._image_part(image.crop(box)))
        if hasattr(self, "request_log"):
            self.request_log.append({"system": instructions, "user_text": content[0]["text"]})
        return self._complete(instructions, content)

    def propose_initial(self, image: Image.Image, target: str, state: dict[str, Any]) -> Any:
        """Ask for a spatial ROI and first search before SAM3 sees the image."""
        instructions = (
            "Plan a search for all visible instances of the target using SAM3. Inspect the whole image. "
            + (f"Target scope: {self.scope} Choose the ROI and searches for only this scope. "
               if getattr(self, "scope", None) else "") +
            "Choose one ROI containing all target-bearing areas, including sparse edge instances; "
            "use null for the full image if unsure. ROI coordinates are integer [x1,y1,x2,y2] "
            "with exclusive right and bottom edges. Do not use exemplars, annotation boxes, or counts. "
            "Select tile_mode force if targets are tiny, crowded, or likely missed by a first crop search; "
            "otherwise select auto so measured detection density decides. "
            "Give at least one positive search prompt faithful to the target. "
            "An action region null means the selected ROI; explicit regions must stay inside it. "
            "Return only JSON {\"roi\":null,\"tile_mode\":\"auto\","
            "\"actions\":[{\"prompt\":\"target noun phrase\",\"region\":null}]}. "
            "Never guess detections or a count."
        )
        content = [{"type": "text", "text": f"Target: {target}. Image size: {image.size}. "
                    f"Maximum actions: {state['max_actions']}."}, self._image_part(image)]
        if hasattr(self, "request_log"):
            self.request_log.append({"system": instructions, "user_text": content[0]["text"]})
        return self._complete(instructions, content)
