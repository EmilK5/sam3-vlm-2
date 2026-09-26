"""Thin real-model adapters for the MVP controller."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import os
from typing import Any

import numpy as np
from PIL import Image

from .core import Detection, Region


class RealSAM3:
    """One text and one crop per request; masks are returned in image coordinates."""

    def __init__(self, model_id: str = "facebook/sam3", device: str | None = None):
        from sam3_vlm.models.sam3 import RealSAM3Sensor

        self.sensor = RealSAM3Sensor(model_id=model_id, device=device)

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
                 api_key: str | None = None):
        from openai import OpenAI

        self.model = model or os.environ.get("QWEN_MODEL")
        endpoint = base_url or os.environ.get("QWEN_BASE_URL")
        if not self.model or not endpoint:
            raise ValueError("QWEN_MODEL and QWEN_BASE_URL are required")
        self.client = OpenAI(base_url=endpoint, api_key=api_key or os.environ.get("QWEN_API_KEY") or "EMPTY",
                             max_retries=0)

    @staticmethod
    def _image_part(image: Image.Image) -> dict[str, Any]:
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=85)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}

    def propose(self, image: Image.Image, target: str, state: dict[str, Any]) -> Any:
        limit = state["max_actions"]
        roi_guidance = (f"The permitted ROI is {state['roi']}; null means this ROI. "
                        "Every explicit region must stay within it. " if "roi" in state else "")
        instructions = (
            "You plan positive text searches for SAM3 in this fixed image. Preserve the user's target meaning. "
            + roi_guidance +
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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": instructions},
                      {"role": "user", "content": content}],
            temperature=0,
        )
        return response.choices[0].message.content

    def propose_initial(self, image: Image.Image, target: str, state: dict[str, Any]) -> Any:
        """Ask for a spatial ROI and first search before SAM3 sees the image."""
        instructions = (
            "Plan a search for all visible instances of the target using SAM3. Inspect the whole image. "
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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": instructions},
                      {"role": "user", "content": content}],
            temperature=0,
        )
        return response.choices[0].message.content
