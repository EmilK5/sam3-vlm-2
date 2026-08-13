"""SAM3 visual sensor interface protocol and adapter implementations (V4 Design Spec §4)."""

import logging
import time
from typing import Any, List, Optional, Protocol, runtime_checkable
import numpy as np

from sam3_vlm.core.geometry import Box, BoxGeometry, GeometryRef
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.core.types import Detection, SpatialMode
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.sensing.observation import SAM3Observation
from sam3_vlm.sensing.tiling import compute_tiles, tile_box_to_image_box, image_box_to_tile_box

try:
    from PIL import Image
except ImportError:  # pragma: no cover - exercised only in minimal installs
    Image = None

logger = logging.getLogger(__name__)


class UnsupportedRealSAM3ActionError(RuntimeError):
    pass


@runtime_checkable
class SAM3Sensor(Protocol):
    def observe(self, image: Any, action: SensingAction) -> SAM3Observation:
        ...


def _image_dimensions(image: Any) -> tuple[int, int]:
    if isinstance(image, (tuple, list)) and len(image) == 2:
        return int(image[0]), int(image[1])
    if hasattr(image, "shape") and len(image.shape) >= 2:
        return int(image.shape[1]), int(image.shape[0])
    if hasattr(image, "size") and isinstance(image.size, tuple):
        return int(image.size[0]), int(image.size[1])
    return 1000, 1000


def _roi_box(roi: Any) -> Optional[Box]:
    if roi is None:
        return None
    if hasattr(roi, "bbox"):
        return roi.bbox()
    if hasattr(roi, "box"):
        return roi.box
    if all(hasattr(roi, attr) for attr in ("x1", "y1", "x2", "y2")):
        return roi
    raise ValueError(f"Unsupported ROI geometry: {type(roi)}")


def _clipped_domain(action: SensingAction, img_w: int, img_h: int) -> Box:
    roi = _roi_box(action.roi)
    if roi is None:
        return Box(0.0, 0.0, float(img_w), float(img_h))
    domain = Box(
        max(0.0, float(roi.x1)),
        max(0.0, float(roi.y1)),
        min(float(img_w), float(roi.x2)),
        min(float(img_h), float(roi.y2)),
    )
    if domain.area <= 0.0:
        raise ValueError(f"Invalid search ROI {roi.as_tuple()} for image size {img_w}x{img_h}")
    return domain


def _tiles_within_domain(domain: Box, tiling) -> List[BoxGeometry]:
    local_tiles = compute_tiles(max(1, int(round(domain.width))), max(1, int(round(domain.height))), tiling)
    return [
        BoxGeometry(
            Box(
                x1=domain.x1 + tile.box.x1,
                y1=domain.y1 + tile.box.y1,
                x2=min(domain.x2, domain.x1 + tile.box.x2),
                y2=min(domain.y2, domain.y1 + tile.box.y2),
            )
        )
        for tile in local_tiles
    ]


class DummySAM3Sensor:
    def __init__(self, call_id_prefix: str = "sam3") -> None:
        self.call_count = 0
        self.call_id_prefix = call_id_prefix
        self.model_id = "dummy-sam3"

    def observe(self, image: Any, action: SensingAction) -> SAM3Observation:
        action.validate()
        self.call_count += 1
        img_w, img_h = _image_dimensions(image)
        domain = _clipped_domain(action, img_w, img_h)
        if action.spatial_mode == SpatialMode.TILED:
            searched = _tiles_within_domain(domain, action.tiling)
        else:
            searched = [BoxGeometry(domain)]
        return SAM3Observation(
            call_id=f"{self.call_id_prefix}_{self.call_count:06d}",
            action_id=action.action_id,
            semantic_key=action.semantic_key,
            detections=[],
            searched_regions=searched,
            runtime_ms=10.0,
            model_metadata={"mock": True, "model_id": self.model_id},
        )


class MockSAM3Adapter:
    """Synthetic adapter with the same locked-ROI spatial semantics as real SAM3."""

    def __init__(
        self,
        id_gen: Optional[IDGenerator] = None,
        synthetic_detections: Optional[List[Detection]] = None,
    ) -> None:
        self.id_gen = id_gen or IDGenerator()
        self.call_count = 0
        self.synthetic_detections = synthetic_detections
        self.model_id = "mock-sam3"

    def observe(self, image: Any, action: SensingAction) -> SAM3Observation:
        action.validate()
        start_time = time.perf_counter()
        self.call_count += 1
        call_id = self.id_gen.next_sam3_call_id()
        img_w, img_h = _image_dimensions(image)
        domain = _clipped_domain(action, img_w, img_h)
        detections: List[Detection] = []

        if action.spatial_mode == SpatialMode.TILED:
            tile_geoms = _tiles_within_domain(domain, action.tiling)
            searched_regions = tile_geoms
            for tile_idx, tile_geom in enumerate(tile_geoms):
                tile_box = tile_geom.box
                tile_id = f"tile_{tile_idx:02d}"
                if self.synthetic_detections:
                    for s_det in self.synthetic_detections:
                        s_box = s_det.geometry.box
                        if s_det.score >= action.threshold and tile_box.intersection(s_box) > 0.0:
                            det_id = self.id_gen.next_detection_id()
                            local_box = image_box_to_tile_box(s_box, tile_box)
                            detections.append(
                                Detection(
                                    detection_id=det_id,
                                    geometry=s_det.geometry,
                                    score=s_det.score,
                                    source_tile_id=tile_id,
                                    local_geometry=GeometryRef(box=local_box),
                                    raw_metadata=s_det.raw_metadata,
                                )
                            )
                else:
                    local_box = Box(
                        10.0,
                        10.0,
                        min(100.0, tile_box.width),
                        min(100.0, tile_box.height),
                        coordinate_space="tile",
                    )
                    detections.append(
                        Detection(
                            detection_id=self.id_gen.next_detection_id(),
                            geometry=GeometryRef(box=tile_box_to_image_box(local_box, tile_box)),
                            score=0.85,
                            source_tile_id=tile_id,
                            local_geometry=GeometryRef(box=local_box),
                        )
                    )
        else:
            searched_regions = [BoxGeometry(domain)]
            if self.synthetic_detections:
                for s_det in self.synthetic_detections:
                    if s_det.score >= action.threshold and domain.intersection(s_det.geometry.box) > 0.0:
                        detections.append(
                            Detection(
                                detection_id=self.id_gen.next_detection_id(),
                                geometry=s_det.geometry,
                                score=s_det.score,
                                raw_metadata=s_det.raw_metadata,
                            )
                        )
            else:
                detections.append(
                    Detection(
                        detection_id=self.id_gen.next_detection_id(),
                        geometry=GeometryRef(
                            box=Box(
                                domain.x1 + 10.0,
                                domain.y1 + 10.0,
                                min(domain.x2, domain.x1 + 50.0),
                                min(domain.y2, domain.y1 + 50.0),
                            )
                        ),
                        score=0.88,
                    )
                )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return SAM3Observation(
            call_id=call_id,
            action_id=action.action_id,
            semantic_key=action.semantic_key,
            detections=detections,
            searched_regions=searched_regions,
            runtime_ms=max(1.0, elapsed_ms),
            model_metadata={
                "mock": True,
                "spatial_mode": action.spatial_mode.value,
                "model_id": self.model_id,
                "search_domain": domain.as_tuple(),
            },
        )


class RealSAM3Sensor:
    """Real SAM3 adapter supporting GLOBAL/TILED execution inside an optional ROI."""

    def __init__(
        self,
        model_id: str = "facebook/sam3",
        device: Optional[str] = None,
        id_gen: Optional[IDGenerator] = None,
        compile_model: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import Sam3Model, Sam3Processor
        except ImportError as exc:
            raise RuntimeError("transformers and torch are required for RealSAM3Sensor") from exc

        self._torch = torch
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.id_gen = id_gen or IDGenerator()
        self.call_count = 0
        self.device = torch.device(resolved_device)
        self.model_id = model_id
        self.compile_model = compile_model

        logger.info(f"Loading real SAM3 model: {model_id} on {self.device}")
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model = Sam3Model.from_pretrained(model_id).to(self.device)
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.processor.model = self.model
        if self.device.type == "cuda" and self.compile_model:
            logger.info("Compiling SAM3 model graph...")
            self.model = torch.compile(self.model)

    def _run_inference(self, image_pil: Any, text_prompt: str, threshold: float) -> tuple[np.ndarray, np.ndarray, list]:
        torch = self._torch
        inputs = self.processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=float(threshold),
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
        raw_boxes = results["boxes"].cpu().numpy() if hasattr(results["boxes"], "cpu") else np.array(results["boxes"])
        raw_scores = results["scores"].cpu().numpy() if hasattr(results["scores"], "cpu") else np.array(results["scores"])
        raw_masks = []
        if "masks" in results and results["masks"] is not None:
            rm = results["masks"]
            rm = rm.cpu().numpy() if hasattr(rm, "cpu") else np.array(rm)
            raw_masks = [np.asarray(rm[k]) > 0.5 for k in range(len(rm))]
        return raw_boxes, raw_scores, raw_masks

    def _append_crop_detections(
        self,
        detections: List[Detection],
        boxes: np.ndarray,
        scores: np.ndarray,
        masks: list,
        crop_region: Box,
        coordinate_space: str,
        source_tile_id: Optional[str] = None,
    ) -> None:
        for i in range(len(boxes)):
            b = boxes[i]
            local_box = Box(
                float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                coordinate_space=coordinate_space,
            )
            global_box = Box(
                crop_region.x1 + local_box.x1,
                crop_region.y1 + local_box.y1,
                crop_region.x1 + local_box.x2,
                crop_region.y1 + local_box.y2,
            )
            raw_meta = {}
            if masks and i < len(masks):
                raw_meta["mask"] = masks[i]
                raw_meta["mask_offset_x"] = crop_region.x1
                raw_meta["mask_offset_y"] = crop_region.y1
            detections.append(
                Detection(
                    detection_id=self.id_gen.next_detection_id(),
                    geometry=GeometryRef(box=global_box),
                    score=float(scores[i]),
                    source_tile_id=source_tile_id,
                    local_geometry=GeometryRef(box=local_box),
                    raw_metadata=raw_meta,
                )
            )

    def observe(self, image: Any, action: SensingAction) -> SAM3Observation:
        action.validate()
        start_time = time.perf_counter()
        self.call_count += 1
        call_id = self.id_gen.next_sam3_call_id()

        if Image is not None and isinstance(image, Image.Image):
            img_pil = image
        elif hasattr(image, "shape") and len(image.shape) >= 2 and Image is not None:
            img_pil = Image.fromarray(image)
        else:
            raise ValueError(f"RealSAM3Sensor requires PIL Image or Numpy array, got {type(image)}")

        img_w, img_h = img_pil.size
        domain = _clipped_domain(action, img_w, img_h)
        detections: List[Detection] = []

        if action.positive_exemplar_ids or action.negative_exemplar_ids:
            raise UnsupportedRealSAM3ActionError("Exemplars are explicitly unsupported in RealSAM3Sensor.")

        if action.spatial_mode == SpatialMode.TILED:
            tile_geoms = _tiles_within_domain(domain, action.tiling)
            searched_regions = tile_geoms
            for tile_idx, tile_geom in enumerate(tile_geoms):
                tile_box = tile_geom.box
                crop = img_pil.crop((int(tile_box.x1), int(tile_box.y1), int(tile_box.x2), int(tile_box.y2)))
                boxes, scores, masks = self._run_inference(crop, action.prompt, action.threshold)
                self._append_crop_detections(
                    detections,
                    boxes,
                    scores,
                    masks,
                    tile_box,
                    coordinate_space="tile",
                    source_tile_id=f"tile_{tile_idx:02d}",
                )
        else:
            # GLOBAL with roi means the whole locked search domain; LOCAL and
            # ROI_BATCH use the same crop mechanics with a controller-owned ROI.
            searched_regions = [BoxGeometry(domain)]
            crop = img_pil.crop((int(domain.x1), int(domain.y1), int(domain.x2), int(domain.y2)))
            boxes, scores, masks = self._run_inference(crop, action.prompt, action.threshold)
            coordinate_space = "image" if domain.as_tuple() == (0.0, 0.0, float(img_w), float(img_h)) else "local"
            self._append_crop_detections(
                detections,
                boxes,
                scores,
                masks,
                domain,
                coordinate_space=coordinate_space,
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return SAM3Observation(
            call_id=call_id,
            action_id=action.action_id,
            semantic_key=action.semantic_key,
            detections=detections,
            searched_regions=searched_regions,
            runtime_ms=max(1.0, elapsed_ms),
            model_metadata={
                "real": True,
                "spatial_mode": action.spatial_mode.value,
                "model_id": self.model_id,
                "search_domain": domain.as_tuple(),
            },
        )
