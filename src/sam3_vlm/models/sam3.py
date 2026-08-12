"""SAM3 visual sensor interface protocol and adapter implementations (V4 Design Spec §4)."""

import time
from typing import Any, List, Optional, Protocol, runtime_checkable
from sam3_vlm.core.geometry import Box, BoxGeometry, GeometryRef
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.core.types import Detection, SpatialMode
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.sensing.observation import SAM3Observation
from sam3_vlm.sensing.tiling import compute_tiles, tile_box_to_image_box, image_box_to_tile_box
import logging
import numpy as np
try:
    from PIL import Image
    import torch
    from transformers import Sam3Model, Sam3Processor
except ImportError:
    pass

logger = logging.getLogger(__name__)

class UnsupportedRealSAM3ActionError(RuntimeError):
    """Raised when an action requests a capability explicitly not supported by the RealSAM3 adapter."""
    pass

@runtime_checkable
class SAM3Sensor(Protocol):
    """Clean SAM3 visual sensor interface contract (V4 Design Spec §4)."""

    def observe(self, image: Any, action: SensingAction) -> SAM3Observation:
        """Execute sensing action on image and return raw sensor observations."""
        ...


class DummySAM3Sensor:
    """Basic mock SAM3 sensor for testing and foundation verification."""

    def __init__(self, call_id_prefix: str = "sam3") -> None:
        self.call_count = 0
        self.call_id_prefix = call_id_prefix

    def observe(self, image: Any, action: SensingAction) -> SAM3Observation:
        action.validate()
        self.call_count += 1
        return SAM3Observation(
            call_id=f"{self.call_id_prefix}_{self.call_count:06d}",
            action_id=action.action_id,
            semantic_key=action.semantic_key,
            detections=[],
            searched_regions=[],
            runtime_ms=10.0,
            model_metadata={"mock": True},
        )


class MockSAM3Adapter:
    """Configurable mock SAM3 adapter producing realistic synthetic detections (V4 Design Spec §4 / §18.3)."""

    def __init__(
        self,
        id_gen: Optional[IDGenerator] = None,
        synthetic_detections: Optional[List[Detection]] = None,
    ) -> None:
        self.id_gen = id_gen or IDGenerator()
        self.call_count = 0
        self.synthetic_detections = synthetic_detections

    def observe(self, image: Any, action: SensingAction) -> SAM3Observation:
        """Execute sensing action, handling global or tiled mode, and translate coordinates to original image space."""
        action.validate()
        start_time = time.perf_counter()
        self.call_count += 1
        call_id = self.id_gen.next_sam3_call_id()

        # Image dimensions (mock fallback if image is None or tuple)
        img_w, img_h = 1000, 1000
        if isinstance(image, (tuple, list)) and len(image) == 2:
            img_w, img_h = int(image[0]), int(image[1])
        elif hasattr(image, "shape") and len(image.shape) >= 2:  # Numpy
            img_h, img_w = int(image.shape[0]), int(image.shape[1])
        elif hasattr(image, "size") and isinstance(image.size, tuple):  # PIL Image
            img_w, img_h = image.size[0], image.size[1]

        detections: List[Detection] = []
        searched_regions = []

        if action.spatial_mode == SpatialMode.TILED:
            tiling_cfg = action.tiling
            if tiling_cfg:
                tile_geoms = compute_tiles(img_w, img_h, tiling_cfg)
                searched_regions = tile_geoms
                for tile_idx, tile_geom in enumerate(tile_geoms):
                    tile_box = tile_geom.box
                    tile_id = f"tile_{tile_idx:02d}"

                    # If synthetic detections provided, map into tile if contained
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
                                    )
                                )
                    else:
                        # Default synthetic detection inside tile
                        det_id = self.id_gen.next_detection_id()
                        local_x1 = 10.0
                        local_y1 = 10.0
                        local_x2 = min(100.0, tile_box.width)
                        local_y2 = min(100.0, tile_box.height)
                        local_box = Box(local_x1, local_y1, local_x2, local_y2, coordinate_space="tile")
                        global_box = tile_box_to_image_box(local_box, tile_box)

                        detections.append(
                            Detection(
                                detection_id=det_id,
                                geometry=GeometryRef(box=global_box),
                                score=0.85,
                                source_tile_id=tile_id,
                                local_geometry=GeometryRef(box=local_box),
                            )
                        )
        else:
            # Global sensing mode
            global_region = Box(0.0, 0.0, float(img_w), float(img_h))
            searched_regions = [BoxGeometry(global_region)]

            if self.synthetic_detections:
                for s_det in self.synthetic_detections:
                    if s_det.score >= action.threshold:
                        det_id = self.id_gen.next_detection_id()
                        detections.append(
                            Detection(
                                detection_id=det_id,
                                geometry=s_det.geometry,
                                score=s_det.score,
                                raw_metadata=s_det.raw_metadata,
                            )
                        )
            else:
                # Default synthetic detection
                det_id = self.id_gen.next_detection_id()
                detections.append(
                    Detection(
                        detection_id=det_id,
                        geometry=GeometryRef(box=Box(10.0, 10.0, 50.0, 50.0)),
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
            model_metadata={"mock": True, "spatial_mode": action.spatial_mode.value},
        )


class RealSAM3Sensor:
    """Real SAM3 visual sensor interface (V4 Design Spec §4)."""

    def __init__(
        self,
        model_id: str = "facebook/sam3",
        device: str | None = None,
        id_gen: Optional[IDGenerator] = None,
        compile_model: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import Sam3Model, Sam3Processor
        except ImportError as e:
            raise RuntimeError(
                "transformers and torch are required for RealSAM3Sensor"
            ) from e

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.id_gen = id_gen or IDGenerator()
        self.call_count = 0
        self.device = torch.device(device)
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

    def _run_inference(self, image_pil: Image.Image, text_prompt: str, threshold: float) -> tuple[np.ndarray, np.ndarray, list]:
        """Runs a single forward pass."""
        inputs = self.processor(
            images=image_pil,
            text=text_prompt,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=float(threshold),
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        raw_boxes = results["boxes"].cpu().numpy() if hasattr(results["boxes"], "cpu") else np.array(results["boxes"])
        raw_scores = results["scores"].cpu().numpy() if hasattr(results["scores"], "cpu") else np.array(results["scores"])
        
        raw_masks = []
        if "masks" in results and results["masks"] is not None:
            rm = results["masks"]
            rm = rm.cpu().numpy() if hasattr(rm, "cpu") else np.array(rm)
            raw_masks = [np.asarray(rm[k]) > 0.5 for k in range(len(rm))]

        return raw_boxes, raw_scores, raw_masks

    def observe(self, image: Any, action: SensingAction) -> SAM3Observation:
        action.validate()
        start_time = time.perf_counter()
        self.call_count += 1
        call_id = self.id_gen.next_sam3_call_id()

        if isinstance(image, Image.Image):
            img_pil = image
        elif hasattr(image, "shape") and len(image.shape) >= 2:
            img_pil = Image.fromarray(image)
        else:
            raise ValueError(f"RealSAM3Sensor requires PIL Image or Numpy array, got {type(image)}")

        img_w, img_h = img_pil.size
        detections: List[Detection] = []
        searched_regions: List[BoxGeometry] = []

        if action.positive_exemplar_ids or action.negative_exemplar_ids:
            raise UnsupportedRealSAM3ActionError("Exemplars are explicitly unsupported in RealSAM3Sensor.")

        if action.spatial_mode == SpatialMode.TILED and action.tiling:
            tile_geoms = compute_tiles(img_w, img_h, action.tiling)
            searched_regions = tile_geoms
            for tile_idx, tile_geom in enumerate(tile_geoms):
                tile_box = tile_geom.box
                tile_id = f"tile_{tile_idx:02d}"
                
                # Crop tile
                crop_box = (int(tile_box.x1), int(tile_box.y1), int(tile_box.x2), int(tile_box.y2))
                tile_pil = img_pil.crop(crop_box)
                
                boxes, scores, masks = self._run_inference(tile_pil, action.prompt, action.threshold)
                
                for i in range(len(boxes)):
                    b = boxes[i]
                    s = float(scores[i])
                    det_id = self.id_gen.next_detection_id()
                    local_box = Box(float(b[0]), float(b[1]), float(b[2]), float(b[3]), coordinate_space="tile")
                    global_box = tile_box_to_image_box(local_box, tile_box)
                    
                    raw_meta = {}
                    if masks and i < len(masks):
                        raw_meta["mask"] = masks[i]
                        
                    detections.append(Detection(
                        detection_id=det_id,
                        geometry=GeometryRef(box=global_box),
                        score=s,
                        source_tile_id=tile_id,
                        local_geometry=GeometryRef(box=local_box),
                        raw_metadata=raw_meta
                    ))
        elif action.spatial_mode in (SpatialMode.LOCAL, SpatialMode.ROI_BATCH):
            if not action.roi:
                raise ValueError(f"Action requires ROI for spatial_mode={action.spatial_mode.name}")
                
            roi_box = action.roi
            if hasattr(roi_box, "bbox"):
                roi_box = roi_box.bbox()
            elif hasattr(roi_box, "x1"):
                pass
            else:
                # If it's a BoxGeometry
                if hasattr(roi_box, "box"):
                    roi_box = roi_box.box
                    
            crop_box = (int(roi_box.x1), int(roi_box.y1), int(roi_box.x2), int(roi_box.y2))
            
            # Ensure crop box is within bounds
            crop_box = (
                max(0, crop_box[0]), max(0, crop_box[1]),
                min(img_w, crop_box[2]), min(img_h, crop_box[3])
            )
            
            if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                raise ValueError(f"Invalid ROI crop box {crop_box} for image size {img_w}x{img_h}")
                
            tile_pil = img_pil.crop(crop_box)
            boxes, scores, masks = self._run_inference(tile_pil, action.prompt, action.threshold)
            
            searched_regions = [BoxGeometry(Box(float(crop_box[0]), float(crop_box[1]), float(crop_box[2]), float(crop_box[3])))]
            
            for i in range(len(boxes)):
                b = boxes[i]
                s = float(scores[i])
                det_id = self.id_gen.next_detection_id()
                
                local_box = Box(float(b[0]), float(b[1]), float(b[2]), float(b[3]), coordinate_space="local")
                global_box = Box(
                    x1=local_box.x1 + float(crop_box[0]),
                    y1=local_box.y1 + float(crop_box[1]),
                    x2=local_box.x2 + float(crop_box[0]),
                    y2=local_box.y2 + float(crop_box[1])
                )
                
                raw_meta = {}
                if masks and i < len(masks):
                    raw_meta["mask"] = masks[i]
                    raw_meta["mask_offset_x"] = float(crop_box[0])
                    raw_meta["mask_offset_y"] = float(crop_box[1])
                    
                detections.append(Detection(
                    detection_id=det_id,
                    geometry=GeometryRef(box=global_box),
                    score=s,
                    local_geometry=GeometryRef(box=local_box),
                    raw_metadata=raw_meta
                ))
        else:
            # Global sensing mode
            global_region = Box(0.0, 0.0, float(img_w), float(img_h))
            searched_regions = [BoxGeometry(global_region)]
            
            boxes, scores, masks = self._run_inference(img_pil, action.prompt, action.threshold)
            
            for i in range(len(boxes)):
                b = boxes[i]
                s = float(scores[i])
                det_id = self.id_gen.next_detection_id()
                global_box = Box(float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                
                raw_meta = {}
                if masks and i < len(masks):
                    raw_meta["mask"] = masks[i]
                    
                detections.append(Detection(
                    detection_id=det_id,
                    geometry=GeometryRef(box=global_box),
                    score=s,
                    raw_metadata=raw_meta
                ))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return SAM3Observation(
            call_id=call_id,
            action_id=action.action_id,
            semantic_key=action.semantic_key,
            detections=detections,
            searched_regions=searched_regions,
            runtime_ms=max(1.0, elapsed_ms),
            model_metadata={"real": True, "spatial_mode": action.spatial_mode.value, "model_id": self.model_id},
        )

