"""SAM3 visual sensor interface protocol and adapter implementations (V4 Design Spec §4)."""

import time
from typing import Any, List, Optional, Protocol, runtime_checkable
from sam3_vlm.core.geometry import Box, BoxGeometry, GeometryRef
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.core.types import Detection, SpatialMode
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.sensing.observation import SAM3Observation
from sam3_vlm.sensing.tiling import compute_tiles, tile_box_to_image_box, image_box_to_tile_box


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
        elif hasattr(image, "size"):  # PIL Image
            img_w, img_h = image.size[0], image.size[1]
        elif hasattr(image, "shape"):  # Numpy array / Tensor
            img_h, img_w = int(image.shape[0]), int(image.shape[1])

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
                            if tile_box.intersection(s_box) > 0.0:
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
