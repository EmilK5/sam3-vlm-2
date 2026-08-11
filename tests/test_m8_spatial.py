import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock

from sam3_vlm.models.sam3 import RealSAM3Sensor, UnsupportedRealSAM3ActionError
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.core.types import ActionFamily, SpatialMode
from sam3_vlm.core.geometry import Box

def fake_run_inference(*args, **kwargs):
    image_pil = args[0] if len(args) > 0 else kwargs.get("image_pil")
    # Returns (boxes, scores, masks)
    # Give back a dummy box proportional to the image to ensure coordinate math works
    w, h = image_pil.size
    # Fake box: 10% from edges
    b = [w * 0.1, h * 0.1, w * 0.9, h * 0.9]
    return np.array([b]), np.array([0.95]), []

@pytest.fixture
def sensor_patched():
    # Construct RealSAM3Sensor but avoid importing transformers or loading weights
    with patch("sam3_vlm.models.sam3.RealSAM3Sensor.__init__", lambda self, **kwargs: None):
        sensor = RealSAM3Sensor()
        sensor.id_gen = MagicMock()
        sensor.id_gen.next_sam3_call_id.return_value = "call_0"
        sensor.id_gen.next_detection_id.return_value = "det_0"
        sensor.model_id = "test_model"
        sensor.call_count = 0
        return sensor

def test_real_sam3_global(sensor_patched):
    img = Image.new("RGB", (1000, 800), color="blue")
    
    action = SensingAction(
        action_id="global_1",
        semantic_key="test",
        prompt="blue",
        family=ActionFamily.DISCOVERY,
        threshold=0.5,
        spatial_mode=SpatialMode.GLOBAL
    )
    
    with patch.object(sensor_patched, "_run_inference", new=fake_run_inference):
        obs = sensor_patched.observe(img, action)
        
    assert obs.model_metadata["spatial_mode"] == "GLOBAL"
    assert len(obs.searched_regions) == 1
    
    reg_box = obs.searched_regions[0].box
    assert reg_box.x1 == 0 and reg_box.y1 == 0 and reg_box.x2 == 1000 and reg_box.y2 == 800
    
    assert len(obs.detections) == 1
    det = obs.detections[0]
    # Box returned by fake inference is 100,80,900,720
    assert det.geometry.box.x1 == 100
    assert det.geometry.box.y1 == 80
    assert det.geometry.box.x2 == 900
    assert det.geometry.box.y2 == 720

def test_real_sam3_tiled(sensor_patched):
    img = Image.new("RGB", (1000, 1000), color="blue")
    
    from sam3_vlm.core.config import TilingConfig
    action = SensingAction(
        action_id="tiled_1",
        semantic_key="test",
        prompt="blue",
        family=ActionFamily.DISCOVERY,
        threshold=0.5,
        spatial_mode=SpatialMode.TILED,
        tiling=TilingConfig(grid_rows=2, grid_cols=2, overlap_ratio=0.0, tile_min_size=100)
    )
    
    # Track calls
    calls = []
    def tracking_inference(*args, **kwargs):
        img_pil = args[0] if len(args) > 0 else kwargs.get("image_pil")
        calls.append(img_pil.size)
        return fake_run_inference(*args, **kwargs)
        
    with patch.object(sensor_patched, "_run_inference", tracking_inference):
        obs = sensor_patched.observe(img, action)
        
    assert len(calls) == 4
    for size in calls:
        assert size == (500, 500)
        
    assert len(obs.searched_regions) == 4
    assert len(obs.detections) == 4
    
    # For tile 0 (x:0-500, y:0-500)
    # fake_inference returns 50,50,450,450 inside tile.
    # Global should match this for tile 0
    det0 = obs.detections[0]
    assert det0.geometry.box.x1 == 50
    assert det0.geometry.box.y1 == 50
    assert det0.geometry.box.x2 == 450
    assert det0.geometry.box.y2 == 450

def test_real_sam3_local(sensor_patched):
    img = Image.new("RGB", (1000, 1000), color="blue")
    roi = Box(x1=200, y1=300, x2=400, y2=600)
    
    action = SensingAction(
        action_id="local_1",
        semantic_key="test",
        prompt="blue",
        family=ActionFamily.CONFOUNDER,
        threshold=0.5,
        spatial_mode=SpatialMode.LOCAL,
        roi=roi
    )
    
    calls = []
    def tracking_inference(*args, **kwargs):
        img_pil = args[0] if len(args) > 0 else kwargs.get("image_pil")
        calls.append(img_pil.size)
        return fake_run_inference(*args, **kwargs)
        
    with patch.object(sensor_patched, "_run_inference", tracking_inference):
        obs = sensor_patched.observe(img, action)
        
    assert len(calls) == 1
    # Crop size is 200x300
    assert calls[0] == (200, 300)
    
    assert len(obs.searched_regions) == 1
    assert obs.searched_regions[0].box.x1 == 200
    assert obs.searched_regions[0].box.y1 == 300
    
    det = obs.detections[0]
    # Local box: 10% of 200x300 -> x1=20, y1=30
    assert det.local_geometry.box.x1 == 20
    assert det.local_geometry.box.y1 == 30
    
    # Global box: x1=200+20=220, y1=300+30=330
    assert det.geometry.box.x1 == 220
    assert det.geometry.box.y1 == 330
    assert det.geometry.box.x2 == 380 # 200 + 180
    assert det.geometry.box.y2 == 570 # 300 + 270

def test_real_sam3_roi_batch(sensor_patched):
    img = Image.new("RGB", (1000, 1000), color="blue")
    # Same code path as LOCAL in our RealSAM3Sensor, but let's test it explicitly
    roi = Box(x1=100, y1=100, x2=500, y2=500)
    
    action = SensingAction(
        action_id="roi_batch_1",
        semantic_key="test",
        prompt="blue",
        family=ActionFamily.VERIFICATION,
        threshold=0.5,
        spatial_mode=SpatialMode.ROI_BATCH,
        roi=roi
    )
    
    with patch.object(sensor_patched, "_run_inference", new=fake_run_inference):
        obs = sensor_patched.observe(img, action)
        
    assert obs.model_metadata["spatial_mode"] == "ROI_BATCH"
    assert len(obs.searched_regions) == 1
    assert obs.searched_regions[0].box.width == 400

def test_real_sam3_exemplar_error(sensor_patched):
    img = Image.new("RGB", (100, 100))
    action = SensingAction(
        action_id="ex_1",
        semantic_key="test",
        prompt="test",
        family=ActionFamily.DISCOVERY,
        threshold=0.5,
        spatial_mode=SpatialMode.GLOBAL,
        positive_exemplar_ids=["node_1"]
    )
    
    with pytest.raises(UnsupportedRealSAM3ActionError):
        sensor_patched.observe(img, action)
