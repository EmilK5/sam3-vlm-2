import pytest
from unittest.mock import patch
import os
import json
from PIL import Image

from sam3_vlm.experiments.m8_smoke import (
    m8_0_validate_adapters,
    m8_1_sam3_smoke,
    m8_2_qwen_smoke,
    m8_3_full_run,
    m8_4_and_5_pilot,
    preflight
)
from sam3_vlm.models.sam3 import MockSAM3Adapter
from sam3_vlm.models.qwen import MockQwenPlanner

@pytest.fixture
def mock_models():
    sam3 = MockSAM3Adapter()
    qwen = MockQwenPlanner()
    with patch("sam3_vlm.experiments.m8_smoke._get_models", return_value=(sam3, qwen)), \
         patch("sam3_vlm.experiments.m8_smoke._get_sam3_only", return_value=sam3), \
         patch("sam3_vlm.experiments.m8_smoke._get_qwen_only", return_value=qwen):
        yield sam3, qwen

class DummyArgs:
    def __init__(self, **kwargs):
        self.require_cuda = False
        self.compile_sam3 = False
        self.dry_run = False
        self.sam3_model = "fake"
        self.qwen_model = "fake"
        self.qwen_base_url = "fake"
        self.image = "fake.jpg"
        self.target = "green citrus"
        self.output_dir = "fake_out"
        self.manifest = None
        self.max_samples = 2
        self.stage = "all"
        for k, v in kwargs.items():
            setattr(self, k, v)

def test_m8_0_validate(mock_models):
    assert m8_0_validate_adapters(DummyArgs()) is True

def test_m8_1_smoke(mock_models, tmp_path):
    img = Image.new("RGB", (64, 64))
    p = str(tmp_path / "test.jpg")
    img.save(p)
    assert m8_1_sam3_smoke(DummyArgs(image=p)) is True

def test_m8_2_smoke(mock_models):
    assert m8_2_qwen_smoke(DummyArgs()) is True

def test_m8_3_full_run_with_mocks(mock_models, tmp_path):
    img = Image.new("RGB", (64, 64))
    p = str(tmp_path / "test.jpg")
    img.save(p)
    args = DummyArgs(image=p, output_dir=str(tmp_path / "runs"))
    
    assert m8_3_full_run(args) is True

def test_m8_4_and_5_pilot_with_mocks(mock_models, tmp_path):
    img = Image.new("RGB", (64, 64))
    p1 = str(tmp_path / "img1.jpg")
    img.save(p1)
    p2 = str(tmp_path / "img2.jpg")
    img.save(p2)
    
    manifest_path = str(tmp_path / "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump([
            {"sample_id": "i1", "image_path": p1, "target": "target", "gt_count": 5},
            {"sample_id": "i2", "image_path": p2, "target": "target", "gt_count": 10},
        ], f)
        
    args = DummyArgs(manifest=manifest_path, output_dir=str(tmp_path / "runs"))
    assert m8_4_and_5_pilot(args) is True
    
    report_p = str(tmp_path / "runs" / "pilot_report.json")
    assert os.path.exists(report_p)
    
    with open(report_p) as f:
        report = json.load(f)
        
    assert len(report) == 6 # 3 variants * 2 images
    
    for r in report:
        assert r["status"] == "SUCCESS"
        assert "predicted_count" in r
        assert "absolute_error" in r
