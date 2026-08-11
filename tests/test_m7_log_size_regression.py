import pytest
import os
import tempfile
import json
from pathlib import Path

from sam3_vlm.pipeline.runner import Runner
from sam3_vlm.logging.writer import RunRecorder
from sam3_vlm.logging.artifacts import RunArtifactPaths
from sam3_vlm.logging.schema import RunManifest
from sam3_vlm.core.config import V4Config
from sam3_vlm.models.sam3 import MockSAM3Adapter
from sam3_vlm.models.qwen import MockQwenPlanner

def test_log_size_and_mask_externalization():
    config = V4Config()
    
    import numpy as np
    from sam3_vlm.core.types import Detection
    from sam3_vlm.core.geometry import GeometryRef, Box
    
    dense_mask = np.ones((1024, 1024), dtype=bool)
    synth_det = Detection(
        detection_id="det_with_mask",
        geometry=GeometryRef(box=Box(10.0, 10.0, 60.0, 60.0)),
        score=0.9,
        raw_metadata={"mask": dense_mask}
    )
    
    sensor = MockSAM3Adapter(synthetic_detections=[synth_det])
    planner = MockQwenPlanner()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir) / "run_artifacts"
        paths = RunArtifactPaths(base_dir=base_dir)
        manifest = RunManifest(run_id="test_size", image_id="test_image.jpg", target_class="apple")
        recorder = RunRecorder(paths, manifest)
        
        runner = Runner(config, sensor, planner, recorder=recorder)
        runner.run(image="mock", image_id="test_image.jpg", user_prompt="Find apples", target_class="apple")
        
        # Check events size
        event_count = sum(1 for _ in open(paths.events_jsonl))
        assert event_count > 0
        assert event_count < 1000 # ensure it's not dumping whole state per step
        
        # Ensure no giant nested arrays in events.jsonl
        # The line length should be reasonably small
        with open(paths.events_jsonl, "r") as f:
            for line in f:
                assert len(line) < 50000, "Event line too large, likely contains raw data!"
                
        # Masks should be externalized
        masks_dir = paths.masks_dir
        assert masks_dir.exists(), "Masks directory was not created!"
        npz_files = list(masks_dir.glob("*.npz"))
        assert len(npz_files) > 0, "No mask npz files found!"
        for npz_file in npz_files:
            assert npz_file.stat().st_size > 0
            with np.load(npz_file) as data:
                assert data["mask"].shape == (1024, 1024)
