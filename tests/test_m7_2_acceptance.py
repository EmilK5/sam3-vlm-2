import pytest
import os
import tempfile
import json
from pathlib import Path
import numpy as np

from sam3_vlm.pipeline.runner import Runner
from sam3_vlm.logging.writer import RunRecorder
from sam3_vlm.logging.artifacts import RunArtifactPaths
from sam3_vlm.logging.schema import RunManifest
from sam3_vlm.core.config import V4Config
from sam3_vlm.models.sam3 import MockSAM3Adapter
from sam3_vlm.models.qwen import MockQwenPlanner
from sam3_vlm.logging.validator import RunValidator
from sam3_vlm.core.types import Detection
from sam3_vlm.core.geometry import GeometryRef, Box

def test_m7_2_acceptance():
    """Verify that a full mock run yields a 100% valid event stream according to RunValidator,
    which now implies referential integrity and replay equivalence."""
    config = V4Config()
    
    dense_mask = np.ones((10, 10), dtype=bool)
    synth_det = Detection(
        detection_id="det_m7_2",
        geometry=GeometryRef(box=Box(10.0, 10.0, 60.0, 60.0)),
        score=0.95,
        raw_metadata={"mask": dense_mask}
    )
    
    sensor = MockSAM3Adapter(synthetic_detections=[synth_det])
    planner = MockQwenPlanner()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir) / "run_artifacts"
        paths = RunArtifactPaths(base_dir=base_dir)
        manifest = RunManifest(run_id="test_m7_2_acceptance", image_id="test.jpg", target_class="target")
        recorder = RunRecorder(paths, manifest)
        
        runner = Runner(config, sensor, planner, recorder=recorder)
        runner.run(image="mock", image_id="test.jpg", user_prompt="Find target", target_class="target")
        
        validator = RunValidator(paths)
        result = validator.validate()
        
        if not result.valid:
            print("Validation Errors:", result.errors)
            with open(paths.events_jsonl) as f:
                print("Events:")
                print(f.read())
                
        assert result.valid, f"Run validation failed: {result.errors}"
