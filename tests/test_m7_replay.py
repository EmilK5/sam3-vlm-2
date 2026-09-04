import pytest
import tempfile
from pathlib import Path

from sam3_vlm.pipeline.runner import Runner
from sam3_vlm.logging.writer import RunRecorder
from sam3_vlm.logging.artifacts import RunArtifactPaths
from sam3_vlm.logging.schema import RunManifest
from sam3_vlm.core.config import BeliefConfig, V4Config
from sam3_vlm.models.sam3 import MockSAM3Adapter
from sam3_vlm.models.qwen import MockQwenPlanner

from sam3_vlm.logging.replay import ReplayEngine

def test_replay_engine():
    config = V4Config(
        budget=__import__('sam3_vlm').core.config.BudgetConfig(max_sam3_calls=5),
        belief=BeliefConfig(target_count_commit_threshold=0.9),
    )
    
    # We want to use MockSAM3Adapter to yield the same synthetic detection
    # during both global and tiled phases, forcing the 'match existing node' path.
    import numpy as np
    from sam3_vlm.core.types import Detection
    from sam3_vlm.core.geometry import GeometryRef, Box
    
    synth_det = Detection(
        detection_id="det_shared",
        geometry=GeometryRef(box=Box(10.0, 10.0, 60.0, 60.0)),
        score=0.95,
        raw_metadata={"mask": np.ones((10, 10), dtype=bool)}
    )
    
    sensor = MockSAM3Adapter(synthetic_detections=[synth_det])
    planner = MockQwenPlanner()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir) / "run_artifacts"
        paths = RunArtifactPaths(base_dir=base_dir)
        manifest = RunManifest(run_id="test_replay", image_id="test_image.jpg", target_class="apple")
        recorder = RunRecorder(paths, manifest)
        
        runner = Runner(config, sensor, planner, recorder=recorder)
        runner.run(image="mock", image_id="test_image.jpg", user_prompt="Find apples", target_class="apple")
        
        orig_state = runner.scene_state
        
        # Now replay
        engine = ReplayEngine(paths)
        replayed_state = engine.replay_state()
        
        # We use canonical_scene_state instead of custom asserts
        from sam3_vlm.logging.replay import canonical_scene_state
        
        c_orig = canonical_scene_state(orig_state)
        c_repl = canonical_scene_state(replayed_state)
        
        assert c_orig == c_repl
