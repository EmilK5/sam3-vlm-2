import pytest
import tempfile
from pathlib import Path

from sam3_vlm.pipeline.runner import Runner
from sam3_vlm.logging.writer import RunRecorder
from sam3_vlm.logging.artifacts import RunArtifactPaths
from sam3_vlm.logging.schema import RunManifest
from sam3_vlm.core.config import V4Config
from sam3_vlm.models.sam3 import MockSAM3Adapter
from sam3_vlm.models.qwen import MockQwenPlanner

from sam3_vlm.logging.replay import ReplayEngine

def test_replay_engine():
    config = V4Config()
    sensor = MockSAM3Adapter()
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
        
        # Replayed state should have same budget
        assert replayed_state.budget.sam3_calls == orig_state.budget.sam3_calls
        
        # Should have same count estimate
        assert abs(replayed_state.count_estimate.mean_count - orig_state.count_estimate.mean_count) < 1e-6
        assert abs(replayed_state.count_estimate.variance - orig_state.count_estimate.variance) < 1e-6
        
        # Stop reason should match
        assert replayed_state.stop_reason == orig_state.stop_reason
        
        # Graphs should match
        orig_nodes = orig_state.graph.to_dict()["nodes"]
        replayed_nodes = replayed_state.graph.to_dict()["nodes"]
        
        assert len(orig_nodes) == len(replayed_nodes)
        for nid in orig_nodes:
            assert nid in replayed_nodes
            # Deep equality
            assert orig_nodes[nid] == replayed_nodes[nid]
