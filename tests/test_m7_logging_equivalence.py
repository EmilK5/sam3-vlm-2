import pytest
import os
import shutil
import tempfile
from pathlib import Path

from sam3_vlm.pipeline.runner import Runner
from sam3_vlm.logging.writer import RunRecorder
from sam3_vlm.logging.artifacts import RunArtifactPaths
from sam3_vlm.logging.schema import RunManifest
from sam3_vlm.core.config import V4Config
from sam3_vlm.models.sam3 import SAM3Sensor
from sam3_vlm.models.qwen import QwenPlanner

from sam3_vlm.models.sam3 import MockSAM3Adapter
from sam3_vlm.models.qwen import MockQwenPlanner

def test_runner_equivalence_with_and_without_logging():
    config = V4Config()
    sensor = MockSAM3Adapter()
    planner = MockQwenPlanner()
    
    # Run WITHOUT logging
    runner1 = Runner(config, sensor, planner)
    runner1.run(image="mock", image_id="test_image.jpg", user_prompt="Find apples", target_class="apple")
    nodes_without = {n.node_id: n for n in runner1.scene_state.graph.active_nodes()}
    
    # Run WITH logging
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir) / "run_artifacts"
        paths = RunArtifactPaths(base_dir=base_dir)
        manifest = RunManifest(run_id="test_run_123", image_id="test_image.jpg", target_class="apple")
        recorder = RunRecorder(paths, manifest)
        
        sensor2 = MockSAM3Adapter()
        planner2 = MockQwenPlanner()
        runner2 = Runner(config, sensor2, planner2, recorder=recorder)
        runner2.run(image="mock", image_id="test_image.jpg", user_prompt="Find apples", target_class="apple")
        nodes_with = {n.node_id: n for n in runner2.scene_state.graph.active_nodes()}
        
        assert len(nodes_without) == len(nodes_with)
        for nid in nodes_without:
            assert nid in nodes_with
            # Check belief equality
            assert nodes_without[nid].class_belief.probabilities == nodes_with[nid].class_belief.probabilities
            
        assert paths.events_jsonl.exists()
        assert paths.summary_json.exists()
        assert paths.run_json.exists()
