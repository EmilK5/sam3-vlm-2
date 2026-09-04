import pytest
import os
import numpy as np
from PIL import Image
import tempfile
import uuid
from pathlib import Path

# M8 STRICT GATE
RUN_REAL_MODELS = os.environ.get("RUN_REAL_MODELS") == "1"
if not RUN_REAL_MODELS:
    pytest.skip("Skipping real model tests because RUN_REAL_MODELS=1 is not set.", allow_module_level=True)

from sam3_vlm.models.sam3 import RealSAM3Sensor
from sam3_vlm.models.qwen import RealQwenPlanner
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.core.types import ActionFamily, SpatialMode
from sam3_vlm.sensing.evidence import QwenEvidencePack, ContactSheet
from sam3_vlm.core.config import V4Config
from sam3_vlm.core.geometry import Box

@pytest.mark.real_models
def test_real_sam3_global():
    sensor = RealSAM3Sensor(compile_model=False)
    
    img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    img_pil = Image.fromarray(img)
    
    action = SensingAction(
        action_id="global_01",
        semantic_key="test_target",
        prompt="green object",
        family=ActionFamily.DISCOVERY,
        threshold=0.1,
        spatial_mode=SpatialMode.GLOBAL
    )
    
    obs = sensor.observe(img_pil, action)
    assert obs.model_metadata["spatial_mode"] == "GLOBAL"
    assert len(obs.searched_regions) == 1
    assert obs.searched_regions[0].box.x2 == 1024

@pytest.mark.real_models
def test_real_sam3_tiled():
    from sam3_vlm.core.config import TilingConfig
    sensor = RealSAM3Sensor(compile_model=False)
    
    img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    img_pil = Image.fromarray(img)
    
    action = SensingAction(
        action_id="tiled_01",
        semantic_key="test_target",
        prompt="green object",
        family=ActionFamily.DISCOVERY,
        threshold=0.1,
        spatial_mode=SpatialMode.TILED,
        tiling=TilingConfig(grid_rows=2, grid_cols=2, overlap_ratio=0.0, tile_min_size=256)
    )
    
    obs = sensor.observe(img_pil, action)
    assert obs.model_metadata["spatial_mode"] == "TILED"
    assert len(obs.searched_regions) == 4

@pytest.mark.real_models
def test_real_sam3_local():
    sensor = RealSAM3Sensor(compile_model=False)
    
    img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    img_pil = Image.fromarray(img)
    
    action = SensingAction(
        action_id="local_01",
        semantic_key="test_target",
        prompt="green object",
        family=ActionFamily.CONFOUNDER,
        threshold=0.1,
        spatial_mode=SpatialMode.LOCAL,
        roi=Box(x1=50, y1=50, x2=250, y2=250)
    )
    
    obs = sensor.observe(img_pil, action)
    assert obs.model_metadata["spatial_mode"] == "LOCAL"
    assert len(obs.searched_regions) == 1
    assert obs.searched_regions[0].box.x1 == 50

@pytest.mark.real_models
def test_real_qwen_multimodal():
    planner = RealQwenPlanner(strict_model_errors=True)
    
    img = Image.new("RGB", (64, 64), color="red")
    fd1, p1 = tempfile.mkstemp(suffix=".jpg")
    os.close(fd1)
    img.save(p1)
    
    fd2, p2 = tempfile.mkstemp(suffix=".png")
    os.close(fd2)
    img.save(p2)
    
    evidence = QwenEvidencePack(
        original_image_id="test_img",
        user_prompt="red square",
        target_class="target",
        image_path=p1,
        contact_sheet=ContactSheet(crops=[], total_candidates=0, contact_sheet_image_path=p2)
    )
    
    from sam3_vlm.planning.qwen_planner import QwenPlannerService
    from sam3_vlm.core.types import BudgetState
    service = QwenPlannerService(planner)
    
    output = service.plan_scene(evidence, BudgetState(), V4Config())
    
    os.remove(p1)
    os.remove(p2)
    
    assert output is not None
    assert isinstance(output.proposed_actions, list)

@pytest.mark.real_models
def test_real_e2e_bounded(tmp_path):
    sensor = RealSAM3Sensor(compile_model=False)
    planner = RealQwenPlanner(strict_model_errors=True)
    
    from sam3_vlm.pipeline.runner import Runner
    from sam3_vlm.core.config import BudgetConfig, StoppingConfig, ReplanningConfig
    from sam3_vlm.logging.writer import RunRecorder, RunArtifactPaths, RunManifest
    from sam3_vlm.logging.validator import RunValidator
    from sam3_vlm.logging.replay import ReplayEngine
    
    config = V4Config(
        budget=BudgetConfig(max_qwen_calls=1, max_sam3_calls=3, max_sam3_tiles=4, max_cleanup_calls=0),
        stopping=StoppingConfig(max_iterations=1),
        replanning=ReplanningConfig(max_replans=0),
        assets_dir=str(tmp_path / "assets"),
    )
    
    from sam3_vlm.experiments.m8_smoke import assemble_e2e_runner, _run_validator_and_replay
    
    run_id = f"test_{uuid.uuid4().hex[:6]}"
    paths = RunArtifactPaths(base_dir=Path(os.path.join(tmp_path, run_id)))
    
    runner, recorder = assemble_e2e_runner(paths, config, sensor, planner, run_id, "green square", "target", "test")
    
    img = Image.new("RGB", (256, 256), color="black")
    count = runner.run(image=img, user_prompt="green square", target_class="target", image_id="test")
    
    assert count >= 0
    
    # Use production validator and replay helper
    valid_run = _run_validator_and_replay(paths, runner.scene_state)
    assert valid_run, "Validator or canonical replay equality failed"
