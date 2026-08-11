import pytest
import numpy as np
from PIL import Image

from sam3_vlm.models.sam3 import RealSAM3Sensor
from sam3_vlm.models.qwen import RealQwenPlanner
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.core.types import ActionFamily, SpatialMode
from sam3_vlm.sensing.evidence import QwenEvidencePack, ContactSheet

@pytest.mark.real_models
def test_real_sam3_sensor_global():
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sensor = RealSAM3Sensor(device=device)
    except Exception as e:
        pytest.skip(f"RealSAM3Sensor could not be initialized: {e}")

    # Create synthetic image
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img[100:200, 100:200] = [0, 255, 0] # A green square
    img_pil = Image.fromarray(img)

    action = SensingAction(
        action_id="test_01",
        semantic_key="test_target",
        prompt="green object",
        family=ActionFamily.DISCOVERY,
        threshold=0.1,
        spatial_mode=SpatialMode.GLOBAL,
    )

    obs = sensor.observe(img_pil, action)

    assert obs.action_id == "test_01"
    assert obs.model_metadata["real"] is True
    assert obs.model_metadata["spatial_mode"] == "GLOBAL"

@pytest.mark.real_models
def test_real_qwen_planner():
    try:
        planner = RealQwenPlanner()
    except Exception as e:
        pytest.skip(f"RealQwenPlanner could not be initialized: {e}")

    evidence = QwenEvidencePack(
        original_image_id="test_img",
        user_prompt="green square",
        target_class="target",
        contact_sheet=ContactSheet(crops=[], total_candidates=0)
    )

    from sam3_vlm.core.config import V4Config
    from sam3_vlm.core.types import BudgetState
    from sam3_vlm.planning.qwen_planner import QwenPlannerService

    service = QwenPlannerService(planner)
    
    try:
        output = service.plan_scene(evidence, BudgetState(), V4Config())
        assert output is not None
        assert isinstance(output.proposed_actions, list)
    except Exception as e:
        pytest.fail(f"QwenPlannerService failed with real planner: {e}")
