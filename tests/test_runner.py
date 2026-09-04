import pytest
from sam3_vlm.core.config import V4Config
from sam3_vlm.core.types import ActionSource
from sam3_vlm.pipeline.runner import Runner, RunnerState
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.planning.qwen_planner import PlannerOutput
from sam3_vlm.scene.state import SceneState
from sam3_vlm.models.qwen import MockQwenPlanner
from sam3_vlm.models.sam3 import MockSAM3Adapter


class FixedMockPlanner(MockQwenPlanner):
    """Planner that generates a couple valid actions, then stops planning more."""
    
    def __init__(self):
        super().__init__()
        self.call_count = 0
        
    def plan_scene(self, evidence, budget, config):
        self.call_count += 1
        if self.call_count == 1:
            return super().plan_scene(evidence, budget, config)
        return PlannerOutput(
            scene_summary="No more actions",
            proposed_actions=[]
        )


def test_runner_end_to_end_mock(tmp_path):
    import dataclasses
    from sam3_vlm.core.config import BudgetConfig
    
    config = V4Config(
        budget=BudgetConfig(max_sam3_calls=5, max_qwen_calls=2),
        assets_dir=str(tmp_path / "assets"),
    )
    
    sensor = MockSAM3Adapter()
    planner = FixedMockPlanner()
    
    runner = Runner(config=config, sensor=sensor, planner=planner)
    
    import numpy as np
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Run the machine
    count = runner.run(
        image=dummy_image,
        user_prompt="green citrus",
        target_class="target",
        image_id="img1"
    )
    
    assert runner.state == RunnerState.DONE
    
    state = runner.scene_state
    assert state is not None
    assert state.budget.sam3_calls >= 1
    assert state.budget.qwen_calls >= 1
    
    # We should have some nodes created from the mock sensor
    assert len(state.graph.active_nodes()) > 0
    
    # The count should be populated
    assert count >= 0.0
    
    # Check stopping condition was met
    assert runner.scene_state.stop_reason is not None or count > 0
