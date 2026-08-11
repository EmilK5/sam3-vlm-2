import pytest
import numpy as np
from typing import Optional

from sam3_vlm.core.config import V4Config, BudgetConfig, ReplanningConfig, CleanupConfig, StoppingConfig
from sam3_vlm.core.geometry import Box, BoxGeometry
from sam3_vlm.core.types import (
    ActionFamily, ActionSource, SpatialMode, StopReason, ClassBelief, NodeStatus, Detection
)
from sam3_vlm.pipeline.runner import Runner, RunnerState
from sam3_vlm.pipeline.bootstrap import BootstrapPipeline
from sam3_vlm.scene.state import SceneState, SceneGraph
from sam3_vlm.scene.node import Node, RegistrationDiagnostics
from sam3_vlm.scene.belief import SemanticMemory
from sam3_vlm.planning.qwen_planner import PlannerOutput, ProposedAction
from sam3_vlm.sensing.observation import SAM3Observation
from sam3_vlm.core.id_generator import IDGenerator

class FixedMockPlanner:
    def __init__(self, action_batches=None):
        self.action_batches = action_batches or []
        self.call_count = 0
        self.last_evidence_pack = None
        
    def plan_scene(self, evidence_pack, budget, config):
        batch = self.action_batches[self.call_count] if self.call_count < len(self.action_batches) else []
        self.call_count += 1
        self.last_evidence_pack = evidence_pack
        return PlannerOutput(scene_summary="mock", proposed_actions=batch, missing_appearance_modes=[], likely_confounders=[])

class MockSAM3Adapter:
    def __init__(self):
        self.call_count = 0
        self.observe_fn = None

    def observe(self, image, action):
        self.call_count += 1
        if self.observe_fn:
            return self.observe_fn(image, action)
        dets = [Detection("d1", BoxGeometry(Box(0,0,10,10)), 0.9)]
        return SAM3Observation(
            call_id=f"c{self.call_count}",
            action_id=action.action_id,
            semantic_key=action.semantic_key,
            detections=dets,
            runtime_ms=10.0
        )

def test_m6_3_full_production_path():
    # 8, 10, 11: Real production path from Bootstrap through Replan and Cleanup
    config = V4Config(
        budget=BudgetConfig(max_qwen_calls=2, max_sam3_calls=5),
        replanning=ReplanningConfig(max_replans=1, min_actions_between_replans=0, discovery_plateau_steps=999),
        cleanup=CleanupConfig(roi_batch_size=2)
    )
    
    sensor = MockSAM3Adapter()
    planner = FixedMockPlanner(action_batches=[
        [ProposedAction("disc1", "find disc1", ActionFamily.DISCOVERY, suggested_spatial_mode=SpatialMode.GLOBAL)],
        [ProposedAction("disc2", "find disc2", ActionFamily.DISCOVERY, suggested_spatial_mode=SpatialMode.GLOBAL)]
    ])
    
    # We pass a string as the image to mimic a path
    image_input = "path/to/real_image.jpg"
    
    bootstrap = BootstrapPipeline(sensor, config=config)
    bootstrap_result = bootstrap.execute_bootstrap("img1", image_input, "test prompt", "test")
    
    # 11: Assert bootstrap correctly propagated image_path
    assert bootstrap_result.state.image_path == "path/to/real_image.jpg"
    assert bootstrap_result.qwen_evidence_pack.image_path == "path/to/real_image.jpg"
    
    runner = Runner(config, sensor, planner)
    runner.scene_state = bootstrap_result.state
    runner.evidence_pack = bootstrap_result.qwen_evidence_pack
    runner.image = image_input
    runner.user_prompt = "test prompt"
    runner.target_class = "test"
    runner.image_id = "img1"
    runner.id_gen = bootstrap.id_gen
    
    from sam3_vlm.planning.action_bank import ActionBank
    runner.scene_state.action_bank = ActionBank()
    
    # Manually transition through state machine exactly like run() would
    runner.state = RunnerState.PLAN
    runner._step()
    
    # Initial Qwen call happened
    assert runner.state == RunnerState.GLOBAL_SENSING
    assert planner.call_count == 1
    assert runner.scene_state.replans_executed == 0
    assert runner.scene_state.budget.qwen_calls == 1
    
    # Execute the GLOBAL_SENSING action
    runner._step() 
    assert runner.state == RunnerState.ASSESS
    
    # Assume it wants to replan (bank is now exhausted)
    runner._step()
    assert runner.state == RunnerState.REPLAN
    
    # REPLAN happens
    runner._step()
    assert runner.state == RunnerState.GLOBAL_SENSING
    assert planner.call_count == 2
    assert runner.scene_state.replans_executed == 1
    assert runner.scene_state.budget.qwen_calls == 2
    
    # Verify second evidence pack retained the image path!
    replan_pack = planner.last_evidence_pack
    assert replan_pack.image_path == "path/to/real_image.jpg"
    
    # Empty bank again
    runner._step() # Executes the action from replan
    assert runner.state == RunnerState.ASSESS
    
    # Now it wants to replan again, but budget/replans exhausted
    runner._step() 
    assert runner.state == RunnerState.REPLAN
    
    runner._step()
    # Should fall back to CLEANUP since no valid actions are in the bank
    assert runner.state == RunnerState.CLEANUP
    
    # Add a mock residual node so cleanup doesn't just instantly say CLEANUP_COMPLETE
    node = Node("r1", BoxGeometry(Box(0,0,10,10)), status=NodeStatus.ACTIVE, class_belief=ClassBelief({"test": 0.5, "other": 0.5}), observations=[], created_by_call_id="x", diagnostics=RegistrationDiagnostics())
    runner.scene_state.graph.add_node(node)
    
    runner._step() # Generates cleanup action and executes it
    assert runner.state == RunnerState.ASSESS_CLEANUP
    
    runner._step() # Loops back to CLEANUP
    assert runner.state == RunnerState.CLEANUP
    
    runner.cleanup_controller.batch_attempts.clear()
    # Exhaust SAM3 budget
    runner.scene_state.budget.sam3_calls = 5
    runner._step() # Fails hard budget check
    assert runner.state == RunnerState.FINALIZE
    assert runner.scene_state.stop_reason == StopReason.SAM3_BUDGET


def test_m6_3_cleanup_complete():
    # 12. CLEANUP_COMPLETE when no residuals remain
    config = V4Config()
    runner = Runner(config, MockSAM3Adapter(), FixedMockPlanner())
    runner.image = np.zeros((10,10,3))
    runner.scene_state = SceneState("img", "t", "t", SceneGraph(), SemanticMemory())
    # Bank empty, fall straight to cleanup
    runner.state = RunnerState.CLEANUP
    runner._step()
    
    assert runner.state == RunnerState.FINALIZE
    assert runner.scene_state.stop_reason == StopReason.CLEANUP_COMPLETE


def test_m6_3_low_marginal_utility():
    # 13. LOW_MARGINAL_UTILITY when utility drops below threshold
    config = V4Config(cleanup=CleanupConfig(cleanup_min_utility=0.9)) # Impossible threshold
    runner = Runner(config, MockSAM3Adapter(), FixedMockPlanner())
    runner.image = np.zeros((10,10,3))
    runner.scene_state = SceneState("img", "t", "t", SceneGraph(), SemanticMemory())
    runner.id_gen = IDGenerator()
    
    # Add residual
    node = Node("r1", BoxGeometry(Box(0,0,10,10)), status=NodeStatus.ACTIVE, class_belief=ClassBelief({"t": 0.5, "other": 0.5}), observations=[], created_by_call_id="x", diagnostics=RegistrationDiagnostics())
    runner.scene_state.graph.add_node(node)
    
    runner.state = RunnerState.CLEANUP
    runner._step()
    
    assert runner.state == RunnerState.FINALIZE
    assert runner.scene_state.stop_reason == StopReason.LOW_MARGINAL_UTILITY


def test_m6_3_qwen_budget_encountered_but_run_continues():
    # 15. Qwen budget exhausted, but useful action remains
    config = V4Config(budget=BudgetConfig(max_qwen_calls=1), replanning=ReplanningConfig(max_replans=1))
    runner = Runner(config, MockSAM3Adapter(), FixedMockPlanner())
    runner.image = np.zeros((10,10,3))
    
    from sam3_vlm.planning.action_bank import ActionBank
    runner.scene_state = SceneState("img", "t", "t", SceneGraph(), SemanticMemory(), action_bank=ActionBank())
    runner.id_gen = IDGenerator()
    
    # We have reached max qwen calls
    runner.scene_state.budget.qwen_calls = 1
    
    # But we have a valid action
    action = ProposedAction("valid", "valid", ActionFamily.DISCOVERY, suggested_spatial_mode=SpatialMode.GLOBAL)
    from sam3_vlm.sensing.action import SensingAction
    sa = SensingAction(runner.id_gen.next_action_id(), "valid", "valid", ActionFamily.DISCOVERY, 0.5, SpatialMode.GLOBAL, ActionSource.QWEN)
    entry = runner.scene_state.action_bank.add_action(sa, qwen_priority=1.0)
    entry.total_utility = 0.5
    
    runner.state = RunnerState.REPLAN
    runner._step()
    
    # Should fall back to GLOBAL_SENSING without setting stop_reason
    assert runner.state == RunnerState.GLOBAL_SENSING
    assert runner.scene_state.stop_reason is None


def test_m6_3_semantic_history_sign():
    # 16. Semantic history uses proper delta logic
    runner = Runner(V4Config(), MockSAM3Adapter(), FixedMockPlanner())
    
    # Setup history where entropy decreased (improved)
    # pre = 1.0, post = 0.6 => change (post - pre) = -0.4
    runner.scene_state = SceneState("img", "t", "t", SceneGraph(), SemanticMemory())
    from sam3_vlm.sensing.action import SensingAction
    action = SensingAction("a1", "t", "t", ActionFamily.DISCOVERY, 0.5, SpatialMode.GLOBAL, ActionSource.QWEN)
    
    # Mock record_execution to simulate
    runner.scene_state.semantic_memory.record_execution(
        action=action, sam3_call_id="c1", entropy_change=-0.4, variance_change=-0.3
    )
    
    # Run the ReplanEvidenceBuilder
    from sam3_vlm.planning.replanning import ReplanEvidenceBuilder
    from sam3_vlm.sensing.evidence import ContactSheetBuilder
    
    class DummyCSB(ContactSheetBuilder):
        def build_contact_sheet(self, *args, **kwargs):
            from sam3_vlm.sensing.evidence import ContactSheet
            return ContactSheet([], 0, {}, "dummy.jpg")
            
    builder = ReplanEvidenceBuilder(DummyCSB())
    pack = builder.build(runner.scene_state, np.zeros((10,10,3)))
    
    # Ensure it's correctly logged as delta, not reduction, and respects the sign!
    assert "avg_ent_delta=-0.40" in pack.scene_summary
    assert "avg_var_delta=-0.30" in pack.scene_summary
