import pytest
import numpy as np

from sam3_vlm.core.config import V4Config, BudgetConfig, CleanupConfig, StoppingConfig, ReplanningConfig
from sam3_vlm.core.types import StopReason, SpatialMode, ActionFamily, ActionSource, CleanupDecision, ClassBelief
from sam3_vlm.core.geometry import BoxGeometry, Box
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.pipeline.runner import Runner, RunnerState
from sam3_vlm.pipeline.cleanup import CleanupController
from sam3_vlm.scene.state import SceneState, SceneGraph, SemanticMemory
from sam3_vlm.scene.graph import Node, NodeStatus
from sam3_vlm.sensing.action import SensingAction


class MockSAM3Adapter:
    def observe(self, image, action):
        from sam3_vlm.sensing.observation import SAM3Observation
        return SAM3Observation("call1", [], 100)


class MockPlanner:
    def __init__(self, count=0):
        self.call_count = count

    def plan_scene(self, evidence_pack, budget, config):
        from sam3_vlm.planning.action_bank import PlannerOutput
        self.call_count += 1
        return PlannerOutput(scene_summary="mock", proposed_actions=[], missing_appearance_modes=[], likely_confounders=[])


def test_m6_4_sam3_budget_outranks_cleanup_budget():
    config = V4Config(budget=BudgetConfig(max_sam3_calls=3, max_cleanup_calls=2))
    runner = Runner(config, MockSAM3Adapter(), MockPlanner())
    runner.scene_state = SceneState("img", "t", "t", SceneGraph(), SemanticMemory())
    
    # Establish simultaneous exhaustion
    runner.scene_state.budget.sam3_calls = 3
    runner.scene_state.budget.cleanup_calls = 2
    
    runner.state = RunnerState.CLEANUP
    runner._step()
    
    assert runner.state == RunnerState.FINALIZE
    assert runner.scene_state.stop_reason == StopReason.SAM3_BUDGET


def test_m6_4_tile_budget_outranks_cleanup_budget():
    config = V4Config(budget=BudgetConfig(max_sam3_tiles=10, max_cleanup_calls=2))
    runner = Runner(config, MockSAM3Adapter(), MockPlanner())
    runner.scene_state = SceneState("img", "t", "t", SceneGraph(), SemanticMemory())
    runner.image = np.zeros((10,10,3))
    
    # Establish simultaneous exhaustion
    runner.scene_state.budget.sam3_tiles = 11
    runner.scene_state.budget.cleanup_calls = 2
    
    runner.state = RunnerState.CLEANUP
    runner._step()
    
    assert runner.state == RunnerState.FINALIZE
    assert runner.scene_state.stop_reason == StopReason.TILE_BUDGET


def test_runtime_finalization_is_monotonic_and_covers_model_latency():
    runner = Runner(V4Config(), MockSAM3Adapter(), MockPlanner())
    runner.scene_state = SceneState("img", "t", "t", SceneGraph(), SemanticMemory())
    budget = runner.scene_state.budget
    budget.sam3_runtime_ms = 700.0
    budget.qwen_runtime_ms = 500.0
    budget.total_runtime_ms = 1500.0
    runner._elapsed_wall_ms = lambda: 1000.0

    runner._finalize_runtime_accounting()

    assert budget.model_runtime_ms == 1200.0
    assert budget.wall_runtime_ms == 1000.0
    assert budget.total_runtime_ms == 1500.0
    assert budget.total_runtime_ms >= budget.model_runtime_ms
    assert budget.total_runtime_ms >= budget.wall_runtime_ms


def test_m6_4_hard_reason_cannot_be_overwritten_by_cleanup_complete():
    config = V4Config()
    runner = Runner(config, MockSAM3Adapter(), MockPlanner())
    runner.scene_state = SceneState("img", "t", "t", SceneGraph(), SemanticMemory())
    runner.image = np.zeros((10,10,3))
    
    # Pre-establish a hard reason
    runner.scene_state.stop_reason = StopReason.SAM3_BUDGET
    
    runner.state = RunnerState.CLEANUP
    runner._step()
    
    assert runner.state == RunnerState.FINALIZE
    # Even though cleanup has no residuals (CLEANUP_COMPLETE), the original reason must survive
    assert runner.scene_state.stop_reason == StopReason.SAM3_BUDGET


def test_m6_4_hard_reason_cannot_be_overwritten_by_low_utility():
    config = V4Config()
    runner = Runner(config, MockSAM3Adapter(), MockPlanner())
    runner.scene_state = SceneState("img", "t", "t", SceneGraph(), SemanticMemory())
    runner.image = np.zeros((10,10,3))
    
    # Pre-establish a hard reason
    runner.scene_state.stop_reason = StopReason.MAX_ITERATIONS
    
    runner.cleanup_controller.generate_cleanup_action = lambda *args, **kwargs: CleanupDecision(action=None, reason=StopReason.LOW_MARGINAL_UTILITY)
    
    runner.state = RunnerState.CLEANUP
    runner._step()
    
    assert runner.state == RunnerState.FINALIZE
    assert runner.scene_state.stop_reason == StopReason.MAX_ITERATIONS


def test_m6_4_roi_batch_does_not_consume_tile_budget():
    config = V4Config(budget=BudgetConfig(max_sam3_tiles=10))
    runner = Runner(config, MockSAM3Adapter(), MockPlanner())
    runner.scene_state = SceneState("img", "t", "t", SceneGraph(), SemanticMemory())
    runner.image = np.zeros((10,10,3))
    runner.scene_state.budget.sam3_tiles = 9
    
    # Add a residual to allow cleanup action generation
    from sam3_vlm.core.types import RegistrationDiagnostics
    n1 = Node("n1", BoxGeometry(Box(0,0,10,10)), status=NodeStatus.ACTIVE, class_belief=ClassBelief({"t": 0.5, "o": 0.5}), observations=[], created_by_call_id="c1", diagnostics=RegistrationDiagnostics())
    runner.scene_state.graph.add_node(n1)
    
    runner.state = RunnerState.CLEANUP
    runner._step() # generates and executes ROI_BATCH
    
    assert runner.scene_state.budget.sam3_calls == 1
    assert runner.scene_state.budget.cleanup_calls == 1
    # Tile cost is 0 for ROI_BATCH
    assert runner.scene_state.budget.sam3_tiles == 9


def test_m6_4_local_does_not_consume_tile_budget():
    config = V4Config(budget=BudgetConfig(max_sam3_tiles=10))
    runner = Runner(config, MockSAM3Adapter(), MockPlanner())
    runner.scene_state = SceneState("img", "t", "t", SceneGraph(), SemanticMemory())
    runner.image = np.zeros((10,10,3))
    runner.scene_state.budget.sam3_tiles = 9
    
    # Mock CleanupDecision to return a LOCAL action
    action = SensingAction(
        action_id="cleanup1", 
        semantic_key="t", 
        prompt="t", 
        family=ActionFamily.VERIFICATION, 
        threshold=0.5, 
        spatial_mode=SpatialMode.LOCAL, 
        source=ActionSource.CONTROLLER
    )
    runner.cleanup_controller.generate_cleanup_action = lambda *args, **kwargs: CleanupDecision(action=action)
    
    runner.state = RunnerState.CLEANUP
    runner._step()
    
    assert runner.scene_state.budget.sam3_tiles == 9
    assert runner.scene_state.budget.sam3_calls == 1


def test_m6_4_tiled_does_consume_real_tile_budget():
    from sam3_vlm.core.config import TilingConfig
    config = V4Config(budget=BudgetConfig(max_sam3_tiles=10), tiling=TilingConfig(grid_rows=3, grid_cols=3))
    runner = Runner(config, MockSAM3Adapter(), MockPlanner())
    
    from sam3_vlm.planning.action_bank import ActionBank
    runner.scene_state = SceneState("img", "t", "t", SceneGraph(), SemanticMemory(), action_bank=ActionBank())
    runner.image = np.zeros((10,10,3))
    runner.scene_state.budget.sam3_tiles = 0
    runner.id_gen = IDGenerator()
    
    action = SensingAction(
        action_id=runner.id_gen.next_action_id(), 
        semantic_key="t", 
        prompt="t", 
        family=ActionFamily.DISCOVERY, 
        threshold=0.5, 
        spatial_mode=SpatialMode.TILED, 
        source=ActionSource.QWEN,
        tiling=TilingConfig(grid_rows=3, grid_cols=3)
    )
    entry = runner.scene_state.action_bank.add_action(action, 1.0)
    entry.total_utility = 0.5
    
    runner.state = RunnerState.GLOBAL_SENSING
    runner._step()
    
    assert runner.scene_state.budget.sam3_tiles == 9 # 3x3 tiles
    assert runner.scene_state.budget.sam3_calls == 1


def test_m6_4_tiled_near_tile_cap_is_rejected():
    from sam3_vlm.core.config import TilingConfig
    config = V4Config(budget=BudgetConfig(max_sam3_tiles=10), tiling=TilingConfig(grid_rows=3, grid_cols=3))
    
    class InstrumentedAdapter:
        called = False
        def observe(self, *args, **kwargs):
            self.called = True
            from sam3_vlm.sensing.observation import SAM3Observation
            return SAM3Observation("x", [], 0)
            
    adapter = InstrumentedAdapter()
    runner = Runner(config, adapter, MockPlanner())
    
    from sam3_vlm.planning.action_bank import ActionBank
    runner.scene_state = SceneState("img", "t", "t", SceneGraph(), SemanticMemory(), action_bank=ActionBank())
    runner.image = np.zeros((10,10,3))
    runner.scene_state.budget.sam3_tiles = 5 # 5 + 9 = 14 > 10
    runner.id_gen = IDGenerator()
    
    action = SensingAction(
        action_id=runner.id_gen.next_action_id(), 
        semantic_key="t", 
        prompt="t", 
        family=ActionFamily.DISCOVERY, 
        threshold=0.5, 
        spatial_mode=SpatialMode.TILED, 
        source=ActionSource.QWEN,
        tiling=TilingConfig(grid_rows=3, grid_cols=3)
    )
    entry = runner.scene_state.action_bank.add_action(action, 1.0)
    entry.total_utility = 0.5
    
    runner.state = RunnerState.GLOBAL_SENSING
    runner._step()
    
    assert runner.state == RunnerState.CLEANUP # Transition triggered by hard limit
    assert runner.scene_state.stop_reason == StopReason.TILE_BUDGET
    assert not adapter.called # Ensure observe was NOT called
    assert not entry.executed # Action should not be marked as executed
