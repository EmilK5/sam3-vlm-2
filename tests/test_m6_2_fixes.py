import pytest
import numpy as np

from sam3_vlm.core.config import V4Config, BudgetConfig, StoppingConfig, CleanupConfig, ReplanningConfig
from sam3_vlm.core.types import ActionFamily, SpatialMode, StopReason, ActionSource
from sam3_vlm.pipeline.runner import Runner, RunnerState
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.planning.action_bank import ActionBankEntry
from sam3_vlm.scene.state import SceneState
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.belief import SemanticMemory
from sam3_vlm.scene.node import Node
from sam3_vlm.core.geometry import Box, BoxGeometry
from sam3_vlm.core.types import NodeStatus, ClassBelief, RegistrationDiagnostics
from sam3_vlm.sensing.observation import SAM3Observation
from sam3_vlm.planning.qwen_planner import PlannerOutput, ProposedAction
from sam3_vlm.pipeline.cleanup import CleanupController
from sam3_vlm.core.id_generator import IDGenerator

class MockSAM3Adapter:
    def observe(self, image, action):
        return SAM3Observation(
            call_id="call1",
            action_id=action.action_id,
            semantic_key=action.semantic_key,
            detections=[],
            runtime_ms=10.0
        )

class MockPlanner:
    def __init__(self):
        self.call_count = 0
        self.last_evidence_pack = None

    def plan_scene(self, evidence_pack, budget_state, config):
        self.call_count += 1
        self.last_evidence_pack = evidence_pack
        return PlannerOutput(
            scene_summary="mock summary",
            proposed_actions=[
                ProposedAction("disc1", "find disc1", ActionFamily.DISCOVERY, 0.9, None, 0.25, SpatialMode.GLOBAL, None, ""),
            ],
            missing_appearance_modes=[],
            likely_confounders=[]
        )

def test_m6_2_a_initial_qwen_uses_bootstrap_evidence():
    # A. Initial Qwen uses bootstrap evidence
    config = V4Config()
    runner = Runner(config, MockSAM3Adapter(), MockPlanner())
    runner.image = np.zeros((100, 100, 3))
    runner.user_prompt = "test"
    runner.target_class = "t"
    runner.image_id = "img1"

    # Fast forward to PLAN
    from sam3_vlm.sensing.evidence import QwenEvidencePack, ContactSheet
    from sam3_vlm.planning.action_bank import ActionBank
    runner.scene_state = SceneState(image_id="img1", user_prompt="test", target_class="t", graph=SceneGraph(), semantic_memory=SemanticMemory(), action_bank=ActionBank())
    runner.evidence_pack = QwenEvidencePack(
        original_image_id="img1",
        user_prompt="test",
        target_class="t",
        contact_sheet=ContactSheet([], 0, {}, "path/to/bootstrap_sheet.jpg"),
        image_path="path/to/original.jpg"
    )
    runner.state = RunnerState.PLAN

    runner._step()

    # Initial plan should execute and go to GLOBAL_SENSING
    assert runner.state == RunnerState.GLOBAL_SENSING
    assert runner.planner.call_count == 1
    
    # Assert bootstrap evidence was preserved
    last_pack = runner.planner.last_evidence_pack
    assert last_pack.image_path == "path/to/original.jpg"
    assert last_pack.contact_sheet.contact_sheet_image_path == "path/to/bootstrap_sheet.jpg"

    # Assert replans_executed is NOT incremented!
    assert runner.scene_state.replans_executed == 0
    assert runner.scene_state.budget.qwen_calls == 1

def test_m6_2_b_and_c_bank_exhaustion_max_replans():
    # B & C. Bank exhaustion rebuilds fresh evidence and respects max_replans
    config = V4Config(
        replanning=ReplanningConfig(max_replans=1, min_actions_between_replans=0),
        budget=BudgetConfig(max_qwen_calls=5)
    )
    runner = Runner(config, MockSAM3Adapter(), MockPlanner())
    runner.image = np.zeros((100, 100, 3))
    runner.user_prompt = "test"
    runner.target_class = "t"
    runner.image_id = "img1"

    # Fast forward to ASSESS
    from sam3_vlm.planning.action_bank import ActionBank
    runner.scene_state = SceneState(
        image_id="img1", user_prompt="test", target_class="t", image_path="path/to/original.jpg",
        graph=SceneGraph(), semantic_memory=SemanticMemory(), action_bank=ActionBank()
    )
    # Give it one initial qwen call already (not a replan)
    runner.scene_state.budget.qwen_calls = 1
    runner.scene_state.replans_executed = 0
    
    runner.state = RunnerState.ASSESS
    
    # Empty bank -> should trigger replan
    runner._step()
    assert runner.state == RunnerState.REPLAN
    
    runner._step()
    # REPLAN should build fresh evidence and call Qwen, then go to GLOBAL_SENSING
    assert runner.state == RunnerState.GLOBAL_SENSING
    assert runner.scene_state.replans_executed == 1
    assert runner.scene_state.budget.qwen_calls == 2
    last_pack = runner.planner.last_evidence_pack
    # Fresh evidence uses replan image id
    assert last_pack.original_image_id == "img1"
    assert last_pack.image_path == "path/to/original.jpg"

    # Now empty bank again
    runner.scene_state.action_bank.entries.clear()
    runner.state = RunnerState.ASSESS
    runner._step()
    assert runner.state == RunnerState.REPLAN

    # This time, max_replans=1 is hit
    runner._step()
    # Should fallback to valid action (none exist), so goes to CLEANUP
    assert runner.state == RunnerState.CLEANUP
    assert runner.scene_state.replans_executed == 1
    assert runner.scene_state.budget.qwen_calls == 2

def test_m6_2_f_qwen_budget_exhausted_useful_action():
    # F. Qwen budget exhausted + useful action remains
    config = V4Config(
        budget=BudgetConfig(max_qwen_calls=1),
        replanning=ReplanningConfig(max_replans=5, min_actions_between_replans=0),
        stopping=StoppingConfig(utility_min_threshold=0.1)
    )
    runner = Runner(config, MockSAM3Adapter(), MockPlanner())
    runner.image = np.zeros((100, 100, 3))
    runner.user_prompt = "test"
    runner.target_class = "t"
    
    from sam3_vlm.planning.action_bank import ActionBank
    runner.scene_state = SceneState(image_id="img1", user_prompt="t", target_class="t", graph=SceneGraph(), semantic_memory=SemanticMemory(), action_bank=ActionBank())
    runner.scene_state.budget.qwen_calls = 1
    
    # Put a useful action in bank
    action1 = SensingAction(action_id="act1", semantic_key="disc", prompt="disc", family=ActionFamily.DISCOVERY, spatial_mode=SpatialMode.GLOBAL)
    entry = ActionBankEntry(action=action1, qwen_priority=0.9, total_utility=0.5)
    runner.scene_state.action_bank.entries.append(entry)
    
    # We trigger a REPLAN directly (e.g. from discovery plateau)
    runner.state = RunnerState.REPLAN
    runner._step()
    
    # Should fallback to GLOBAL_SENSING because budget is exhausted but useful action remains
    assert runner.state == RunnerState.GLOBAL_SENSING
    assert runner.scene_state.budget.qwen_calls == 1 # didn't increment

def test_m6_2_g_and_h_cleanup_iterations():
    # G & H. Cleanup increments total iteration count and respects max_iterations
    config = V4Config(
        stopping=StoppingConfig(max_iterations=2),
        cleanup=CleanupConfig(roi_batch_size=4)
    )
    runner = Runner(config, MockSAM3Adapter(), MockPlanner())
    runner.image = np.zeros((100, 100, 3))
    
    runner.scene_state = SceneState(image_id="img1", user_prompt="t", target_class="t", graph=SceneGraph(), semantic_memory=SemanticMemory())
    
    # Create ambiguous node
    node = Node("n1", BoxGeometry(Box(10, 10, 20, 20)), status=NodeStatus.ACTIVE, class_belief=ClassBelief({"t": 0.5, "other": 0.5}), observations=[], created_by_call_id="c1", diagnostics=RegistrationDiagnostics())
    runner.scene_state.graph.add_node(node)
    
    runner.state = RunnerState.CLEANUP
    runner.scene_state.iteration = 1
    
    runner._step()
    # It executes cleanup, iteration goes to 2, state -> ASSESS_CLEANUP
    assert runner.state == RunnerState.ASSESS_CLEANUP
    assert runner.scene_state.iteration == 2
    
    runner._step() # -> CLEANUP
    
    # Trick the cleanup controller into thinking it's a new batch to avoid it returning None
    runner.cleanup_controller.batch_attempts.clear()
    
    # Now in CLEANUP, but next action will hit iteration budget if we tried to execute it?
    # Actually, hard budget checks are BEFORE sensor execution.
    runner._step()
    # _check_hard_budgets sees iteration >= max_iterations (2 >= 2)
    # returns StopReason.MAX_ITERATIONS
    # state goes to FINALIZE
    assert runner.state == RunnerState.FINALIZE
    assert runner.scene_state.stop_reason == StopReason.MAX_ITERATIONS

def test_m6_2_i_and_j_cleanup_stop_reasons():
    # I & J. CLEANUP_BUDGET vs CLEANUP_COMPLETE
    config = V4Config(budget=BudgetConfig(max_cleanup_calls=1))
    runner = Runner(config, MockSAM3Adapter(), MockPlanner())
    runner.image = np.zeros((100, 100, 3))
    
    runner.scene_state = SceneState(image_id="img1", user_prompt="t", target_class="t", graph=SceneGraph(), semantic_memory=SemanticMemory())
    node = Node("n1", BoxGeometry(Box(10, 10, 20, 20)), status=NodeStatus.ACTIVE, class_belief=ClassBelief({"t": 0.5, "other": 0.5}), observations=[], created_by_call_id="c1", diagnostics=RegistrationDiagnostics())
    runner.scene_state.graph.add_node(node)
    
    # Exhaust cleanup budget manually
    runner.scene_state.budget.cleanup_calls = 1
    runner.state = RunnerState.CLEANUP
    runner._step()
    
    assert runner.state == RunnerState.FINALIZE
    assert runner.scene_state.stop_reason == StopReason.CLEANUP_BUDGET
    
    # Now give budget back but remove residual
    runner.scene_state.budget.cleanup_calls = 0
    node.class_belief = ClassBelief({"t": 0.99, "other": 0.01}) # not ambiguous
    runner.state = RunnerState.CLEANUP
    runner._step()
    
    assert runner.state == RunnerState.FINALIZE
    assert runner.scene_state.stop_reason == StopReason.CLEANUP_COMPLETE

def test_m6_2_l_cleanup_batch_rotation():
    # L. Cleanup batch rotation
    config = V4Config(cleanup=CleanupConfig(roi_batch_size=2, cleanup_min_utility=0.01))
    id_gen = IDGenerator()
    cleanup_controller = CleanupController(id_gen)
    
    graph = SceneGraph()
    # 3 identical ambiguous nodes
    n1 = Node("n1", BoxGeometry(Box(0, 0, 10, 10)), status=NodeStatus.ACTIVE, class_belief=ClassBelief({"t": 0.5, "other": 0.5}), observations=[], created_by_call_id="c1", diagnostics=RegistrationDiagnostics())
    n2 = Node("n2", BoxGeometry(Box(20, 20, 30, 30)), status=NodeStatus.ACTIVE, class_belief=ClassBelief({"t": 0.5, "other": 0.5}), observations=[], created_by_call_id="c1", diagnostics=RegistrationDiagnostics())
    n3 = Node("n3", BoxGeometry(Box(40, 40, 50, 50)), status=NodeStatus.ACTIVE, class_belief=ClassBelief({"t": 0.5, "other": 0.5}), observations=[], created_by_call_id="c1", diagnostics=RegistrationDiagnostics())
    
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    
    residuals = cleanup_controller.select_residual_nodes(graph, config, "t")
    assert len(residuals) == 3
    
    action1 = cleanup_controller.generate_cleanup_action(residuals, graph, "t", config)
    assert action1.action is not None
    # first batch is n1, n2
    
    # If utility doesn't improve, next call should pick n3!
    action2 = cleanup_controller.generate_cleanup_action(residuals, graph, "t", config)
    assert action2.action is not None
    assert action2.action.roi == Box(40, 40, 50, 50) # only n3 is selected since n1,n2 are penalized
