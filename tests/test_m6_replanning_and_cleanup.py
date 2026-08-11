"""Tests for M6: Replanning, Stopping, and Residual Cleanup."""

import pytest
from sam3_vlm.core.config import V4Config
from sam3_vlm.core.geometry import Box, GeometryRef
from sam3_vlm.core.types import ActionFamily, ActionSource, NodeStatus, SpatialMode, Detection, StopReason
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.node import Node
from sam3_vlm.scene.state import SceneState
from sam3_vlm.pipeline.runner import Runner, RunnerState
from sam3_vlm.pipeline.cleanup import CleanupController
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.planning.action_bank import ActionBank
from sam3_vlm.models.sam3 import SAM3Sensor
from sam3_vlm.models.qwen import QwenPlanner
from sam3_vlm.planning.qwen_planner import PlannerOutput, ProposedAction
from sam3_vlm.sensing.observation import SAM3Observation
from sam3_vlm.scene.belief import SemanticMemory


class MockSAM3Sensor(SAM3Sensor):
    def __init__(self):
        self.call_count = 0

    def observe(self, image, action):
        self.call_count += 1
        return SAM3Observation(
            call_id=f"call_{self.call_count}",
            action_id=action.action_id,
            semantic_key=action.semantic_key,
            detections=[
                Detection("det1", GeometryRef(Box(10, 10, 50, 50)), score=0.9),
                Detection("det2", GeometryRef(Box(60, 60, 100, 100)), score=0.8),
            ],
            searched_regions=[],
            runtime_ms=100.0,
            model_metadata={}
        )


class MockQwenPlanner(QwenPlanner):
    def __init__(self):
        self.call_count = 0
        self.proposals = []

    def plan_scene(self, evidence, budget, config):
        self.call_count += 1
        return PlannerOutput(
            scene_summary="Mock summary",
            proposed_actions=self.proposals,
            missing_appearance_modes=[],
            likely_confounders=[]
        )


def test_m6_a_replanning_is_event_driven():
    """Verify Qwen calls << SAM3 actions."""
    config = V4Config()
    sensor = MockSAM3Sensor()
    planner = MockQwenPlanner()
    
    # Propose 3 actions so we don't replan immediately
    planner.proposals = [
        ProposedAction("key1", "prompt 1", ActionFamily.DISCOVERY, 0.9, None, 0.25, SpatialMode.GLOBAL, None, ""),
        ProposedAction("key2", "prompt 2", ActionFamily.DISCOVERY, 0.8, None, 0.25, SpatialMode.GLOBAL, None, ""),
        ProposedAction("key3", "prompt 3", ActionFamily.DISCOVERY, 0.7, None, 0.25, SpatialMode.GLOBAL, None, "")
    ]

    runner = Runner(config, sensor, planner)
    # Fast forward through bootstrap
    runner.state = RunnerState.PLAN
    runner.scene_state = SceneState("img1", "target", "target", SceneGraph(), SemanticMemory(), action_bank=ActionBank())
    from sam3_vlm.sensing.evidence import QwenEvidencePack, ContactSheet
    runner.evidence_pack = QwenEvidencePack(original_image_id="img1", target_class="target", user_prompt="target", contact_sheet=ContactSheet([], 0, {}))

    runner.run("mock_img", "target")

    # Qwen should have been called 1 time (during PLAN)
    # SAM3 should have been called 3 times (GLOBAL_SENSING) + cleanup depending on residuals
    assert planner.call_count == 1
    assert sensor.call_count >= 3


def test_m6_b_bank_exhaustion_triggers_one_replan():
    from sam3_vlm.core.config import StoppingConfig
    config = V4Config(stopping=StoppingConfig(max_iterations=2))
    sensor = MockSAM3Sensor()
    planner = MockQwenPlanner()
    
    # First plan gives 1 action. Second plan gives 1 action.
    planner.proposals = [
        ProposedAction("key1", "prompt 1", ActionFamily.DISCOVERY, 0.9, None, 0.25, SpatialMode.GLOBAL, None, "")
    ]
    
    runner = Runner(config, sensor, planner)
    runner.state = RunnerState.PLAN
    runner.scene_state = SceneState("img1", "target", "target", SceneGraph(), SemanticMemory(), action_bank=ActionBank())
    from sam3_vlm.sensing.evidence import QwenEvidencePack, ContactSheet
    runner.evidence_pack = QwenEvidencePack(original_image_id="img1", target_class="target", user_prompt="target", contact_sheet=ContactSheet([], 0, {}))

    def capture_planner_call(evidence, budget, config):
        planner.call_count += 1
        # Change proposal for 2nd call so it's not deduplicated out
        if planner.call_count == 2:
            planner.proposals = [
                ProposedAction("key2", "prompt 2", ActionFamily.DISCOVERY, 0.9, None, 0.25, SpatialMode.GLOBAL, None, "")
            ]
        return PlannerOutput(
            scene_summary="Mock summary",
            proposed_actions=planner.proposals,
            missing_appearance_modes=[],
            likely_confounders=[]
        )
    planner.plan_scene = capture_planner_call

    runner.run("mock_img", "target")
    
    # Should be 2 qwen calls: first plan, then replan due to bank exhaustion.
    assert planner.call_count == 2
    # Sensor calls: 2 global sensing + maybe cleanup
    assert sensor.call_count >= 2


def test_m6_cleanup_controller_batching():
    id_gen = IDGenerator()
    cleanup_controller = CleanupController(id_gen)
    config = V4Config()
    
    graph = SceneGraph()
    # Create two highly ambiguous nodes
    from sam3_vlm.core.types import ClassBelief, RegistrationDiagnostics
    from sam3_vlm.core.geometry import BoxGeometry
    n1 = Node("n1", BoxGeometry(Box(10, 10, 20, 20)), status=NodeStatus.ACTIVE, class_belief=ClassBelief({"target": 0.5, "leaf": 0.5}), observations=[], created_by_call_id="c1", diagnostics=RegistrationDiagnostics())
    
    n2 = Node("n2", BoxGeometry(Box(30, 30, 40, 40)), status=NodeStatus.ACTIVE, class_belief=ClassBelief({"target": 0.45, "leaf": 0.55}), observations=[], created_by_call_id="c1", diagnostics=RegistrationDiagnostics())
    
    graph.add_node(n1)
    graph.add_node(n2)
    
    residuals = cleanup_controller.select_residual_nodes(graph, config, "target")
    assert len(residuals) == 2
    
    action = cleanup_controller.generate_cleanup_action(residuals, graph, "target", config)
    assert action is not None
    assert action.spatial_mode == SpatialMode.ROI_BATCH
    assert action.roi.x1 == 10
    assert action.roi.y2 == 40
    assert len(action.positive_exemplar_ids) == 0


def test_m6_f_empirical_history_overrides_qwen():
    config = V4Config()
    from sam3_vlm.planning.action_bank import ActionBankGenerator
    generator = ActionBankGenerator()
    
    # Create SemanticMemory with bad history for a specific group
    memory = SemanticMemory()
    action_bad = SensingAction("a1", "bad_key", "bad prompt", ActionFamily.DISCOVERY)
    memory.record_execution(action_bad, "sam1", realized_discrimination_proxy=0.01, predicted_utility=0.01) # Execution 1: utility=0
    # Add fake data for the record
    record = memory.records["bad_key"]
    record.realized_utility_by_execution = [0.01]
    record.execution_count = 1
    
    bank = ActionBank()
    id_gen = IDGenerator()
    
    planner_output = PlannerOutput(
        scene_summary="",
        missing_appearance_modes=[], likely_confounders=[],
        proposed_actions=[
            ProposedAction("bad_key2", "another bad prompt", ActionFamily.DISCOVERY, 0.95, None, 0.25, SpatialMode.GLOBAL, None, "", correlation_group="bad_key"),
            ProposedAction("good_key", "good prompt", ActionFamily.DISCOVERY, 0.60, None, 0.25, SpatialMode.GLOBAL, None, "", correlation_group="good_key"),
        ]
    )
    
    entries = generator.generate_entries(planner_output, memory, bank, id_gen, config=config)
    assert len(entries) == 2
    
    bad_entry = next(e for e in entries if e.action.correlation_group == "bad_key")
    good_entry = next(e for e in entries if e.action.correlation_group == "good_key")
    
    # Qwen gave bad_key 0.95, but it has bad history (< 0.05). So it should be penalized to 0.095.
    assert bad_entry.qwen_priority < 0.1
    # Qwen gave good_key 0.60. No history. So it stays 0.60.
    assert good_entry.qwen_priority == 0.60
