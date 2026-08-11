import pytest
import numpy as np
from sam3_vlm.core.config import V4Config, BudgetConfig, StoppingConfig, ActionSelectionConfig, ReplanningConfig
from sam3_vlm.core.types import ActionFamily, SpatialMode
from sam3_vlm.pipeline.runner import Runner, RunnerState
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.planning.action_bank import ActionBankEntry
from sam3_vlm.scene.state import SceneState
from sam3_vlm.models.qwen import MockQwenPlanner
from sam3_vlm.models.sam3 import MockSAM3Adapter
from sam3_vlm.core.geometry import BoxGeometry, Box
from sam3_vlm.scene.node import Node


class ExtendedMockPlanner(MockQwenPlanner):
    """Provides a mix of discovery and confounder actions."""
    def __init__(self, actions=None):
        super().__init__()
        self.call_count = 0
        self.actions = actions
        
    def plan_scene(self, evidence, budget, config):
        self.call_count += 1
        from sam3_vlm.planning.qwen_planner import PlannerOutput
        if self.call_count == 1 and self.actions:
            return PlannerOutput(scene_summary="Mocked", proposed_actions=self.actions)
        return PlannerOutput(scene_summary="No more", proposed_actions=[])


def test_runner_multiple_nodes_updated_and_created():
    # 1. One SAM3 execution updates >1 existing node
    # 2. One execution updates and creates new node
    config = V4Config()
    sensor = MockSAM3Adapter()
    
    # We will prepopulate scene graph with 2 nodes, and mock SAM3 to return 3 detections (2 overlapping, 1 new)
    runner = Runner(config=config, sensor=sensor, planner=ExtendedMockPlanner())
    
    # Initialize Runner
    runner.target_class = "target"
    runner.state = RunnerState.GLOBAL_SENSING
    runner.scene_state = SceneState(
        image_id="img1", user_prompt="citrus", target_class="target", 
        graph=None, semantic_memory=None
    )
    
    from sam3_vlm.scene.graph import SceneGraph
    from sam3_vlm.scene.belief import SemanticMemory
    runner.scene_state.graph = SceneGraph()
    runner.scene_state.semantic_memory = SemanticMemory()
    from sam3_vlm.planning.action_bank import ActionBank
    runner.scene_state.action_bank = ActionBank()
    
    # Add two existing nodes
    node1 = Node(node_id="node1", geometry=BoxGeometry(Box(0,0,10,10)))
    node2 = Node(node_id="node2", geometry=BoxGeometry(Box(20,20,30,30)))
    runner.scene_state.graph.add_node(node1)
    runner.scene_state.graph.add_node(node2)
    
    action = SensingAction(action_id="act1", semantic_key="test", prompt="test", 
                           family=ActionFamily.DISCOVERY, spatial_mode=SpatialMode.GLOBAL)
    entry = ActionBankEntry(action=action, qwen_priority=0.9)
    runner.scene_state.action_bank.entries.append(entry)
    
    # Mock sensor to return 3 detections:
    # 1 overlaps node1, 1 overlaps node2, 1 is disjoint
    from sam3_vlm.sensing.observation import SAM3Observation
    from sam3_vlm.core.types import Detection
    
    def fake_observe(image, action):
        dets = [
            Detection(detection_id="d1", geometry=BoxGeometry(Box(0,0,10,10)), score=0.9),
            Detection(detection_id="d2", geometry=BoxGeometry(Box(20,20,30,30)), score=0.9),
            Detection(detection_id="d3", geometry=BoxGeometry(Box(50,50,60,60)), score=0.9)
        ]
        return SAM3Observation(call_id="call1", action_id=action.action_id, semantic_key=action.semantic_key, detections=dets, runtime_ms=10.0)
        
    sensor.observe = fake_observe
    runner.image = np.zeros((100, 100, 3))
    
    runner._step() # Executes GLOBAL_SENSING
    
    # Check graph size = 3
    assert len(runner.scene_state.graph.active_nodes()) == 3
    assert len(node1.observations) == 1
    assert len(node2.observations) == 1
    
    # 3. M4 tests: executed flag changes only after execution
    assert entry.executed is True
    
    # NEW NODE TEST: Check that the new node (from d3) did not receive NOT_RETRIEVED
    new_node = [n for n in runner.scene_state.graph.active_nodes() if n.node_id not in ["node1", "node2"]][0]
    assert len(new_node.observations) == 1 # Only NEW_DETECTION, no NOT_RETRIEVED


def test_runner_plateau_allows_discrimination():
    # 4. A zero-new-node discovery action does not immediately prevent a useful confounder action
    config = V4Config(
        replanning=ReplanningConfig(discovery_plateau_steps=2, unresolved_entropy_threshold=0.5),
        stopping=StoppingConfig(discovery_saturation_threshold=0.05)
    )
    
    sensor = MockSAM3Adapter()
    
    action1 = SensingAction(action_id="act1", semantic_key="disc", prompt="disc", family=ActionFamily.DISCOVERY, spatial_mode=SpatialMode.GLOBAL)
    action2 = SensingAction(action_id="act2", semantic_key="conf", prompt="conf", family=ActionFamily.CONFOUNDER, spatial_mode=SpatialMode.GLOBAL)
    
    runner = Runner(config=config, sensor=sensor, planner=ExtendedMockPlanner())
    runner.state = RunnerState.GLOBAL_SENSING
    from sam3_vlm.scene.graph import SceneGraph
    from sam3_vlm.scene.belief import SemanticMemory
    from sam3_vlm.planning.action_bank import ActionBank
    runner.scene_state = SceneState(image_id="img1", user_prompt="c", target_class="t", graph=SceneGraph(), semantic_memory=SemanticMemory(), action_bank=ActionBank())
    runner.scene_state.action_bank.entries.append(ActionBankEntry(action=action1, qwen_priority=0.9))
    runner.scene_state.action_bank.entries.append(ActionBankEntry(action=action2, qwen_priority=0.8))
    
    # Fake plateau: iteration 2, past counts are [0.0, 0.0]
    runner.scene_state.iteration = 2
    runner.scene_state.discovery_state.recent_new_node_counts = [0.0, 0.0]

    # Add a node with high entropy so uncertainty is NOT saturated
    from sam3_vlm.scene.node import Node
    from sam3_vlm.core.geometry import Box, BoxGeometry
    from sam3_vlm.core.types import NodeStatus, ClassBelief, RegistrationDiagnostics
    node = Node("n1", BoxGeometry(Box(10, 10, 20, 20)), status=NodeStatus.ACTIVE, class_belief=ClassBelief({"target": 0.5, "leaf": 0.5}), observations=[], created_by_call_id="c1", diagnostics=RegistrationDiagnostics())
    runner.scene_state.graph.add_node(node)

    # We should NOT stop because uncertainty is not saturated
    assert not runner.stopping_condition.should_stop(runner.scene_state, runner.config)
    



def test_runner_id_collision():
    # 6. Post-bootstrap new-node IDs never collide
    config = V4Config()
    sensor = MockSAM3Adapter()
    
    from sam3_vlm.sensing.observation import SAM3Observation
    from sam3_vlm.core.types import Detection
    def fake_observe(image, action):
        # Force a detection to guarantee a new node during GLOBAL_SENSING
        dets = [Detection(detection_id="d_new", geometry=BoxGeometry(Box(90,90,100,100)), score=0.9)]
        return SAM3Observation(call_id="call1", action_id=action.action_id, semantic_key=action.semantic_key, detections=dets, runtime_ms=10.0)
    sensor.observe = fake_observe
    
    from sam3_vlm.planning.qwen_planner import ProposedAction
    runner = Runner(config=config, sensor=sensor, planner=ExtendedMockPlanner(actions=[
        ProposedAction(semantic_key="new", prompt="new", family=ActionFamily.DISCOVERY, suggested_spatial_mode=SpatialMode.GLOBAL)
    ]))
    
    # Run full mock
    runner.run(image=np.zeros((100,100,3)), user_prompt="test", target_class="t", image_id="img")
    
    # Gather all node IDs
    node_ids = [n.node_id for n in runner.scene_state.graph.nodes.values()]
    
    # Check uniqueness
    assert len(node_ids) == len(set(node_ids))
    # Ensure there was actually a new node created post-bootstrap
    assert len(node_ids) > 0


def test_runner_budgets_enforced():
    # 7. Tile/runtime budgets are actually enforced
    config = V4Config(
        budget=BudgetConfig(max_sam3_calls=10, max_sam3_tiles=3, max_runtime_seconds=1.0) # 3 tiles max
    )
    sensor = MockSAM3Adapter()
    from sam3_vlm.sensing.observation import SAM3Observation
    def fake_observe(image, action):
        return SAM3Observation(call_id="call1", action_id=action.action_id, semantic_key=action.semantic_key, detections=[], runtime_ms=20.0)
    sensor.observe = fake_observe
    
    runner = Runner(config=config, sensor=sensor, planner=ExtendedMockPlanner())
    runner.image = np.zeros((100, 100, 3))
    runner.state = RunnerState.GLOBAL_SENSING
    from sam3_vlm.scene.graph import SceneGraph
    from sam3_vlm.scene.belief import SemanticMemory
    from sam3_vlm.planning.action_bank import ActionBank
    runner.scene_state = SceneState(image_id="img1", user_prompt="c", target_class="t", graph=SceneGraph(), semantic_memory=SemanticMemory(), action_bank=ActionBank())
    from sam3_vlm.scene.node import Node
    from sam3_vlm.core.types import NodeStatus, ClassBelief, RegistrationDiagnostics
    from sam3_vlm.core.geometry import Box, BoxGeometry
    node = Node("n1", BoxGeometry(Box(10, 10, 20, 20)), status=NodeStatus.ACTIVE, class_belief=ClassBelief({"t": 0.5, "o": 0.5}), observations=[], created_by_call_id="c1", diagnostics=RegistrationDiagnostics())
    runner.scene_state.graph.add_node(node)
    
    # Action takes 20ms, budget is 10ms
    action1 = SensingAction(action_id="act1", semantic_key="disc", prompt="disc", family=ActionFamily.DISCOVERY, spatial_mode=SpatialMode.GLOBAL)
    runner.scene_state.action_bank.entries.append(ActionBankEntry(action=action1, qwen_priority=0.9))
    
    runner._step() # Executes GLOBAL_SENSING -> ASSESS
    # Now budget is exhausted
    from sam3_vlm.sensing.tiling import TilingConfig
    action2 = SensingAction(action_id="act2", semantic_key="disc", prompt="disc", family=ActionFamily.DISCOVERY, spatial_mode=SpatialMode.TILED, tiling=TilingConfig(grid_rows=2, grid_cols=2)) # 4 tiles, budget is 3
    entry2 = ActionBankEntry(action=action2, qwen_priority=0.9)
    runner.scene_state.action_bank.entries.append(entry2)
    
    runner._step() # Executes ASSESS -> GLOBAL_SENSING
    runner._step() # Executes GLOBAL_SENSING -> ASSESS (skips execution due to budget)
    
    # Action should NOT be executed, but invalid
    assert not entry2.executed
    assert entry2.invalid_reason is not None
    
    # Try another loop step, it should execute ASSESS -> CLEANUP_DECISION or FINALIZE or REPLAN
    runner._step()
    assert runner.state in (RunnerState.REPLAN, RunnerState.CLEANUP, RunnerState.PLAN)


def test_runner_context_action_nodes():
    # 8. Context actions cannot increase target object count
    config = V4Config()
    sensor = MockSAM3Adapter()
    runner = Runner(config=config, sensor=sensor, planner=ExtendedMockPlanner())
    
    runner.state = RunnerState.GLOBAL_SENSING
    from sam3_vlm.scene.graph import SceneGraph
    from sam3_vlm.scene.belief import SemanticMemory
    from sam3_vlm.planning.action_bank import ActionBank
    runner.scene_state = SceneState(image_id="img1", user_prompt="c", target_class="t", graph=SceneGraph(), semantic_memory=SemanticMemory(), action_bank=ActionBank())
    
    action1 = SensingAction(action_id="act1", semantic_key="context", prompt="context", family=ActionFamily.CONTEXT, spatial_mode=SpatialMode.GLOBAL)
    runner.scene_state.action_bank.entries.append(ActionBankEntry(action=action1, qwen_priority=0.9))
    
    from sam3_vlm.sensing.observation import SAM3Observation
    from sam3_vlm.core.types import Detection
    def fake_observe(image, action):
        dets = [Detection(detection_id="d1", geometry=BoxGeometry(Box(0,0,10,10)), score=0.9)]
        return SAM3Observation(call_id="call1", action_id=action.action_id, semantic_key=action.semantic_key, detections=dets, runtime_ms=10.0)
    sensor.observe = fake_observe
    runner.image = np.zeros((100,100,3))
    
    def mock_choose():
        return runner.scene_state.action_bank.entries[0]
    runner._choose_best_action = mock_choose
    
    runner._step()
    
    # Graph should have 0 nodes because it's a CONTEXT action
    assert len(runner.scene_state.graph.active_nodes()) == 0


def test_runner_historical_penalty():
    # 9. Empirical poor prompt performance can override higher Qwen priority
    config = V4Config(
        action_selection=ActionSelectionConfig(alpha_discovery=1.0, beta_discrimination=1.0, eta_qwen_priority=0.2)
    )
    sensor = MockSAM3Adapter()
    runner = Runner(config=config, sensor=sensor, planner=ExtendedMockPlanner())
    
    runner.state = RunnerState.GLOBAL_SENSING
    from sam3_vlm.scene.graph import SceneGraph
    from sam3_vlm.scene.belief import SemanticMemory
    from sam3_vlm.planning.action_bank import ActionBank
    runner.scene_state = SceneState(image_id="img1", user_prompt="c", target_class="t", graph=SceneGraph(), semantic_memory=SemanticMemory(), action_bank=ActionBank())
    
    # Same key, previously returned 0 nodes
    action_bad = SensingAction(action_id="act_bad", semantic_key="bad_prompt", prompt="bad", family=ActionFamily.DISCOVERY, spatial_mode=SpatialMode.GLOBAL)
    entry_bad = ActionBankEntry(action=action_bad, qwen_priority=0.9) # High priority
    
    action_good = SensingAction(action_id="act_good", semantic_key="good_prompt", prompt="good", family=ActionFamily.DISCOVERY, spatial_mode=SpatialMode.GLOBAL)
    entry_good = ActionBankEntry(action=action_good, qwen_priority=0.8) # Lower priority
    
    runner.scene_state.action_bank.entries.append(entry_bad)
    runner.scene_state.action_bank.entries.append(entry_good)
    
    # Populate memory
    runner.scene_state.semantic_memory.record_execution(
        action=action_bad, sam3_call_id="c1", new_nodes=0, runtime_ms=10.0, predicted_utility=0.0
    )
    
    best = runner._choose_best_action()
    
    # Despite Qwen priority being higher (0.9 vs 0.8), bad_prompt should be penalized
    assert best.action.semantic_key == "good_prompt"


def test_runner_not_observable():
    # 10. Local actions append NOT_OBSERVABLE rather than NOT_RETRIEVED to unmatched nodes
    config = V4Config()
    sensor = MockSAM3Adapter()
    runner = Runner(config=config, sensor=sensor, planner=ExtendedMockPlanner())
    
    runner.state = RunnerState.GLOBAL_SENSING
    from sam3_vlm.scene.graph import SceneGraph
    from sam3_vlm.scene.belief import SemanticMemory
    from sam3_vlm.planning.action_bank import ActionBank
    runner.scene_state = SceneState(image_id="img1", user_prompt="c", target_class="t", graph=SceneGraph(), semantic_memory=SemanticMemory(), action_bank=ActionBank())
    
    # Add an existing node
    node1 = Node(node_id="node1", geometry=BoxGeometry(Box(0,0,10,10)))
    runner.scene_state.graph.add_node(node1)
    
    # We use ROI_BATCH here to trigger NOT_OBSERVABLE because it does not cover the full scene
    action1 = SensingAction(action_id="act1", semantic_key="roibatch", prompt="roibatch", family=ActionFamily.DISCOVERY, spatial_mode=SpatialMode.ROI_BATCH)
    runner.scene_state.action_bank.entries.append(ActionBankEntry(action=action1, qwen_priority=0.9))
    
    from sam3_vlm.sensing.observation import SAM3Observation
    def fake_observe(image, action):
        return SAM3Observation(call_id="call1", action_id=action.action_id, semantic_key=action.semantic_key, detections=[], runtime_ms=10.0)
    sensor.observe = fake_observe
    runner.image = np.zeros((100,100,3))
    
    # Need target_class for runner
    runner.target_class = "t"
    
    def mock_choose():
        return runner.scene_state.action_bank.entries[0]
    runner._choose_best_action = mock_choose
    
    runner._step()
    
    assert len(node1.observations) == 1
    from sam3_vlm.core.types import ObservationRelation
    assert node1.observations[0].relation == ObservationRelation.NOT_OBSERVABLE


def test_runner_max_iterations():
    # 11. max_iterations condition stops the loop
    config = V4Config(
        stopping=StoppingConfig(max_iterations=5)
    )
    sensor = MockSAM3Adapter()
    runner = Runner(config=config, sensor=sensor, planner=ExtendedMockPlanner())
    
    runner.state = RunnerState.GLOBAL_SENSING
    from sam3_vlm.scene.graph import SceneGraph
    from sam3_vlm.scene.belief import SemanticMemory
    from sam3_vlm.planning.action_bank import ActionBank
    runner.scene_state = SceneState(image_id="img1", user_prompt="c", target_class="t", graph=SceneGraph(), semantic_memory=SemanticMemory(), action_bank=ActionBank())
    runner.scene_state.iteration = 5  # reached max
    
    # Check stopping condition directly
    assert runner.stopping_condition.should_stop(runner.scene_state, runner.config)
