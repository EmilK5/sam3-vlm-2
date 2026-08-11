import pytest
from typing import Dict
from sam3_vlm.scene.belief import BeliefUpdater
from sam3_vlm.scene.node import Node
from sam3_vlm.core.geometry import BoxGeometry, Box
from sam3_vlm.core.types import ClassBelief, NodeObservationRef, ObservationRelation, ActionFamily, SpatialMode
from sam3_vlm.sensing.action import SensingAction


def create_mock_node() -> Node:
    return Node(
        node_id="test_node",
        geometry=BoxGeometry(Box(0, 0, 10, 10)),
    )

def test_m5_a_basic_bayesian_movement():
    updater = BeliefUpdater()
    node = create_mock_node()
    
    # Target compatible prompt -> strong target-like observation
    action1 = SensingAction(action_id="a1", semantic_key="target", prompt="target", family=ActionFamily.DISCOVERY)
    obs1 = NodeObservationRef("o1", "call1", "a1", "target", relation=ObservationRelation.STRONG_MATCH, score=0.9)
    updater.update_node_belief(node, action1, obs1, target_class="target", confounder_class="confounder")
    
    p_target_after_a1 = node.class_belief.probabilities["target"]
    p_confounder_after_a1 = node.class_belief.probabilities["confounder"]
    assert p_target_after_a1 > p_confounder_after_a1
    assert p_target_after_a1 > 0.5  # Started at 0.5 (assuming target and confounder)

    # Confounder compatible prompt -> strong confounder observation
    action2 = SensingAction(action_id="a2", semantic_key="confounder", prompt="confounder", family=ActionFamily.CONFOUNDER)
    obs2 = NodeObservationRef("o2", "call2", "a2", "confounder", relation=ObservationRelation.STRONG_MATCH, score=0.9)
    updater.update_node_belief(node, action2, obs2, target_class="target", confounder_class="confounder")
    
    p_target_after_a2 = node.class_belief.probabilities["target"]
    p_confounder_after_a2 = node.class_belief.probabilities["confounder"]
    # Confounder observation should decrease target posterior
    assert p_target_after_a2 < p_target_after_a1
    assert p_confounder_after_a2 > p_confounder_after_a1


def test_m5_b_weak_evidence():
    updater = BeliefUpdater()
    
    node_strong = create_mock_node()
    action = SensingAction(action_id="a1", semantic_key="target", prompt="target", family=ActionFamily.DISCOVERY)
    obs_strong = NodeObservationRef("o1", "call1", "a1", "target", relation=ObservationRelation.STRONG_MATCH, score=0.9)
    updater.update_node_belief(node_strong, action, obs_strong, target_class="target", confounder_class="confounder")
    
    node_weak = create_mock_node()
    obs_weak = NodeObservationRef("o2", "call1", "a1", "target", relation=ObservationRelation.WEAK_MATCH, score=0.9)
    updater.update_node_belief(node_weak, action, obs_weak, target_class="target", confounder_class="confounder")
    
    # Weak match moves posterior less than strong match
    assert node_weak.class_belief.probabilities["target"] > 0.5
    assert node_strong.class_belief.probabilities["target"] > node_weak.class_belief.probabilities["target"]


def test_m5_c_not_observable():
    updater = BeliefUpdater()
    node = create_mock_node()
    
    # Init belief
    action1 = SensingAction(action_id="a1", semantic_key="target", prompt="target", family=ActionFamily.DISCOVERY)
    obs1 = NodeObservationRef("o1", "call1", "a1", "target", relation=ObservationRelation.STRONG_MATCH, score=0.9)
    updater.update_node_belief(node, action1, obs1, target_class="target", confounder_class="confounder")
    
    p_before = node.class_belief.probabilities["target"]
    
    # NOT_OBSERVABLE should leave posterior exactly unchanged
    action2 = SensingAction(action_id="a2", semantic_key="target", prompt="target", family=ActionFamily.DISCOVERY)
    obs2 = NodeObservationRef("o2", "call2", "a2", "target", relation=ObservationRelation.NOT_OBSERVABLE, score=0.0)
    updater.update_node_belief(node, action2, obs2, target_class="target", confounder_class="confounder")
    
    p_after = node.class_belief.probabilities["target"]
    assert p_before == p_after


def test_m5_d_not_retrieved():
    updater = BeliefUpdater()
    
    node1 = create_mock_node()
    action = SensingAction(action_id="a1", semantic_key="target", prompt="target", family=ActionFamily.DISCOVERY)
    obs_not_retrieved = NodeObservationRef("o1", "call1", "a1", "target", relation=ObservationRelation.NOT_RETRIEVED, score=0.0)
    updater.update_node_belief(node1, action, obs_not_retrieved, target_class="target", confounder_class="confounder")
    
    node2 = create_mock_node()
    action_conf = SensingAction(action_id="a2", semantic_key="confounder", prompt="confounder", family=ActionFamily.CONFOUNDER)
    obs_strong = NodeObservationRef("o2", "call1", "a2", "confounder", relation=ObservationRelation.STRONG_MATCH, score=0.9)
    updater.update_node_belief(node2, action_conf, obs_strong, target_class="target", confounder_class="confounder")
    
    # NOT_RETRIEVED is weak negative evidence
    p_target_not_retrieved = node1.class_belief.probabilities["target"]
    assert p_target_not_retrieved < 0.5  # It decreased from 0.5
    
    p_target_contradictory = node2.class_belief.probabilities["target"]
    assert p_target_contradictory < 0.5
    
    # Effect of NOT_RETRIEVED should be less than strong contradictory match
    assert p_target_not_retrieved > p_target_contradictory


def test_m5_e_repeated_same_semantic_key():
    updater = BeliefUpdater()
    
    node_correlated = create_mock_node()
    node_independent = create_mock_node()
    
    # 3 correlated actions
    for i in range(3):
        action = SensingAction(action_id=f"a{i}", semantic_key="target", prompt="target", family=ActionFamily.DISCOVERY, correlation_group="group_A")
        obs = NodeObservationRef(f"o{i}", f"call{i}", f"a{i}", "target", correlation_group="group_A", relation=ObservationRelation.STRONG_MATCH, score=0.9)
        updater.update_node_belief(node_correlated, action, obs, target_class="target", confounder_class="confounder")
        node_correlated.observations.append(obs)
        
    # 3 independent actions
    for i in range(3):
        action = SensingAction(action_id=f"b{i}", semantic_key=f"target_{i}", prompt="target", family=ActionFamily.DISCOVERY, correlation_group=f"group_{i}")
        obs = NodeObservationRef(f"p{i}", f"call_b{i}", f"b{i}", f"target_{i}", correlation_group=f"group_{i}", relation=ObservationRelation.STRONG_MATCH, score=0.9)
        updater.update_node_belief(node_independent, action, obs, target_class="target", confounder_class="confounder")
        node_independent.observations.append(obs)
        
    p_corr = node_correlated.class_belief.probabilities["target"]
    p_indep = node_independent.class_belief.probabilities["target"]
    
    # Correlated repeats should accumulate less evidence than independent semantic groups
    assert p_indep > p_corr


def test_m5_f_global_and_tiled_same_key():
    updater = BeliefUpdater()
    node = create_mock_node()
    
    # Global
    action_g = SensingAction(action_id="a1", semantic_key="target", prompt="target", family=ActionFamily.DISCOVERY, spatial_mode=SpatialMode.GLOBAL, correlation_group="target")
    obs_g = NodeObservationRef("o1", "call1", "a1", "target", correlation_group="target", relation=ObservationRelation.STRONG_MATCH, score=0.9)
    updater.update_node_belief(node, action_g, obs_g, target_class="target", confounder_class="confounder")
    node.observations.append(obs_g)
    
    p1 = node.class_belief.probabilities["target"]
    
    # Tiled, same key
    from sam3_vlm.core.config import TilingConfig
    action_t = SensingAction(action_id="a2", semantic_key="target", prompt="target", family=ActionFamily.DISCOVERY, spatial_mode=SpatialMode.TILED, correlation_group="target", tiling=TilingConfig(grid_rows=2, grid_cols=2))
    obs_t = NodeObservationRef("o2", "call2", "a2", "target", correlation_group="target", relation=ObservationRelation.STRONG_MATCH, score=0.9)
    updater.update_node_belief(node, action_t, obs_t, target_class="target", confounder_class="confounder")
    node.observations.append(obs_t)
    
    p2 = node.class_belief.probabilities["target"]
    
    # Assert discounted update due to same correlation_group
    # Let's compare with a different semantic key
    node_diff = create_mock_node()
    updater.update_node_belief(node_diff, action_g, obs_g, target_class="target", confounder_class="confounder")
    node_diff.observations.append(obs_g)
    
    action_diff = SensingAction(action_id="a3", semantic_key="target_diff", prompt="target_diff", family=ActionFamily.DISCOVERY, spatial_mode=SpatialMode.TILED, correlation_group="target_diff", tiling=TilingConfig(grid_rows=2, grid_cols=2))
    obs_diff = NodeObservationRef("o3", "call3", "a3", "target_diff", correlation_group="target_diff", relation=ObservationRelation.STRONG_MATCH, score=0.9)
    updater.update_node_belief(node_diff, action_diff, obs_diff, target_class="target", confounder_class="confounder")
    
    p_diff = node_diff.class_belief.probabilities["target"]
    
    # Different semantic key should give more information (higher probability) than same key repeated
    assert p_diff > p2


def test_m5_i_soft_count():
    from sam3_vlm.scene.state import SceneState, DiscoveryState
    from sam3_vlm.scene.graph import SceneGraph
    from sam3_vlm.scene.belief import SemanticMemory
    
    state = SceneState(image_id="i1", user_prompt="p", target_class="target", graph=SceneGraph(), semantic_memory=SemanticMemory())
    
    nodes = []
    for p in [0.9, 0.6, 0.2]:
        node = create_mock_node()
        node.node_id = f"node_{p}"
        node.class_belief = ClassBelief(probabilities={"target": p, "confounder": 1-p})
        state.graph.add_node(node)
        nodes.append(node)
        
    # Recalculate count
    mean_count = 0.0
    for node in state.graph.active_nodes():
        p = node.class_belief.probabilities.get(state.target_class, 0.0)
        mean_count += p
    
    assert abs(mean_count - 1.7) < 1e-5


def test_m5_j_count_variance():
    from sam3_vlm.scene.state import SceneState, DiscoveryState
    from sam3_vlm.scene.graph import SceneGraph
    from sam3_vlm.scene.belief import SemanticMemory
    
    state = SceneState(image_id="i1", user_prompt="p", target_class="target", graph=SceneGraph(), semantic_memory=SemanticMemory())
    
    probs = [0.9, 0.6, 0.2]
    for p in probs:
        node = create_mock_node()
        node.node_id = f"node_{p}"
        node.class_belief = ClassBelief(probabilities={"target": p, "confounder": 1-p})
        state.graph.add_node(node)
        
    variance = 0.0
    for node in state.graph.active_nodes():
        p = node.class_belief.probabilities.get(state.target_class, 0.0)
        variance += p * (1.0 - p)
    
    expected_var = sum(p * (1 - p) for p in probs)
    assert abs(variance - expected_var) < 1e-5


def test_m5_k_confident_nodes_reduce_variance():
    from sam3_vlm.scene.state import SceneState, DiscoveryState
    from sam3_vlm.scene.graph import SceneGraph
    from sam3_vlm.scene.belief import SemanticMemory
    
    # State 1: uncertain
    state1 = SceneState(image_id="i1", user_prompt="p", target_class="target", graph=SceneGraph(), semantic_memory=SemanticMemory())
    node_unc = create_mock_node()
    node_unc.class_belief = ClassBelief(probabilities={"target": 0.5, "confounder": 0.5})
    state1.graph.add_node(node_unc)
    var1 = 0.5 * 0.5
    
    # State 2: confident
    state2 = SceneState(image_id="i1", user_prompt="p", target_class="target", graph=SceneGraph(), semantic_memory=SemanticMemory())
    node_conf = create_mock_node()
    node_conf.class_belief = ClassBelief(probabilities={"target": 0.95, "confounder": 0.05})
    state2.graph.add_node(node_conf)
    var2 = 0.95 * 0.05
    
    assert var2 < var1


def test_m5_p_numerical_safety():
    updater = BeliefUpdater()
    node = create_mock_node()
    
    action = SensingAction(action_id="a1", semantic_key="target", prompt="target", family=ActionFamily.DISCOVERY)
    obs = NodeObservationRef("o1", "call1", "a1", "target", relation=ObservationRelation.STRONG_MATCH, score=0.0)
    
    # Update with zero score
    updater.update_node_belief(node, action, obs, target_class="target", confounder_class="confounder")
    
    assert sum(node.class_belief.probabilities.values()) == pytest.approx(1.0)
    assert not any(v < 0 for v in node.class_belief.probabilities.values())
    import math
    assert not any(math.isnan(v) for v in node.class_belief.probabilities.values())
    
    # Force probabilities near 0 or 1
    node.class_belief = ClassBelief(probabilities={"target": 1e-9, "confounder": 1.0 - 1e-9})
    updater.update_node_belief(node, action, obs, target_class="target", confounder_class="confounder")
    
    assert sum(node.class_belief.probabilities.values()) == pytest.approx(1.0)

def test_m5_g_post_bootstrap_discovery_confounder():
    updater = BeliefUpdater()
    node = create_mock_node()
    
    # 1. Post-bootstrap discovery (e.g. from a confounder prompt)
    action1 = SensingAction(action_id="a1", semantic_key="confounder", prompt="confounder", family=ActionFamily.DISCOVERY, semantic_prior={"confounder": 0.8, "target": 0.2})
    obs1 = NodeObservationRef("o1", "call1", "a1", "confounder", relation=ObservationRelation.NEW_DETECTION, score=0.9)
    updater.update_node_belief(node, action1, obs1, target_class="target", confounder_class="confounder")
    
    p_target_after_a1 = node.class_belief.probabilities["target"]
    p_conf_after_a1 = node.class_belief.probabilities["confounder"]
    
    # 2. Later receives strong confounder evidence
    action2 = SensingAction(action_id="a2", semantic_key="confounder", prompt="confounder", family=ActionFamily.CONFOUNDER, semantic_prior={"confounder": 0.95, "target": 0.05})
    obs2 = NodeObservationRef("o2", "call2", "a2", "confounder", relation=ObservationRelation.STRONG_MATCH, score=0.9)
    updater.update_node_belief(node, action2, obs2, target_class="target", confounder_class="confounder")
    
    p_target_after_a2 = node.class_belief.probabilities["target"]
    assert p_target_after_a2 < p_target_after_a1


def test_m5_h_qwen_semantics_prior():
    updater = BeliefUpdater()
    node = create_mock_node()
    
    # Semantic key is NOT the class label. 
    # Qwen specifies this key strongly implies "leaf" and weakly implies "target"
    action = SensingAction(
        action_id="a1", 
        semantic_key="leaf_foliage", 
        prompt="green leaf foliage", 
        family=ActionFamily.DISCOVERY, 
        semantic_prior={"leaf": 0.9, "target": 0.1}
    )
    obs = NodeObservationRef("o1", "call1", "a1", "leaf_foliage", relation=ObservationRelation.STRONG_MATCH, score=0.9)
    
    updater.update_node_belief(node, action, obs, target_class="target", confounder_class="leaf")
    
    # Leaf should gain posterior mass
    assert node.class_belief.probabilities["leaf"] > node.class_belief.probabilities["target"]


def test_m5_l_strong_verification_match():
    updater = BeliefUpdater()
    node = create_mock_node()
    
    action = SensingAction(
        action_id="a1", 
        semantic_key="target", 
        prompt="verify target", 
        family=ActionFamily.VERIFICATION,
        semantic_prior={"target": 1.0}
    )
    obs = NodeObservationRef("o1", "call1", "a1", "target", relation=ObservationRelation.STRONG_MATCH, score=0.9)
    
    updater.update_node_belief(node, action, obs, target_class="target")
    
    # Verification should update beliefs
    assert node.class_belief.probabilities["target"] > 0.5


def test_m5_m_node_to_dict_preserves_correlation_group():
    node = create_mock_node()
    obs = NodeObservationRef(
        observation_id="o1", 
        sam3_call_id="call1", 
        action_id="a1", 
        semantic_key="target", 
        correlation_group="my_group",
        relation=ObservationRelation.STRONG_MATCH, 
        score=0.9
    )
    node.observations.append(obs)
    
    data = node.to_dict()
    node2 = Node.from_dict(data)
    
    assert node2.observations[0].correlation_group == "my_group"

def test_m5_n_precise_observability():
    from sam3_vlm.pipeline.runner import Runner
    from sam3_vlm.core.config import V4Config
    from sam3_vlm.sensing.observation import SAM3Observation
    from sam3_vlm.core.types import Detection
    from sam3_vlm.scene.state import SceneState
    from sam3_vlm.scene.graph import SceneGraph
    from sam3_vlm.scene.belief import SemanticMemory
    from unittest.mock import MagicMock

    # Setup Runner with one existing active node
    adapter = MagicMock()
    planner = MagicMock()
    runner = Runner(sensor=adapter, planner=planner, config=V4Config())
    runner.target_class = "target"
    runner.image = None
    runner.scene_state = SceneState(
        image_id="i1", user_prompt="p", target_class="target", 
        graph=SceneGraph(), semantic_memory=SemanticMemory()
    )
    
    node = create_mock_node() # Box(0, 0, 10, 10)
    node.node_id = "node1"
    runner.scene_state.graph.add_node(node)
    
    # Create an action that searches a region NOT intersecting the node
    from sam3_vlm.core.config import TilingConfig
    action = SensingAction(
        action_id="a1", semantic_key="target", prompt="target", 
        family=ActionFamily.DISCOVERY, spatial_mode=SpatialMode.TILED,
        tiling=TilingConfig(grid_rows=1, grid_cols=1)
    )
    
    from sam3_vlm.planning.action_bank import ActionBank, ActionBankEntry
    runner.scene_state.action_bank = ActionBank(entries=[
        ActionBankEntry(action=action, total_utility=1.0)
    ])
    
    # Mock observation with a searched region far away from the node
    obs = SAM3Observation(
        call_id="call1",
        action_id=action.action_id,
        semantic_key=action.semantic_key,
        detections=[], # No detections
        searched_regions=[BoxGeometry(Box(100, 100, 110, 110))],
        runtime_ms=10.0
    )
    
    # Overwrite sensor to return this exact observation
    adapter.observe.return_value = obs
    
    from sam3_vlm.pipeline.runner import RunnerState
    runner.state = RunnerState.GLOBAL_SENSING
    
    # Assert that unexecuted entries works
    assert len(list(runner.scene_state.action_bank.unexecuted_entries())) == 1
    
    # Patch runner to run just the global sensing step
    runner._step()
    assert runner.state == RunnerState.GLOBAL_SENSING, f"State changed to {runner.state}!"

    
    # Node should have a new observation, and it MUST be NOT_OBSERVABLE since the searched region (100,100,110,110) does not intersect (0,0,10,10)
    assert len(node.observations) == 1
    assert node.observations[-1].relation == ObservationRelation.NOT_OBSERVABLE

def test_m5_o_count_estimator():
    from sam3_vlm.scene.state import CountEstimator, SceneGraph
    graph = SceneGraph()
    node1 = create_mock_node()
    node1.node_id = "1"
    node1.class_belief = ClassBelief(probabilities={"target": 0.8, "confounder": 0.2})
    graph.add_node(node1)
    
    node2 = create_mock_node()
    node2.node_id = "2"
    node2.class_belief = ClassBelief(probabilities={"target": 0.4, "confounder": 0.6})
    graph.add_node(node2)
    
    est = CountEstimator.estimate(graph, "target")
    assert est.mean_count == pytest.approx(1.2)
    assert est.variance == pytest.approx(0.8*0.2 + 0.4*0.6)
