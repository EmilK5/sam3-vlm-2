import pytest
from sam3_vlm.core.types import ActionFamily, SpatialMode
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.planning.action_bank import ActionBankEntry
from sam3_vlm.planning.utility import DefaultUtilityEvaluator
from sam3_vlm.core.config import V4Config, ActionSelectionConfig
from sam3_vlm.scene.state import SceneState
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.belief import SemanticMemory


def create_mock_state_and_config(iteration=0):
    cfg = V4Config(
        action_selection=ActionSelectionConfig(
            alpha_discovery=1.0, beta_discrimination=1.0, gamma_redundancy=1.0, lambda_cost=0.5, eta_qwen_priority=1.0
        )
    )
    state = SceneState(
        image_id="img1", user_prompt="test", target_class="t",
        graph=SceneGraph(), semantic_memory=SemanticMemory()
    )
    state.iteration = iteration
    return state, cfg


def test_default_utility_evaluator_discovery():
    evaluator = DefaultUtilityEvaluator()
    
    action = SensingAction(
        action_id="a1",
        semantic_key="test_target",
        prompt="test target",
        family=ActionFamily.DISCOVERY,
        spatial_mode=SpatialMode.GLOBAL,
    )
    entry = ActionBankEntry(action=action, qwen_priority=0.8, redundancy=0.0)
    
    # Iteration 0
    state, config = create_mock_state_and_config(iteration=0)
    breakdown_it0 = evaluator.evaluate_utility(entry, state, config)
    assert breakdown_it0.discovery_value == 1.0
    assert breakdown_it0.discrimination_value == 0.0
    assert breakdown_it0.compute_cost == 1.0
    # Utility = 1.0(1.0) + 1.0(0.0) - 1.0(0.0) - 0.5(1.0) + 1.0(0.8) = 1.0 - 0.5 + 0.8 = 1.3
    assert breakdown_it0.total_utility == 1.3
    
    # Iteration 5 (decaying discovery)
    state, config = create_mock_state_and_config(iteration=5)
    breakdown_it5 = evaluator.evaluate_utility(entry, state, config)
    assert breakdown_it5.discovery_value == 0.5
    # Utility = 1.0(0.5) + 1.0(0.0) - 1.0(0.0) - 0.5(1.0) + 1.0(0.8) = 0.5 - 0.5 + 0.8 = 0.8
    assert breakdown_it5.total_utility == 0.8


def test_default_utility_evaluator_discrimination():
    evaluator = DefaultUtilityEvaluator()
    
    action = SensingAction(
        action_id="a2",
        semantic_key="test_confounder",
        prompt="test confounder",
        family=ActionFamily.CONFOUNDER,
        spatial_mode=SpatialMode.GLOBAL,
    )
    entry = ActionBankEntry(action=action, qwen_priority=0.5, redundancy=0.2)
    
    # Iteration 0
    state, config = create_mock_state_and_config(iteration=0)
    breakdown_it0 = evaluator.evaluate_utility(entry, state, config)
    assert breakdown_it0.discovery_value == 0.0
    assert breakdown_it0.discrimination_value == 0.5
    assert breakdown_it0.redundancy_cost == 0.2
    assert breakdown_it0.compute_cost == 1.0
    # Utility = 1.0(0.0) + 1.0(0.5) - 1.0(0.2) - 0.5(1.0) + 1.0(0.5) = 0.5 - 0.2 - 0.5 + 0.5 = 0.3
    assert breakdown_it0.total_utility == 0.3
    
    # Iteration 3 (increasing discrimination)
    state, config = create_mock_state_and_config(iteration=3)
    breakdown_it3 = evaluator.evaluate_utility(entry, state, config)
    assert breakdown_it3.discrimination_value == 0.8
    # Utility = 1.0(0.0) + 1.0(0.8) - 1.0(0.2) - 0.5(1.0) + 1.0(0.5) = 0.8 - 0.2 - 0.5 + 0.5 = 0.6
    assert breakdown_it3.total_utility == pytest.approx(0.6)


def test_default_utility_evaluator_tiled_cost():
    evaluator = DefaultUtilityEvaluator()
    
    action = SensingAction(
        action_id="a3",
        semantic_key="test_tiled",
        prompt="test tiled",
        family=ActionFamily.DISCOVERY,
        spatial_mode=SpatialMode.TILED,
    )
    entry = ActionBankEntry(action=action, qwen_priority=0.8, redundancy=0.0)
    
    state, config = create_mock_state_and_config(iteration=0)
    breakdown = evaluator.evaluate_utility(entry, state, config)
    assert breakdown.compute_cost == 4.0
    # Utility = 1.0(1.0) + 1.0(0.0) - 1.0(0.0) - 0.5(4.0) + 1.0(0.8) = 1.0 - 2.0 + 0.8 = -0.2
    assert breakdown.total_utility == pytest.approx(-0.2)
