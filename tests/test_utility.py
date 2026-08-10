import pytest
from sam3_vlm.core.types import ActionFamily, SpatialMode
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.planning.action_bank import ActionBankEntry
from sam3_vlm.planning.utility import DefaultUtilityEvaluator


def test_default_utility_evaluator_discovery():
    evaluator = DefaultUtilityEvaluator(alpha=1.0, beta=1.0, gamma=1.0, lambda_=0.5, eta=1.0)
    
    action = SensingAction(
        action_id="a1",
        semantic_key="test_target",
        prompt="test target",
        family=ActionFamily.DISCOVERY,
        spatial_mode=SpatialMode.GLOBAL,
    )
    entry = ActionBankEntry(action=action, qwen_priority=0.8, redundancy=0.0)
    
    # Iteration 0
    breakdown_it0 = evaluator.evaluate_utility(entry, iteration=0)
    assert breakdown_it0.discovery_value == 1.0
    assert breakdown_it0.discrimination_value == 0.0
    assert breakdown_it0.compute_cost == 1.0
    # Utility = 1.0(1.0) + 1.0(0.0) - 1.0(0.0) - 0.5(1.0) + 1.0(0.8) = 1.0 - 0.5 + 0.8 = 1.3
    assert breakdown_it0.total_utility == 1.3
    
    # Iteration 5 (decaying discovery)
    breakdown_it5 = evaluator.evaluate_utility(entry, iteration=5)
    assert breakdown_it5.discovery_value == 0.5
    # Utility = 1.0(0.5) + 1.0(0.0) - 1.0(0.0) - 0.5(1.0) + 1.0(0.8) = 0.5 - 0.5 + 0.8 = 0.8
    assert breakdown_it5.total_utility == 0.8


def test_default_utility_evaluator_discrimination():
    evaluator = DefaultUtilityEvaluator(alpha=1.0, beta=1.0, gamma=1.0, lambda_=0.5, eta=1.0)
    
    action = SensingAction(
        action_id="a2",
        semantic_key="test_confounder",
        prompt="test confounder",
        family=ActionFamily.CONFOUNDER,
        spatial_mode=SpatialMode.GLOBAL,
    )
    entry = ActionBankEntry(action=action, qwen_priority=0.5, redundancy=0.2)
    
    # Iteration 0
    breakdown_it0 = evaluator.evaluate_utility(entry, iteration=0)
    assert breakdown_it0.discovery_value == 0.0
    assert breakdown_it0.discrimination_value == 0.5
    assert breakdown_it0.redundancy_cost == 0.2
    assert breakdown_it0.compute_cost == 1.0
    # Utility = 1.0(0.0) + 1.0(0.5) - 1.0(0.2) - 0.5(1.0) + 1.0(0.5) = 0.5 - 0.2 - 0.5 + 0.5 = 0.3
    assert breakdown_it0.total_utility == 0.3
    
    # Iteration 3 (increasing discrimination)
    breakdown_it3 = evaluator.evaluate_utility(entry, iteration=3)
    assert breakdown_it3.discrimination_value == 0.8
    # Utility = 1.0(0.0) + 1.0(0.8) - 1.0(0.2) - 0.5(1.0) + 1.0(0.5) = 0.8 - 0.2 - 0.5 + 0.5 = 0.6
    assert breakdown_it3.total_utility == pytest.approx(0.6)


def test_default_utility_evaluator_tiled_cost():
    evaluator = DefaultUtilityEvaluator(alpha=1.0, beta=1.0, gamma=1.0, lambda_=0.5, eta=1.0)
    
    action = SensingAction(
        action_id="a3",
        semantic_key="test_tiled",
        prompt="test tiled",
        family=ActionFamily.DISCOVERY,
        spatial_mode=SpatialMode.TILED,
    )
    entry = ActionBankEntry(action=action, qwen_priority=0.8, redundancy=0.0)
    
    breakdown = evaluator.evaluate_utility(entry, iteration=0)
    assert breakdown.compute_cost == 4.0
    # Utility = 1.0(1.0) + 1.0(0.0) - 1.0(0.0) - 0.5(4.0) + 1.0(0.8) = 1.0 - 2.0 + 0.8 = -0.2
    assert breakdown.total_utility == pytest.approx(-0.2)
