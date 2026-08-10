"""Behavioral unit tests required by V4 Design Spec §18.2 / §36."""

from dataclasses import FrozenInstanceError
import pytest

from sam3_vlm.core.config import (
    ActionSelectionConfig,
    AssociationConfig,
    BeliefConfig,
    BootstrapConfig,
    BudgetConfig,
    CleanupConfig,
    LoggingConfig,
    PlannerConfig,
    ReplanningConfig,
    SAM3Config,
    StoppingConfig,
    TilingConfig,
    V4Config,
)
from sam3_vlm.core.geometry import Box, BoxGeometry
from sam3_vlm.core.types import (
    ActionFamily,
    ActionSource,
    BudgetState,
    ClassBelief,
    NodeStatus,
    SpatialMode,
)
from sam3_vlm.models.sam3 import DummySAM3Sensor
from sam3_vlm.planning.action_bank import ActionBank, ActionBankEntry
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.node import Node
from sam3_vlm.scene.state import CoverageSummary, DiscoveryState, SceneState
from sam3_vlm.scene.belief import SemanticMemory
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.sensing.observation import SAM3Observation


# 1. ActionSource exact values (Spec §7.2)
def test_action_source_exact_values():
    assert set(ActionSource.__members__.keys()) == {
        "USER_BOOTSTRAP",
        "QWEN",
        "CONTROLLER",
        "CLEANUP",
    }
    assert ActionSource.USER_BOOTSTRAP.value == "USER_BOOTSTRAP"
    assert ActionSource.QWEN.value == "QWEN"
    assert ActionSource.CONTROLLER.value == "CONTROLLER"
    assert ActionSource.CLEANUP.value == "CLEANUP"


# 2. ClassBelief invariants (Spec §21.5)
def test_class_belief_sum_to_one():
    cb = ClassBelief(probabilities={"target": 0.7, "confounder": 0.3})
    assert cb.probabilities["target"] == 0.7

    # Non sum-to-one should raise
    with pytest.raises(ValueError, match="sum to 1.0"):
        ClassBelief(probabilities={"target": 0.8, "confounder": 0.5})


def test_class_belief_finiteness_and_nonnegativity():
    # Negative probability
    with pytest.raises(ValueError, match="cannot be negative"):
        ClassBelief(probabilities={"target": -0.1, "confounder": 1.1})

    # NaN / Inf probability
    with pytest.raises(ValueError, match="must be finite"):
        ClassBelief(probabilities={"target": float("nan"), "confounder": 1.0})

    # Empty probabilities is valid
    empty_cb = ClassBelief()
    assert len(empty_cb.probabilities) == 0


# 3. DummySAM3Sensor call counting and observation generation (Spec §4)
def test_dummy_sam3_sensor_execution():
    sensor = DummySAM3Sensor(call_id_prefix="sam3")
    action = SensingAction(
        action_id="act_000001",
        semantic_key="green_citrus",
        prompt="green citrus fruit",
        family=ActionFamily.DISCOVERY,
    )

    obs1 = sensor.observe(image=None, action=action)
    assert sensor.call_count == 1
    assert obs1.call_id == "sam3_000001"
    assert obs1.action_id == "act_000001"
    assert isinstance(obs1, SAM3Observation)

    obs2 = sensor.observe(image=None, action=action)
    assert sensor.call_count == 2
    assert obs2.call_id == "sam3_000002"


# 4. Sensor non-mutation invariant (Spec §4.3)
def test_sensor_does_not_mutate_scene_graph():
    sensor = DummySAM3Sensor()
    graph = SceneGraph()
    initial_node_count = len(graph.nodes)

    action = SensingAction(
        action_id="act_001",
        semantic_key="fruit",
        prompt="citrus fruit",
        family=ActionFamily.DISCOVERY,
    )

    obs = sensor.observe(image=None, action=action)

    # Sensor returns observations only; graph is unchanged
    assert len(graph.nodes) == initial_node_count
    assert isinstance(obs, SAM3Observation)


# 5. ActionBank entry lifecycle & invalid action handling (Spec §7 / §24.1)
def test_action_bank_lifecycle():
    bank = ActionBank()

    valid_action = SensingAction(
        action_id="act_001",
        semantic_key="fruit",
        prompt="citrus fruit",
        family=ActionFamily.DISCOVERY,
    )
    entry1 = bank.add_action(valid_action, qwen_priority=0.9)
    assert entry1 is not None
    assert entry1.executed is False

    # Pop next marks executed
    popped = bank.pop_next()
    assert popped is not None
    assert popped.action.action_id == "act_001"
    assert popped.executed is True

    # No more unexecuted entries
    assert bank.pop_next() is None
    assert len(bank.executed_entries()) == 1
    assert len(bank.unexecuted_entries()) == 0


def test_action_bank_invalid_action_rejection():
    bank = ActionBank()

    # Invalid action (empty prompt)
    invalid_action = SensingAction(
        action_id="act_002",
        semantic_key="fruit",
        prompt="",
        family=ActionFamily.DISCOVERY,
    )
    entry = bank.add_action(invalid_action)
    assert entry is None
    assert len(bank.entries) == 1
    assert bank.entries[0].invalid_reason is not None

    # Invalid action cannot be popped for execution
    assert bank.pop_next() is None


# 6. SceneGraph.active_nodes() filtering (Spec §3.3)
def test_scene_graph_active_nodes_filtering():
    graph = SceneGraph()
    geom = BoxGeometry(Box(10.0, 10.0, 50.0, 50.0))

    n1 = Node(node_id="n1", geometry=geom, status=NodeStatus.ACTIVE)
    n2 = Node(node_id="n2", geometry=geom, status=NodeStatus.RESOLVED)
    n3 = Node(node_id="n3", geometry=geom, status=NodeStatus.REJECTED)
    n4 = Node(node_id="n4", geometry=geom, status=NodeStatus.ACTIVE)

    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    graph.add_node(n4)

    active = graph.active_nodes()
    assert len(active) == 2
    assert {n.node_id for n in active} == {"n1", "n4"}


# 7. BudgetState counter independence (Spec §15)
def test_budget_state_counter_independence():
    budget = BudgetState()

    budget.sam3_calls += 1
    assert budget.sam3_calls == 1
    assert budget.qwen_calls == 0
    assert budget.sam3_tiles == 0
    assert budget.model_runtime_ms == 0.0

    budget.qwen_calls += 2
    assert budget.qwen_calls == 2
    assert budget.sam3_calls == 1


# 8. SensingAction.validate() tiling & exemplar checks (Spec §21.7)
def test_sensing_action_tiling_mode_mismatch():
    tiling_cfg = TilingConfig()

    # Mode TILED without tiling config raises
    with pytest.raises(ValueError, match="must specify tiling configuration"):
        action1 = SensingAction(
            action_id="a1",
            semantic_key="k1",
            prompt="p1",
            family=ActionFamily.DISCOVERY,
            spatial_mode=SpatialMode.TILED,
            tiling=None,
        )
        action1.validate()

    # Tiling config specified but mode GLOBAL raises
    with pytest.raises(ValueError, match="must use TILED spatial_mode"):
        action2 = SensingAction(
            action_id="a2",
            semantic_key="k2",
            prompt="p2",
            family=ActionFamily.DISCOVERY,
            spatial_mode=SpatialMode.GLOBAL,
            tiling=tiling_cfg,
        )
        action2.validate()


# 9. V4Config sub-configs & immutability (Spec §31)
def test_v4_config_subconfigs_and_immutability():
    cfg = V4Config()

    assert isinstance(cfg.tiling, TilingConfig)
    assert isinstance(cfg.budget, BudgetConfig)
    assert isinstance(cfg.stopping, StoppingConfig)
    assert isinstance(cfg.bootstrap, BootstrapConfig)
    assert isinstance(cfg.planner, PlannerConfig)
    assert isinstance(cfg.sam3, SAM3Config)
    assert isinstance(cfg.action_selection, ActionSelectionConfig)
    assert isinstance(cfg.association, AssociationConfig)
    assert isinstance(cfg.belief, BeliefConfig)
    assert isinstance(cfg.replanning, ReplanningConfig)
    assert isinstance(cfg.cleanup, CleanupConfig)
    assert isinstance(cfg.logging, LoggingConfig)

    # Immutability check
    with pytest.raises(FrozenInstanceError):
        cfg.device = "cpu"  # type: ignore


# 10. SceneState structure (Spec §3.2)
def test_scene_state_structure():
    graph = SceneGraph()
    mem = SemanticMemory()
    disc = DiscoveryState()

    state = SceneState(
        image_id="img_001",
        user_prompt="count green citrus",
        target_class="green_citrus",
        graph=graph,
        semantic_memory=mem,
        discovery_state=disc,
    )

    assert state.image_id == "img_001"
    assert state.target_class == "green_citrus"
    assert state.iteration == 0
