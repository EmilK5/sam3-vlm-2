"""Regression tests for M8 action-bank and graph-expansion semantics."""

from types import SimpleNamespace

import pytest

from sam3_vlm.core.config import V4Config
from sam3_vlm.core.geometry import Box, BoxGeometry
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.core.types import (
    ActionFamily,
    ActionSource,
    ClassBelief,
    SpatialMode,
)
from sam3_vlm.models.sam3 import MockSAM3Adapter
from sam3_vlm.pipeline.runner import Runner
from sam3_vlm.planning.action_bank import (
    ActionBank,
    ActionBankGenerator,
    ActionRejectionReason,
)
from sam3_vlm.planning.qwen_planner import PlannerOutput, ProposedAction
from sam3_vlm.scene.belief import SemanticMemory
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.node import Node
from sam3_vlm.scene.state import SceneState
from sam3_vlm.sensing.action import SensingAction


class _NoopPlanner:
    model = "noop-qwen"

    def plan_scene(self, evidence, budget, config):
        return PlannerOutput(scene_summary="noop", proposed_actions=[])


def _runner_with_empty_state() -> Runner:
    runner = Runner(V4Config(), MockSAM3Adapter(), _NoopPlanner())
    runner.scene_state = SceneState(
        image_id="img",
        user_prompt="green fruit",
        target_class="target",
        graph=SceneGraph(),
        semantic_memory=SemanticMemory(),
    )
    runner.target_class = "target"
    return runner


def _node(node_id: str) -> Node:
    return Node(
        node_id=node_id,
        geometry=BoxGeometry(Box(10.0, 10.0, 30.0, 30.0)),
        class_belief=ClassBelief(),
    )


def test_same_semantic_key_different_prompts_are_valid_distinct_actions():
    """`target` is a belief coordinate, not a one-action-only identifier."""
    cfg = V4Config()
    bank = ActionBank()
    generator = ActionBankGenerator()

    output = PlannerOutput(
        proposed_actions=[
            ProposedAction(
                semantic_key="target",
                prompt="shadowed green fruit",
                family=ActionFamily.DISCOVERY,
                priority=0.8,
                semantic_prior={"target": 0.9},
                suggested_threshold=0.5,
                suggested_spatial_mode=SpatialMode.TILED,
            ),
            ProposedAction(
                semantic_key="target",
                prompt="round green fruit",
                family=ActionFamily.DISCOVERY,
                priority=0.9,
                semantic_prior={"target": 0.9},
                suggested_threshold=0.5,
                suggested_spatial_mode=SpatialMode.TILED,
            ),
        ]
    )

    entries = generator.generate_entries(
        output,
        SemanticMemory(),
        bank,
        IDGenerator(),
        config=cfg,
        enforce_qwen_contract=True,
        allowed_belief_classes=["target", "confounder1", "confounder2"],
    )

    assert len(entries) == 2
    assert [entry.action.semantic_key for entry in entries] == ["target", "target"]
    assert [entry.action.prompt for entry in entries] == [
        "shadowed green fruit",
        "round green fruit",
    ]
    assert not generator.last_rejections

    # Same semantic coordinate => correlated evidence, not hard rejection.
    assert entries[1].redundancy >= 0.5


def test_exact_prompt_duplicate_is_still_rejected():
    """Removing same-key rejection must not permit an exact repeated sensor query."""
    cfg = V4Config()
    bank = ActionBank()
    generator = ActionBankGenerator()

    output = PlannerOutput(
        proposed_actions=[
            ProposedAction(
                semantic_key="target",
                prompt="round green fruit",
                family=ActionFamily.DISCOVERY,
                semantic_prior={"target": 0.9},
                suggested_spatial_mode=SpatialMode.TILED,
            ),
            ProposedAction(
                semantic_key="target",
                prompt="round green fruit",
                family=ActionFamily.DISCOVERY,
                semantic_prior={"target": 0.9},
                suggested_spatial_mode=SpatialMode.TILED,
            ),
        ]
    )

    entries = generator.generate_entries(
        output,
        SemanticMemory(),
        bank,
        IDGenerator(),
        config=cfg,
        enforce_qwen_contract=True,
        allowed_belief_classes=["target", "confounder1", "confounder2"],
    )

    assert len(entries) == 1
    assert len(generator.last_rejections) == 1
    rejection = generator.last_rejections[0]
    assert rejection.reason == ActionRejectionReason.DUPLICATE_SEMANTIC_KEY.value
    assert "exact SAM3 prompt" in rejection.detail


@pytest.mark.parametrize(
    "family,prompt,semantic_key",
    [
        (ActionFamily.CONFOUNDER, "green leaf", "confounder1"),
        (ActionFamily.VERIFICATION, "round green fruit", "target"),
        (ActionFamily.CONTEXT, "tree canopy", "locked_context_region"),
    ],
)
def test_non_discovery_unmatched_detections_do_not_expand_graph(
    family,
    prompt,
    semantic_key,
):
    """Association's provisional nodes are rolled back for discriminative actions."""
    runner = _runner_with_empty_state()
    provisional = _node("node_provisional")
    runner.scene_state.graph.add_node(provisional)

    action = SensingAction(
        action_id="action_test",
        semantic_key=semantic_key,
        prompt=prompt,
        family=family,
        spatial_mode=SpatialMode.GLOBAL,
        source=ActionSource.QWEN,
        semantic_prior={"confounder1": 1.0} if family == ActionFamily.CONFOUNDER else None,
    )
    assoc_result = SimpleNamespace(
        matched_observations=[],
        new_nodes=[provisional],
    )
    observation = SimpleNamespace(
        call_id="sam3_test",
        searched_regions=[],
    )

    new_nodes_count, not_retrieved_count = runner._project_observations(
        action,
        observation,
        assoc_result,
    )

    assert new_nodes_count == 0
    assert not_retrieved_count == 0
    assert runner.scene_state.graph.get_node("node_provisional") is None
    assert len(runner.scene_state.graph.nodes) == 0


def test_discovery_action_keeps_provisional_unmatched_nodes():
    """Discovery remains the only family allowed to enlarge graph dimensionality."""
    runner = _runner_with_empty_state()
    provisional = _node("node_discovered")
    runner.scene_state.graph.add_node(provisional)

    action = SensingAction(
        action_id="action_discovery",
        semantic_key="target",
        prompt="round green fruit",
        family=ActionFamily.DISCOVERY,
        spatial_mode=SpatialMode.GLOBAL,
        source=ActionSource.QWEN,
        semantic_prior={"target": 1.0},
    )
    assoc_result = SimpleNamespace(
        matched_observations=[],
        new_nodes=[provisional],
    )

    admitted = runner._admitted_new_nodes_for_action(action, assoc_result)

    assert admitted == [provisional]
    assert runner.scene_state.graph.get_node("node_discovered") is provisional

