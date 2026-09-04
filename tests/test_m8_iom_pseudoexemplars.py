"""Regression coverage for IoM deduplication and M8 pseudoexemplars."""

from dataclasses import replace

import pytest

from sam3_vlm.core.config import BootstrapConfig, V4Config
from sam3_vlm.core.geometry import Box, BoxGeometry, GeometryRef
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.core.types import ActionFamily, ClassBelief, Detection, SpatialMode
from sam3_vlm.models.sam3 import _localize_exemplar_boxes
from sam3_vlm.pipeline.bootstrap import BootstrapPipeline
from sam3_vlm.pipeline.runner import Runner
from sam3_vlm.planning.qwen_planner import PlannerOutput
from sam3_vlm.scene.association_dual import IoUIoMAssociationPolicy, box_iom
from sam3_vlm.scene.belief import SemanticMemory
from sam3_vlm.scene.exemplars import select_target_pseudoexemplars
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.node import Node
from sam3_vlm.scene.state import SceneState
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.sensing.evidence import ContactSheet
from sam3_vlm.sensing.observation import SAM3Observation


def _det(det_id, box, score=0.9):
    return Detection(
        detection_id=det_id,
        geometry=GeometryRef(Box(*box)),
        score=score,
    )


def _node(node_id, box, score, target_prob=0.5):
    node = Node(
        node_id=node_id,
        geometry=BoxGeometry(Box(*box)),
        class_belief=ClassBelief(
            probabilities={"target": target_prob, "confounder1": 1.0 - target_prob}
        ),
        created_by_call_id="sam_seed",
    )
    from sam3_vlm.core.types import NodeObservationRef, ObservationRelation
    node.observations.append(
        NodeObservationRef(
            observation_id=f"obs_{node_id}",
            sam3_call_id="sam_seed",
            action_id="a_seed",
            semantic_key="target",
            detection_id=f"det_{node_id}",
            relation=ObservationRelation.NEW_DETECTION,
            score=score,
            association_score=None,
        )
    )
    return node


def test_iom_is_one_for_nested_box_even_when_iou_is_small():
    outer = Box(0, 0, 40, 40)
    inner = Box(10, 10, 20, 20)
    assert outer.iou(inner) == pytest.approx(0.0625)
    assert box_iom(outer, inner) == pytest.approx(1.0)


def test_full_containment_deduplicates_even_when_iou_is_tiny():
    graph = SceneGraph()
    policy = IoUIoMAssociationPolicy()
    result = policy.associate(
        graph=graph,
        detections=[
            _det("large", (0, 0, 100, 100), 0.95),
            _det("tiny", (45, 45, 55, 55), 0.90),
        ],
        sam3_call_id="sam1",
        action_id="a1",
        semantic_key="target",
        id_gen=IDGenerator(),
        config=replace(V4Config().association, enable_iom_dedup=True),
    )
    # This is the failure mode IoU misses: IoU=0.01 but IoM=1.0.
    assert len(result.new_nodes) == 1
    assert len(graph.nodes) == 1


def test_same_observation_nested_boxes_create_only_one_node():
    graph = SceneGraph()
    policy = IoUIoMAssociationPolicy()
    result = policy.associate(
        graph=graph,
        detections=[
            _det("large", (0, 0, 40, 40), 0.95),
            _det("small", (10, 10, 20, 20), 0.90),
        ],
        sam3_call_id="sam_1",
        action_id="a_1",
        semantic_key="target",
        id_gen=IDGenerator(),
        config=V4Config().association,
    )
    assert len(result.new_nodes) == 1
    assert len(graph.active_nodes()) == 1


def test_cross_pass_nested_box_matches_existing_node():
    graph = SceneGraph()
    graph.add_node(_node("node_1", (0, 0, 40, 40), 0.95))
    result = IoUIoMAssociationPolicy().associate(
        graph=graph,
        detections=[_det("nested", (10, 10, 20, 20), 0.80)],
        sam3_call_id="sam_2",
        action_id="a_2",
        semantic_key="target",
        id_gen=IDGenerator(),
        config=V4Config().association,
    )
    assert len(result.new_nodes) == 0
    assert len(result.matched_observations) == 1
    assert len(graph.active_nodes()) == 1


def test_adjacent_fruits_are_not_merged_by_iom_gate():
    graph = SceneGraph()
    result = IoUIoMAssociationPolicy().associate(
        graph=graph,
        detections=[
            _det("left", (0, 0, 20, 20), 0.95),
            _det("right", (16, 0, 36, 20), 0.94),
        ],
        sam3_call_id="sam_1",
        action_id="a_1",
        semantic_key="target",
        id_gen=IDGenerator(),
        config=V4Config().association,
    )
    assert len(result.new_nodes) == 2


def test_pseudoexemplar_selection_keeps_only_strongest_seed_nodes():
    graph = SceneGraph()
    graph.add_node(_node("low", (0, 0, 10, 10), 0.40, 0.9))
    graph.add_node(_node("strong_a", (20, 0, 30, 10), 0.95, 0.7))
    graph.add_node(_node("strong_b", (40, 0, 50, 10), 0.85, 0.8))
    selection = select_target_pseudoexemplars(
        graph,
        max_count=2,
        min_score=0.60,
    )
    assert selection.node_ids == ("strong_a", "strong_b")
    assert selection.scores == pytest.approx((0.95, 0.85))


def test_exemplar_boxes_are_translated_to_crop_and_not_clipped():
    region = Box(100, 200, 300, 400)
    boxes = (
        (120, 230, 150, 260),
        (90, 230, 150, 260),  # partially outside: reject instead of clipping
    )
    assert _localize_exemplar_boxes(boxes, region) == [[20.0, 30.0, 50.0, 60.0]]


class _PseudoSensor:
    model_id = "pseudo-test-sam3"

    def __init__(self):
        self.actions = []
        self.call_count = 0

    def observe(self, image, action):
        action.validate()
        self.actions.append(action)
        self.call_count += 1
        if action.semantic_key == "locked_context_region":
            detections = [_det("canopy", (100, 100, 900, 900), 0.95)]
            searched = [BoxGeometry(Box(0, 0, 1000, 1000))]
        elif not action.positive_exemplar_boxes:
            detections = [
                _det("seed_hi", (200, 200, 240, 240), 0.95),
                _det("seed_lo", (300, 300, 340, 340), 0.40),
            ]
            searched = [action.search_region]
        else:
            detections = [_det("refined", (200, 200, 240, 240), 0.98)]
            searched = [action.search_region]
        return SAM3Observation(
            call_id=f"sam3_{self.call_count}",
            action_id=action.action_id,
            semantic_key=action.semantic_key,
            detections=detections,
            searched_regions=searched,
            runtime_ms=1.0,
            model_metadata={"model_id": self.model_id},
        )


def test_bootstrap_refines_with_pseudoexemplar_before_qwen(monkeypatch):
    sensor = _PseudoSensor()
    cfg = V4Config(
        bootstrap=BootstrapConfig(
            enable_tiled_bootstrap=False,
            locked_context_prompt="tree canopy",
            enable_pseudoexemplar_refinement=True,
            pseudoexemplar_max_count=3,
            pseudoexemplar_min_score=0.60,
        )
    )
    monkeypatch.setattr(
        "sam3_vlm.pipeline.bootstrap.ContactSheetBuilder.build_contact_sheet",
        lambda self, **kwargs: ContactSheet(
            crops=[], total_candidates=len(kwargs["graph"].active_nodes())
        ),
    )
    BootstrapPipeline(sensor=sensor, config=cfg).execute_bootstrap(
        image_id="img",
        image=(1000, 1000),
        user_prompt="green fruit",
        target_class="target",
    )
    assert len(sensor.actions) == 3
    assert sensor.actions[0].prompt == "tree canopy"
    assert sensor.actions[1].prompt == "green fruit"
    assert sensor.actions[1].positive_exemplar_boxes == ()
    assert sensor.actions[2].prompt == "green fruit"
    assert sensor.actions[2].positive_exemplar_ids
    assert sensor.actions[2].positive_exemplar_boxes == ((200.0, 200.0, 240.0, 240.0),)


class _NoopPlanner:
    model = "noop"

    def plan_scene(self, evidence, budget, config):
        return PlannerOutput(scene_summary="noop", proposed_actions=[])


def test_runner_inherits_pseudoexemplars_only_for_target_oriented_actions():
    cfg = V4Config(
        bootstrap=BootstrapConfig(
            enable_pseudoexemplar_refinement=True,
            pseudoexemplar_max_count=2,
            pseudoexemplar_min_score=0.60,
        )
    )
    runner = Runner(cfg, _PseudoSensor(), _NoopPlanner())
    graph = SceneGraph()
    graph.add_node(_node("fruit", (10, 10, 30, 30), 0.95, 0.8))
    runner.scene_state = SceneState(
        image_id="img",
        user_prompt="green fruit",
        target_class="target",
        graph=graph,
        semantic_memory=SemanticMemory(),
        belief_classes=["target", "confounder1", "confounder2"],
    )

    target_action = SensingAction(
        action_id="a_target",
        semantic_key="round_green_fruit",
        prompt="round green fruit",
        family=ActionFamily.DISCOVERY,
        semantic_prior={"target": 0.9, "confounder1": 0.1},
    )
    grounded = runner._attach_target_pseudoexemplars(target_action)
    assert grounded.positive_exemplar_ids == ("fruit",)

    confounder_action = replace(
        target_action,
        action_id="a_leaf",
        semantic_key="flat_leaf",
        prompt="flat green leaf",
        family=ActionFamily.CONFOUNDER,
        semantic_prior={"target": 0.1, "confounder1": 0.9},
    )
    untouched = runner._attach_target_pseudoexemplars(confounder_action)
    assert untouched.positive_exemplar_ids == ()


def test_generic_default_keeps_m8_geometry_and_pseudo_features_disabled():
    assert V4Config().bootstrap.enable_pseudoexemplar_refinement is False
    assert V4Config().association.enable_iom_dedup is False
