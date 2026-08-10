"""Test core types, dataclasses, validation logic, and schema invariants."""

import pytest
from sam3_vlm.core.types import (
    NodeStatus,
    ObservationRelation,
    ActionFamily,
    SpatialMode,
    ActionSource,
    BudgetState,
    ClassBelief,
    RegistrationDiagnostics,
    NodeObservationRef,
    Detection,
)
from sam3_vlm.core.geometry import Box, BoxGeometry, GeometryRef
from sam3_vlm.sensing.action import SensingAction
from sam3_vlm.sensing.observation import SAM3Observation


def test_class_belief_validation():
    cb = ClassBelief(probabilities={"target": 0.8, "leaf": 0.2})
    assert cb.probabilities["target"] == 0.8

    with pytest.raises(ValueError):
        ClassBelief(probabilities={"target": -0.1})


def test_sensing_action_validation():
    action = SensingAction(
        action_id="act_001",
        semantic_key="green_citrus",
        prompt="green citrus fruit",
        family=ActionFamily.DISCOVERY,
    )
    assert action.validate() is True

    # Empty prompt raises error
    with pytest.raises(ValueError):
        invalid_action = SensingAction(
            action_id="act_002",
            semantic_key="green_citrus",
            prompt="",
            family=ActionFamily.DISCOVERY,
        )
        invalid_action.validate()

    # Non-disjoint exemplars raise error
    with pytest.raises(ValueError):
        invalid_action = SensingAction(
            action_id="act_003",
            semantic_key="green_citrus",
            prompt="fruit",
            family=ActionFamily.VERIFICATION,
            positive_exemplar_ids=("node_001", "node_002"),
            negative_exemplar_ids=("node_002", "node_003"),
        )
        invalid_action.validate()


def test_detection_schema(sample_box: Box):
    geom_ref = GeometryRef(box=sample_box, mask_artifact="masks/mask_001.npz")
    det = Detection(
        detection_id="det_000001",
        geometry=geom_ref,
        score=0.92,
        source_tile_id="tile_01",
        local_geometry=None,
        mask_artifact="masks/mask_001.npz",
    )
    assert det.detection_id == "det_000001"
    assert det.score == 0.92
    assert det.geometry.mask_artifact == "masks/mask_001.npz"


def test_sam3_observation_schema():
    obs = SAM3Observation(
        call_id="sam3_000001",
        action_id="act_000001",
        semantic_key="green_citrus",
    )
    assert obs.call_id == "sam3_000001"
    assert len(obs.detections) == 0
