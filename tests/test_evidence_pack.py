"""Unit tests for QwenEvidencePack serialization and compact prompt text formatting (V4 Design Spec §6.1)."""

import pytest
from sam3_vlm.core.geometry import Box
from sam3_vlm.sensing.evidence import (
    ContactSheet,
    CropCandidateAnnotation,
    QwenEvidencePack,
)


def test_evidence_pack_serialization_roundtrip():
    crop = CropCandidateAnnotation(
        node_id="n1",
        box=Box(10.0, 10.0, 50.0, 50.0),
        target_support_score=0.88,
        support_count=2,
        target_support_semantic_key="target",
        target_support_call_id="sam3_000001",
        target_support_action_id="action_000001",
        latest_observation_score=0.0,
        latest_observation_semantic_key="green_leaf",
        latest_observation_relation="NOT_RETRIEVED",
        latest_observation_call_id="sam3_000002",
        class_belief={"target": 0.8, "leaf": 0.2},
    )
    cs = ContactSheet(crops=[crop], total_candidates=1, strata_counts={"high": 1})

    pack = QwenEvidencePack(
        original_image_id="img_001",
        user_prompt="count green citrus",
        target_class="green_citrus",
        contact_sheet=cs,
        scene_summary="Bootstrap finished.",
    )

    data = pack.to_dict()
    assert data["original_image_id"] == "img_001"
    assert len(data["contact_sheet"]["crops"]) == 1
    crop_data = data["contact_sheet"]["crops"][0]
    assert crop_data["target_support_score"] == 0.88
    assert crop_data["latest_observation_score"] == 0.0
    assert crop_data["target_posterior"] == 0.8
    assert crop_data["sam3_score"] == 0.88

    restored = QwenEvidencePack.from_dict(data)
    assert restored.original_image_id == "img_001"
    assert restored.contact_sheet.crops[0].node_id == "n1"

    json_str = pack.to_json()
    assert "img_001" in json_str

    restored_json = QwenEvidencePack.from_json(json_str)
    assert restored_json.target_class == "green_citrus"


def test_evidence_pack_to_prompt_text():
    crop = CropCandidateAnnotation(
        node_id="node_000001",
        box=Box(10.0, 10.0, 50.0, 50.0),
        target_support_score=0.92,
        support_count=3,
        target_support_semantic_key="target",
        target_support_call_id="sam3_000001",
        latest_observation_score=0.0,
        latest_observation_semantic_key="green_leaf",
        latest_observation_relation="NOT_RETRIEVED",
        class_belief={"green_citrus": 0.85},
        target_posterior=0.85,
    )
    cs = ContactSheet(crops=[crop], total_candidates=5)

    pack = QwenEvidencePack(
        original_image_id="img_002",
        user_prompt="green citrus fruit",
        target_class="green_citrus",
        contact_sheet=cs,
    )

    prompt_text = pack.to_prompt_text()
    assert "=== SCENE EVIDENCE PACK ===" in prompt_text
    assert "img_002" in prompt_text
    assert "green citrus fruit" in prompt_text
    assert "node_000001" in prompt_text
    assert "target_support_score=0.92" in prompt_text
    assert "latest_observation_score=0.00" in prompt_text
    assert "latest_semantic_key=green_leaf" in prompt_text
    assert "latest_relation=NOT_RETRIEVED" in prompt_text
    assert "target_posterior=0.850" in prompt_text
    assert "class_belief={green_citrus:0.850}" in prompt_text


def test_legacy_evidence_pack_deserialization_maps_score_to_target_support():
    legacy = {
        "node_id": "legacy_node",
        "box": [0.0, 0.0, 10.0, 10.0],
        "sam3_score": 0.73,
        "support_count": 1,
        "provenance": "sam3_legacy",
        "class_belief": {"target": 0.7},
    }

    crop = CropCandidateAnnotation.from_dict(legacy)

    assert crop.target_support_score == 0.73
    assert crop.target_support_call_id == "sam3_legacy"
    assert crop.target_posterior == 0.7
    assert crop.sam3_score == 0.73
