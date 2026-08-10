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
        sam3_score=0.88,
        support_count=2,
        provenance="sam3_000001",
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
        sam3_score=0.92,
        support_count=3,
        provenance="sam3_000001",
        class_belief={"green_citrus": 0.85},
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
    assert "system_belief=green_citrus" in prompt_text
