"""Unit tests for multimodal evidence pack visual asset references and prompt text instructions (V4 Design Spec §5.3 / §6.1)."""

import pytest
from sam3_vlm.core.geometry import Box
from sam3_vlm.sensing.evidence import (
    ContactSheet,
    CropCandidateAnnotation,
    QwenEvidencePack,
)


def test_multimodal_evidence_pack_visual_assets():
    crop = CropCandidateAnnotation(
        node_id="n1",
        box=Box(10.0, 10.0, 50.0, 50.0),
        sam3_score=0.9,
        support_count=2,
        provenance="sam3_1",
        crop_image_path="crops/n1.jpg",
    )
    cs = ContactSheet(
        crops=[crop],
        total_candidates=1,
        contact_sheet_image_path="contact_sheets/sheet_1.jpg",
    )
    pack = QwenEvidencePack(
        original_image_id="img_1",
        user_prompt="citrus",
        target_class="citrus",
        contact_sheet=cs,
        image_path="images/img_1.jpg",
    )

    data = pack.to_dict()
    assert data["image_path"] == "images/img_1.jpg"
    assert data["contact_sheet"]["contact_sheet_image_path"] == "contact_sheets/sheet_1.jpg"
    assert data["contact_sheet"]["crops"][0]["crop_image_path"] == "crops/n1.jpg"

    restored = QwenEvidencePack.from_dict(data)
    assert restored.image_path == "images/img_1.jpg"
    assert restored.contact_sheet.crops[0].crop_image_path == "crops/n1.jpg"


def test_qwen_unverified_candidate_prompt_instructions():
    crop = CropCandidateAnnotation(
        node_id="n1",
        box=Box(10.0, 10.0, 50.0, 50.0),
        sam3_score=0.9,
        support_count=2,
        provenance="sam3_1",
    )
    cs = ContactSheet(crops=[crop], total_candidates=1)
    pack = QwenEvidencePack(
        original_image_id="img_1",
        user_prompt="citrus",
        target_class="citrus",
        contact_sheet=cs,
    )

    prompt_text = pack.to_prompt_text()
    assert "=== IMPORTANT QWEN INSTRUCTIONS ===" in prompt_text
    assert "UNVERIFIED visual sensor candidates" in prompt_text
    assert "Do NOT label them as ground truth" in prompt_text
    assert "Do NOT attempt to output final object counts" in prompt_text
