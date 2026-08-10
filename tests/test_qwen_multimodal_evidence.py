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


def test_bootstrap_physical_asset_generation(tmp_path):
    """Verify that bootstrap physically generates crop and contact sheet images (Spec M3.5 Phase 2)."""
    import numpy as np
    from pathlib import Path
    from sam3_vlm.core.config import V4Config
    from sam3_vlm.pipeline.bootstrap import BootstrapPipeline
    from sam3_vlm.models.sam3 import MockSAM3Adapter
    from sam3_vlm.core.types import Detection
    from sam3_vlm.core.geometry import BoxGeometry

    # Create dummy image array (100x100 RGB)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[10:50, 10:50] = (255, 0, 0)  # Red box
    image[50:90, 50:90] = (0, 255, 0)  # Green box

    # Mock sensor
    d1 = Detection("d1", BoxGeometry(Box(10, 10, 50, 50)), score=0.9)
    d2 = Detection("d2", BoxGeometry(Box(50, 50, 90, 90)), score=0.8)
    adapter = MockSAM3Adapter(synthetic_detections=[d1, d2])

    config = V4Config(assets_dir=str(tmp_path / "assets"))
    pipeline = BootstrapPipeline(sensor=adapter, config=config)

    result = pipeline.execute_bootstrap(
        image_id="test_img",
        image=image,
        user_prompt="boxes"
    )

    pack = result.qwen_evidence_pack
    import cv2
    
    assert pack.image_path is not None
    assert Path(pack.image_path).exists()
    assert cv2.imread(pack.image_path) is not None
    
    assert pack.contact_sheet.contact_sheet_image_path is not None
    assert Path(pack.contact_sheet.contact_sheet_image_path).exists()
    assert cv2.imread(pack.contact_sheet.contact_sheet_image_path) is not None

    assert len(pack.contact_sheet.crops) == 2
    for crop in pack.contact_sheet.crops:
        assert crop.crop_image_path is not None
        assert Path(crop.crop_image_path).exists()
        assert cv2.imread(crop.crop_image_path) is not None
