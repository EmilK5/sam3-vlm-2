"""Integration test for BootstrapPipeline execution and evidence pack generation (V4 Design Spec §5)."""

import pytest
from sam3_vlm.core.config import BootstrapConfig, V4Config
from sam3_vlm.core.geometry import Box, GeometryRef
from sam3_vlm.core.types import Detection
from sam3_vlm.models.sam3 import MockSAM3Adapter
from sam3_vlm.models.qwen import DummyQwenPlanner
from sam3_vlm.pipeline.bootstrap import BootstrapPipeline, BootstrapResult


def test_bootstrap_pipeline_execution(tmp_path):
    """Verify end-to-end bootstrap pass: global pass + tiled pass -> initial SceneState + QwenEvidencePack."""
    synth_dets = [
        Detection("d1", GeometryRef(Box(10.0, 10.0, 50.0, 50.0)), score=0.88),
        Detection("d2", GeometryRef(Box(200.0, 200.0, 250.0, 250.0)), score=0.75),
    ]
    sensor = MockSAM3Adapter(synthetic_detections=synth_dets)
    planner = DummyQwenPlanner()  # Used only to verify zero calls

    pipeline = BootstrapPipeline(
        sensor=sensor,
        config=V4Config(assets_dir=str(tmp_path / "assets")),
    )

    result = pipeline.execute_bootstrap(
        image_id="img_001",
        image=(2000, 2000),
        user_prompt="count green citrus",
        target_class="green_citrus",
    )

    assert isinstance(result, BootstrapResult)
    assert result.state.image_id == "img_001"
    assert result.state.target_class == "green_citrus"

    # Verify registered candidate nodes
    active_nodes = result.state.graph.active_nodes()
    assert len(active_nodes) >= 2

    # Verify budget accounting
    assert result.state.budget.sam3_calls >= 1
    assert result.state.budget.total_runtime_ms > 0.0

    # Verify provenance tracing in SemanticMemory uses real SAM3 call IDs
    mem = result.state.semantic_memory
    assert "green_citrus" in mem.records
    rec = mem.records["green_citrus"]
    assert len(rec.sam3_call_ids) >= 1
    for call_id in rec.sam3_call_ids:
        assert call_id.startswith("sam3_")
        assert call_id not in ("global_bootstrap", "tiled_bootstrap")

    # Verify QwenEvidencePack assembly
    evidence_pack = result.qwen_evidence_pack
    assert evidence_pack.original_image_id == "img_001"
    assert evidence_pack.user_prompt == "count green citrus"
    assert evidence_pack.target_class == "green_citrus"
    assert len(evidence_pack.contact_sheet.crops) >= 2

    # Invariant §5.4: Bootstrap MUST NOT call Qwen internally
    assert planner.call_count == 0
