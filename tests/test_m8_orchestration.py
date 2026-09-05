import pytest
from unittest.mock import patch
import os
import json
import logging
from PIL import Image

from sam3_vlm.experiments.m8_smoke import (
    _load_pilot_samples,
    _pilot_variants,
    load_m8_config,
    m8_0_validate_adapters,
    m8_1_sam3_smoke,
    m8_2_qwen_smoke,
    m8_3_full_run,
    m8_4_and_5_pilot,
    preflight,
    main
)
from sam3_vlm.models.sam3 import MockSAM3Adapter
from sam3_vlm.models.qwen import MockQwenPlanner
from sam3_vlm.planning.qwen_planner import PlannerOutput

@pytest.fixture
def mock_models():
    sam3 = MockSAM3Adapter()
    qwen = MockQwenPlanner()
    with patch("sam3_vlm.experiments.m8_smoke._get_models", return_value=(sam3, qwen)), \
         patch("sam3_vlm.experiments.m8_smoke._get_sam3_only", return_value=sam3), \
         patch("sam3_vlm.experiments.m8_smoke._get_qwen_only", return_value=qwen):
        yield sam3, qwen

class DummyArgs:
    def __init__(self, **kwargs):
        self.require_cuda = False
        self.compile_sam3 = False
        self.dry_run = False
        self.sam3_model = "fake"
        self.qwen_model = "fake"
        self.qwen_base_url = "fake"
        self.image = "fake.jpg"
        self.target = "green citrus"
        self.output_dir = "fake_out"
        self.manifest = None
        self.max_samples = 2
        self.stage = "all"
        for k, v in kwargs.items():
            setattr(self, k, v)

def test_m8_0_validate(mock_models):
    assert m8_0_validate_adapters(DummyArgs()) is True

def test_m8_1_smoke(mock_models, tmp_path):
    img = Image.new("RGB", (64, 64))
    p = str(tmp_path / "test.jpg")
    img.save(p)
    assert m8_1_sam3_smoke(DummyArgs(image=p)) is True

def test_m8_2_smoke(mock_models, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    assert m8_2_qwen_smoke(DummyArgs(output_dir=str(tmp_path / "runs"))) is True
    assert not list(tmp_path.rglob("summary.json"))
    assert "planner-only smoke test; no summary.json is created" in caplog.text


def test_m8_2_reports_empty_unsaturated_contract_failure(mock_models, tmp_path, caplog):
    _, planner = mock_models
    planner.strict_model_errors = True
    planner.custom_output = PlannerOutput(scene_summary="No actions")
    assert m8_2_qwen_smoke(DummyArgs(output_dir=str(tmp_path / "runs"))) is False
    assert "EMPTY_UNSATURATED_PLAN" in caplog.text
    assert planner.call_count == 1
    assert not list(tmp_path.rglob("summary.json"))

def test_m8_3_full_run_with_mocks(mock_models, tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    img = Image.new("RGB", (64, 64))
    p = str(tmp_path / "test.jpg")
    img.save(p)
    monkeypatch.chdir(tmp_path)
    args = DummyArgs(image=p, output_dir="unused/../runs")

    def load_models_after_path_log(args):
        assert "M8.3 artifact directory:" in caplog.text
        return mock_models

    with patch("sam3_vlm.experiments.m8_smoke._get_models", side_effect=load_models_after_path_log):
        assert m8_3_full_run(args) is True
    assert list((tmp_path / "runs").rglob("assets/m8_test_img.jpg"))
    summary_path, = (tmp_path / "runs").rglob("summary.json")
    assert summary_path.is_absolute()
    assert json.loads(summary_path.read_text())["qwen_calls"] >= 1
    assert f"M8.3 artifact directory: {summary_path.parent}" in caplog.text
    assert f"Summary: {summary_path}" in caplog.text


def test_m8_3_does_not_report_validated_summary_on_validation_failure(mock_models, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    image = tmp_path / "test.jpg"
    Image.new("RGB", (64, 64)).save(image)
    args = DummyArgs(image=str(image), output_dir=str(tmp_path / "runs"))
    with patch("sam3_vlm.experiments.m8_smoke._run_validator_and_replay", return_value=False):
        assert m8_3_full_run(args) is False
    assert "M8.3 artifact directory:" in caplog.text
    assert "Summary:" not in caplog.text

def test_m8_4_and_5_pilot_with_mocks(mock_models, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    img = Image.new("RGB", (64, 64))
    p1 = str(tmp_path / "img1.jpg")
    img.save(p1)
    p2 = str(tmp_path / "img2.jpg")
    img.save(p2)
    p3 = str(tmp_path / "img3.jpg")
    img.save(p3)
    
    manifest_path = str(tmp_path / "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump([
            {"sample_id": "i1", "image_path": p1, "target": "target", "gt_count": 5},
            {"sample_id": "i2", "image_path": p2, "target": "target", "gt_count": 10},
            {"sample_id": "i3", "image_path": p3, "target": "target", "gt_count": 15},
        ], f)
        
    args = DummyArgs(manifest=manifest_path, output_dir=str(tmp_path / "runs"), max_samples=2)
    assert m8_4_and_5_pilot(args) is True
    
    report_p = str(tmp_path / "runs" / "pilot_report.json")
    assert os.path.exists(report_p)
    assert f"Report at {report_p}" in caplog.text
    
    with open(report_p) as f:
        report = json.load(f)
        
    assert "metadata" in report
    assert "aggregates" in report
    assert "samples" in report
    
    samples = report["samples"]
    assert len(samples) == 8  # 4 variants * 2 images (limit enforced)
    
    one_shot = [r for r in samples if r["variant"] == "A_SAM3_Global"]
    assert len(one_shot) == 2
    for r in one_shot:
        assert r["sam3_calls"] == 1
        assert r["qwen_calls"] == 0
        assert r["replans"] == 0
        assert r["cleanup_calls"] == 0
        assert r["count_type"] == "hard_candidate_count"

    sam3_bootstrap = [
        r for r in samples if r["variant"] == "B_SAM3_Bootstrap"
    ]
    assert len(sam3_bootstrap) == 2
    assert all(r["qwen_calls"] == 0 for r in sam3_bootstrap)

    one_qwen = [r for r in samples if r["variant"] == "C_Qwen_OneRound"]
    assert len(one_qwen) == 2
    assert all(r["qwen_calls"] == 1 for r in one_qwen)

    assert report["metadata"]["variants"] == [
        "A_SAM3_Global",
        "B_SAM3_Bootstrap",
        "C_Qwen_OneRound",
        "D_Qwen_TwoRound",
    ]


def test_pilot_variants_isolate_sam3_and_qwen_costs():
    base = load_m8_config(DummyArgs()).v4_config
    variants = {variant.name: variant for variant in _pilot_variants(base)}

    global_only = variants["A_SAM3_Global"]
    assert global_only.config.budget.max_qwen_calls == 0
    assert global_only.config.bootstrap.locked_context_prompt is None
    assert global_only.config.bootstrap.enable_tiled_bootstrap is False
    assert global_only.config.bootstrap.enable_pseudoexemplar_refinement is False

    sam3_bootstrap = variants["B_SAM3_Bootstrap"]
    assert sam3_bootstrap.config.budget.max_qwen_calls == 0
    assert sam3_bootstrap.config.bootstrap.locked_context_prompt == "tree canopy"

    one_round = variants["C_Qwen_OneRound"]
    assert one_round.config.budget.max_qwen_calls == 1
    assert one_round.config.replanning.max_replans == 0

    two_round = variants["D_Qwen_TwoRound"]
    assert two_round.config.budget.max_qwen_calls == 2
    assert two_round.config.replanning.max_replans == 1


def test_pilot_manifest_requires_ground_truth(tmp_path):
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (16, 16)).save(image_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [{"sample_id": "i1", "image_path": str(image_path)}]
        )
    )

    with pytest.raises(ValueError, match="gt_count"):
        _load_pilot_samples(
            DummyArgs(manifest=str(manifest_path)),
            limit=5,
        )

@patch("sam3_vlm.experiments.m8_smoke.preflight", return_value=True)
@patch("sam3_vlm.experiments.m8_smoke.m8_0_validate_adapters", return_value=False)
@patch("sam3_vlm.experiments.m8_smoke.m8_1_sam3_smoke")
def test_stage_fail_fast(mock_1, mock_0, mock_preflight, tmp_path):
    import sys
    test_args = ["--stage", "all", "--dry-run", "--output_dir", str(tmp_path / "out"), "--qwen-base-url", "fake", "--allow-cpu"]
    with patch.object(sys, 'argv', ["m8_smoke.py"] + test_args):
        assert main() == 1
    mock_preflight.assert_called_once()
    mock_0.assert_called_once()
    mock_1.assert_not_called()

@patch("sam3_vlm.experiments.m8_smoke.preflight", return_value=True)
@patch("sam3_vlm.experiments.m8_smoke.m8_0_validate_adapters", return_value=True)
@patch("sam3_vlm.experiments.m8_smoke.m8_1_sam3_smoke", return_value=True)
@patch("sam3_vlm.experiments.m8_smoke.m8_2_qwen_smoke", return_value=True)
@patch("sam3_vlm.experiments.m8_smoke.m8_3_full_run", return_value=True)
@patch("sam3_vlm.experiments.m8_smoke.m8_4_and_5_pilot")
def test_stage_all_excludes_pilot(
    mock_pilot,
    mock_full,
    mock_qwen_smoke,
    mock_sam3_smoke,
    mock_validate,
    mock_preflight,
    tmp_path,
):
    import sys
    test_args = ["--stage", "all", "--dry-run", "--output_dir", str(tmp_path / "out"), "--qwen-base-url", "fake", "--allow-cpu"]
    with patch.object(sys, 'argv', ["m8_smoke.py"] + test_args):
        assert main() == 0
            
    mock_preflight.assert_called_once()
    mock_validate.assert_called_once()
    mock_sam3_smoke.assert_called_once()
    mock_qwen_smoke.assert_called_once()
    mock_full.assert_called_once()
    mock_pilot.assert_not_called()


@pytest.mark.parametrize("output_dir", ["", " \t"])
def test_cli_rejects_empty_output_even_for_dry_run(output_dir, capsys):
    with patch("sys.argv", ["m8_smoke.py", "--stage", "M8.3", "--dry-run", "--output_dir", output_dir]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 2
    assert "output directory" in capsys.readouterr().err
