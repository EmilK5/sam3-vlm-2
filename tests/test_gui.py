"""GUI integration uses deterministic adapters; no weights or endpoint needed."""

import csv
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

from PIL import Image
import pytest

from sam3_vlm.core.types import ActionFamily, SpatialMode
from sam3_vlm.experiments.m8_smoke import load_m8_config, _run_validator_and_replay
from sam3_vlm.gui.service import RunModels, SingleImageService
from sam3_vlm.gui.settings import CONTROLS, build_config, control_values, presets
from sam3_vlm.logging.artifacts import RunArtifactPaths
from sam3_vlm.models.sam3 import MockSAM3Adapter
from sam3_vlm.sensing.action import SensingAction
from test_m8_rejection_correction import SequencePlanner, proposal


@pytest.fixture
def deployment(tmp_path):
    path = Path(__file__).resolve().parents[1] / "configs/m8_real_smoke.json"
    return load_m8_config(SimpleNamespace(output_dir=str(tmp_path), require_cuda=False), str(path))


def change(config, enabled=True, **overrides):
    values = control_values(config)
    for index, control in enumerate(CONTROLS):
        if control.path in overrides:
            values[index] = overrides[control.path]
    return build_config(config, enabled, values)


def test_presets_keep_current_qwen_instructions_and_derive_replans(deployment):
    variants = presets(deployment.v4_config)
    assert len(variants) == 5
    for name, variant in variants.items():
        config = build_config(variant.config, variant.uses_qwen, control_values(variant.config))
        assert config.planner.prompt_version == "old"
        assert config.planner.target_scope == deployment.v4_config.planner.target_scope
        if variant.uses_qwen:
            assert config == variant.config
        else:
            assert config.budget.max_qwen_calls == config.budget.max_cleanup_calls == 0
            assert not config.planner.execute_confounder_prompts
    base = variants["D_Qwen_TwoRound"].config
    config = change(base, **{"budget.max_qwen_calls": 7})
    assert config.replanning.max_replans == 6
    assert config.planner == base.planner
    assert not any(c.path in {"planner.prompt_version", "planner.target_scope", "planner.max_actions_per_prompt"} for c in CONTROLS)


@pytest.mark.parametrize("key,value", [
    ("sam3.default_threshold", -0.1), ("sam3.qwen_discovery_threshold", 1.1),
    ("sam3.qwen_confounder_threshold", float("nan")), ("budget.max_runtime_seconds", 0),
    ("budget.max_qwen_calls", 1.5), ("budget.max_qwen_calls", 101),
    ("budget.max_sam3_calls", 1001), ("tiling.grid_rows", 0),
    ("tiling.overlap_ratio", 1), ("planner.temperature", float("inf")),
    ("belief.num_confounders", 0), ("planner.execute_confounder_prompts", "false"),
    ("budget.max_sam3_calls", True), ("sam3.default_threshold", None),
])
def test_invalid_controls_fail_before_model_loading(deployment, key, value):
    with pytest.raises(ValueError):
        change(deployment.v4_config, **{key: value})


def test_counts_cross_field_validation_and_optional_caps(deployment):
    base = deployment.v4_config
    with pytest.raises(ValueError, match="mutually exclusive"):
        change(base, **{"belief.target_count_hard_threshold": .5, "belief.target_count_commit_threshold": .8})
    with pytest.raises(ValueError, match="IoU"):
        change(base, **{"association.new_node_iou_threshold": .9})
    with pytest.raises(ValueError, match="at least one"):
        change(base, **{"budget.max_qwen_calls": 0})
    config = change(base, False, **{"belief.target_count_hard_threshold": .5,
                                  "bootstrap.locked_context_prompt": "", "budget.max_sam3_tiles": None})
    assert config.belief.target_count_hard_threshold is None
    assert config.bootstrap.locked_context_prompt is None
    assert config.budget.max_sam3_tiles is None


def test_baseline_never_loads_qwen_and_each_run_has_fresh_state(deployment):
    loaded = []
    def sensor_factory():
        loaded.append("sam3")
        return MockSAM3Adapter()
    def forbidden_planner():
        pytest.fail("Qwen must not load for a baseline")
    service = SingleImageService(deployment, sensor_factory, forbidden_planner)
    config = presets(deployment.v4_config)["A_SAM3_Global"].config
    config = build_config(config, False, control_values(config))
    outputs = [service.run(Image.new("RGB", (160, 100)), "green fruit", config, False, gt_count=0)
               for _ in range(2)]
    assert loaded == ["sam3"]
    assert outputs[0][0]["run_id"] != outputs[1][0]["run_id"]
    for row, settings, overlay, downloads in outputs:
        assert row["qwen_calls"] == 0 and row["sam3_calls"] == 1
        assert row["predicted_count"] == row["candidate_count"]
        assert row["count_type"] == "hard_candidate_count"
        assert row["relative_error"] is None
        assert Image.open(overlay).size == (160, 100)
        with open(downloads[0], encoding="utf-8-sig") as stream:
            assert next(csv.DictReader(stream))["relative_error"] == ""
        with ZipFile(downloads[1]) as bundle:
            assert {"settings.json", "input.png", "bboxes.png", "result.json", "result.csv", "run.json", "summary.json"} <= set(bundle.namelist())
            assert "run_bundle.zip" not in bundle.namelist()
        assert not (Path(deployment.output_root) / "assets").exists()


@pytest.mark.parametrize("hard,commit", [(None, None), (.5, None), (None, .5)])
def test_qwen_integration_thresholds_negatives_count_and_replay(deployment, hard, commit):
    class CapturingSensor(MockSAM3Adapter):
        def __init__(self):
            super().__init__()
            self.actions = []
        def observe(self, image, action):
            self.actions.append(action)
            return super().observe(image, action)
    sensor = CapturingSensor()
    planner = SequencePlanner([proposal("dark green fruit", ("leaf", "branch"))])
    planner.last_request_text = {"system": "unchanged test instructions"}
    service = SingleImageService(deployment, lambda: sensor, lambda: planner)
    config = presets(deployment.v4_config)["C_Qwen_OneRound"].config
    config = change(config, **{"belief.target_count_hard_threshold": hard,
                               "belief.target_count_commit_threshold": commit})
    row, settings, overlay, files = service.run(Image.new("RGB", (200, 200)), "green fruit", config, gt_count=7)
    assert row["qwen_calls"] == 1
    assert row["absolute_error"] == abs(row["predicted_count"] - 7)
    positives = [a for a in sensor.actions if a.prompt == "dark green fruit"]
    negatives = [a for a in sensor.actions if a.family == ActionFamily.CONFOUNDER]
    assert positives and positives[0].threshold == .2 and not positives[0].positive_exemplar_boxes
    assert len(negatives) == 2 and all(a.threshold == .5 for a in negatives)
    root = Path(overlay).parent
    graph = json.loads((root / "artifacts/graph/final_graph.json").read_text())
    probs = [n["class_belief"]["probabilities"]["target"] for n in graph["nodes"].values() if n["status"] == "ACTIVE"]
    expected = sum(float(p > hard) if hard is not None else 1 if commit is not None and p >= commit else p for p in probs)
    assert row["predicted_count"] == pytest.approx(expected)
    assert _run_validator_and_replay(RunArtifactPaths(root))
    artifact = json.loads(next((root / "artifacts/qwen").glob("*.json")).read_text())
    assert artifact["input"]["request_text"] == planner.last_request_text
    assert settings["v4_config"]["planner"]["target_scope"] == config.planner.target_scope


def test_failure_keeps_artifacts_and_next_submission_can_recover(deployment):
    attempts = []
    def load_sensor():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("test load failure")
        return MockSAM3Adapter()
    service = SingleImageService(deployment, load_sensor)
    config = presets(deployment.v4_config)["A_SAM3_Global"].config
    config = build_config(config, False, control_values(config))
    with pytest.raises(RuntimeError, match="test load failure"):
        service.run(Image.new("RGB", (100, 100)), "fruit", config, False)
    failure = next(Path(deployment.output_root).glob("*/failure.json"))
    assert json.loads(failure.read_text())["success"] is False
    assert (failure.parent / "settings.json").exists()
    assert not (failure.parent / "result.json").exists()
    assert service.run(Image.new("RGB", (100, 100)), "fruit", config, False)[0]["success"]


def test_budget_guard_includes_bootstrap_and_does_not_invoke_over_limit(deployment):
    config = replace(deployment.v4_config, budget=replace(deployment.v4_config.budget,
        max_sam3_calls=1, max_sam3_tiles=0, max_qwen_calls=1, max_runtime_seconds=10))
    sensor = MockSAM3Adapter()
    planner = SequencePlanner([proposal("fruit")])
    clock = [0]
    models = RunModels(sensor, planner, config, clock=lambda: clock[0])
    action = SensingAction("a", "target", "fruit", ActionFamily.DISCOVERY)
    tiled = replace(action, spatial_mode=SpatialMode.TILED, tiling=config.tiling)
    with pytest.raises(RuntimeError, match="Tile budget"):
        models.observe((100, 100), tiled)
    assert sensor.call_count == 0
    models.observe((100, 100), action)
    with pytest.raises(RuntimeError, match="SAM3 budget"):
        models.observe((100, 100), action)
    assert sensor.call_count == 1
    models.plan_scene(None, None, config)
    with pytest.raises(RuntimeError, match="Qwen call budget"):
        models.plan_scene(None, None, config)
    assert len(planner.evidence) == 1
    clock[0] = 10
    with pytest.raises(RuntimeError, match="Runtime budget"):
        models.observe((100, 100), action)


def test_gradio_form_presets_and_failure_clear_stale_results(deployment):
    pytest.importorskip("gradio")
    from sam3_vlm.gui.app import create_app
    service = SingleImageService(deployment, lambda: MockSAM3Adapter())
    app = create_app(deployment, service)
    components = {c.get("props", {}).get("label"): c for c in app.config["components"]}
    for label in ("GT count (optional)", "Hard count cutoff (blank = off)", "Soft commitment cutoff (blank = off)"):
        assert components[label]["type"] == "textbox"
        assert components[label]["props"]["value"] == ""
    callbacks = {fn.fn.__name__: fn.fn for fn in app.fns.values() if fn.fn}
    values = callbacks["load_preset"]("A_SAM3_Global")
    assert len(values) == len(CONTROLS) + 1 and values[0] is False
    result = callbacks["execute"](Image.new("RGB", (100, 100)), "fruit", values[0], None, 42, *values[1:])
    assert result[2]["success"] and result[0] is not None
    failed = callbacks["execute"](None, "fruit", values[0], None, 42, *values[1:])
    assert failed[0] is None and failed[2]["success"] is False and failed[3] == [] and failed[4] is None


def test_launch_allows_only_resolved_output_directory(deployment, monkeypatch):
    from sam3_vlm.gui import app
    calls = []
    class FakeApp:
        def launch(self, **kwargs):
            calls.append(kwargs)
    monkeypatch.setattr(app, "create_app", lambda dep: FakeApp())
    monkeypatch.setattr(app, "load_m8_config", lambda *args, **kwargs: deployment)
    monkeypatch.setattr("sys.argv", ["app", "--config", str(Path(__file__).resolve().parents[1] / "configs/m8_real_smoke.json")])
    app.main()
    assert calls == [{"server_name": "127.0.0.1", "server_port": 7860, "share": False,
                      "allowed_paths": [str(Path(deployment.output_root).resolve())]}]
