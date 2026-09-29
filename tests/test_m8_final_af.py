"""A–F fruit comparison with the separate VLM-first controller as F."""

import csv
import json
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from PIL import Image

from sam3_vlm.experiments.m8_smoke import _pilot_variants, m8_4_and_5_pilot
from sam3_vlm.experiments.mvp_fruit_arm import FRUIT_ARM
from sam3_vlm.experiments import mvp_fruit_arm
from sam3_vlm.models.sam3 import MockSAM3Adapter
from sam3_vlm.mvp.core import Config as MVPConfig, Detection
from sam3_vlm.mvp.core import Node as MVPNode, Result as MVPResult
from sam3_vlm.experiments.final_outputs import render_mvp_boxes
from test_m8_low_discovery import production
from test_m8_orchestration import DummyArgs
from test_m8_rejection_correction import SequencePlanner, proposal


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class FruitSensor:
    def __init__(self):
        self.calls = []

    def search(self, image, prompt, region, exemplar_boxes=()):
        self.calls.append((prompt, region, exemplar_boxes))
        if len(self.calls) > 1:
            return []
        mask = np.zeros((image.height, image.width), dtype=bool)
        mask[10:20, 10:20] = True
        return [Detection(mask, .9)]


class FruitVLM:
    def __init__(self):
        self.initial_calls = 0

    def propose_initial(self, image, target, state):
        self.initial_calls += 1
        assert "gt_count" not in state and "ground_truth" not in state
        return {"roi": [0, 0, 1000, 1000], "tile_mode": "auto",
                "actions": [{"prompt": target, "region": None}]}

    def propose(self, image, target, state):
        return {"actions": []}


def test_final_af_appends_f_without_changing_original_arms():
    base = production()
    old = _pilot_variants(base, "final-ae")
    expanded = _pilot_variants(base, "final-af")
    assert expanded[:5] == old
    assert [variant.name[0] for variant in expanded] == list("ABCDEF")
    f = expanded[-1]
    assert f.name == FRUIT_ARM and f.count_type == "hard_belief_count"
    assert isinstance(f.config, MVPConfig)
    assert f.config.vlm_first and not f.config.bootstrap_regions
    assert f.config.vlm_coordinate_mode == "normalized_1000"
    assert f.config.enable_adaptive_tiling and f.config.enable_exemplar_refinement


def test_f_reuses_loaded_sensor_and_sends_tree_only_scope(monkeypatch):
    class LoadedSensor:
        def _run_inference(self, *args, **kwargs):
            return np.empty((0, 4)), np.empty(0), []

    captured = {}

    class FakeVLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(mvp_fruit_arm, "RealVLM", FakeVLM)
    loaded = LoadedSensor()
    deployment = SimpleNamespace(qwen_base_url="http://qwen/v1", qwen_model="qwen",
                                 v4_config=SimpleNamespace(planner=SimpleNamespace(
                                     target_scope="fruit on trees only")))
    sam3, vlm = mvp_fruit_arm.make_adapters(loaded, deployment)
    assert sam3.sensor is loaded and isinstance(vlm, FakeVLM)
    assert captured["scope"] == "fruit on trees only"
    assert captured["coordinate_mode"] == "normalized_1000"


def test_f_overlay_selects_exact_hard_count_boxes(tmp_path):
    image = Image.new("RGB", (64, 64), "white")
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:20, 10:20] = True
    strong = MVPNode("n1", mask, (10, 10, 20, 20), "d1", .9, "a1", "a1", belief=.8)
    weak = MVPNode("n2", mask, (30, 30, 40, 40), "d2", .2, "a1", "a1", belief=.3)
    result = MVPResult({"n1": strong, "n2": weak}, [], [], {"hard": 1, "soft": 1.1},
                       {}, "qwen_budget", False, [], [], [], 0., {})
    path = tmp_path / "boxes.png"
    export = render_mvp_boxes(image, result, path, .5)
    assert export["bbox_count"] == result.counts["hard"] == 1
    with Image.open(path) as rendered:
        assert rendered.getpixel((10, 10)) == (255, 48, 48)
        assert rendered.getpixel((30, 30)) == (255, 255, 255)


def test_final_af_runs_f_on_same_fruit_and_exports_all_six(tmp_path, monkeypatch):
    image_path = tmp_path / "fruit.png"
    Image.new("RGB", (64, 64), (20, 80, 20)).save(image_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"sample_id": "fruit_1", "image_path": str(image_path),
                                    "target": "green fruit", "gt_count": 1}]))
    sensor, vlm = FruitSensor(), FruitVLM()
    monkeypatch.setattr("sam3_vlm.experiments.m8_smoke._get_models",
                        lambda args: (MockSAM3Adapter(), SequencePlanner([proposal("dark green fruit")])))
    monkeypatch.setattr("sam3_vlm.experiments.mvp_fruit_arm.make_adapters",
                        lambda legacy, deployment, config: (sensor, vlm))
    output_dir = tmp_path / "results"
    args = DummyArgs(manifest=str(manifest), max_samples=1,
                     pilot_suite="final-af", output_dir=str(output_dir))
    assert m8_4_and_5_pilot(args)
    report = json.loads((output_dir / "pilot_report.json").read_text())
    assert report["metadata"]["variants"] == [v.name for v in _pilot_variants(production(), "final-af")]
    assert report["metadata"]["variant_engines"][FRUIT_ARM] == "mvp_vlm_first"
    assert len(report["samples"]) == 6 and all(row["success"] for row in report["samples"])
    f = next(row for row in report["samples"] if row["variant"] == FRUIT_ARM)
    assert f["predicted_count"] == 1 and f["bbox_count"] == 1
    assert f["roi"] == [0, 0, 64, 64] and f["candidate_count"] == 1
    assert f["validator_status"] == "NOT_APPLICABLE_MVP"
    assert vlm.initial_calls == 1
    assert sensor.calls[0] == ("green fruit", (0, 0, 64, 64), ())
    trace = json.loads((output_dir / "pilot" / FRUIT_ARM / f["run_id"] / "mvp_result.json").read_text())
    assert trace["actions"][0]["source"] == "vlm_initial"
    assert not any(action["source"] == "bootstrap" for action in trace["actions"])
    assert len(read_csv(output_dir / "aggregate_results.csv")) == 6
    assert len(read_csv(output_dir / "per_image_results.csv")) == 6
    assert FRUIT_ARM in read_csv(output_dir / "counts_by_image.csv")[0]
    with ZipFile(output_dir / "bbox_images.zip") as archive:
        assert len(archive.namelist()) == 6
    with ZipFile(output_dir / "compact_review.zip") as archive:
        assert "aggregate_results.csv" in archive.namelist()
        assert len(json.loads(archive.read("counts.json"))) == 6


def test_final_af_marks_f_incomplete_without_fabricating_metrics(tmp_path, monkeypatch):
    image_path = tmp_path / "fruit.png"
    Image.new("RGB", (64, 64)).save(image_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"sample_id": "fruit_1", "image_path": str(image_path),
                                    "target": "green fruit", "gt_count": 1}]))

    class InvalidVLM(FruitVLM):
        def propose_initial(self, image, target, state):
            return {"roi": [-1, 0, 64, 64], "tile_mode": "auto",
                    "actions": [{"prompt": target, "region": None}]}

    monkeypatch.setattr("sam3_vlm.experiments.m8_smoke._get_models",
                        lambda args: (MockSAM3Adapter(), SequencePlanner([proposal("dark green fruit")])))
    monkeypatch.setattr("sam3_vlm.experiments.mvp_fruit_arm.make_adapters",
                        lambda legacy, deployment: (FruitSensor(), InvalidVLM()))
    output_dir = tmp_path / "results"
    args = DummyArgs(manifest=str(manifest), max_samples=1,
                     pilot_suite="final-af", output_dir=str(output_dir))
    assert not m8_4_and_5_pilot(args)
    report = json.loads((output_dir / "pilot_report.json").read_text())
    assert sum(row["success"] for row in report["samples"]) == 5
    f = read_csv(output_dir / "aggregate_results.csv")[-1]
    assert f["variant"] == FRUIT_ARM and f["MAE"] == "" and f["complete"] == "False"
    assert read_csv(output_dir / "counts_by_image.csv")[0][FRUIT_ARM] == ""
