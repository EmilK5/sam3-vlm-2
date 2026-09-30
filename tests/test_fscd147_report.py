import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
import pytest

from sam3_vlm.experiments.fscd147_report import build_summary, render_summary, write_summary
from sam3_vlm.experiments.fscd147_smoke import main


@pytest.fixture
def saved(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "Train_Test_Val_FSC_147.json").write_text(json.dumps({"val": ["a.jpg", "b.jpg"]}))
    (dataset / "ImageClasses_FSC147.txt").write_text("a.jpg bottle caps\nb.jpg green apples\n")
    (dataset / "instances_val.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
        "annotations": [{"id": i, "image_id": 2, "category_id": 1, "bbox": [1, 1, 2, 2]} for i in range(2)],
        "categories": [{"id": 1, "name": "object"}]}))
    (dataset / "sample_manifest.json").write_text(json.dumps({"seed": 42, "image_ids": ["a.jpg", "b.jpg"]}))
    run = tmp_path / "run"
    run.mkdir()
    variants = {"A_Global": {"count_type": "hard_candidate_count", "config": {"association": {"mask_only": True}}},
                "B_Adaptive": {"count_type": "hard_candidate_count"},
                "C_Qwen": {"count_type": "soft_posterior_count"}}
    (run / "metadata.json").write_text(json.dumps({"schema_version": 1, "dataset": "FSCD-147", "split": "val",
        "image_ids": ["a.jpg", "b.jpg"], "variants": variants,
        "model_settings": {"sam3": "facebook/sam3", "qwen": "local-qwen", "seed": 42, "api_key": "secret"}}))
    rows = []
    for variant, counts in zip(variants, ([1, 1], [0, 2], [.25, 1.5])):
        for index, (image_id, count) in enumerate(zip(["a.jpg", "b.jpg"], counts)):
            rows.append({"image_id": image_id, "variant": variant, "target": "bottle caps" if index == 0 else "green apples",
                "success": True, "predicted_count": count, "runtime_seconds": 2+index,
                "count_type": variants[variant]["count_type"], "image_size": [10, 10],
                "nodes": [{"box": [1, 1, 3, 3], "score": .25}],
                "artifact_directory": "/unavailable/old/location",
                "budget": {"sam3_calls": 3, "qwen_calls": 1 if variant.startswith("C") else 0, "sam3_tiles": 2,
                           "sam3_runtime_ms": 1000, "qwen_runtime_ms": 500},
                "stop_reason": "QWEN_BUDGET" if variant.startswith("C") else "SAM3_BASELINE_COMPLETE",
                "adaptive_tiling": {"trigger": True, "density_score": .75, "object_count": 50,
                                    "tile_size": 97, "tiles": [[0, 0, 97, 97]]*2}})
    path = run / "predictions.jsonl"
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    return dataset, path, rows


def artifacts(path, row):
    stem = hashlib.sha256(row["image_id"].encode()).hexdigest()[:20]
    root = path.parent / "runs" / row["variant"] / stem
    (root / "artifacts" / "graph").mkdir(parents=True)
    (root / "artifacts" / "qwen").mkdir()
    events = [
        {"event_type": "SAM3_ACTION_COMPLETED", "data": {"observation": {"num_detections": 5, "prompt": "green apple"}}},
        {"event_type": "ASSOCIATION_COMPLETED", "data": {"matched_nodes": 3, "new_nodes": 2}},
        {"event_type": "BUDGET_UPDATED", "data": {"sam3_calls": 7, "qwen_calls": 2, "sam3_tiles": 5}},
        {"event_type": "STOP_DECIDED", "data": {"reason": "RUNTIME_BUDGET"}},
    ]
    (root / "events.jsonl").write_text("".join(json.dumps(event)+"\n" for event in events))
    (root / "summary.json").write_text(json.dumps({"sam3_calls": 4, "qwen_calls": 1,
        "discovery_statistics": {"confidence_trace": [{"raw_soft_count": 2}, {"raw_soft_count": .25}]}}))
    graph = {"nodes": {"n1": {"status": "ACTIVE", "mask_geometry": {"pixel_area": 4},
        "class_belief": {"probabilities": {"target": .25}}, "diagnostics": {"duplicate_risk": .4}},
        "n2": {"status": "REJECTED", "class_belief": {"probabilities": {"target": .9}}}}}
    (root / "artifacts" / "graph" / "final_graph.json").write_text(json.dumps(graph))
    payload = {"input": {"request_text": "PRIVATE LONG EVIDENCE"}, "output": {"proposed_actions": [{"sam3_prompt": "green apples"}]},
        "metadata": {"accepted_action_count": 1, "correction_of": "previous-call", "rejections": [{"reason": "DUPLICATE"}],
                     "contract_diagnostic": "repair needed"}}
    (root / "artifacts" / "qwen" / "call.json").write_text(json.dumps(payload))
    return root


def save_rows(path, rows):
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))


def test_complete_report_computes_soft_counts_zero_gt_and_paired_gains(saved):
    dataset, path, _ = saved
    report = build_summary(dataset, path)
    a, b, c = (report["aggregates"][name] for name in ("A_Global", "B_Adaptive", "C_Qwen"))
    assert a["mae"] == a["rmse"] == 1 and a["mean_signed_error"] == 0
    assert a["undercount_images"] == a["overcount_images"] == 1
    assert b["mae"] == b["rmse"] == 0
    assert c["mae"] == .375 and c["count_type"] == "soft_posterior_count"
    assert c["mean_runtime_seconds"] == 2.5 and c["mean_qwen_calls"] == 1
    assert c["tiling_triggered"] == c["tiling_rows"] == 2
    assert report["paired_comparisons"][0]["mean_absolute_error_reduction"] == 1
    assert report["paired_comparisons"][0]["right_better"] == 2
    assert report["paired_comparisons"][1]["mean_absolute_error_reduction"] == -.375
    assert report["paired_comparisons"][1]["right_worse"] == 2
    assert report["per_image"][0]["gt_count"] == 0
    assert report["models"] == {"sam3": "facebook/sam3", "qwen": "local-qwen", "seed": 42}
    assert "secret" not in json.dumps(report)
    assert len(report["predictions_sha256"]) == len(report["config_sha256"]) == 64


def test_partial_report_never_reports_complete_mae_and_recovers_failed_budget(saved):
    dataset, path, rows = saved
    rows = [row for row in rows if not (row["variant"] == "B_Adaptive" and row["image_id"] == "b.jpg")]
    failed = rows[-1]
    failed.update(success=False, predicted_count=None, error="<GPU out of memory>")
    failed.pop("budget")
    failed.pop("stop_reason")
    artifacts(path, failed)
    save_rows(path, rows)
    report = build_summary(dataset, path)
    b, c = (report["aggregates"][name] for name in ("B_Adaptive", "C_Qwen"))
    assert b["mae"] is None and b["missing_runs"] == 1 and b["successful_subset_mae"] == 0
    assert c["mae"] is None and c["failed_runs"] == 1 and c["successful_subset_mae"] == .25
    record = report["per_image"][1]["variants"]["C_Qwen"]
    assert record["sam3_calls"] == 7 and record["qwen_calls"] == 2
    assert record["stop_reason"] == "RUNTIME_BUDGET"
    assert report["paired_comparisons"][0]["paired_images"] == 1
    assert not report["paired_comparisons"][0]["complete"]
    text = render_summary(report)
    assert "FAILED" in text and "MISSING" in text and "GPU out of memory" in text


def test_artifact_diagnostics_find_moved_runs_and_omit_full_evidence(saved):
    dataset, path, rows = saved
    artifacts(path, rows[-1])
    report = build_summary(dataset, path)
    record = report["per_image"][1]["variants"]["C_Qwen"]
    diag = record["diagnostics"]
    assert diag["available"] and diag["association_matches"] == 3 and diag["association_new_nodes"] == 2
    assert diag["raw_detections"] == 5 and diag["mask_nodes"] == diag["active_nodes"] == 1
    assert diag["target_probability_mean"] == .25 and diag["nodes_below_half"] == 1
    assert diag["bootstrap_target_mass"] == 2 and diag["final_target_mass"] == .25
    assert diag["qwen_correction_calls"] == 1 and diag["qwen_first_last"][0]["prompts"] == ["green apples"]
    assert report["aggregates"]["C_Qwen"]["qwen_rejections"] == {"DUPLICATE": 1}
    assert report["aggregates"]["C_Qwen"]["qwen_contract_diagnostics"] == {"repair needed": 1}
    assert "PRIVATE LONG EVIDENCE" not in json.dumps(report)
    # Canonical prediction budget takes precedence over log snapshots on successful rows.
    assert record["sam3_calls"] == 3
    assert diag['executed_sam3_prompts'] == [{'prompt': 'green apple', 'calls': 1}]


def test_soft_count_summary_separates_discovery_and_negative_mass(saved):
    dataset, path, rows = saved
    root = artifacts(path, rows[-1])
    (root / 'summary.json').write_text(json.dumps({'discovery_statistics': {'confidence_trace': [
        {'raw_soft_count': 2, 'new_nodes': 4, 'new_node_target_mass': 2},
        {'family': 'DISCOVERY', 'raw_soft_count': 2.5, 'new_nodes': 2, 'new_node_target_mass': .7,
         'existing_node_target_mass_change': -.2, 'existing_target_mass_lost': .3,
         'existing_target_mass_change_by_relation': {'NOT_RETRIEVED': -.2}},
        {'family': 'CONFOUNDER', 'raw_soft_count': .25, 'existing_node_target_mass_change': -2.25,
         'existing_target_mass_lost': 2.25,
         'existing_target_mass_change_by_relation': {'STRONG_MATCH': -2.25}}]}}))
    report = build_summary(dataset, path)
    diag = report['per_image'][1]['variants']['C_Qwen']['diagnostics']
    assert diag['candidate_to_soft_count_gap'] == .75
    positive = diag['confidence_by_family']['DISCOVERY']
    assert positive['new_nodes'] == 2 and positive['new_target_mass'] == .7
    assert positive['existing_target_mass_change'] == -.2
    assert positive['mass_change_by_relation'] == {'NOT_RETRIEVED': -.2}
    assert diag['confidence_by_family']['CONFOUNDER']['existing_target_mass_lost'] == 2.25
    rendered = render_summary(report)
    assert 'Candidates minus soft count=0.75' in rendered
    assert 'CONFOUNDER: new candidates=0' in rendered


def test_corrupt_artifacts_are_warned_without_losing_valid_predictions(saved):
    dataset, path, rows = saved
    root = artifacts(path, rows[-1])
    (root / "summary.json").write_text("{")
    with (root / "events.jsonl").open("a") as stream:
        stream.write("{unfinished\n")
    report = build_summary(dataset, path)
    a = report["aggregates"]["C_Qwen"]
    assert a["complete"] and a["mae"] == .375
    warnings = report["per_image"][1]["variants"]["C_Qwen"]["diagnostics"]["warnings"]
    assert any("summary.json" in warning for warning in warnings)
    assert any("Invalid event JSON" in warning for warning in warnings)
    assert "Artifact warnings" in render_summary(report)


def test_export_is_small_text_only_preserves_notes_and_never_opens_images(saved, monkeypatch):
    dataset, path, rows = saved
    artifacts(path, rows[-1])
    monkeypatch.setattr(Image, "open", lambda *a, **kw: pytest.fail("Report opened an image"))
    before = path.read_bytes()
    evaluation = path.parent / "evaluation"
    evaluation.mkdir()
    (evaluation / "metrics.json").write_text("existing AP metrics")
    result = write_summary(dataset, path)
    assert path.read_bytes() == before
    assert (evaluation / "metrics.json").read_text() == "existing AP metrics"
    with ZipFile(result["archive"]) as bundle:
        assert set(bundle.namelist()) == {"summary.md", "summary.json", "visual_notes.md"}
    assert result["archive_bytes"] < 20_000
    Path(result["visual_notes"]).write_text("My observation: D missed the bottle caps.\n")
    result = write_summary(dataset, path)
    assert "D missed the bottle caps" in Path(result["summary"]).read_text()
    assert "D missed the bottle caps" in Path(result["visual_notes"]).read_text()


def test_details_are_bounded_and_prioritize_missing_runs(saved):
    dataset, path, rows = saved
    save_rows(path, rows[:-1])
    report = build_summary(dataset, path, max_images=1, top_k=1, include_artifacts=False)
    assert len(report["per_image"]) == 1 and report["per_image"][0]["image_id"] == "b.jpg"
    assert report["omitted_image_details"] == 1
    assert all(len(a["worst_images"]) <= 1 for a in report["aggregates"].values())
    assert all(a["artifact_warning_runs"] == 0 for a in report["aggregates"].values())


@pytest.mark.parametrize("corruption", ["duplicate", "unknown_image", "unknown_variant", "count_type", "nan", "split", "json"])
def test_invalid_predictions_are_rejected(saved, corruption):
    dataset, path, rows = saved
    if corruption == "duplicate": rows.append(rows[0])
    elif corruption == "unknown_image": rows[0]["image_id"] = "unknown.jpg"
    elif corruption == "unknown_variant": rows[0]["variant"] = "F_Removed"
    elif corruption == "count_type": rows[0]["count_type"] = "different_rule"
    elif corruption == "nan": rows[0]["predicted_count"] = float("nan")
    elif corruption == "split":
        metadata = json.loads((path.parent / "metadata.json").read_text())
        metadata["split"] = "test"
        (path.parent / "metadata.json").write_text(json.dumps(metadata))
    save_rows(path, rows)
    if corruption == "json": path.write_text("{unfinished")
    with pytest.raises(ValueError):
        build_summary(dataset, path)
    assert not (path.parent / "report").exists()


def test_summary_cli(saved, capsys):
    dataset, path, _ = saved
    assert main(["summary", str(dataset), str(path), "--no-artifacts"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert Path(result["summary"]).exists() and Path(result["archive"]).exists()


def test_annotation_audit_is_separate_and_never_changes_official_scores(saved, tmp_path):
    dataset, path, _ = saved
    original = build_summary(dataset, path, include_artifacts=False)
    audit_path = tmp_path / 'audit.json'
    audit_path.write_text(json.dumps({'b.jpg': {'note': 'Some visible objects lack boxes', 'audited_count': 3}}))
    gt_before = (dataset / 'instances_val.json').read_bytes()
    predictions_before = path.read_bytes()
    audited = build_summary(dataset, path, include_artifacts=False, annotation_audit=audit_path)
    assert audited['aggregates'] == original['aggregates']
    assert audited['per_image'] == original['per_image']
    assert audited['annotation_audit']['count_metrics']['B_Adaptive']['mae'] == 1
    assert audited['annotation_audit']['count_metrics']['B_Adaptive']['complete']
    assert (dataset / 'instances_val.json').read_bytes() == gt_before
    assert path.read_bytes() == predictions_before
    assert 'Separate annotation audit' in render_summary(audited)


@pytest.mark.parametrize('content', [[], {'unknown.jpg': {'note': 'gap'}},
    {'b.jpg': {'audited_count': True}}, {'b.jpg': {'audited_count': -1}},
    {'b.jpg': {'audited_count': 1.5}}, {'b.jpg': {'note': 4}}, {'b.jpg': {'count': 3}}])
def test_invalid_annotation_audit_fails(saved, tmp_path, content):
    dataset, path, _ = saved
    audit_path = tmp_path / 'audit.json'
    audit_path.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        build_summary(dataset, path, annotation_audit=audit_path)


def test_audited_metrics_require_all_audited_images_to_succeed(saved, tmp_path):
    dataset, path, rows = saved
    rows[-1].update(success=False, predicted_count=None)
    save_rows(path, rows)
    audit_path = tmp_path / 'audit.json'
    audit_path.write_text(json.dumps({'b.jpg': {'audited_count': 3}}))
    audited = build_summary(dataset, path, annotation_audit=audit_path)
    metrics = audited['annotation_audit']['count_metrics']['C_Qwen']
    assert not metrics['complete'] and metrics['mae'] is None and metrics['rmse'] is None


def test_optional_ap_uses_only_complete_arms_and_fresh_predictions(saved, monkeypatch):
    from sam3_vlm.experiments import fscd147
    dataset, path, rows = saved
    rows[-1].update(success=False, predicted_count=None, error="failed")
    save_rows(path, rows)
    calls = []
    def fake_ap(annotation_path, detections):
        calls.append((annotation_path, detections))
        return {"AP": .4, "AP50": .6, "max_detections": 1000}
    monkeypatch.setattr(fscd147, "coco_ap", fake_ap)
    report = build_summary(dataset, path, compute_ap=True, include_artifacts=False)
    assert len(calls) == 2
    assert all(annotation_path == dataset / "instances_val.json" for annotation_path, _ in calls)
    assert all(len(detections) == 2 for _, detections in calls)
    assert report["aggregates"]["A_Global"]["AP"] == .4
    assert "AP" not in report["aggregates"]["C_Qwen"]
