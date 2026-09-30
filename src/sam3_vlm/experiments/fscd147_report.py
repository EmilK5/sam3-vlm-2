"""Compact, evaluation-only FSCD reports from frozen predictions and text logs."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from zipfile import ZIP_DEFLATED, ZipFile

from sam3_vlm.datasets.fscd147 import FSCD147, evaluate_counts


def _number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _mean(values):
    values = [value for value in values if _number(value)]
    return mean(values) if values else None


def _text(value, limit=500):
    value = str(value).replace("\n", " ").replace("\r", " ").replace("|", "\\|")
    return value if len(value) <= limit else value[:limit] + "…"


def _artifact_diagnostics(prediction_path, row):
    """Read JSON only, never image bytes, mask arrays or repeated evidence exports."""
    run_key = hashlib.sha256(row["image_id"].encode()).hexdigest()[:20]
    # Prefer the portable layout; absolute paths in old rows may refer to another machine.
    candidates = [prediction_path.parent / "runs" / row["variant"] / run_key]
    if row.get("artifact_directory"):
        candidates.append(Path(row["artifact_directory"]))
    run_dir = next((path for path in candidates if path.is_dir()), None)
    result = {"available": run_dir is not None, "warnings": [], "qwen_artifacts": 0,
              "qwen_rejections": {}, "qwen_contract_diagnostics": {}}
    if run_dir is None:
        result["warnings"].append("Run artifacts unavailable; prediction-level statistics only")
        return result

    def read_json(path):
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError("Expected a JSON object")
            return value
        except (OSError, ValueError) as exc:
            result["warnings"].append(f"Cannot read {path.name}: {_text(exc)}")
            return None

    summary_path = run_dir / "summary.json"
    audit_path = run_dir / "mask_audit.json"
    if audit_path.exists():
        result["mask_audit"] = read_json(audit_path)
    if summary_path.exists():
        summary = read_json(summary_path)
        if summary is not None:
            result["budget"] = {key: summary.get(key) for key in
                                ("sam3_calls", "qwen_calls", "sam3_tiles", "sam3_runtime_ms", "qwen_runtime_ms")}
            result["stop_reason"] = summary.get("final_stop_reason")
            discovery = summary.get("discovery_statistics", {})
            result["adaptive_tiling"] = discovery.get("adaptive_tiling")
            trace = discovery.get("confidence_trace", [])
            if trace:
                result["bootstrap_target_mass"] = trace[0].get("raw_soft_count")
                result["final_target_mass"] = trace[-1].get("raw_soft_count")
                by_family = {}
                for step in trace:
                    family = step.get("family") or "BOOTSTRAP"
                    totals = by_family.setdefault(family, {"steps": 0, "new_nodes": 0,
                        "new_target_mass": 0.0, "existing_target_mass_change": 0.0,
                        "existing_target_mass_lost": None, "removed_target_mass": 0.0,
                        "mass_change_by_relation": {}})
                    totals["steps"] += 1
                    for dest, source in (("new_nodes", "new_nodes"),
                            ("new_target_mass", "new_node_target_mass"),
                            ("existing_target_mass_change", "existing_node_target_mass_change"),
                            ("existing_target_mass_lost", "existing_target_mass_lost"),
                            ("removed_target_mass", "removed_node_target_mass")):
                        if _number(step.get(source)):
                            totals[dest] = (totals[dest] or 0) + step[source]
                    for relation, mass in step.get("existing_target_mass_change_by_relation", {}).items():
                        if _number(mass):
                            totals["mass_change_by_relation"][relation] = totals["mass_change_by_relation"].get(relation, 0.0) + mass
                result["confidence_by_family"] = by_family

    events_path = run_dir / "events.jsonl"
    event_count = matched = new = raw_detections = 0
    empty_masks_skipped = empty_mask_calls = 0
    executed_prompts = Counter()
    if events_path.exists():
        with events_path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict) or not isinstance(event.get("data", {}), dict):
                        raise ValueError("Invalid event structure")
                except ValueError:
                    result["warnings"].append(f"Invalid event JSON at line {line_number}")
                    continue
                event_count += 1
                data = event.get("data", {})
                kind = event.get("event_type")
                if kind == "BUDGET_UPDATED":
                    result["budget"] = data
                elif kind == "STOP_DECIDED":
                    result["stop_reason"] = data.get("reason")
                elif kind == "DISCOVERY_STATE_UPDATED":
                    if data.get("adaptive_tiling") is not None:
                        result["adaptive_tiling"] = data["adaptive_tiling"]
                elif kind == "SAM3_ACTION_COMPLETED":
                    observation = data.get("observation", {})
                    validation = observation.get("model_metadata", {}).get("mask_validation", {})
                    raw_detections += validation.get("raw_detections", observation.get("num_detections", 0))
                    empty_masks_skipped += validation.get("empty_masks_skipped", 0)
                    empty_mask_calls += validation.get("empty_masks_skipped", 0) > 0
                    if isinstance(observation.get("prompt"), str):
                        executed_prompts[observation["prompt"]] += 1
                elif kind == "ASSOCIATION_COMPLETED":
                    matched += data.get("matched_nodes", 0)
                    new += data.get("new_nodes", 0)
        result.update(event_count=event_count, raw_detections=raw_detections,
                      empty_masks_skipped=empty_masks_skipped, empty_mask_calls=empty_mask_calls,
                      association_matches=matched, association_new_nodes=new)
        result["executed_sam3_prompts"] = [{"prompt": _text(prompt, 150), "calls": calls}
                                           for prompt, calls in list(executed_prompts.items())[:12]]
        result["omitted_sam3_prompts"] = max(0, len(executed_prompts) - 12)

    graph_path = run_dir / "artifacts" / "graph" / "final_graph.json"
    if graph_path.exists():
        graph = read_json(graph_path)
        if graph is not None:
            active = [node for node in graph.get("nodes", {}).values() if node.get("status") == "ACTIVE"]
            probabilities = [node.get("class_belief", {}).get("probabilities", {}).get("target") for node in active]
            result.update(active_nodes=len(active), mask_nodes=sum("mask_geometry" in node for node in active),
                          target_probability_mean=_mean(probabilities),
                          nodes_below_half=sum(_number(p) and p < .5 for p in probabilities),
                          mean_duplicate_risk=_mean([node.get("diagnostics", {}).get("duplicate_risk") for node in active]))
            if all(_number(p) for p in probabilities):
                result["candidate_to_soft_count_gap"] = len(active) - sum(probabilities)
            result["rejected_mask_nodes"] = sum(node.get("status") == "REJECTED" and
                "mask_geometry" in node for node in graph.get("nodes", {}).values())

    rejections, contracts = Counter(), Counter()
    correction_calls = 0
    finish_reasons = Counter()
    malformed_attempts = failed_qwen_calls = 0
    response_failures = []
    examples = []
    for path in sorted((run_dir / "artifacts" / "qwen").glob("*.json")):
        artifact = read_json(path)
        if artifact is None:
            continue
        result["qwen_artifacts"] += 1
        metadata = artifact.get("metadata", {})
        failed_qwen_calls += metadata.get("status") == "FAILED"
        for attempt in metadata.get("call_attempts", []):
            finish_reasons[str(attempt.get("finish_reason") or "unavailable")] += 1
            malformed_attempts += attempt.get("parse_valid") is False and not attempt.get("error")
            if attempt.get("parse_valid") is False:
                raw = attempt.get("raw_output")
                text = raw if isinstance(raw, str) else json.dumps(raw)
                failure = {key: attempt.get(key) for key in (
                    "phase", "finish_reason", "usage", "error", "max_output_tokens", "temperature", "sampling_seed")}
                failure.update(qwen_call_id=artifact.get("qwen_call_id"), raw_output=text[:8000],
                               raw_output_chars=len(text), raw_output_truncated=len(text) > 8000)
                if len(response_failures) < 2:
                    response_failures.append(failure)
                else:
                    response_failures[-1] = failure
        rejections.update(str(item.get("reason", "UNKNOWN")) for item in metadata.get("rejections", []))
        if metadata.get("contract_diagnostic"):
            contracts[str(metadata["contract_diagnostic"])] += 1
        correction_calls += metadata.get("correction_of") is not None
        actions = artifact.get("output", {}).get("proposed_actions", [])
        example = {"prompts": [_text(action.get("sam3_prompt", action.get("prompt", "")), 150)
                               for action in actions[:3]],
                   "accepted_actions": metadata.get("accepted_action_count")}
        assessments = artifact.get("output", {}).get("confounder_assessments", [])
        if assessments:
            example["confounder_assessments"] = [
                {key: _text(item.get(key, ""), 200) for key in ("label", "relationship", "reason")}
                for item in assessments[:4] if isinstance(item, dict)]
        unsafe = [item for item in metadata.get("rejections", []) if item.get("reason") == "UNSAFE_CONFOUNDER"]
        if unsafe:
            example["unsafe_confounders"] = [{key: _text(item.get(key, ""), 200)
                for key in ("sam3_prompt", "detail")} for item in unsafe[:4]]
        if not examples:
            examples.append(example)
        elif len(examples) == 1:
            examples.append(example)
        else:
            examples[-1] = example
    result.update(qwen_rejections=dict(rejections), qwen_contract_diagnostics=dict(contracts),
                  qwen_finish_reasons=dict(finish_reasons), qwen_malformed_attempts=malformed_attempts,
                  qwen_failed_calls=failed_qwen_calls, qwen_response_failures=response_failures,
                  qwen_correction_calls=correction_calls, qwen_first_last=examples)
    if not summary_path.exists() and not events_path.exists() and not graph_path.exists() and not result["qwen_artifacts"]:
        result["available"] = False
        result["warnings"].append("Run directory has no readable diagnostic artifacts")
    expected_qwen = (row.get("budget") or result.get("budget", {})).get("qwen_calls", 0)
    if _number(expected_qwen) and expected_qwen > 0 and not result["qwen_artifacts"]:
        result["warnings"].append("Qwen calls recorded but no readable Qwen artifacts (requests may have failed)")
    return result


def build_summary(dataset_root, prediction_path, *, split="val", top_k=3, max_images=10,
                  include_artifacts=True, compute_ap=False, annotation_audit=None):
    """Count failures and missing rows explicitly; never infer visual quality from counts."""
    if top_k < 1 or max_images < 1:
        raise ValueError("top_k and max_images must be positive")
    dataset = FSCD147(dataset_root, split)
    prediction_path = Path(prediction_path)
    metadata = json.loads((prediction_path.parent / "metadata.json").read_text())
    if metadata.get("schema_version") != 1 or metadata.get("dataset") != "FSCD-147":
        raise ValueError("Invalid FSCD prediction metadata")
    if metadata["split"] != split or metadata["image_ids"] != list(dataset.image_ids):
        raise ValueError("Predictions do not match the summary split; use the same dataset root as inference")
    variants = metadata["variants"]
    if not variants:
        raise ValueError("Prediction metadata has no variants")
    truth = dataset.ground_truth()
    audit = {}
    if annotation_audit is not None:
        audit = json.loads(Path(annotation_audit).read_text())
        if not isinstance(audit, dict):
            raise ValueError("Annotation audit must map image IDs to notes and optional audited_count")
        for image_id, item in audit.items():
            if image_id not in truth or not isinstance(item, dict):
                raise ValueError("Unknown image or invalid annotation audit entry")
            if set(item) - {"note", "audited_count"} or not isinstance(item.get("note", ""), str):
                raise ValueError("Audit entries accept a note and optional audited_count only")
            if "audited_count" in item and (type(item["audited_count"]) is not int or item["audited_count"] < 0):
                raise ValueError("audited_count must be a nonnegative integer")
    rows = {variant: {} for variant in variants}
    with prediction_path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"Invalid prediction JSON at line {line_number}") from exc
            variant, image_id = row["variant"], row["image_id"]
            if variant not in variants or image_id not in truth or image_id in rows[variant]:
                raise ValueError("Unknown or duplicate prediction")
            if row["count_type"] != variants[variant]["count_type"]:
                raise ValueError("Prediction count rule differs from frozen metadata")
            if type(row["success"]) is not bool:
                raise ValueError("Prediction success must be boolean")
            if row["success"] and (not _number(row["predicted_count"]) or row["predicted_count"] < 0):
                raise ValueError("Successful prediction has an invalid count")
            diagnostic = _artifact_diagnostics(prediction_path, row) if include_artifacts else {}
            if row.get("mask_audit") is not None:
                diagnostic["mask_audit"] = row["mask_audit"]
            budget = dict(diagnostic.get("budget", {}))
            budget.update(row.get("budget", {}))
            tiling = row.get("adaptive_tiling") or diagnostic.get("adaptive_tiling")
            signed_error = row["predicted_count"] - truth[image_id].count if row["success"] else None
            record = {"image_id": image_id, "target": row.get("target"), "gt_count": truth[image_id].count,
                      "inference_target": row.get("inference_target", row.get("target")),
                      "success": row["success"], "predicted_count": row["predicted_count"] if row["success"] else None,
                      "signed_error": signed_error, "absolute_error": abs(signed_error) if signed_error is not None else None,
                      "candidate_count": len(row.get("nodes", [])) if row["success"] else None,
                      "runtime_seconds": row.get("runtime_seconds"),
                      "stop_reason": row.get("stop_reason") or diagnostic.get("stop_reason"),
                      "error": _text(row.get("error", "Unknown failure")) if not row["success"] else None,
                      "sam3_calls": budget.get("sam3_calls"), "qwen_calls": budget.get("qwen_calls"),
                      "sam3_tiles": budget.get("sam3_tiles"),
                      "sam3_seconds": budget.get("sam3_runtime_ms", 0) / 1000 if _number(budget.get("sam3_runtime_ms")) else None,
                      "qwen_seconds": budget.get("qwen_runtime_ms", 0) / 1000 if _number(budget.get("qwen_runtime_ms")) else None,
                      "tiling": ({"trigger": tiling.get("trigger"), "density_score": tiling.get("density_score"),
                                  "seed_count": tiling.get("object_count"), "tile_size": tiling.get("tile_size"),
                                  "trigger_reason": tiling.get("trigger_reason"),
                                  "candidate_count": tiling.get("candidate_count"),
                                  "small_candidate_count": tiling.get("small_candidate_count"),
                                  "planned_tiles": len(tiling.get("tiles", []))} if tiling else None),
                      "diagnostics": {key: value for key, value in diagnostic.items()
                                      if key not in {"budget", "adaptive_tiling", "stop_reason"}}}
            rows[variant][image_id] = (record, row)

    aggregates = {}
    for variant, entries in rows.items():
        records = [item[0] for item in entries.values()]
        valid = [row for row in records if row["success"]]
        metrics = evaluate_counts({row["image_id"]: row["predicted_count"] for row in records}, truth)
        runtimes = [row["runtime_seconds"] for row in records if _number(row["runtime_seconds"])]
        rejections, contracts = Counter(), Counter()
        for row in records:
            rejections.update(row["diagnostics"].get("qwen_rejections", {}))
            contracts.update(row["diagnostics"].get("qwen_contract_diagnostics", {}))
        aggregates[variant] = dict(metrics, count_type=variants[variant]["count_type"],
            failed_runs=sum(not row["success"] for row in records), missing_runs=len(truth)-len(records),
            successful_subset_mae=_mean([row["absolute_error"] for row in valid]),
            mean_signed_error=_mean([row["signed_error"] for row in valid]),
            undercount_images=sum(row["signed_error"] < -1e-9 for row in valid),
            overcount_images=sum(row["signed_error"] > 1e-9 for row in valid),
            mean_runtime_seconds=_mean(runtimes), median_runtime_seconds=median(runtimes) if runtimes else None,
            max_runtime_seconds=max(runtimes) if runtimes else None, total_runtime_seconds=sum(runtimes),
            mean_sam3_calls=_mean([row["sam3_calls"] for row in records]),
            mean_qwen_calls=_mean([row["qwen_calls"] for row in records]),
            mean_sam3_tiles=_mean([row["sam3_tiles"] for row in records]),
            budget_rows=sum(row["sam3_calls"] is not None and row["qwen_calls"] is not None for row in records),
            tiling_triggered=sum(bool(row["tiling"]) and row["tiling"]["trigger"] is True for row in records),
            tiling_rows=sum(row["tiling"] is not None for row in records),
            artifact_rows=sum(row["diagnostics"].get("available", False) for row in records),
            artifact_warning_runs=sum(bool(row["diagnostics"].get("warnings")) for row in records),
            artifact_warnings=[{"image_id": row["image_id"], "warnings": row["diagnostics"]["warnings"]}
                               for row in records if row["diagnostics"].get("warnings")][:max_images],
            stop_reasons=dict(Counter(row["stop_reason"] or ("UNKNOWN" if row["success"] else "FAILED") for row in records)),
            qwen_rejections=dict(rejections), qwen_contract_diagnostics=dict(contracts),
            empty_masks_skipped=sum(row["diagnostics"].get("empty_masks_skipped", 0) for row in records),
            qwen_malformed_attempts=sum(row["diagnostics"].get("qwen_malformed_attempts", 0) for row in records),
            qwen_failed_calls=sum(row["diagnostics"].get("qwen_failed_calls", 0) for row in records),
            failures=[{"image_id": row["image_id"], "error": row["error"]}
                      for row in records if not row["success"]][:max_images],
            worst_images=sorted(valid, key=lambda row: (-row["absolute_error"], row["image_id"]))[:top_k])
        if compute_ap and metrics["complete"]:
            from sam3_vlm.experiments.fscd147 import coco_ap, export_detections
            aggregates[variant].update(coco_ap(dataset.root / f"instances_{split}.json",
                                              export_detections({key: value[1] for key, value in entries.items()}, truth)))

    comparisons = []
    ordered_variants = sorted(variants)
    for left, right in zip(ordered_variants, ordered_variants[1:]):
        pairs = [(rows[left][key][0], rows[right][key][0]) for key in dataset.image_ids
                 if key in rows[left] and key in rows[right]
                 and rows[left][key][0]["success"] and rows[right][key][0]["success"]]
        reductions = [a["absolute_error"]-b["absolute_error"] for a, b in pairs]
        comparisons.append({"left": left, "right": right, "paired_images": len(pairs),
            "complete": len(pairs) == len(truth), "mean_absolute_error_reduction": _mean(reductions),
            "right_better": sum(value > 1e-9 for value in reductions),
            "right_worse": sum(value < -1e-9 for value in reductions),
            "tied": sum(abs(value) <= 1e-9 for value in reductions)})

    def priority(image_id):
        image_rows = [rows[variant].get(image_id, ({"success": False}, None))[0] for variant in variants]
        errors = [row.get("absolute_error") for row in image_rows]
        return (-sum(not row["success"] for row in image_rows), -(_mean(errors) or 0), image_id)

    selected = (list(dataset.image_ids) if len(truth) <= max_images
                else sorted(dataset.image_ids, key=priority)[:max_images])
    configs = {variant: details.get("config", {}) for variant, details in variants.items()}
    config_hash = hashlib.sha256(json.dumps(configs, sort_keys=True).encode()).hexdigest()
    sample_path = dataset.root / "sample_manifest.json"
    sample = json.loads(sample_path.read_text()) if sample_path.exists() else None
    audit_metrics = {}
    audited_ids = [key for key, item in audit.items() if "audited_count" in item]
    for variant in variants:
        errors = [rows[variant][key][0]["predicted_count"] - audit[key]["audited_count"]
                  for key in audited_ids if key in rows[variant] and rows[variant][key][0]["success"]]
        complete = bool(audited_ids) and len(errors) == len(audited_ids)
        audit_metrics[variant] = {"expected_images": len(audited_ids), "successful_images": len(errors),
            "complete": complete, "mae": _mean([abs(error) for error in errors]) if complete else None,
            "rmse": math.sqrt(mean(error**2 for error in errors)) if complete else None}
    return {"schema_version": 1, "dataset": "FSCD-147", "split": split, "images_per_arm": len(truth),
            "predictions_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
            "config_sha256": config_hash, "models": {key: value for key, value in metadata.get("model_settings", {}).items()
                                                     if key in {"sam3", "qwen", "seed"}},
            "sample": {key: value for key, value in sample.items()
                       if key in {"seed", "split", "count", "image_ids"}} if sample is not None else None,
            "aggregates": aggregates, "paired_comparisons": comparisons,
            "annotation_audit": {"entries": audit, "count_metrics": audit_metrics,
                "note": "User-audited counts are separate from official metrics; inference and official annotations are unchanged."},
            "per_image": [{"image_id": key, "gt_count": truth[key].count,
                           "variants": {variant: rows[variant].get(key, (None, None))[0] for variant in variants}}
                          for key in selected],
            "omitted_image_details": len(truth)-len(selected),
            "configurations": {variant: {key: config.get(key) for key in
                                ("budget", "bootstrap", "tiling", "association", "belief", "sam3", "planner", "replanning")}
                               for variant, config in configs.items()},
            "notes": ["Ground-truth counts come from FSCD bounding-box annotations.",
                      "A/B count candidates; C/D/E sum target probabilities.",
                      "Full declared-split MAE/RMSE remain unavailable if any image failed or is missing.",
                      "Successful-subset metrics and paired comparisons exclude failed/missing runs.",
                      "Count agreement and association matches do not establish localization quality or correct deduplication.",
                      "No images, masks, full graphs, Qwen input evidence, or credentials are exported."]}


def render_summary(report):
    def num(value):
        return f"{value:.2f}" if _number(value) else "—"

    aggregates = report["aggregates"]
    lines = ["# FSCD-147 results", "", f"Split: {report['split']}; images per arm: {report['images_per_arm']}.",
             f"Models / seed: `{json.dumps(report['models'], sort_keys=True)}`.",
             f"Prediction SHA256: `{report['predictions_sha256']}`.",
             f"Configuration SHA256: `{report['config_sha256']}`.", ""]
    if report["sample"]:
        lines += [f"Random sample seed: {report['sample'].get('seed')}. These are smoke-test results.", ""]
    lines += ["| Arm | OK / expected | Failed / missing | MAE | RMSE | Mean signed error* | Mean / max seconds† | Mean SAM / Qwen calls† |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for variant, a in aggregates.items():
        lines.append(f"| {_text(variant)} | {a['evaluated_images']}/{a['total_images']} | {a['failed_runs']}/{a['missing_runs']} | "
                     f"{num(a['mae'])} | {num(a['rmse'])} | {num(a['mean_signed_error'])} | "
                     f"{num(a['mean_runtime_seconds'])}/{num(a['max_runtime_seconds'])} | "
                     f"{num(a['mean_sam3_calls'])}/{num(a['mean_qwen_calls'])} |")
    lines += ["", "* Signed error = prediction − ground truth; successful runs only. Negative means undercounting.",
              "† Runtime/calls include failed runs when recorded; missing measurements are excluded, not zeroed.",
              "A/B count candidates; C–E use soft counts. MAE/RMSE require a complete declared split.", "",
              "## Per-image counts", "", "| Image / target | GT | " + " | ".join(_text(v) for v in aggregates) + " |",
              "|---|---:|" + "---:|" * len(aggregates)]
    for image in report["per_image"]:
        target = next((row["target"] for row in image["variants"].values() if row and row["target"]), "")
        cells = ["MISSING" if row is None else "FAILED" if not row["success"] else num(row["predicted_count"])
                 for row in image["variants"].values()]
        lines.append(f"| {_text(image['image_id'])} / {_text(target)} | {image['gt_count']} | " + " | ".join(cells) + " |")
    if report["omitted_image_details"]:
        lines += ["", f"{report['omitted_image_details']} image details omitted; shown cases prioritize failures and large errors."]
    lines += ["", "## Paired comparisons", "", "Positive error reduction favors the right-hand arm.", "",
              "| Comparison | Paired / expected | Error reduction | Right better / worse / tied |",
              "|---|---:|---:|---:|"]
    for pair in report["paired_comparisons"]:
        lines.append(f"| {_text(pair['left'])} → {_text(pair['right'])} | {pair['paired_images']}/{report['images_per_arm']} | "
                     f"{num(pair['mean_absolute_error_reduction'])} | {pair['right_better']}/{pair['right_worse']}/{pair['tied']} |")
    for variant, a in aggregates.items():
        lines += ["", f"## {_text(variant)} diagnostics", "",
                  f"Count type: {a['count_type']}. Undercount / overcount images: {a['undercount_images']}/{a['overcount_images']}.",
                  f"Artifact rows: {a['artifact_rows']}; budget rows: {a['budget_rows']}; "
                  f"adaptive tiling triggered: {a['tiling_triggered']}/{a['tiling_rows']} recorded decisions.",
                  f"Stop reasons: `{json.dumps(a['stop_reasons'], sort_keys=True)}`.",
                  f"Qwen rejections: `{json.dumps(a['qwen_rejections'], sort_keys=True)}`.",
                  f"Qwen contract diagnostics: `{json.dumps(a['qwen_contract_diagnostics'], sort_keys=True)}`."]
        if "AP" in a:
            lines.append(f"BBox AP / AP50 (max 1000 detections): {num(a['AP'])}/{num(a['AP50'])}.")
        if not a["complete"]:
            lines.append(f"Successful-subset MAE: {num(a['successful_subset_mae'])}; excludes failures and missing predictions.")
        if a["failures"]:
            lines.append("")
        for failure in a["failures"]:
            lines.append(f"- FAILED {_text(failure['image_id'])}: {failure['error']}")
        lines += ["", "Largest successful count errors:", ""]
        for row in a["worst_images"]:
            tile = row["tiling"]
            diag = row["diagnostics"]
            lines.append(f"- {_text(row['image_id'])} ({_text(row['target'])}): GT {row['gt_count']}, "
                         f"pred {num(row['predicted_count'])}, error {num(row['signed_error'])}, "
                         f"candidates {row['candidate_count']}, {num(row['runtime_seconds'])}s, "
                         f"SAM/Qwen {num(row['sam3_calls'])}/{num(row['qwen_calls'])}.")
            if tile:
                lines.append(f"  Tiling: trigger={tile['trigger']}, reason={tile.get('trigger_reason') or 'unrecorded'}, density={num(tile['density_score'])}, "
                             f"seeds={tile['seed_count']}, planned tiles={tile['planned_tiles']}.")
            if "target_probability_mean" in diag:
                lines.append(f"  Mean target probability={num(diag['target_probability_mean'])}, "
                             f"nodes below 0.5={diag['nodes_below_half']}, mask nodes={diag['mask_nodes']}/{diag['active_nodes']}.")
            if "candidate_to_soft_count_gap" in diag:
                lines.append(f"  Candidates minus soft count={num(diag['candidate_to_soft_count_gap'])}.")
            if diag.get("mask_audit"):
                audit = diag["mask_audit"]
                lines.append(f"  Mask audit: complete={audit['complete']}, high-IoU pairs={audit['high_iou_pairs']}, "
                             f"high-IoM pairs={audit['high_iom_pairs']}, large-ratio containment pairs={audit['large_area_ratio_containment_pairs']}.")
            for family, mass in diag.get("confidence_by_family", {}).items():
                lines.append(f"  {family}: new candidates={mass['new_nodes']}, new target mass={num(mass['new_target_mass'])}, "
                             f"existing mass change={num(mass['existing_target_mass_change'])}, "
                             f"existing mass lost={num(mass['existing_target_mass_lost'])}, "
                             f"removed mass={num(mass['removed_target_mass'])}.")
            if diag.get("empty_masks_skipped"):
                lines.append(f"  Empty masks skipped={diag['empty_masks_skipped']} across {diag['empty_mask_calls']} sensor calls.")
            if diag.get("qwen_finish_reasons"):
                lines.append(f"  Qwen finish reasons={json.dumps(diag['qwen_finish_reasons'])}; "
                             f"malformed attempts={diag['qwen_malformed_attempts']}; failed calls={diag['qwen_failed_calls']}.")
            for failure in diag.get("qwen_response_failures", []):
                lines.append(f"  Qwen {failure['phase']} response failure: finish_reason={failure['finish_reason']}, "
                             f"usage={failure['usage']}, error={_text(failure['error'])}. Raw response is in summary.json.")
            for example in diag.get("qwen_first_last", []):
                if example["prompts"]:
                    lines.append(f"  Qwen proposals: {'; '.join(example['prompts'])}; accepted actions={example['accepted_actions']}.")
                for assessment in example.get("confounder_assessments", []):
                    lines.append(f"  Negative assessment: {assessment['label']} / {assessment['relationship']}: {assessment['reason']}.")
                for rejection in example.get("unsafe_confounders", []):
                    lines.append(f"  Rejected negative: {rejection['sam3_prompt']}: {rejection['detail']}.")
            if diag.get("executed_sam3_prompts"):
                lines.append("  Executed SAM3 queries: " + "; ".join(
                    f"{item['prompt']} ({item['calls']} calls)" for item in diag['executed_sam3_prompts']) + ".")
        if a["artifact_warning_runs"]:
            lines += ["", f"Artifact warnings in {a['artifact_warning_runs']} runs:", ""]
            for warning in a["artifact_warnings"]:
                lines.append(f"- {_text(warning['image_id'])}: " + "; ".join(warning["warnings"][:5]))
    audit = report.get("annotation_audit", {})
    if audit.get("entries"):
        lines += ["", "## Separate annotation audit", "", audit["note"], ""]
        for image_id, item in audit["entries"].items():
            lines.append(f"- {_text(image_id)}: {_text(item.get('note', ''))}; audited count={item.get('audited_count', 'not supplied')}.")
        if any("audited_count" in item for item in audit["entries"].values()):
            lines += ["", "| Arm | Successful / audited images | Audited MAE | Audited RMSE |", "|---|---:|---:|---:|"]
            for variant, metrics in audit["count_metrics"].items():
                lines.append(f"| {_text(variant)} | {metrics['successful_images']}/{metrics['expected_images']} | "
                             f"{num(metrics['mae'])} | {num(metrics['rmse'])} |")
    lines += ["", "Count agreement and association matches do not prove correct masks or deduplication. "
              "Add visual observations about misses, false positives, duplicates, fragments, and slow behavior."]
    return "\n".join(lines) + "\n"


def write_summary(dataset_root, prediction_path, *, output_dir=None, **kwargs):
    report = build_summary(dataset_root, prediction_path, **kwargs)
    output_dir = Path(output_dir or Path(prediction_path).parent / "report")
    output_dir.mkdir(parents=True, exist_ok=True)
    notes_path = output_dir / "visual_notes.md"
    if not notes_path.exists():
        lines = ["# My visual observations", "", "Overall pain points:", "", "",
                 "For each image, note affected arms and missed objects / false positives / duplicates / fragments / slowness.", ""]
        for image in report["per_image"]:
            lines += [f"- {image['image_id']} (GT {image['gt_count']}): "]
        notes_path.write_text("\n".join(lines) + "\n")
    summary_path = output_dir / "summary.md"
    summary_path.write_text(render_summary(report) + "\n" + notes_path.read_text())
    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    archive_path = output_dir / "summary.zip"
    with ZipFile(archive_path, "w", ZIP_DEFLATED) as bundle:
        for path in (summary_path, json_path, notes_path):
            bundle.write(path, path.name)
    return {"summary": str(summary_path), "json": str(json_path), "visual_notes": str(notes_path),
            "archive": str(archive_path), "archive_bytes": archive_path.stat().st_size}
