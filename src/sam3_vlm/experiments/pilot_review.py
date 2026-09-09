"""Small, deterministic review exports; full run artifacts remain on disk."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PROMPT_CONTRASTS = (
    ("D_negative_miss_policy", "D_CurrentNegativeEvidence", "D_NeutralNegativeMisses"),
    ("D_discovery_050_vs_025", "D_Discovery_050_Exemplars", "D_Discovery_025_NoExemplars"),
    ("D_discovery_050_vs_020", "D_Discovery_050_Exemplars", "D_Discovery_020_NoExemplars"),
    ("D_discovery_025_vs_020", "D_Discovery_025_NoExemplars", "D_Discovery_020_NoExemplars"),
    ("D_discovery_020_vs_015", "D_Discovery_020_NoExemplars", "D_Discovery_015_NoExemplars"),
    ("D_rejection_correction", "D_ReferenceOld", "D_OldWithCorrection"),
    ("D_evidence_prompt", "D_OldWithCorrection", "D_V4WithCorrection"),
    ("C_prompt", "C_OldPrompt", "C_NewPrompt"),
    ("D_prompt", "D_OldPrompt", "D_NewPrompt"),
    ("E_prompt_without_negatives", "E1_OldPrompt_NoNegatives", "E3_NewPrompt_NoNegatives"),
    ("E_prompt_with_negatives", "E2_OldPrompt_WithNegatives", "E4_NewPrompt_WithNegatives"),
    ("E_negatives_old_prompt", "E1_OldPrompt_NoNegatives", "E2_OldPrompt_WithNegatives"),
    ("E_negatives_new_prompt", "E3_NewPrompt_NoNegatives", "E4_NewPrompt_WithNegatives"),
)


def _paired_rows(report, left, right):
    arms = [
        {r["sample_id"]: r for r in report["samples"]
         if r["variant"] == name and r.get("success")}
        for name in (left, right)
    ]
    return [(arms[0][key], arms[1][key]) for key in sorted(arms[0].keys() & arms[1].keys())
            if arms[0][key]["gt_count"] == arms[1][key]["gt_count"]]


def prompt_comparisons(report):
    """Positive error reduction means the right-hand variant is better."""
    variants = report["metadata"]["variants"]
    expected = report["metadata"]["sample_count"]
    result = {}
    for label, left, right in PROMPT_CONTRASTS:
        if left not in variants or right not in variants:
            continue
        pairs = _paired_rows(report, left, right)
        changes = [a["absolute_error"] - b["absolute_error"] for a, b in pairs]

        def mean(values):
            return sum(values) / len(values) if values else None

        result[label] = {
            "left": left, "right": right,
            "n_paired": len(pairs), "n_expected": expected, "complete": len(pairs) == expected,
            "left_MAE": mean([a["absolute_error"] for a, _ in pairs]),
            "right_MAE": mean([b["absolute_error"] for _, b in pairs]),
            "mean_absolute_error_reduction": mean(changes),
            "right_better_images": sum(v > 1e-9 for v in changes),
            "right_worse_images": sum(v < -1e-9 for v in changes),
            "tied_images": sum(abs(v) <= 1e-9 for v in changes),
            "mean_extra_runtime_ms": mean([b["runtime_ms"] - a["runtime_ms"] for a, b in pairs]),
            "mean_extra_sam3_calls": mean([b["sam3_calls"] - a["sam3_calls"] for a, b in pairs]),
            "mean_extra_qwen_calls": mean([b["qwen_calls"] - a["qwen_calls"] for a, b in pairs]),
        }
    return result


def _select_cases(report, limit=3):
    """At most three images: failures, prompt gains/regressions, worst error."""
    selected = {}

    def add(sample, reason):
        if sample in selected:
            selected[sample].append(reason)
        elif len(selected) < limit:
            selected[sample] = [reason]

    failed = sorted({r["sample_id"] for r in report["samples"] if not r.get("success")})
    if failed:
        add(failed[0], "execution or validation failure")
    changes = [
        (a["absolute_error"] - b["absolute_error"], a["sample_id"])
        for label, left, right in PROMPT_CONTRASTS if ("_prompt" in label or "_discovery_" in label or label == "D_negative_miss_policy") and not label.startswith("E_negatives")
        for a, b in _paired_rows(report, left, right)
    ]
    if changes:
        best = max(changes, key=lambda item: (item[0], item[1]))
        worst = min(changes)
        if best[0] > 1e-9:
            add(best[1], "largest improvement from the comparison variant")
        if worst[0] < -1e-9:
            add(worst[1], "largest regression from the comparison variant")
    for row in sorted(report["samples"], key=lambda r: (-r.get("absolute_error", -1), r["sample_id"])):
        add(row["sample_id"], "large remaining count error")
        if len(selected) >= limit:
            break
    return selected


def final_evaluation_summary(report):
    """Presentation notes generated from the measured report, including failures."""
    def number(value):
        return f"{value:.3f}" if value is not None else "unavailable"

    n = report["metadata"]["sample_count"]
    lines = ["# Final negative-evidence comparison", "",
        f"Images per variant: {n}. Two D variants, at most two Qwen calls per image.", "",
        "Both use positive threshold 0.20 without target exemplars, negative threshold 0.50, "
        "soft counts, IoU+IoM, and the same corrected scope/prompt instructions.", "",
        "Current: a missed confounder can raise target probability. "
        "Neutral: a missed confounder has likelihood 1; matched confounder evidence still lowers target probability.", "",
        "| Variant | Successful / expected | MAE | RMSE | Signed error | Mean runtime (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in report["metadata"]["variants"]:
        a = report["aggregates"].get(variant, {})
        runtime = a.get("avg_runtime_ms")
        lines.append(f"| {variant} | {a.get('n_samples', 0)} / {a.get('n_expected', n)} | "
                     f"{number(a.get('MAE'))} | {number(a.get('RMSE'))} | "
                     f"{number(a.get('mean_signed_error'))} | "
                     f"{number(runtime / 1000 if runtime is not None else None)} |")
    pair = report.get("paired_comparisons", {}).get("D_negative_miss_policy", {})
    lines += ["", f"Paired successful images: {pair.get('n_paired', 0)} / {n}. "
              f"Complete comparison: {pair.get('complete', False)}.",
              "MAE reduction with neutral misses (positive favors neutral): "
              f"{number(pair.get('mean_absolute_error_reduction'))}.",
              f"Neutral better / worse / tied: {pair.get('right_better_images', 0)} / "
              f"{pair.get('right_worse_images', 0)} / {pair.get('tied_images', 0)}.", "",
              "Counts are sums of uncalibrated target probabilities. New candidates are not confirmed fruit. "
              "Qwen proposals may differ between runs; this compares the complete adaptive policies.",
              "This is a development-set evaluation, including the earlier diagnostic images; "
              "it is not an independent held-out accuracy estimate.",
              "The old core prompt now receives tree-only scope and reinforced target/grammar instructions; "
              "earlier reports are not an identical baseline."]
    return "\n".join(lines) + "\n"


def write_compact_review(report, output_dir):
    """Write one ZIP with summaries, small numeric rows and selected Qwen outputs.

    No masks, full scene graphs, event streams or repeated evidence packs are
    copied. Missing/corrupt artifacts are reported, not silently called complete.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = _select_cases(report)
    diagnostics = {}
    cases = []
    prompt_examples = {}
    image_paths = {}
    warnings = []
    prompt_stats = defaultdict(lambda: {"executions": 0, "new_nodes": 0, "zero_gain_executions": 0})
    for variant in report["metadata"]["variants"]:
        rows = [r for r in report["samples"] if r["variant"] == variant]
        rejections, contracts = Counter(), Counter()
        artifact_count = 0
        correction_count = 0
        successful_corrections = 0
        target_corrections = 0
        for row in rows:
            for outcome in row.get("prompt_outcomes", []):
                stat = prompt_stats[(variant, outcome["family"], outcome["prompt"])]
                stat["executions"] += 1
                stat["new_nodes"] += outcome["new_nodes"]
                stat["zero_gain_executions"] += outcome["new_nodes"] == 0
            run_dir = Path(row["artifact_directory"])
            paths = sorted((run_dir / "artifacts/qwen").glob("*.json"))
            if row.get("qwen_calls", 0) and not paths:
                warnings.append(f"No Qwen artifacts for {row['run_id']}")
            examples = []
            for index, path in enumerate(paths):
                try:
                    artifact = json.loads(path.read_text())
                except (OSError, ValueError) as exc:
                    warnings.append(f"Cannot read {path}: {exc}")
                    continue
                artifact_count += 1
                meta = artifact.get("metadata", {})
                correction_count += meta.get("correction_of") is not None
                successful_corrections += (
                    meta.get("correction_of") is not None and meta.get("accepted_action_count", 0) > 0
                )
                target_corrections += (
                    meta.get("correction_of") is not None and meta.get("accepted_target_action_count", 0) > 0
                )
                rejections.update(r.get("reason", "UNKNOWN") for r in meta.get("rejections", []))
                if meta.get("contract_diagnostic"):
                    contracts[str(meta["contract_diagnostic"])] += 1
                request = artifact.get("input", {}).get("request_text")
                if request and variant not in prompt_examples:
                    prompt_examples[variant] = {
                        "sample_id": row["sample_id"], "qwen_call_id": artifact.get("qwen_call_id"),
                        "request_text": request,
                    }
                if row["sample_id"] in selected:
                    pack = artifact.get("input", {}).get("evidence_pack", {})
                    if pack.get("image_path"):
                        image_paths.setdefault(row["sample_id"], (pack["image_path"], run_dir))
                    if index in {0, len(paths) - 1}:
                        examples.append({
                            "qwen_call_id": artifact.get("qwen_call_id"),
                            "output": artifact.get("output"), "metadata": meta,
                        })
            if row["sample_id"] in selected:
                cases.append({
                    "sample_id": row["sample_id"], "variant": variant,
                    "selection_reasons": selected[row["sample_id"]],
                    "gt_count": row["gt_count"], "predicted_count": row.get("predicted_count"),
                    "stop_reason": row.get("stop_reason"),
                    "failure_message": row.get("failure_message"),
                    "qwen_first_and_last": examples,
                    "confidence_trace": row.get("confidence_trace", []),
                })
        diagnostics[variant] = {
            "stop_reasons": dict(Counter(r.get("stop_reason") or "FAILED" for r in rows)),
            "rejection_reasons": dict(rejections), "contract_diagnostics": dict(contracts),
            "qwen_artifacts_read": artifact_count,
            "rejection_correction_calls": correction_count,
            "corrections_with_accepted_actions": successful_corrections,
            "corrections_with_accepted_target": target_corrections,
            "confidence_trace_rows_available": sum("confidence_trace" in r for r in rows),
            "prompt_outcomes_rows_available": sum("prompt_outcomes" in r for r in rows),
            "failed_images": [r["sample_id"] for r in rows if not r.get("success")],
        }

    summary = {
        "metadata": report["metadata"], "aggregates": report["aggregates"],
        "paired_comparisons": report.get("paired_comparisons", {}),
        "paired_comparison": report.get("paired_comparison"),
        "diagnostics": diagnostics,
        "selected_images": selected,
        "notes": ["Positive paired error reduction favors the right-hand variant.",
                  "Prompt yields include bootstrap queries; new nodes are candidates, not confirmed fruit.",
                  "Selected examples are diagnostic cases, not a representative accuracy sample.",
                  "Qwen artifact counts can differ from API call counts because of repairs."],
    }
    numeric_keys = ("sample_id", "variant", "gt_count", "predicted_count", "raw_soft_count",
                    "count_type", "success", "absolute_error", "runtime_ms", "qwen_calls",
                    "sam3_calls", "stop_reason", "failure_message")
    confidence_totals = []
    for row in report["samples"]:
        trace = row.get("confidence_trace", [])
        if not trace:
            continue
        sensing = [step for step in trace if step["stage"] == "sensing"]
        confidence_totals.append({
            "sample_id": row["sample_id"], "variant": row["variant"],
            "bootstrap_target_mass": trace[0]["raw_soft_count"],
            "final_target_mass": trace[-1]["raw_soft_count"],
            "final_node_count": trace[-1]["node_count"],
            "final_nodes_below_half": trace[-1]["nodes_below_half"],
            "new_nodes_after_bootstrap": sum(s["new_nodes"] for s in sensing),
            "new_node_target_mass_at_creation": sum(s["new_node_target_mass"] for s in sensing),
            "existing_node_target_mass_change": sum(s["existing_node_target_mass_change"] for s in sensing),
            "removed_node_target_mass": sum(s["removed_node_target_mass"] for s in sensing),
        })
    path = output_dir / "compact_review.zip"
    with ZipFile(path, "w", ZIP_DEFLATED) as bundle:
        # Previews are bounded in number and resolution; full-resolution originals
        # stay local for a later focused inspection if needed.
        from PIL import Image
        from io import BytesIO
        for index, sample in enumerate(selected, 1):
            if sample not in image_paths:
                continue
            raw_path, run_dir = image_paths[sample]
            source = Path(raw_path)
            if not source.is_file():
                source = run_dir / raw_path
            try:
                with Image.open(source) as original:
                    preview = original.convert("RGB")
                    preview.thumbnail((1280, 1280))
                    data = BytesIO()
                    preview.save(data, format="JPEG", quality=80)
                name = f"previews/case_{index}.jpg"
                bundle.writestr(name, data.getvalue())
                summary.setdefault("preview_files", {})[sample] = name
            except (OSError, ValueError) as exc:
                warnings.append(f"Preview unavailable for {sample}: {exc}")
        summary["export_warnings"] = warnings
        members = {
            "review_summary.json": summary,
            "counts.json": [{k: r.get(k) for k in numeric_keys} for r in report["samples"]],
            "selected_cases.json": cases,
            "confidence_totals.json": confidence_totals,
            "prompt_examples.json": prompt_examples,
            "prompt_yields.json": [dict(variant=v, family=f, prompt=p, **stats)
                                   for (v, f, p), stats in sorted(prompt_stats.items())],
        }
        if report["metadata"].get("pilot_suite") == "final-ablation":
            notes = final_evaluation_summary(report)
            (output_dir / "mentor_summary.md").write_text(notes)
            bundle.writestr("mentor_summary.md", notes)
        for name, data in members.items():
            bundle.writestr(name, json.dumps(data, indent=2))
    return path
