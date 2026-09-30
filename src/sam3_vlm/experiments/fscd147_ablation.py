"""Controlled FSCD-147 prompt, evidence and search experiments on the same ten images."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from zipfile import ZipFile, ZIP_DEFLATED

from sam3_vlm.datasets.fscd147 import FSCD147
from sam3_vlm.experiments.fscd147 import run_dataset, select_variants
from sam3_vlm.experiments.fscd147_report import write_summary
from sam3_vlm.experiments.m8_smoke import load_m8_config


@dataclass
class Trial:
    name: str
    deployment: object
    arms: str
    target_overrides: dict
    reference: str | None


def build_trials(deployment, suite="policies"):
    base = deployment.v4_config
    # Freeze a common control, including disabling every new experimental switch.
    base = replace(base, sam3=replace(base.sam3, singularize_prompts=True),
        planner=replace(base.planner, validate_confounders=False),
        belief=replace(base.belief, neutral_appearance_misses=False),
        replanning=replace(base.replanning, adaptive_e_zero_gain_patience=None))
    if suite == "final":
        base = replace(base, planner=replace(base.planner, temperature=0.0,
            sampling_seed=deployment.seed, compact_json=True, max_output_tokens=1024))
        return [
            Trial("final_reference", replace(deployment, v4_config=base), "DE",
                  {"donuts tray": "donut"}, None),
            Trial("final_adaptive", replace(deployment, v4_config=replace(base,
                replanning=replace(base.replanning, adaptive_e_zero_gain_patience=3))),
                "E", {"donuts tray": "donut"}, "final_reference"),
        ]
    if suite != "policies":
        raise ValueError("suite must be policies or final")
    configs = [
        ("plural_control", replace(base, sam3=replace(base.sam3, singularize_prompts=False)), "ABCDE", {}, None),
        ("singular_control", base, "ABCDE", {}, "plural_control"),
        ("safe_negatives", replace(base, planner=replace(base.planner, validate_confounders=True)), "CDE", {}, "singular_control"),
        ("neutral_appearance_misses", replace(base, belief=replace(base.belief, neutral_appearance_misses=True)), "CDE", {}, "singular_control"),
        ("adaptive_e", replace(base, replanning=replace(base.replanning, adaptive_e_zero_gain_patience=3)), "E", {}, "singular_control"),
        ("counting_unit", base, "ABCDE", {"donuts tray": "donut"}, "singular_control"),
        ("combined", replace(base, planner=replace(base.planner, validate_confounders=True),
            belief=replace(base.belief, neutral_appearance_misses=True),
            replanning=replace(base.replanning, adaptive_e_zero_gain_patience=3)),
            "ABCDE", {"donuts tray": "donut"}, "singular_control"),
    ]
    return [Trial(name, replace(deployment, v4_config=config), arms, overrides, reference)
            for name, config, arms, overrides, reference in configs]


def trial_manifest(dataset, trials):
    profiles = {}
    for trial in trials:
        variants = {variant.name: asdict(variant.config)
                    for variant in select_variants(trial.deployment.v4_config) if variant.name[0] in trial.arms}
        profiles[trial.name] = {"arms": trial.arms, "reference": trial.reference,
            "target_overrides": trial.target_overrides, "variants": variants,
            "config_sha256": hashlib.sha256(json.dumps(variants, sort_keys=True).encode()).hexdigest()}
    return {"schema_version": 1, "split": dataset.split, "image_ids": list(dataset.image_ids),
        "profiles": profiles, "expected_runs": len(dataset.image_ids) * sum(len(trial.arms) for trial in trials),
        "model_settings": {"sam3": trials[0].deployment.sam3_model, "qwen": trials[0].deployment.qwen_model,
                           "seed": trials[0].deployment.seed},
        "note": "Smoke-test ablations; fixed split, seed, models and thresholds. Ground truth is evaluation-only."}


def paired_result(current, reference, image_ids):
    pairs = []
    for image_id in image_ids:
        left, right = current.get(image_id), reference.get(image_id)
        if left and right and left["success"] and right["success"]:
            pairs.append({"image_id": image_id, "count_before": right["predicted_count"],
                "count_after": left["predicted_count"],
                "absolute_error_reduction": right["absolute_error"] - left["absolute_error"],
                "runtime_change_seconds": left["runtime_seconds"] - right["runtime_seconds"]})
    complete = len(pairs) == len(image_ids)
    return {"complete": complete, "paired_images": len(pairs), "expected_images": len(image_ids),
        "mean_absolute_error_reduction": mean(pair["absolute_error_reduction"] for pair in pairs) if complete else None,
        "mean_runtime_change_seconds": mean(pair["runtime_change_seconds"] for pair in pairs) if complete else None,
        "improved_images": sum(pair["absolute_error_reduction"] > 1e-9 for pair in pairs),
        "worsened_images": sum(pair["absolute_error_reduction"] < -1e-9 for pair in pairs),
        "unchanged_images": sum(abs(pair["absolute_error_reduction"]) <= 1e-9 for pair in pairs),
        "per_image": pairs}


def report_suite(dataset_root, output_dir, *, split="val"):
    output_dir = Path(output_dir)
    dataset = FSCD147(dataset_root, split)
    manifest = json.loads((output_dir / "suite.json").read_text())
    if manifest["split"] != split or manifest["image_ids"] != list(dataset.image_ids):
        raise ValueError("Suite and dataset split differ")
    reports, missing, files = {}, {}, []
    for name, profile in manifest["profiles"].items():
        directory = output_dir / name
        if not (directory / "predictions.jsonl").is_file():
            missing[name] = "Predictions missing"
            continue
        metadata = json.loads((directory / "metadata.json").read_text())
        configs = {key: value["config"] for key, value in metadata["variants"].items()}
        if (metadata["model_settings"] != manifest["model_settings"] or configs != profile["variants"]
                or metadata.get("target_overrides", {}) != profile["target_overrides"]):
            raise ValueError(f"Frozen profile differs from suite manifest: {name}")
        paths = write_summary(dataset_root, directory / "predictions.jsonl", split=split,
                              max_images=max(10, len(dataset.image_ids)))
        reports[name] = json.loads(Path(paths["json"]).read_text())
        files.extend((name, Path(paths[key])) for key in ("summary", "json", "visual_notes"))
    comparisons = {}
    for name, report in reports.items():
        reference_name = manifest["profiles"][name]["reference"]
        if reference_name is None:
            continue
        if reference_name not in reports:
            comparisons[name] = {"reference": reference_name, "available": False}
            continue
        reference = reports[reference_name]
        arms = {}
        for arm in report["aggregates"]:
            if arm not in reference["aggregates"]:
                continue
            current_rows = {image["image_id"]: image["variants"][arm] for image in report["per_image"]}
            reference_rows = {image["image_id"]: image["variants"][arm] for image in reference["per_image"]}
            arms[arm] = paired_result(current_rows, reference_rows, dataset.image_ids)
        comparisons[name] = {"reference": reference_name, "available": True, "arms": arms}
    cross_arm = {}
    if "final_reference" in reports:
        reference = reports["final_reference"]
        d = next((arm for arm in reference["aggregates"] if arm.startswith("D_")), None)
        e = next((arm for arm in reference["aggregates"] if arm.startswith("E_")), None)
        if d and e:
            cross_arm["reference_D_to_E"] = paired_result(
                {image["image_id"]: image["variants"][e] for image in reference["per_image"]},
                {image["image_id"]: image["variants"][d] for image in reference["per_image"]}, dataset.image_ids)
    result = {"schema_version": 1, "image_ids": list(dataset.image_ids), "expected_runs": manifest["expected_runs"],
        "complete": not missing and all(arm["complete"] for report in reports.values() for arm in report["aggregates"].values()),
        "missing_profiles": missing,
        "aggregates": {name: report["aggregates"] for name, report in reports.items()},
        "paired_comparisons": comparisons,
        "cross_arm_comparisons": cross_arm,
        "note": "Positive error reduction means improved accuracy; negative runtime change means faster. Ten images are a smoke test."}
    json_path = output_dir / "comparison.json"
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# FSCD-147 controlled experiments", "", f"Complete: {result['complete']}. Expected runs: {manifest['expected_runs']}.", "",
        "Each profile uses the same images and seed. New policies are experimental. Official labels are unchanged.", "",
        "| Profile | Arm | MAE | Mean seconds |", "|---|---|---:|---:|"]
    for name, report in reports.items():
        for arm, aggregate in report["aggregates"].items():
            lines.append(f"| {name} | {arm} | {aggregate['mae']} | {aggregate['mean_runtime_seconds']} |")
    lines += ["", "Paired comparisons: positive error reduction is better; negative runtime change is faster.", "",
              "| Profile | Reference | Arm | Mean error reduction | Mean runtime change | Improved / worse |",
              "|---|---|---|---:|---:|---:|"]
    for name, comparison in comparisons.items():
        for arm, paired in comparison.get("arms", {}).items():
            lines.append(f"| {name} | {comparison['reference']} | {arm} | {paired['mean_absolute_error_reduction']} | "
                         f"{paired['mean_runtime_change_seconds']} | {paired['improved_images']} / {paired['worsened_images']} |")
    if missing:
        lines += ["", f"Missing profiles: {missing}"]
    if cross_arm:
        paired = cross_arm["reference_D_to_E"]
        lines += ["", "Reference D → E on the same images:",
                  f"Mean error reduction: {paired['mean_absolute_error_reduction']}; "
                  f"mean runtime change: {paired['mean_runtime_change_seconds']} seconds; "
                  f"improved / worse: {paired['improved_images']} / {paired['worsened_images']}.",
                  "Both final profiles use compact JSON, 1024 output tokens, temperature 0 and an API sampling seed. "
                  "Compare these fresh controls within this suite; older sampling settings differ."]
    lines += ["", "Mask overlap flags in profile summaries identify pairs for visual review; counts alone do not establish dedup quality.",
              "Negative semantic assessments rely on Qwen and can be wrong. Check the book, cap and donut examples.", ""]
    markdown = output_dir / "comparison.md"
    markdown.write_text("\n".join(lines))
    archive = output_dir / "summary.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
        bundle.write(json_path, json_path.name)
        bundle.write(markdown, markdown.name)
        bundle.writestr("suite.json", json.dumps(manifest, indent=2) + "\n")
        for profile, path in files:
            bundle.write(path, f"{profile}/{path.name}")
    return {"complete": result["complete"], "comparison": str(markdown), "archive": str(archive)}


def run_suite(dataset, trials, sam3, qwen, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "suite.json").write_text(json.dumps(trial_manifest(dataset, trials), indent=2) + "\n")
    for index, trial in enumerate(trials, 1):
        print(f"[{index}/{len(trials)}] {trial.name}: {trial.arms}", flush=True)
        def progress(row):
            print(f"  {row['image_id']} {row['variant']}: success={row['success']}, "
                  f"count={row['predicted_count']}, seconds={row['runtime_seconds']:.1f}", flush=True)
        run_dataset(dataset, trial.deployment, sam3, qwen, output_dir / trial.name,
                    arms=trial.arms, target_overrides=trial.target_overrides, audit=True, progress=progress)
    return report_suite(dataset.root, output_dir, split=dataset.split)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("dataset_root", type=Path)
    run.add_argument("output_dir", type=Path)
    run.add_argument("--config", type=Path, default=Path("configs/fscd147.json"))
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--suite", choices=["policies", "final"], default="policies",
                     help="policies: original seven profiles; final: D, E and adaptive E (30 runs)")
    run.add_argument("--profiles", nargs="+", help="Selected names within the chosen suite")
    run.add_argument("--allow-full-split", action="store_true", help="Explicitly allow a split size other than ten")
    run.add_argument("--sam3-model")
    run.add_argument("--qwen-model")
    run.add_argument("--qwen-base-url")
    report = commands.add_parser("report", help="Rebuild the comparison and zip from saved runs")
    report.add_argument("dataset_root", type=Path)
    report.add_argument("output_dir", type=Path)
    for command in (run, report):
        command.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args(argv)
    if args.command == "report":
        result = report_suite(args.dataset_root, args.output_dir, split=args.split)
    else:
        dataset = FSCD147(args.dataset_root, args.split)
        if len(dataset.image_ids) != 10 and not args.allow_full_split:
            parser.error("Use the prepared ten-image subset, or explicitly pass --allow-full-split")
        if not args.config.is_file():
            parser.error(f"Missing configuration: {args.config}")
        deployment = load_m8_config(SimpleNamespace(**{**vars(args), "output_dir": str(args.output_dir)}),
                                    config_path=args.config)
        trials = build_trials(deployment, args.suite)
        if args.profiles:
            if set(args.profiles) - {trial.name for trial in trials}:
                parser.error("Unknown experiment profile")
            trials = [trial for trial in trials if trial.name in args.profiles]
        if args.dry_run:
            # Validate image paths; never load models or ground truth.
            list(dataset.samples())
            manifest = trial_manifest(dataset, trials)
            print(json.dumps({"images": len(dataset.image_ids), "expected_runs": manifest["expected_runs"],
                              "profiles": {name: profile["arms"] for name, profile in manifest["profiles"].items()}}, indent=2))
            return 0
        if args.output_dir.exists():
            parser.error("Use a fresh output directory for the suite")
        import torch
        from sam3_vlm.models.sam3 import RealSAM3Sensor
        from sam3_vlm.models.qwen import RealQwenPlanner
        if deployment.require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by the benchmark configuration")
        sam3 = RealSAM3Sensor(model_id=deployment.sam3_model, device=deployment.v4_config.device,
                              compile_model=deployment.compile_sam3)
        qwen = RealQwenPlanner(model=deployment.qwen_model, base_url=deployment.qwen_base_url,
                              strict_model_errors=True)
        result = run_suite(dataset, trials, sam3, qwen, args.output_dir)
    print(json.dumps(result, indent=2))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
