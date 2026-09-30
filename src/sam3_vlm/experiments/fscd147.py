"""Run the A–E pipeline on FSCD-147; score saved predictions separately."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
from itertools import islice
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image

from sam3_vlm.core.types import StopReason
from sam3_vlm.datasets.fscd147 import FSCD147, evaluate_counts
from sam3_vlm.experiments.m8_smoke import (
    _pilot_variants, _run_sam3_baseline, _run_validator_and_replay,
    assemble_e2e_runner, load_m8_config,
)
from sam3_vlm.logging.artifacts import RunArtifactPaths


def select_variants(config, arm=None):
    variants = _pilot_variants(config, "final-ae")
    if arm is not None:
        variants = [v for v in variants if v.name.startswith(arm + "_")]
        if not variants:
            raise ValueError("arm must be one of A, B, C, D, E")
    return variants


def run_dataset(dataset, deployment, sam3, qwen, output_dir, *, arm=None, max_images=None):
    """Inference reads only the image, split and class name; never annotations."""
    if max_images is not None and (type(max_images) is not int or max_images < 1):
        raise ValueError("max_images must be positive")
    variants = select_variants(deployment.v4_config, arm)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.jsonl"
    # Refuse to destroy a previous benchmark, including its frozen configuration.
    if prediction_path.exists() or (output_dir / "metadata.json").exists():
        raise FileExistsError("Use a fresh output directory for each benchmark run")
    metadata = {"schema_version": 1, "dataset": "FSCD-147", "split": dataset.split,
                "image_ids": dataset.image_ids, "max_images": max_images,
                "variants": {v.name: {"config": asdict(v.config), "count_type": v.count_type}
                             for v in variants},
                "model_settings": {"sam3": deployment.sam3_model, "qwen": deployment.qwen_model,
                                   "seed": deployment.seed},
                "dataset_root": str(dataset.root.resolve())}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with prediction_path.open("x") as stream:
        for sample in islice(dataset.samples(), max_images):
            for variant in variants:
                run_key = hashlib.sha256(sample.image_id.encode()).hexdigest()[:20]
                run_id = f"{variant.name}_{run_key}"
                paths = RunArtifactPaths(output_dir / "runs" / variant.name / run_key)
                config = replace(variant.config, assets_dir=str(paths.base_dir / "assets"))
                row = {"image_id": sample.image_id, "variant": variant.name,
                       "target": sample.target, "count_type": variant.count_type,
                       "success": False, "predicted_count": None, "nodes": [],
                       "artifact_directory": str(paths.base_dir)}
                started = time.perf_counter()
                try:
                    random.seed(deployment.seed)
                    np.random.seed(deployment.seed)
                    import torch
                    torch.manual_seed(deployment.seed)
                    with Image.open(sample.image_path) as source:
                        image = source.convert("RGB")
                    row["image_size"] = list(image.size)
                    if variant.uses_qwen:
                        if qwen is None:
                            raise ValueError("C–E require a Qwen planner")
                        runner, _ = assemble_e2e_runner(
                            paths, config, sam3, qwen, run_id, sample.target,
                            "target", sample.image_id, seed=deployment.seed,
                            experiment_name=f"FSCD-147:{variant.name}",
                        )
                        count = runner.run(image, sample.target, image_id=sample.image_id)
                        state = runner.scene_state
                        valid = _run_validator_and_replay(paths, state)
                        stop = state.stop_reason.value if state.stop_reason else None
                        exhausted = stop in {StopReason.SAM3_BUDGET.value, StopReason.TILE_BUDGET.value,
                                             StopReason.RUNTIME_BUDGET.value, StopReason.MAX_ITERATIONS.value}
                        if not valid or exhausted:
                            raise RuntimeError(f"Incomplete run: {stop}; replay valid={valid}")
                    else:
                        count, state = _run_sam3_baseline(
                            paths=paths, config=config, sensor=sam3, run_id=run_id,
                            prompt=sample.target, image_id=sample.image_id, image=image,
                            seed=deployment.seed, experiment_name=f"FSCD-147:{variant.name}",
                        )
                        stop = "SAM3_BASELINE_COMPLETE"
                    nodes = []
                    for node in state.graph.active_nodes():
                        score = (node.class_belief.probabilities.get("target", 0.0) if variant.uses_qwen
                                 else max((obs.score or 0 for obs in node.observations), default=0.0))
                        nodes.append({"box": list(node.geometry.bbox().as_tuple()), "score": score})
                    row.update(success=True, predicted_count=count, nodes=nodes, stop_reason=stop,
                               budget=asdict(state.budget), adaptive_tiling=state.discovery_state.adaptive_tiling)
                except Exception as exc:
                    row["error"] = str(exc)
                row["runtime_seconds"] = time.perf_counter() - started
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
    return prediction_path


def evaluate_saved(dataset, prediction_path, *, output_dir=None, compute_ap=False):
    """Evaluate each frozen arm on the complete split; failed/partial runs stay explicit."""
    prediction_path = Path(prediction_path)
    metadata = json.loads((prediction_path.parent / "metadata.json").read_text())
    if metadata.get("schema_version") != 1 or metadata.get("dataset") != "FSCD-147":
        raise ValueError("Invalid FSCD-147 prediction metadata")
    if metadata["split"] != dataset.split or metadata["image_ids"] != list(dataset.image_ids):
        raise ValueError("Prediction metadata does not match the requested split")
    variants = metadata["variants"]
    rows = {variant: {} for variant in variants}
    for line in prediction_path.read_text().splitlines():
        row = json.loads(line)
        variant, image_id = row["variant"], row["image_id"]
        if variant not in rows or image_id not in dataset.image_ids:
            raise ValueError("Unknown variant or image ID in predictions")
        if image_id in rows[variant]:
            raise ValueError(f"Duplicate prediction: {variant}/{image_id}")
        if row["count_type"] != variants[variant]["count_type"]:
            raise ValueError("Prediction count rule differs from frozen metadata")
        rows[variant][image_id] = row
    truth = dataset.ground_truth()
    output_dir = Path(output_dir or prediction_path.parent / "evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"split": dataset.split, "variants": {}}
    for variant, predictions in rows.items():
        counts = {key: row["predicted_count"] if row["success"] is True else None
                  for key, row in predictions.items()}
        metrics = evaluate_counts(counts, truth)
        metrics["count_type"] = variants[variant]["count_type"]
        metrics["failures"] = {key: row.get("error") for key, row in predictions.items()
                               if row["success"] is not True}
        if metrics["complete"]:
            detections = export_detections(predictions, truth)
            path = output_dir / f"{variant}_detections.json"
            path.write_text(json.dumps(detections, allow_nan=False) + "\n")
            metrics["coco_output"] = str(path)
            if compute_ap:
                metrics.update(coco_ap(dataset.root / f"instances_{dataset.split}.json", detections))
        report["variants"][variant] = metrics
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def export_detections(predictions, truth):
    """Use all candidate boxes for AP, scored by each arm's existing counting policy."""
    detections = []
    for image_id, row in predictions.items():
        gt = truth[image_id]
        width, height = row["image_size"]
        if type(width) is not int or type(height) is not int or min(width, height) <= 0:
            raise ValueError("Invalid prediction image size")
        scale_x = gt.width / width if gt.width is not None else 1.0
        scale_y = gt.height / height if gt.height is not None else 1.0
        for node in row["nodes"]:
            x1, y1, x2, y2 = node["box"]
            score = node["score"]
            if (not all(np.isfinite(v) for v in (x1, y1, x2, y2, score)) or
                    not 0 <= score <= 1 or not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height):
                raise ValueError(f"Invalid detection for {image_id}")
            detections.append({"image_id": gt.coco_image_id, "category_id": gt.category_id,
                               "bbox": [x1*scale_x, y1*scale_y, (x2-x1)*scale_x, (y2-y1)*scale_y],
                               "score": score})
    return detections


def coco_ap(annotation_path, detections):
    """Optional official COCO bbox metrics; no COCO 100-detection truncation in dense scenes."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO(str(annotation_path))
    gt.dataset.setdefault("info", {})
    if detections:
        dt = gt.loadRes(detections)
    else:
        dt = COCO()
        dt.dataset = {"images": gt.dataset["images"], "categories": gt.dataset["categories"], "annotations": []}
        dt.createIndex()
    evaluator = COCOeval(gt, dt, "bbox")
    # Keep the dense benchmark's usual 1000 detection limit explicit in the results.
    evaluator.params.maxDets = [1, 10, 1000]
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"][:, :, :, 0, -1]
    valid = precision[precision > -1]
    at_50 = precision[0][precision[0] > -1]
    return {"AP": float(valid.mean()) if valid.size else None,
            "AP50": float(at_50.mean()) if at_50.size else None,
            "max_detections": 1000, "ap_count_rule": "all_candidates_ranked_by_score"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run A–E without reading ground truth")
    run.add_argument("dataset_root", type=Path)
    run.add_argument("output_dir", type=Path)
    run.add_argument("--split", choices=["val", "test"], default="val")
    run.add_argument("--config", type=Path, default=Path("configs/fscd147.json"))
    run.add_argument("--arm", choices=list("ABCDE"))
    run.add_argument("--max-images", type=int)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--sam3-model")
    run.add_argument("--qwen-model")
    run.add_argument("--qwen-base-url")
    run.add_argument("--singular-prompts", action=argparse.BooleanOptionalAction, default=None,
                     help="Override prompt singularization for a controlled comparison")
    evaluate = commands.add_parser("evaluate", help="Score a saved complete split")
    evaluate.add_argument("dataset_root", type=Path)
    evaluate.add_argument("prediction_path", type=Path)
    evaluate.add_argument("--split", choices=["val", "test"], default="val")
    evaluate.add_argument("--output-dir", type=Path)
    evaluate.add_argument("--ap", action="store_true", help="Requires pip install -e '.[evaluation]'")
    args = parser.parse_args(argv)
    dataset = FSCD147(args.dataset_root, args.split)
    if args.command == "evaluate":
        print(json.dumps(evaluate_saved(dataset, args.prediction_path,
                                       output_dir=args.output_dir, compute_ap=args.ap), indent=2))
        return 0
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be positive")
    if not args.config.is_file():
        parser.error(f"Configuration file does not exist: {args.config}")
    args.output_dir = str(args.output_dir)
    deployment = load_m8_config(args, config_path=args.config)
    if args.singular_prompts is not None:
        deployment = replace(deployment, v4_config=replace(deployment.v4_config,
            sam3=replace(deployment.v4_config.sam3, singularize_prompts=args.singular_prompts)))
    variants = select_variants(deployment.v4_config, args.arm)
    if args.dry_run:
        # Consume the image iterator to validate paths without loading models/annotations.
        count = sum(1 for _ in dataset.samples())
        print(json.dumps({"split_images": count, "run_images": min(count, args.max_images or count),
                          "variants": [v.name for v in variants]}, indent=2))
        return 0
    import torch
    from sam3_vlm.models.sam3 import RealSAM3Sensor
    from sam3_vlm.models.qwen import RealQwenPlanner
    if deployment.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the benchmark configuration")
    sam3 = RealSAM3Sensor(model_id=deployment.sam3_model, device=deployment.v4_config.device,
                          compile_model=deployment.compile_sam3)
    qwen = (RealQwenPlanner(model=deployment.qwen_model, base_url=deployment.qwen_base_url,
                            strict_model_errors=True) if any(v.uses_qwen for v in variants) else None)
    path = run_dataset(dataset, deployment, sam3, qwen, args.output_dir,
                       arm=args.arm, max_images=args.max_images)
    print(path)
    # A process failure remains visible to cluster schedulers despite saved artifacts.
    with path.open() as stream:
        return 0 if all(json.loads(line)["success"] for line in stream) else 1


if __name__ == "__main__":
    raise SystemExit(main())
