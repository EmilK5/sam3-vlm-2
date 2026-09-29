"""Run VLM-first FSCD-147 inference and evaluate saved predictions separately."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .adapters import RealSAM3, RealVLM
from .cli import result_dict
from .core import Config, Controller
from .fscd147 import FSCD147, coco_detections, evaluate_counts


def run_dataset(dataset: FSCD147, config: Config, sam3: object, vlm: object,
                output_dir: Path, max_images: int | None = None) -> Path:
    """Write inference artifacts without opening any annotation file."""
    if not config.vlm_first:
        raise ValueError("FSCD-147 requires the VLM-first, no-bootstrap configuration")
    if max_images is not None and max_images < 1:
        raise ValueError("max_images must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"split": dataset.split, "image_ids": dataset.image_ids,
                "config": config.__dict__, "max_images": max_images,
                "dataset_root": str(dataset.root)}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=list) + "\n")
    prediction_path = output_dir / "predictions.jsonl"
    controller = Controller(sam3, vlm, config)
    with prediction_path.open("w") as out:
        for index, sample in enumerate(dataset.samples()):
            if max_images is not None and index >= max_images:
                break
            with Image.open(sample.image_path) as source:
                image = source.convert("RGB")
            result = controller.run(image, sample.target)
            row = {"image_id": sample.image_id, "target": sample.target,
                   "result": result_dict(result)}
            out.write(json.dumps(row) + "\n")
            out.flush()
    return prediction_path


def evaluate_saved(dataset: FSCD147, prediction_path: Path, config: Config,
                   coco_output: Path | None = None) -> dict:
    """Load FSCD annotations only after inference and export paper-format boxes."""
    rows: dict[str, dict] = {}
    for line in prediction_path.read_text().splitlines():
        row = json.loads(line)
        image_id = row["image_id"]
        if image_id in rows:
            raise ValueError(f"duplicate prediction ID: {image_id}")
        rows[image_id] = row["result"]
    truth = dataset.ground_truth()
    counts = {image_id: (result["counts"]["hard"] if not result["partial"] and
                          result["stop_reason"] not in {"error", "time_budget", "sam3_budget"}
                          else None)
              for image_id, result in rows.items()}
    metrics = evaluate_counts(counts, truth)
    metrics.update({"split": dataset.split, "count_rule": "hard", "detection_export_complete": False})
    if metrics["complete"]:
        detections = coco_detections(rows, truth,
                                     hard_count_threshold=config.hard_count_threshold)
        if coco_output is not None:
            coco_output.write_text(json.dumps(detections, indent=2) + "\n")
        metrics["detection_export_complete"] = True
        metrics["detections"] = len(detections)
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run models without reading annotations")
    run.add_argument("dataset_root", type=Path)
    run.add_argument("output_dir", type=Path)
    run.add_argument("--split", choices=["val", "test"], default="val")
    run.add_argument("--config", type=Path, default=Path("configs/fscd147.json"))
    run.add_argument("--max-images", type=int)
    run.add_argument("--sam3-model", default="facebook/sam3")
    run.add_argument("--device")
    evaluate = commands.add_parser("evaluate", help="Score saved counts and export COCO detections")
    evaluate.add_argument("dataset_root", type=Path)
    evaluate.add_argument("prediction_path", type=Path)
    evaluate.add_argument("--split", choices=["val", "test"], default="val")
    evaluate.add_argument("--config", type=Path, default=Path("configs/fscd147.json"))
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--coco-output", type=Path)
    args = parser.parse_args(argv)
    config = Config(**json.loads(args.config.read_text()))
    dataset = FSCD147(args.dataset_root, args.split)
    if args.command == "run":
        path = run_dataset(dataset, config, RealSAM3(args.sam3_model, args.device),
                           RealVLM(), args.output_dir, args.max_images)
        print(path)
    else:
        metrics = evaluate_saved(dataset, args.prediction_path, config, args.coco_output)
        output = json.dumps(metrics, indent=2)
        if args.output:
            args.output.write_text(output + "\n")
        else:
            print(output)


if __name__ == "__main__":
    main()
