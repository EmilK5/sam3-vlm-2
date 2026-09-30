"""Prepare a seeded FSCD-147 subset, review predictions, and export compact summaries."""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import random
import shutil

from PIL import Image, ImageDraw

from sam3_vlm.datasets.fscd147 import FSCD147


def prepare_subset(source, destination, *, split="val", count=10, seed=42):
    """Select by split IDs alone; filter evaluation files after selection."""
    source, destination = Path(source).resolve(), Path(destination)
    dataset = FSCD147(source, split)
    if type(count) is not int or not 1 <= count <= len(dataset.image_ids):
        raise ValueError("count must be between 1 and the split size")
    chosen = set(random.Random(seed).sample(list(dataset.image_ids), count))
    ids = [image_id for image_id in dataset.image_ids if image_id in chosen]
    paths = []
    for image_id in ids:
        if Path(image_id).name != image_id or image_id in {".", ".."}:
            raise ValueError(f"invalid image ID: {image_id!r}")
        path = source / "images_384_VarV2" / image_id
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
    # COCO filtering makes optional AP refer to these images only.
    coco = json.loads((source / f"instances_{split}.json").read_text())
    coco["images"] = [record for record in coco["images"] if record["file_name"] in chosen]
    if {record["file_name"] for record in coco["images"]} != chosen:
        raise ValueError("COCO annotations missing selected images")
    coco_ids = {record["id"] for record in coco["images"]}
    coco["annotations"] = [record for record in coco["annotations"] if record["image_id"] in coco_ids]
    points_path = source / "annotation_FSC147_384.json"
    points = json.loads(points_path.read_text()) if points_path.exists() else None
    destination.mkdir(parents=True, exist_ok=False)
    image_dir = destination / "images_384_VarV2"
    image_dir.mkdir()
    for path in paths:
        (image_dir / path.name).symlink_to(path)
    shutil.copyfile(source / "ImageClasses_FSC147.txt", destination / "ImageClasses_FSC147.txt")
    (destination / "Train_Test_Val_FSC_147.json").write_text(json.dumps({split: ids}, indent=2) + "\n")
    (destination / f"instances_{split}.json").write_text(json.dumps(coco) + "\n")
    if points is not None:
        (destination / points_path.name).write_text(json.dumps({key: points[key] for key in ids if key in points}) + "\n")
    manifest = {"source": str(source), "split": split, "seed": seed, "count": count, "image_ids": ids}
    (destination / "sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def review_predictions(dataset_root, prediction_path, *, split="val", output_dir=None):
    """Read ground truth only after inference; create a portable HTML/PNG gallery."""
    dataset = FSCD147(dataset_root, split)
    prediction_path = Path(prediction_path)
    metadata = json.loads((prediction_path.parent / "metadata.json").read_text())
    if metadata["split"] != split or metadata["image_ids"] != list(dataset.image_ids):
        raise ValueError("Predictions do not match the review split")
    variants = list(metadata["variants"])
    rows = {}
    for line in prediction_path.read_text().splitlines():
        row = json.loads(line)
        key = (row["image_id"], row["variant"])
        if key in rows or key[0] not in dataset.image_ids or key[1] not in variants:
            raise ValueError("Duplicate or unknown prediction in review")
        rows[key] = row
    truth = dataset.ground_truth()
    output_dir = Path(output_dir or prediction_path.parent / "review")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "images").mkdir()
    cards = []
    success_count = 0
    for sample in dataset.samples():
        with Image.open(sample.image_path) as source_image:
            image = source_image.convert("RGB")
        gt = truth[sample.image_id]
        stem = hashlib.sha256(sample.image_id.encode()).hexdigest()[:20]
        gt_image = image.copy()
        draw = ImageDraw.Draw(gt_image)
        scale_x = image.width / gt.width if gt.width else 1
        scale_y = image.height / gt.height if gt.height else 1
        for x, y, w, h in gt.boxes:
            draw.rectangle((x*scale_x, y*scale_y, (x+w)*scale_x, (y+h)*scale_y), outline="lime", width=2)
        gt_name = f"images/{stem}_gt.png"
        gt_image.save(output_dir / gt_name)
        cells = [f'<figure><img src="{gt_name}"><figcaption>Ground truth: {gt.count}</figcaption></figure>']
        for index, variant in enumerate(variants):
            row = rows.get((sample.image_id, variant))
            overlay = image.copy()
            draw = ImageDraw.Draw(overlay)
            if row and row["success"] is True:
                success_count += 1
                width, height = row["image_size"]
                for node in row["nodes"]:
                    x1, y1, x2, y2 = node["box"]
                    draw.rectangle((x1*image.width/width, y1*image.height/height,
                                    x2*image.width/width, y2*image.height/height), outline="red", width=2)
                caption = (f'{variant}: count {row["predicted_count"]:.2f}; '
                           f'{len(row["nodes"])} candidates; {row["runtime_seconds"]:.1f}s')
            else:
                caption = f'{variant}: FAILED — {row.get("error", "unknown error") if row else "missing prediction"}'
            name = f"images/{stem}_{index}.png"
            overlay.save(output_dir / name)
            cells.append(f'<figure><img src="{name}"><figcaption>{escape(caption)}</figcaption></figure>')
        cards.append(f'<section><h2>{escape(sample.image_id)} — {escape(sample.target)}</h2><div class="row">'
                     + "".join(cells) + '</div></section>')
    expected = len(dataset.image_ids) * len(variants)
    header = (f'<h1>FSCD-147 {escape(split)} review</h1><p>{success_count}/{expected} successful runs. '
              'Green: ground-truth boxes. Red: predicted candidate boxes. '
              'C–E counts sum target probabilities, so their counts can differ from the number of red boxes. '
              'A sampled subset measures smoke-test behavior only.</p>')
    html = ('<!doctype html><html><head><meta charset="utf-8"><title>FSCD-147 review</title><style>'
            'body{font:16px sans-serif;margin:24px;background:#f4f4f4;color:#222}'
            '.row{display:flex;gap:12px;overflow-x:auto}figure{margin:0;min-width:280px;max-width:380px}'
            'img{width:100%;display:block}figcaption{padding:8px;background:white}section{margin:28px 0}'
            '</style></head><body>' + header + "".join(cards) + '</body></html>')
    path = output_dir / "index.html"
    path.write_text(html)
    return {"gallery": str(path), "successful_runs": success_count, "expected_runs": expected}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Create a seeded random subset without copying image bytes")
    prepare.add_argument("source", type=Path)
    prepare.add_argument("destination", type=Path)
    prepare.add_argument("--split", choices=["val", "test"], default="val")
    prepare.add_argument("--count", type=int, default=10)
    prepare.add_argument("--seed", type=int, default=42)
    review = commands.add_parser("review", help="Create an evaluation-only gallery from saved predictions")
    review.add_argument("dataset_root", type=Path)
    review.add_argument("prediction_path", type=Path)
    review.add_argument("--split", choices=["val", "test"], default="val")
    review.add_argument("--output-dir", type=Path)
    summary = commands.add_parser("summary", help="Export a small text/JSON report without images or masks")
    summary.add_argument("dataset_root", type=Path)
    summary.add_argument("prediction_path", type=Path)
    summary.add_argument("--split", choices=["val", "test"], default="val")
    summary.add_argument("--output-dir", type=Path)
    summary.add_argument("--top-k", type=int, default=3, help="Largest errors per arm")
    summary.add_argument("--max-images", type=int, default=10, help="Maximum per-image detail rows")
    summary.add_argument("--no-artifacts", action="store_true", help="Read predictions/ground truth only")
    summary.add_argument("--ap", action="store_true", help="Compute fresh bbox AP; requires evaluation extra")
    summary.add_argument("--annotation-audit", type=Path, help="Separate JSON notes and optional manual counts; official metrics stay unchanged")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_subset(args.source, args.destination, split=args.split, count=args.count, seed=args.seed)
    elif args.command == "review":
        result = review_predictions(args.dataset_root, args.prediction_path, split=args.split, output_dir=args.output_dir)
    else:
        from sam3_vlm.experiments.fscd147_report import write_summary
        result = write_summary(args.dataset_root, args.prediction_path, split=args.split, output_dir=args.output_dir,
                               top_k=args.top_k, max_images=args.max_images,
                               include_artifacts=not args.no_artifacts, compute_ap=args.ap,
                               annotation_audit=args.annotation_audit)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
