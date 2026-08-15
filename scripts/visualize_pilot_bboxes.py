#!/usr/bin/env python3
"""Visualize final pilot graph boxes for A/B/C variants.

Usage:
    python scripts/visualize_pilot_bboxes.py \
        --pilot-report results/pilot_report.json \
        --manifest pilot_manifest.json \
        --output-dir results/bbox_visualizations

The script reads each run's artifacts/graph/final_graph.json and draws the
final graph nodes over the original image. It also creates an A/B/C comparison
image per sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


VARIANT_ORDER = ["A_OneShot", "B_FixedBank", "C_V4_NoExemplarCleanup"]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _manifest_map(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    data = _load_json(manifest_path)
    if not isinstance(data, list):
        raise ValueError("Pilot manifest must be a JSON list.")
    out: Dict[str, Dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, dict) or "sample_id" not in item:
            continue
        out[str(item["sample_id"])] = item
    return out


def _nodes_from_graph(graph: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    nodes = graph.get("nodes", {})
    if isinstance(nodes, dict):
        return [(str(node_id), node) for node_id, node in nodes.items() if isinstance(node, dict)]
    if isinstance(nodes, list):
        out = []
        for i, node in enumerate(nodes):
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("node_id", f"node_{i:06d}"))
            out.append((node_id, node))
        return out
    return []


def _extract_box(node: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    candidates: List[Any] = [
        node.get("box"),
        node.get("bbox"),
    ]
    geom = node.get("geometry")
    if isinstance(geom, dict):
        candidates.extend([geom.get("box"), geom.get("bbox")])

    for value in candidates:
        if isinstance(value, dict):
            keys = ("x1", "y1", "x2", "y2")
            if all(k in value for k in keys):
                return tuple(float(value[k]) for k in keys)  # type: ignore[return-value]
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return tuple(float(v) for v in value)  # type: ignore[return-value]
    return None


def _extract_target_probability(node: Dict[str, Any]) -> Optional[float]:
    belief = node.get("class_belief")
    if isinstance(belief, dict):
        value = belief.get("target")
        if isinstance(value, (int, float)):
            return float(value)
        probs = belief.get("probabilities")
        if isinstance(probs, dict) and isinstance(probs.get("target"), (int, float)):
            return float(probs["target"])

    value = node.get("target_probability")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _box_color(target_p: Optional[float], variant: str) -> Tuple[int, int, int]:
    # A is a hard one-shot baseline; use a single neutral color.
    if variant == "A_OneShot" or target_p is None:
        return (0, 140, 255)
    if target_p >= 0.70:
        return (20, 180, 70)
    if target_p >= 0.40:
        return (255, 170, 0)
    return (220, 50, 50)


def _font(size: int = 16):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def _text_box(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font, fill=(255, 255, 255), bg=(0, 0, 0)):
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 3
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=bg)
    draw.text((x, y), text, font=font, fill=fill)


def _draw_variant(
    image: Image.Image,
    graph_path: Path,
    variant: str,
    predicted_count: Optional[float],
    gt_count: Optional[float],
    line_width: int,
    show_node_ids: bool,
    show_probabilities: bool,
) -> Image.Image:
    graph = _load_json(graph_path)
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = _font(14)
    header_font = _font(20)

    nodes = _nodes_from_graph(graph)
    drawn = 0
    for node_id, node in nodes:
        box = _extract_box(node)
        if box is None:
            continue
        target_p = _extract_target_probability(node)
        color = _box_color(target_p, variant)
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

        parts: List[str] = []
        if show_node_ids:
            parts.append(node_id.replace("node_", "#"))
        if show_probabilities and variant != "A_OneShot" and target_p is not None:
            parts.append(f"p={target_p:.2f}")
        if parts:
            label = " ".join(parts)
            label_y = max(0, int(y1) - 18)
            _text_box(draw, (int(x1), label_y), label, font, bg=color)
        drawn += 1

    pred_text = "?" if predicted_count is None else f"{predicted_count:.2f}"
    gt_text = "?" if gt_count is None else f"{gt_count:g}"
    header = f"{variant} | pred={pred_text} | GT={gt_text} | boxes={drawn}"
    _text_box(draw, (10, 10), header, header_font)

    return canvas


def _make_comparison(images: List[Tuple[str, Image.Image]]) -> Image.Image:
    if not images:
        raise ValueError("No images supplied for comparison.")

    max_h = max(img.height for _, img in images)
    resized: List[Tuple[str, Image.Image]] = []
    for name, img in images:
        if img.height != max_h:
            scale = max_h / img.height
            img = img.resize((int(round(img.width * scale)), max_h), Image.Resampling.LANCZOS)
        resized.append((name, img))

    total_w = sum(img.width for _, img in resized)
    out = Image.new("RGB", (total_w, max_h), "white")
    x = 0
    for _, img in resized:
        out.paste(img, (x, 0))
        x += img.width
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw A/B/C pilot final graph bounding boxes.")
    parser.add_argument("--pilot-report", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--no-node-ids", action="store_true")
    parser.add_argument("--no-probabilities", action="store_true")
    args = parser.parse_args()

    report = _load_json(args.pilot_report)
    manifest = _manifest_map(args.manifest)
    output_dir = args.output_dir or (args.pilot_report.parent / "bbox_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = report.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("pilot_report.json has no valid 'samples' list.")

    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in samples:
        if not isinstance(row, dict) or not row.get("success"):
            continue
        sample_id = str(row.get("sample_id", ""))
        variant = str(row.get("variant", ""))
        if not sample_id or not variant:
            continue
        grouped.setdefault(sample_id, {})[variant] = row

    generated = 0
    for sample_id, variants in grouped.items():
        manifest_row = manifest.get(sample_id)
        if manifest_row is None:
            print(f"WARNING: sample {sample_id!r} not found in manifest; skipping")
            continue

        image_path = Path(manifest_row["image_path"])
        if not image_path.exists():
            print(f"WARNING: image missing for {sample_id}: {image_path}")
            continue

        original = Image.open(image_path).convert("RGB")
        comparison_parts: List[Tuple[str, Image.Image]] = []

        for variant in VARIANT_ORDER:
            row = variants.get(variant)
            if row is None:
                continue
            artifact_dir = Path(row["artifact_directory"])
            graph_path = artifact_dir / "artifacts" / "graph" / "final_graph.json"
            if not graph_path.exists():
                print(f"WARNING: graph missing: {graph_path}")
                continue

            rendered = _draw_variant(
                original,
                graph_path,
                variant,
                row.get("predicted_count"),
                row.get("gt_count", manifest_row.get("gt_count")),
                line_width=max(1, args.line_width),
                show_node_ids=not args.no_node_ids,
                show_probabilities=not args.no_probabilities,
            )
            out_path = output_dir / f"{sample_id}__{variant}.jpg"
            rendered.save(out_path, quality=95)
            comparison_parts.append((variant, rendered))
            generated += 1
            print(f"WROTE {out_path}")

        if comparison_parts:
            comparison = _make_comparison(comparison_parts)
            comparison_path = output_dir / f"{sample_id}__ABC.jpg"
            comparison.save(comparison_path, quality=95)
            print(f"WROTE {comparison_path}")

    print(f"Generated {generated} individual overlays in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
