"""FSCD-147 image/class adapter with ground truth isolated to evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class FSCDSample:
    image_id: str
    image_path: Path
    target: str


@dataclass(frozen=True)
class FSCDGroundTruth:
    image_id: str
    count: int
    points: tuple[tuple[float, float], ...]
    coco_image_id: int
    category_id: int
    boxes: tuple[tuple[float, float, float, float], ...]
    width: int | None = None
    height: int | None = None

    @property
    def point_count(self) -> int:
        return len(self.points)


class FSCD147:
    """Read split and class files for inference; load annotations only on request.

    Expected root: images_384_VarV2/, ImageClasses_FSC147.txt,
    Train_Test_Val_FSC_147.json, annotation_FSC147_384.json, and
    instances_val.json / instances_test.json for evaluation.
    """

    def __init__(self, root: str | Path, split: str = "val"):
        if split not in {"val", "test"}:
            raise ValueError("FSCD evaluation split must be 'val' or 'test'")
        self.root = Path(root)
        self.split = split
        splits = json.loads((self.root / "Train_Test_Val_FSC_147.json").read_text())
        ids = splits.get(split)
        if not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids):
            raise ValueError(f"missing or invalid {split} split")
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate image ID in split")
        self.image_ids = tuple(ids)
        classes: dict[str, str] = {}
        for line in (self.root / "ImageClasses_FSC147.txt").read_text().splitlines():
            parts = line.strip().split(maxsplit=1)
            if not parts:
                continue
            if len(parts) != 2 or not parts[1].strip():
                raise ValueError(f"invalid class line: {line!r}")
            if parts[0] in classes:
                raise ValueError(f"duplicate class entry: {parts[0]}")
            classes[parts[0]] = parts[1].strip()
        missing = set(self.image_ids) - set(classes)
        if missing:
            raise ValueError(f"missing class names for {len(missing)} images")
        self._classes = classes

    def samples(self) -> Iterator[FSCDSample]:
        for image_id in self.image_ids:
            candidate = Path(image_id)
            if candidate.name != image_id or candidate.is_absolute() or image_id in {".", ".."}:
                raise ValueError(f"invalid image ID: {image_id!r}")
            image_path = self.root / "images_384_VarV2" / image_id
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            yield FSCDSample(image_id, image_path, self._classes[image_id])

    def ground_truth(self) -> dict[str, FSCDGroundTruth]:
        """Evaluation only. Never pass this output to a planner or controller."""
        points_path = self.root / "annotation_FSC147_384.json"
        annotations = json.loads(points_path.read_text()) if points_path.exists() else {}
        coco = json.loads((self.root / f"instances_{self.split}.json").read_text())
        if not isinstance(coco, dict) or not isinstance(coco.get("images"), list) or not isinstance(coco.get("annotations"), list):
            raise ValueError("invalid FSCD COCO annotations")
        images = {item["file_name"]: item["id"] for item in coco["images"]
                  if isinstance(item, dict) and "file_name" in item and "id" in item}
        image_records = {item["file_name"]: item for item in coco["images"]}
        if len(images) != len(coco["images"]) or len(set(images.values())) != len(images):
            raise ValueError("duplicate or invalid COCO image IDs")
        if any(image_id not in images for image_id in self.image_ids):
            raise ValueError("FSCD COCO annotations missing split images")
        by_image: dict[int, list[dict]] = {images[image_id]: [] for image_id in self.image_ids}
        for item in coco["annotations"]:
            if not isinstance(item, dict):
                raise ValueError("invalid FSCD box annotation")
            if item.get("image_id") in by_image:
                by_image[item["image_id"]].append(item)
        result: dict[str, FSCDGroundTruth] = {}
        for image_id in self.image_ids:
            record = annotations.get(image_id, {"points": []})
            if not isinstance(record, dict) or not isinstance(record.get("points"), list):
                raise ValueError(f"missing point annotations for {image_id}")
            points = []
            for point in record["points"]:
                if (not isinstance(point, (list, tuple)) or len(point) != 2 or
                        any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                            not math.isfinite(v) for v in point)):
                    raise ValueError(f"invalid point annotation for {image_id}")
                points.append((float(point[0]), float(point[1])))
            box_records = by_image[images[image_id]]
            categories = {item.get("category_id") for item in box_records}
            if not box_records:
                categories = {category["id"] for category in coco.get("categories", [])}
            if len(categories) != 1 or any(type(v) is not int for v in categories):
                raise ValueError(f"expected one FSCD category for {image_id}")
            boxes = []
            for item in box_records:
                box = item.get("bbox")
                if (not isinstance(box, list) or len(box) != 4 or
                        any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                            not math.isfinite(v) for v in box) or box[2] <= 0 or box[3] <= 0):
                    raise ValueError(f"invalid FSCD box for {image_id}")
                boxes.append(tuple(float(v) for v in box))
            result[image_id] = FSCDGroundTruth(
                image_id, len(boxes), tuple(points), images[image_id],
                next(iter(categories)), tuple(boxes),
                image_records[image_id].get("width"), image_records[image_id].get("height"))
        return result


def evaluate_counts(predictions: dict[str, float | int | None],
                    ground_truth: dict[str, FSCDGroundTruth]) -> dict[str, float | int | bool | None]:
    """Return official count metrics only when every split image is valid."""
    unknown = set(predictions) - set(ground_truth)
    if unknown:
        raise ValueError(f"unknown prediction IDs: {sorted(unknown)[:3]}")
    valid = {image_id: float(predictions[image_id]) for image_id in ground_truth
             if image_id in predictions and predictions[image_id] is not None and
             not isinstance(predictions[image_id], bool) and
             isinstance(predictions[image_id], (int, float)) and
             math.isfinite(predictions[image_id]) and predictions[image_id] >= 0}
    errors = [valid[image_id] - ground_truth[image_id].count for image_id in valid]
    n = len(errors)
    complete = n == len(ground_truth) and n > 0
    return {"total_images": len(ground_truth), "evaluated_images": n,
            "complete": complete,
            "mae": sum(abs(e) for e in errors) / n if complete else None,
            "rmse": math.sqrt(sum(e * e for e in errors) / n) if complete else None}

