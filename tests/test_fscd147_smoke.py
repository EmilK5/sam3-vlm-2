import json

from PIL import Image
import pytest

from sam3_vlm.datasets.fscd147 import FSCD147
from sam3_vlm.experiments.fscd147 import evaluate_saved
from sam3_vlm.experiments.fscd147_smoke import main, prepare_subset, review_predictions


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "dataset"
    (root / "images_384_VarV2").mkdir(parents=True)
    ids = [f"{index}.jpg" for index in range(12)]
    for name in ids:
        Image.new("RGB", (32, 32)).save(root / "images_384_VarV2" / name)
    (root / "Train_Test_Val_FSC_147.json").write_text(json.dumps({"val": ids, "test": ids[:3]}))
    (root / "ImageClasses_FSC147.txt").write_text("".join(f"{name} green apples\n" for name in ids))
    coco = {"images": [{"id": i, "file_name": name, "width": 64, "height": 64} for i, name in enumerate(ids)],
            "annotations": [{"id": i, "image_id": i, "category_id": 1, "bbox": [4, 4, 16, 16]}
                            for i in range(12)], "categories": [{"id": 1, "name": "object"}]}
    (root / "instances_val.json").write_text(json.dumps(coco))
    (root / "annotation_FSC147_384.json").write_text(json.dumps({name: {"points": [[4, 4]]} for name in ids}))
    return root


def predictions(root, dataset, *, failed=False):
    root.mkdir()
    variant = "C_Test"
    metadata = {"schema_version": 1, "dataset": "FSCD-147", "split": "val",
                "image_ids": list(dataset.image_ids), "variants": {variant: {"count_type": "soft_posterior_count"}}}
    (root / "metadata.json").write_text(json.dumps(metadata))
    rows = [{"image_id": name, "variant": variant, "success": not (failed and i == 0),
             "count_type": "soft_posterior_count", "predicted_count": .75, "runtime_seconds": 1.5,
             "image_size": [32, 32], "nodes": [{"box": [2, 2, 10, 10], "score": .75}],
             "error": "<model failed>"} for i, name in enumerate(dataset.image_ids)]
    path = root / "predictions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_random_subset_is_reproducible_filtered_and_evaluable(source, tmp_path):
    original = (source / "Train_Test_Val_FSC_147.json").read_bytes()
    first = prepare_subset(source, tmp_path / "first", seed=42)
    second = prepare_subset(source, tmp_path / "second", seed=42)
    third = prepare_subset(source, tmp_path / "third", seed=43)
    assert len(first["image_ids"]) == 10 and len(set(first["image_ids"])) == 10
    assert first == second and first["image_ids"] != third["image_ids"]
    assert first["image_ids"] != list(FSCD147(source).image_ids[:10])
    assert original == (source / "Train_Test_Val_FSC_147.json").read_bytes()
    subset = FSCD147(tmp_path / "first")
    samples = list(subset.samples())
    assert len(samples) == 10 and all(sample.image_path.is_symlink() for sample in samples)
    coco = json.loads((subset.root / "instances_val.json").read_text())
    assert len(coco["images"]) == len(coco["annotations"]) == 10
    assert {record["image_id"] for record in coco["annotations"]} == {record["id"] for record in coco["images"]}
    assert set(json.loads((subset.root / "annotation_FSC147_384.json").read_text())) == set(first["image_ids"])
    path = predictions(tmp_path / "run", subset)
    metrics = evaluate_saved(subset, path)["variants"]["C_Test"]
    assert metrics["complete"] and metrics["total_images"] == 10 and metrics["mae"] == .25
    with pytest.raises(FileExistsError):
        prepare_subset(source, subset.root)


@pytest.mark.parametrize("count", [0, -1, 13, True])
def test_invalid_sample_size_leaves_no_destination(source, tmp_path, count):
    destination = tmp_path / "invalid"
    with pytest.raises(ValueError):
        prepare_subset(source, destination, count=count)
    assert not destination.exists()


def test_gallery_shows_counts_failure_and_scaled_ground_truth(source, tmp_path):
    prepare_subset(source, tmp_path / "subset")
    dataset = FSCD147(tmp_path / "subset")
    path = predictions(tmp_path / "run", dataset, failed=True)
    result = review_predictions(dataset.root, path)
    assert result["successful_runs"] == 9 and result["expected_runs"] == 10
    html = (path.parent / "review" / "index.html").read_text()
    assert "count 0.75" in html and "1 candidates" in html
    assert "FAILED" in html and "&lt;model failed&gt;" in html and "<model failed>" not in html
    gt_images = list((path.parent / "review" / "images").glob("*_gt.png"))
    assert len(gt_images) == 10
    with Image.open(gt_images[0]) as overlay:
        assert overlay.getpixel((2, 2)) == (0, 255, 0)  # GT 64px -> image 32px.
    with pytest.raises(FileExistsError):
        review_predictions(dataset.root, path)


def test_review_rejects_split_mismatch_before_writing(source, tmp_path):
    prepare_subset(source, tmp_path / "subset")
    subset = FSCD147(tmp_path / "subset")
    path = predictions(tmp_path / "run", subset)
    with pytest.raises(ValueError, match="review split"):
        review_predictions(source, path)
    assert not (path.parent / "review").exists()


def test_prepare_cli(source, tmp_path, capsys):
    assert main(["prepare", str(source), str(tmp_path / "subset"), "--count", "10", "--seed", "42"]) == 0
    assert json.loads(capsys.readouterr().out)["count"] == 10
