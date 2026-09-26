import json

import pytest
from PIL import Image

from sam3_vlm.mvp.core import Node, Result
from sam3_vlm.mvp.core import Config
import numpy as np

from sam3_vlm.mvp.fscd147 import FSCD147, coco_detections, evaluate_counts
from sam3_vlm.mvp.fscd_cli import evaluate_saved, run_dataset


def dataset(tmp_path):
    (tmp_path / "images_384_VarV2").mkdir()
    Image.new("RGB", (20, 20)).save(tmp_path / "images_384_VarV2" / "a.jpg")
    Image.new("RGB", (20, 20)).save(tmp_path / "images_384_VarV2" / "b.jpg")
    (tmp_path / "Train_Test_Val_FSC_147.json").write_text(json.dumps({"val": ["a.jpg", "b.jpg"]}))
    (tmp_path / "ImageClasses_FSC147.txt").write_text("a.jpg green apples\nb.jpg bottle caps\n")
    (tmp_path / "annotation_FSC147_384.json").write_text(json.dumps({
        "a.jpg": {"points": [[1, 2], [3, 4]]}, "b.jpg": {"points": [[1, 2]]}}))
    (tmp_path / "instances_val.json").write_text(json.dumps({
        "images": [{"id": 11, "file_name": "a.jpg"}, {"id": 12, "file_name": "b.jpg"}],
        "annotations": [
            {"id": 1, "image_id": 11, "category_id": 1, "bbox": [1, 2, 3, 4]},
            {"id": 2, "image_id": 11, "category_id": 1, "bbox": [4, 5, 2, 3]},
            {"id": 3, "image_id": 12, "category_id": 1, "bbox": [1, 2, 3, 4]}],
        "categories": [{"id": 1, "name": "object"}]}))
    return FSCD147(tmp_path)


def test_inference_sample_contains_only_image_and_target(tmp_path):
    adapter = dataset(tmp_path)
    samples = list(adapter.samples())
    assert [(s.image_id, s.target) for s in samples] == [
        ("a.jpg", "green apples"), ("b.jpg", "bottle caps")]
    assert not hasattr(samples[0], "ground_truth")
    (tmp_path / "annotation_FSC147_384.json").unlink()
    assert len(list(adapter.samples())) == 2  # Inference never reads annotations.


def test_evaluation_is_separate_and_reports_completeness(tmp_path):
    adapter = dataset(tmp_path)
    truth = adapter.ground_truth()
    assert truth["a.jpg"].count == 2
    assert truth["a.jpg"].coco_image_id == 11
    assert truth["a.jpg"].boxes == ((1., 2., 3., 4.), (4., 5., 2., 3.))
    complete = evaluate_counts({"a.jpg": 3, "b.jpg": 1}, truth)
    assert complete == {"total_images": 2, "evaluated_images": 2,
                        "complete": True, "mae": .5, "rmse": 2**-.5}
    partial = evaluate_counts({"a.jpg": None, "b.jpg": 0}, truth)
    assert partial["complete"] is False and partial["evaluated_images"] == 1
    assert partial["mae"] is None and partial["rmse"] is None


def test_coco_export_uses_predicted_boxes_and_evaluation_only_ids(tmp_path):
    truth = dataset(tmp_path).ground_truth()
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:6, 1:4] = True
    node = Node("n1", mask, (1, 2, 4, 6), "d1", .9, "a1", "a1", belief=.8)
    weak = Node("n2", mask, (5, 2, 8, 6), "d2", .3, "a1", "a1", belief=.2)
    def result(nodes):
        return Result(nodes, [], [], {"hard": sum(n.belief >= .5 for n in nodes.values()), "soft": .8}, {}, "qwen_budget",
                      False, [], [], [], 0.0, {})
    predictions = {"a.jpg": result({"n1": node, "n2": weak}), "b.jpg": result({})}
    assert coco_detections(predictions, truth, hard_count_threshold=.5) == [
        {"image_id": 11, "category_id": 1, "bbox": [1, 2, 3, 4], "score": .8}]
    with pytest.raises(ValueError, match="every split image"):
        coco_detections({"a.jpg": predictions["a.jpg"]}, truth, hard_count_threshold=.5)
    predictions["b.jpg"].partial = True
    with pytest.raises(ValueError, match="incomplete"):
        coco_detections(predictions, truth, hard_count_threshold=.5)
    predictions["b.jpg"].partial = False
    with pytest.raises(ValueError, match="threshold mismatch"):
        coco_detections(predictions, truth, hard_count_threshold=.9)


def test_missing_class_and_invalid_split_fail_closed(tmp_path):
    dataset(tmp_path)
    (tmp_path / "ImageClasses_FSC147.txt").write_text("a.jpg green apples\n")
    with pytest.raises(ValueError, match="missing class"):
        FSCD147(tmp_path)
    with pytest.raises(ValueError, match="split"):
        FSCD147(tmp_path, split="train")


def test_runner_keeps_ground_truth_out_of_inference_and_blocks_partial_scores(tmp_path):
    adapter = dataset(tmp_path)
    class Sensor:
        def search(self, image, prompt, region, exemplar_boxes=()):
            return []
    class Planner:
        def propose_initial(self, image, target, state):
            return {"roi": None, "tile_mode": "auto",
                    "actions": [{"prompt": target, "region": None}]}
    config = Config(vlm_first=True, max_vlm_calls=1, enable_adaptive_tiling=False,
                    enable_exemplar_refinement=False)
    path = run_dataset(adapter, config, Sensor(), Planner(), tmp_path / "run")
    assert len(path.read_text().splitlines()) == 2
    coco_path = tmp_path / "detections.json"
    metrics = evaluate_saved(adapter, path, config, coco_path)
    assert metrics["complete"] and metrics["mae"] == 1.5
    assert metrics["detection_export_complete"] and json.loads(coco_path.read_text()) == []
    partial_path = run_dataset(adapter, config, Sensor(), Planner(), tmp_path / "partial", max_images=1)
    partial = evaluate_saved(adapter, partial_path, config, tmp_path / "partial_detections.json")
    assert not partial["complete"] and partial["mae"] is None
    assert not (tmp_path / "partial_detections.json").exists()
    (tmp_path / "annotation_FSC147_384.json").unlink()
    (tmp_path / "instances_val.json").unlink()
    assert len(run_dataset(adapter, config, Sensor(), Planner(), tmp_path / "no_gt").read_text().splitlines()) == 2
