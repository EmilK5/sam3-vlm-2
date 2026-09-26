import json

import numpy as np
from PIL import Image

from sam3_vlm.mvp.adapters import RealSAM3, RealVLM
from sam3_vlm.mvp.cli import result_dict
from sam3_vlm.mvp.core import Config, Controller
from sam3_vlm.mvp.evaluation import evaluate_count


def test_sam3_crop_mask_returns_global_coordinates():
    adapter = RealSAM3.__new__(RealSAM3)

    class FakeSensor:
        def _run_inference(self, crop, prompt, threshold):
            assert crop.size == (4, 4)
            assert prompt == "fruit" and threshold == 0.0
            return np.array([[0, 0, 4, 4]]), np.array([0.8]), [np.ones((2, 2), dtype=bool)]

    adapter.sensor = FakeSensor()
    detections = adapter.search(Image.new("RGB", (10, 10)), "fruit", (3, 2, 7, 6))
    assert len(detections) == 1
    assert detections[0].mask.shape == (10, 10)
    assert detections[0].mask.sum() == 16
    assert detections[0].mask[2:6, 3:7].all()


def test_vlm_sends_image_candidates_and_action_limit():
    adapter = RealVLM.__new__(RealVLM)
    adapter.model = "fake-qwen"
    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return type("Response", (), {"choices": [type("Choice", (), {
                "message": type("Message", (), {"content": '{"actions":[]}'})()
            })()]})()

    adapter.client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    state = {"max_actions": 2, "candidate_boxes": [(0, 0, 2, 2)], "nodes": [],
             "history": [], "successful_coverage": [], "remaining_sam3_calls": 3}
    assert adapter.propose(Image.new("RGB", (4, 4)), "fruit", state) == '{"actions":[]}'
    content = captured["messages"][1]["content"]
    assert len([part for part in content if part["type"] == "image_url"]) == 2
    assert "Maximum actions: 2" in content[0]["text"]


def test_evaluation_does_not_change_inference_result():
    class EmptySensor:
        def search(self, image, prompt, region):
            return []

    result = Controller(EmptySensor(), None, Config(max_vlm_calls=0)).run(Image.new("RGB", (4, 4)), "fruit")
    before = json.dumps(result_dict(result), sort_keys=True)
    assert evaluate_count(result, 1)["hard_absolute_error"] == 1
    assert evaluate_count(result, 3)["hard_absolute_error"] == 3
    assert json.dumps(result_dict(result), sort_keys=True) == before
