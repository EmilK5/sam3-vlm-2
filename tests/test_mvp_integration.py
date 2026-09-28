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
        def _run_inference(self, crop, prompt, threshold, positive_boxes=()):
            assert crop.size == (4, 4)
            assert prompt == "fruit" and threshold == 0.0
            assert positive_boxes == []
            return np.array([[0, 0, 4, 4]]), np.array([0.8]), [np.ones((2, 2), dtype=bool)]

    adapter.sensor = FakeSensor()
    detections = adapter.search(Image.new("RGB", (10, 10)), "fruit", (3, 2, 7, 6))
    assert len(detections) == 1
    assert detections[0].mask.shape == (10, 10)
    assert detections[0].mask.sum() == 16
    assert detections[0].mask[2:6, 3:7].all()


def test_sam3_exemplar_boxes_are_localized_to_crop():
    adapter = RealSAM3.__new__(RealSAM3)

    class FakeSensor:
        def _run_inference(self, crop, prompt, threshold, positive_boxes=()):
            assert crop.size == (10, 10)
            assert positive_boxes == [[2, 2, 5, 5]]
            return np.empty((0, 4)), np.empty(0), []

    adapter.sensor = FakeSensor()
    image = Image.new("RGB", (30, 30))
    assert adapter.search(image, "fruit", (10, 10, 20, 20), ((12, 12, 15, 15),)) == []


def test_mvp_reuses_loaded_legacy_sam3_sensor():
    class LoadedSensor:
        def _run_inference(self, *args, **kwargs):
            return np.empty((0, 4)), np.empty(0), []

    loaded = LoadedSensor()
    adapter = RealSAM3(sensor=loaded)
    assert adapter.sensor is loaded
    assert adapter.search(Image.new("RGB", (8, 8)), "fruit", (0, 0, 8, 8)) == []


def test_vlm_sends_image_candidates_and_action_limit():
    adapter = RealVLM.__new__(RealVLM)
    adapter.model = "fake-qwen"
    adapter.scope = "Only count fruit on trees."
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
    assert "Only count fruit on trees" in captured["messages"][0]["content"]
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["extra_body"] == {"reasoning_effort": "none"}


def test_vlm_first_prompt_requests_roi_and_force_tiling():
    adapter = RealVLM.__new__(RealVLM)
    adapter.model = "fake-qwen"
    adapter.scope = "Focus on fruit on trees; exclude fallen fruit."
    adapter.request_log = []
    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return type("Response", (), {"choices": [type("Choice", (), {
                "message": type("Message", (), {"content": '{"roi":null,"tile_mode":"auto","actions":[]}'})()
            })()]})()

    adapter.client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    raw = adapter.propose_initial(Image.new("RGB", (40, 30)), "berries", {"max_actions": 2})
    assert '"tile_mode":"auto"' in raw
    system = captured["messages"][0]["content"]
    assert "ROI" in system and "force" in system
    assert "exclude fallen fruit" in system
    assert "Image size: (40, 30)" in captured["messages"][1]["content"][0]["text"]
    assert adapter.request_log[0]["system"] == system
    assert "Image size: (40, 30)" in adapter.request_log[0]["user_text"]


def test_evaluation_does_not_change_inference_result():
    class EmptySensor:
        def search(self, image, prompt, region):
            return []

    result = Controller(EmptySensor(), None, Config(max_vlm_calls=0)).run(Image.new("RGB", (4, 4)), "fruit")
    before = json.dumps(result_dict(result), sort_keys=True)
    assert evaluate_count(result, 1)["hard_absolute_error"] == 1
    assert evaluate_count(result, 3)["hard_absolute_error"] == 3
    assert json.dumps(result_dict(result), sort_keys=True) == before
