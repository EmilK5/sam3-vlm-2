import json

import numpy as np
from PIL import Image

from sam3_vlm.mvp.core import Config, Controller, Detection, parse_batch, parse_initial_plan


IMAGE = Image.new("RGB", (400, 400))
ROI = [50, 50, 350, 350]


def detection(box):
    mask = np.zeros((400, 400), dtype=bool)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = True
    return Detection(mask, .9)


class Sensor:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def search(self, image, prompt, region, exemplar_boxes=()):
        self.calls.append((prompt, region, exemplar_boxes))
        return next(self.responses)


class Planner:
    def __init__(self, initial, followups=()):
        self.initial = initial
        self.followups = iter(followups)
        self.states = []

    def propose_initial(self, image, target, state):
        self.states.append(("initial", state))
        return self.initial

    def propose(self, image, target, state):
        self.states.append(("followup", state))
        return next(self.followups)


def initial(tile_mode="auto", roi=ROI):
    return {"roi": roi, "tile_mode": tile_mode,
            "actions": [{"prompt": "red berries", "region": None}]}


def test_initial_vlm_plan_precedes_every_sam3_call_and_refines_inside_roi():
    sensor = Sensor([[detection((70, 70, 80, 80))], []])
    planner = Planner(initial())
    config = Config(vlm_first=True, max_vlm_calls=1, max_sam3_calls=2,
                    enable_adaptive_tiling=False)
    result = Controller(sensor, planner, config).run(IMAGE, "berries")
    assert result.roi == tuple(ROI)
    assert [a["source"] for a in result.actions] == ["vlm_initial", "exemplar_refinement"]
    assert sensor.calls == [
        ("red berries", tuple(ROI), ()),
        ("berries", tuple(ROI), ((70, 70, 80, 80),))]
    assert result.calls["vlm_attempted"] == 1
    assert planner.states[0][0] == "initial" and planner.states[0][1]["nodes"] == []


def test_vlm_can_force_roi_tiles_after_empty_initial_search():
    sensor = Sensor([[], [detection((55, 55, 60, 60))]])
    planner = Planner(initial(tile_mode="force"))
    config = Config(vlm_first=True, max_vlm_calls=1, max_sam3_calls=2,
                    enable_exemplar_refinement=False)
    result = Controller(sensor, planner, config).run(IMAGE, "berries")
    assert result.tiling["trigger"] and result.tiling["forced_by_vlm"]
    assert result.actions[1]["source"] == "adaptive_tile"
    assert result.counts["hard"] == 1
    assert all(ROI[0] <= region[0] < region[2] <= ROI[2] and
               ROI[1] <= region[1] < region[3] <= ROI[3]
               for _, region, _ in sensor.calls)


def test_invalid_initial_roi_fails_before_sam3_and_followups_stay_inside_roi():
    bad_sensor = Sensor([])
    bad = Controller(bad_sensor, Planner(initial(roi=[-1, 0, 20, 20])),
                     Config(vlm_first=True, max_vlm_calls=1)).run(IMAGE, "berries")
    assert bad.stop_reason == "error" and bad_sensor.calls == []
    sensor = Sensor([[], []])
    planner = Planner(initial(), [{"actions": [
        {"prompt": "berries", "region": [0, 0, 100, 100]},
        {"prompt": "small berries", "region": None}]}])
    result = Controller(sensor, planner, Config(vlm_first=True, max_vlm_calls=2,
                        max_actions_per_vlm_call=2, enable_adaptive_tiling=False,
                        enable_exemplar_refinement=False)).run(IMAGE, "berries")
    assert result.proposals[1]["rejected"] == [{"index": 0, "reason": "outside_roi"}]
    assert sensor.calls[-1][1] == tuple(ROI)


def test_initial_plan_contract_rejects_no_search_and_outside_roi():
    try:
        parse_initial_plan({"roi": ROI, "tile_mode": "force", "actions": []}, 400, 400, 1)
    except ValueError as exc:
        assert "no valid positive search" in str(exc)
    else:
        assert False
    try:
        Config(vlm_first=True, bootstrap_regions=((0, 0, 10, 10),))
    except ValueError as exc:
        assert "forbids bootstrap" in str(exc)
    else:
        assert False


def test_initial_plan_accepts_cluster_qwen_fence_and_roi_default():
    raw = '''```json
{
    "roi": [4, 36, 995, 786],
    "tile_mode": "auto",
    "actions": [
        {"prompt": "green fruit on trees"}
    ]
}
```'''
    roi, force_tiles, batch, rejected = parse_initial_plan(
        raw, 720, 1280, 2, coordinate_mode="normalized_1000")
    assert roi == (2, 46, 717, 1007)
    assert force_tiles is False
    assert batch == [(0, "green fruit on trees", roi)]
    assert rejected == []
    sensor = Sensor([[]])
    result = Controller(sensor, Planner(raw), Config(vlm_first=True, max_vlm_calls=1,
                        vlm_coordinate_mode="normalized_1000", enable_adaptive_tiling=False,
                        enable_exemplar_refinement=False)).run(
                            Image.new("RGB", (720, 1280)), "green fruit")
    assert sensor.calls == [("green fruit on trees", roi, ())]
    assert result.actions[0]["source"] == "vlm_initial"


def test_initial_plan_still_rejects_prose_and_ambiguous_actions():
    try:
        parse_initial_plan('Here is the plan: {"roi":null,"tile_mode":"auto","actions":[]}',
                           400, 400, 2)
    except ValueError as exc:
        assert "malformed initial VLM JSON" in str(exc)
    else:
        assert False
    try:
        parse_initial_plan({"roi": ROI, "tile_mode": "auto",
                            "actions": [{"prompt": "berries", "unexpected": True}]},
                           400, 400, 2)
    except ValueError as exc:
        assert "no valid positive search" in str(exc)
    else:
        assert False


def test_followup_action_without_region_uses_vlm_roi():
    batch, rejected = parse_batch({"actions": [{"prompt": "more green fruit"}]},
                                  400, 400, 2, allowed_region=tuple(ROI))
    assert batch == [(0, "more green fruit", tuple(ROI))]
    assert rejected == []


def test_explicit_normalized_region_is_converted_before_roi_check():
    roi = (2, 46, 717, 1007)
    batch, rejected = parse_batch({"actions": [{"prompt": "small green fruit",
                                                  "region": [100, 100, 900, 700]}]},
                                  720, 1280, 2, allowed_region=roi,
                                  coordinate_mode="normalized_1000")
    assert batch == [(0, "small green fruit", (72, 128, 648, 896))]
    assert rejected == []
    outside, rejected = parse_batch({"actions": [{"prompt": "fruit",
                                                   "region": [0, 0, 1000, 1000]}]},
                                    720, 1280, 2, allowed_region=roi,
                                    coordinate_mode="normalized_1000")
    assert outside == [] and rejected == [{"index": 0, "reason": "outside_roi"}]
    try:
        parse_initial_plan({"roi": [0, 0, 1001, 900], "tile_mode": "auto",
                            "actions": [{"prompt": "fruit"}]},
                           720, 1280, 2, coordinate_mode="normalized_1000")
    except ValueError as exc:
        assert "invalid VLM ROI" in str(exc)
    else:
        assert False


def test_malformed_initial_reply_is_saved_for_diagnosis():
    sensor = Sensor([])
    result = Controller(sensor, Planner("```json\nnot valid\n```"),
                        Config(vlm_first=True, max_vlm_calls=1)).run(IMAGE, "berries")
    assert result.stop_reason == "error"
    assert sensor.calls == []
    assert result.proposals[0]["raw"] == "```json\nnot valid\n```"
    assert result.proposals[0]["status"] == "invalid"
    assert "response starts with '```json" in result.errors[0]


def test_malformed_initial_reply_is_written_to_qwen_artifact(tmp_path):
    from sam3_vlm.experiments.mvp_fruit_arm import run_fruit_arm

    result = run_fruit_arm(IMAGE, "berries", Config(vlm_first=True, max_vlm_calls=1),
                           Sensor([]), Planner("not JSON"), tmp_path)
    assert result.partial
    saved = json.loads((tmp_path / "artifacts" / "qwen" / "call_001.json").read_text())
    assert saved["output"] == "not JSON"
