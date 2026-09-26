import numpy as np
from PIL import Image

from sam3_vlm.mvp.core import Config, Controller, Detection, parse_initial_plan


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
