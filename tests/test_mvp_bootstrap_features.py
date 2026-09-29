import numpy as np
from PIL import Image

from sam3_vlm.mvp.core import Action, Config, Controller, Detection, select_exemplars


def mask(width, height, box):
    out = np.zeros((height, width), dtype=bool)
    x1, y1, x2, y2 = box
    out[y1:y2, x1:x2] = True
    return out


class Sensor:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def search(self, image, prompt, region, exemplar_boxes=()):
        self.calls.append((prompt, region, exemplar_boxes))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def test_strong_seed_runs_conditioned_refinement_without_negative_penalty():
    image = Image.new("RGB", (20, 20))
    seed = Detection(mask(20, 20, (2, 2, 6, 6)), .8)
    sensor = Sensor([[seed], []])
    result = Controller(sensor, None, Config(max_vlm_calls=0, enable_adaptive_tiling=False)).run(image, "fruit")
    assert result.calls["sam3_attempted"] == 2
    assert sensor.calls[1] == ("fruit", (0, 0, 20, 20), ((2, 2, 6, 6),))
    assert result.actions[1]["source"] == "exemplar_refinement"
    assert result.actions[1]["exemplar_node_ids"] == ["n0001"]
    assert result.nodes["n0001"].negatives == {}
    assert result.nodes["n0001"].belief > .6


def test_weak_seed_skips_refinement():
    image = Image.new("RGB", (20, 20))
    sensor = Sensor([[Detection(mask(20, 20, (2, 2, 6, 6)), .5)]])
    result = Controller(sensor, None, Config(max_vlm_calls=0, enable_adaptive_tiling=False)).run(image, "fruit")
    assert result.calls["sam3_attempted"] == 1


def test_refinement_failure_keeps_stage1_nodes():
    image = Image.new("RGB", (20, 20))
    sensor = Sensor([[Detection(mask(20, 20, (2, 2, 6, 6)), .8)], RuntimeError("offline")])
    result = Controller(sensor, None, Config(max_vlm_calls=0, enable_adaptive_tiling=False)).run(image, "fruit")
    assert result.stop_reason == "error" and result.partial
    assert result.calls["sam3_successful"] == 1
    assert result.counts["hard"] == 1
    assert result.nodes["n0001"].negatives == {}


def test_dense_stage1_triggers_individual_budgeted_tiles():
    image = Image.new("RGB", (400, 400))
    boxes = [(60 + 5 * (i % 10), 60 + 5 * (i // 10),
              62 + 5 * (i % 10), 62 + 5 * (i // 10)) for i in range(60)]
    first = [Detection(mask(400, 400, box), .8) for box in boxes]
    sensor = Sensor([first, [Detection(mask(400, 400, boxes[0]), .9)]])
    result = Controller(sensor, None, Config(max_vlm_calls=0, max_sam3_calls=2,
                        enable_exemplar_refinement=False)).run(image, "fruit")
    assert result.tiling["trigger"] is True
    assert len(result.tiling["tiles"]) > 1
    assert result.calls["sam3_attempted"] == 2
    assert result.actions[1]["source"] == "adaptive_tile"
    assert len(result.nodes) == 60  # A duplicate from the first tile associates.
    assert result.stop_reason == "sam3_budget" and result.partial
    assert result.unexecuted_bootstrap == list(result.tiling["tiles"])[1:]


def test_cross_tile_box_nms_associates_disjoint_partial_masks():
    grid = np.indices((20, 20)).sum(axis=0) % 2 == 0
    first = np.zeros((20, 20), dtype=bool)
    second = np.zeros((20, 20), dtype=bool)
    first[2:12, 2:12] = grid[2:12, 2:12]
    second[2:12, 2:12] = ~grid[2:12, 2:12]
    controller = Controller(Sensor([]), None, Config(max_vlm_calls=0))
    nodes, applied, events = {}, set(), []
    controller.apply_observation(Action("a1", "bootstrap", "fruit", (0, 0, 20, 20)),
                                 [Detection(first, .8)], nodes, applied, events, (20, 20))
    controller.apply_observation(Action("a2", "adaptive_tile", "fruit", (0, 0, 20, 20)),
                                 [Detection(second, .7)], nodes, applied, events, (20, 20))
    assert len(nodes) == 1
    assert len(nodes["n0001"].detection_ids) == 2


def test_density_uses_clean_high_score_stage1_nodes():
    image = Image.new("RGB", (400, 400))
    first = [Detection(mask(400, 400, (60 + 5 * (i % 10), 60 + 5 * (i // 10),
                                        62 + 5 * (i % 10), 62 + 5 * (i // 10))), .3)
             for i in range(60)]
    sensor = Sensor([first])
    result = Controller(sensor, None, Config(max_vlm_calls=0)).run(image, "fruit")
    assert result.tiling["trigger"] is False
    assert result.calls["sam3_attempted"] == 1


def test_tiles_use_refreshed_strong_exemplars_after_refinement():
    image = Image.new("RGB", (400, 400))
    boxes = [(60 + 5 * (i % 10), 60 + 5 * (i // 10),
              62 + 5 * (i % 10), 62 + 5 * (i // 10)) for i in range(60)]
    first = [Detection(mask(400, 400, box), .8) for box in boxes]
    new_box = (90, 90, 92, 92)
    sensor = Sensor([first, [Detection(mask(400, 400, new_box), .95)], []])
    result = Controller(sensor, None, Config(max_vlm_calls=0, max_sam3_calls=3)).run(image, "fruit")
    assert result.actions[1]["source"] == "exemplar_refinement"
    assert result.actions[2]["source"] == "adaptive_tile"
    assert result.actions[2]["exemplar_node_ids"][0] == "n0061"
    assert result.actions[2]["exemplar_boxes"][0] == new_box


def test_exemplar_eligibility_uses_strongest_evidence_with_largest_mask_box():
    controller = Controller(Sensor([]), None, Config(max_vlm_calls=0))
    nodes, applied, events = {}, set(), []
    controller.apply_observation(Action("a1", "bootstrap", "fruit", (0, 0, 20, 20)),
                                 [Detection(mask(20, 20, (2, 2, 6, 6)), .9),
                                  Detection(mask(20, 20, (2, 2, 10, 10)), .4)],
                                 nodes, applied, events, (20, 20))
    assert nodes["n0001"].canonical_score == .4
    assert select_exemplars(nodes, controller.config) == (("n0001", (2, 2, 10, 10)),)
