import math

import numpy as np
from PIL import Image

from sam3_vlm.mvp.core import Action, Config as MVPConfig, Controller, Detection, parse_batch


def Config(**kwargs):
    """Keep controller-contract tests focused on the original plain-search path."""
    return MVPConfig(enable_exemplar_refinement=False, enable_adaptive_tiling=False, **kwargs)


IMAGE = Image.new("RGB", (20, 20))


def mask(x1, y1, x2, y2):
    out = np.zeros((20, 20), dtype=bool)
    out[y1:y2, x1:x2] = True
    return out


class Sensor:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def search(self, image, prompt, region):
        self.calls.append((prompt, region))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class Planner:
    def __init__(self, batches):
        self.batches = iter(batches)
        self.states = []

    def propose(self, image, target, state):
        self.states.append(state)
        return next(self.batches)


def batch(*actions):
    return {"actions": [{"prompt": p, "region": r} for p, r in actions]}


def test_final_vlm_batch_runs_sequentially_and_uses_budget():
    sensor = Sensor([[Detection(mask(1, 1, 5, 5), .8)], [], [Detection(mask(10, 10, 14, 14), .8)]])
    planner = Planner([batch(("round fruit", None), ("green citrus", None))])
    result = Controller(sensor, planner, Config(max_vlm_calls=1, max_actions_per_vlm_call=2)).run(IMAGE, "green fruit")
    assert result.calls == {"sam3_attempted": 3, "sam3_successful": 3, "vlm_attempted": 1, "vlm_successful": 1}
    assert result.stop_reason == "qwen_budget"
    assert len(result.nodes) == 2
    assert result.actions[1]["events"][0]["type"] == "non_retrieval"
    assert planner.states[0]["remaining_sam3_calls"] == 15


def test_partial_invalid_batch_duplicate_and_truncation():
    sensor = Sensor([[], []])
    planner = Planner([{"actions": [
        {"prompt": "target", "region": None},
        {"prompt": "second", "region": [0, 0, 10, 10]},
        {"prompt": "bad", "region": [99, 0, 100, 10]},
        {"prompt": "third", "region": None},
    ]}])
    result = Controller(sensor, planner, Config(max_vlm_calls=1, max_actions_per_vlm_call=3)).run(IMAGE, "target")
    assert result.calls["sam3_attempted"] == 2
    assert result.proposals[0]["rejected"] == [
        {"index": 3, "reason": "batch_truncated"},
        {"index": 2, "reason": "invalid_region"},
        {"index": 0, "reason": "duplicate_completed"},
    ]
    assert result.counts == {"soft": 0, "hard": 0}


def test_non_retrieval_requires_coverage_and_is_once_per_prompt():
    sensor = Sensor([[Detection(mask(1, 1, 5, 5), .8)], [], [], []])
    planner = Planner([batch(("target", [0, 0, 2, 2]),
                             ("target", [0, 0, 19, 20]),
                             ("target", [0, 0, 18, 20]))])
    result = Controller(sensor, planner, Config(max_vlm_calls=1, max_actions_per_vlm_call=3)).run(IMAGE, "target")
    node = result.nodes["n0001"]
    assert len(node.negatives) == 1
    assert node.negatives["target"] == "a0003"
    assert result.actions[1]["events"] == []
    assert result.actions[2]["events"][0]["type"] == "non_retrieval"
    assert result.actions[3]["events"] == []
    assert node.belief < .659073


def test_ambiguous_detection_blocks_negative_and_mask_matching():
    sensor = Sensor([[Detection(mask(1, 1, 11, 11), .8)],
                     [Detection(mask(1, 1, 2, 2), .8)]])
    planner = Planner([batch(("other target", None))])
    result = Controller(sensor, planner, Config(max_vlm_calls=1)).run(IMAGE, "target")
    assert len(result.nodes) == 1
    assert result.nodes["n0001"].negatives == {}
    assert result.actions[1]["events"][0]["type"] == "ambiguous_containment"


def test_partial_mask_associates_and_replay_is_idempotent():
    sensor = Sensor([])
    controller = Controller(sensor, None)
    nodes, applied, events = {}, set(), []
    controller.apply_observation(Action("a1", "bootstrap", "target", (0, 0, 20, 20)),
                                 [Detection(mask(1, 1, 11, 11), .8)], nodes, applied, events, (20, 20))
    controller.apply_observation(Action("a2", "vlm", "  TARGET ", (0, 0, 20, 20)),
                                 [Detection(mask(1, 1, 5, 11), .7)], nodes, applied, events, (20, 20))
    prior, count = nodes["n0001"].belief, len(events)
    controller.apply_observation(Action("a2", "vlm", "  TARGET ", (0, 0, 20, 20)),
                                 [Detection(mask(1, 1, 5, 11), .7)], nodes, applied, events, (20, 20))
    assert len(nodes) == 1
    assert nodes["n0001"].belief == prior
    assert len(events) == count
    assert len(nodes["n0001"].positives) == 1
    assert nodes["n0001"].box == (1, 1, 11, 11)


def test_failed_and_empty_calls_have_distinct_counts_and_honest_bootstrap():
    failed = Controller(Sensor([RuntimeError("sensor down")]), None, Config(max_vlm_calls=0)).run(IMAGE, "target")
    assert failed.stop_reason == "error" and failed.counts == {"soft": None, "hard": None}
    assert failed.unexecuted_bootstrap == []
    empty = Controller(Sensor([[]]), None, Config(max_vlm_calls=0)).run(IMAGE, "target")
    assert empty.counts == {"soft": 0, "hard": 0}
    partial = Controller(Sensor([[]]), None, Config(max_vlm_calls=0, max_sam3_calls=1,
                          bootstrap_regions=((0, 0, 10, 10),))).run(IMAGE, "target")
    assert partial.stop_reason == "sam3_budget" and partial.partial
    assert partial.unexecuted_bootstrap == [(0, 0, 10, 10)]


def test_negative_is_region_specific_and_failure_never_penalizes():
    sensor = Sensor([[Detection(mask(1, 1, 5, 5), .8)], RuntimeError("failed")])
    planner = Planner([batch(("other target", None))])
    result = Controller(sensor, planner, Config(max_vlm_calls=1)).run(IMAGE, "target")
    assert result.stop_reason == "error" and result.partial
    assert result.nodes["n0001"].negatives == {}
    assert result.counts["soft"] > .6


def test_budget_truncates_final_batch_and_preserves_first_result():
    sensor = Sensor([[], [Detection(mask(1, 1, 5, 5), .8)]])
    planner = Planner([batch(("first", None), ("second", None))])
    result = Controller(sensor, planner, Config(max_vlm_calls=1, max_sam3_calls=2,
                          max_actions_per_vlm_call=2)).run(IMAGE, "target")
    assert result.stop_reason == "sam3_budget" and result.partial
    assert result.unexecuted_batch == [{"prompt": "second", "region": (0, 0, 20, 20)}]
    assert len(result.nodes) == 1


def test_ground_truth_is_not_an_inference_input():
    sensor = Sensor([[]])
    result = Controller(sensor, None, Config(max_vlm_calls=0)).run(IMAGE, "target")
    assert result.counts == {"soft": 0, "hard": 0}
    assert "ground_truth" not in Controller.run.__code__.co_varnames
    assert "ground_truth" not in str(result)


def test_batch_contract_rejects_malformed_json():
    assert parse_batch("{oops", 20, 20, 1)[1][0]["reason"] == "malformed_json"
    assert parse_batch({"actions": [{"prompt": "x", "region": None, "role": "negative"}]}, 20, 20, 1)[1][0]["reason"] == "invalid_action"
    valid, rejected = parse_batch({"actions": [42, {"prompt": "x", "region": None}]}, 20, 20, 2)
    assert valid == [(1, "x", (0, 0, 20, 20))]
    assert rejected == [{"index": 0, "reason": "invalid_action"}]


def test_malformed_detection_is_discarded_without_failing_search():
    sensor = Sensor([[Detection([[1], [1, 0]], .9),
                      Detection(mask(1, 1, 5, 5), np.complex128(.7 + .1j)),
                      Detection(mask(1, 1, 5, 5), 10**1000),
                      Detection(mask(1, 1, 5, 5), .8)]])
    result = Controller(sensor, None, Config(max_vlm_calls=0)).run(IMAGE, "target")
    assert result.calls["sam3_successful"] == 1
    assert result.stop_reason == "qwen_budget"
    assert len(result.nodes) == 1
    assert result.actions[0]["events"][0] == {"type": "discarded_detection", "index": 0}
    assert result.actions[0]["events"][1] == {"type": "discarded_detection", "index": 1}
    assert result.actions[0]["events"][2] == {"type": "discarded_detection", "index": 2}


def test_invalid_vlm_proposal_is_successful_request_but_no_action():
    result = Controller(Sensor([[]]), Planner(["{oops"]), Config(max_vlm_calls=1)).run(IMAGE, "target")
    assert result.calls["vlm_attempted"] == result.calls["vlm_successful"] == 1
    assert result.stop_reason == "no_valid_action"


def test_positive_retrieval_clears_same_prompt_negative_but_not_other_prompt():
    sensor = Sensor([[Detection(mask(1, 1, 5, 5), .8)], [], [],
                     [Detection(mask(1, 1, 5, 5), .8)]])
    planner = Planner([batch(("fruit", [0, 0, 19, 20]),
                             ("round fruit", None),
                             ("fruit", [0, 0, 18, 20]))])
    result = Controller(sensor, planner, Config(max_vlm_calls=1, max_actions_per_vlm_call=3)).run(IMAGE, "fruit")
    node = result.nodes["n0001"]
    assert set(node.negatives) == {"round fruit"}
    assert any(e["type"] == "negative_cleared" for e in result.actions[-1]["events"])


def test_duplicate_within_batch_and_different_region():
    sensor = Sensor([[], []])
    planner = Planner([batch(("target", [0, 0, 10, 10]),
                             (" TARGET ", [0, 0, 10, 10]),
                             ("target", [10, 0, 20, 10]))])
    result = Controller(sensor, planner, Config(max_vlm_calls=1, max_actions_per_vlm_call=3)).run(IMAGE, "target")
    assert result.calls["sam3_attempted"] == 3
    assert {r["reason"] for r in result.proposals[0]["rejected"]} == {"duplicate_batch"}


def test_failed_later_search_keeps_previous_result_and_no_negative():
    sensor = Sensor([[Detection(mask(1, 1, 5, 5), .8)], RuntimeError("offline")])
    planner = Planner([batch(("round target", None))])
    result = Controller(sensor, planner, Config(max_vlm_calls=1)).run(IMAGE, "target")
    assert result.calls["sam3_attempted"] == 2
    assert result.calls["sam3_successful"] == 1
    assert result.nodes["n0001"].negatives == {}
    assert result.counts["hard"] == 1


def test_runtime_budget_stops_between_batch_items_and_leaves_pending_action():
    now = [0.0]

    class TimedSensor:
        def search(self, image, prompt, region):
            now[0] += 0.6
            return []

    planner = Planner([batch(("first", None), ("second", None))])
    result = Controller(TimedSensor(), planner, Config(max_vlm_calls=1,
                        max_actions_per_vlm_call=2, max_runtime_seconds=1.0),
                        clock=lambda: now[0]).run(IMAGE, "target")
    assert result.stop_reason == "time_budget"
    assert result.calls["sam3_successful"] == 2
    assert result.unexecuted_batch == [{"prompt": "second", "region": (0, 0, 20, 20)}]
    assert result.proposals[0]["accepted"][1]["status"] == "pending"
    assert result.partial
