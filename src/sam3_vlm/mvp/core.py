"""Small, model-independent SAM3 + VLM counting controller.

All masks and regions use original-image coordinates. The two adapters only
return observations and proposals; this module owns every state change.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import time
from typing import Any, Protocol

import numpy as np

from .tiling import plan_adaptive_tiles

Region = tuple[int, int, int, int]


def normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.lower().split())


def mask_box(mask: np.ndarray) -> Region:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("empty mask")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


@dataclass(frozen=True)
class Config:
    sam3_score_threshold: float = 0.25
    match_iou_threshold: float = 0.50
    match_iom_threshold: float = 0.80
    min_match_area_ratio: float = 0.25
    belief_prior: float = 0.25
    evidence_base: float = 9.0
    evidence_discount: float = 0.50
    negative_strength: float = 0.50
    negative_coverage: float = 0.90
    hard_count_threshold: float = 0.50
    bootstrap_regions: tuple[Region, ...] = ()
    vlm_first: bool = False
    enable_exemplar_refinement: bool = True
    exemplar_min_score: float = 0.60
    exemplar_max_count: int = 5
    enable_adaptive_tiling: bool = True
    adaptive_density_threshold: float = 0.69
    adaptive_density_seed_score: float = 0.50
    tile_box_iou_threshold: float = 0.50
    max_vlm_calls: int = 2
    max_sam3_calls: int = 16
    max_actions_per_vlm_call: int = 1
    max_candidate_crops: int = 4
    max_runtime_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in ("sam3_score_threshold", "match_iou_threshold", "match_iom_threshold",
                     "hard_count_threshold", "exemplar_min_score", "adaptive_density_threshold",
                     "adaptive_density_seed_score", "tile_box_iou_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0 < self.min_match_area_ratio <= 1:
            raise ValueError("min_match_area_ratio must be in (0, 1]")
        if not math.isfinite(self.negative_coverage) or not 0 < self.negative_coverage <= 1:
            raise ValueError("negative_coverage must be in (0, 1]")
        if not 0 < self.belief_prior < 1 or not math.isfinite(self.evidence_base) or self.evidence_base <= 1:
            raise ValueError("invalid belief prior or evidence base")
        if (not math.isfinite(self.negative_strength) or not 0 <= self.evidence_discount < 1
                or not 0 < self.negative_strength):
            raise ValueError("invalid evidence discount or negative strength")
        for name in ("max_vlm_calls", "max_sam3_calls", "max_candidate_crops", "exemplar_max_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if type(self.max_actions_per_vlm_call) is not int or self.max_actions_per_vlm_call < 1:
            raise ValueError("max_actions_per_vlm_call must be a positive integer")
        if self.max_runtime_seconds is not None and (
            not math.isfinite(self.max_runtime_seconds) or self.max_runtime_seconds <= 0
        ):
            raise ValueError("max_runtime_seconds must be positive or null")
        object.__setattr__(self, "bootstrap_regions", tuple(tuple(r) for r in self.bootstrap_regions))
        if (type(self.enable_exemplar_refinement) is not bool or
                type(self.enable_adaptive_tiling) is not bool or type(self.vlm_first) is not bool):
            raise ValueError("feature switches must be boolean")
        if self.vlm_first and (self.bootstrap_regions or self.max_vlm_calls < 1):
            raise ValueError("VLM-first mode requires a VLM call and forbids bootstrap regions")


@dataclass(frozen=True)
class Detection:
    mask: np.ndarray
    score: float
    detection_id: str = ""


@dataclass(frozen=True)
class Action:
    action_id: str
    source: str
    prompt: str
    region: Region
    exemplar_boxes: tuple[Region, ...] = ()


@dataclass
class Node:
    node_id: str
    mask: np.ndarray
    box: Region
    canonical_detection_id: str
    canonical_score: float
    created_at: str
    latest_at: str
    detection_ids: list[str] = field(default_factory=list)
    positives: dict[str, tuple[float, str]] = field(default_factory=dict)
    negatives: dict[str, str] = field(default_factory=dict)
    belief: float = 0.0


@dataclass
class Result:
    nodes: dict[str, Node]
    actions: list[dict[str, Any]]
    proposals: list[dict[str, Any]]
    counts: dict[str, float | int | None]
    calls: dict[str, int]
    stop_reason: str
    partial: bool
    errors: list[str]
    unexecuted_bootstrap: list[Region]
    unexecuted_batch: list[dict[str, Any]]
    elapsed_seconds: float
    model_seconds: dict[str, float]
    tiling: dict[str, Any] | None = None
    roi: Region | None = None


class SAM3(Protocol):
    def search(self, image: Any, prompt: str, region: Region,
               exemplar_boxes: tuple[Region, ...] = ()) -> list[Detection]: ...


class VLM(Protocol):
    def propose(self, image: Any, target: str, state: dict[str, Any]) -> Any: ...


def belief(node: Node, config: Config) -> float:
    pos = sum((config.evidence_discount ** i) * score for i, score in
              enumerate(sorted((entry[0] for entry in node.positives.values()), reverse=True)))
    neg = sum(config.evidence_discount ** i for i in range(len(node.negatives)))
    log_odds = math.log(config.belief_prior / (1 - config.belief_prior))
    log_odds += math.log(config.evidence_base) * (pos - config.negative_strength * neg)
    if log_odds >= 0:
        return 1 / (1 + math.exp(-log_odds))
    odds = math.exp(log_odds)
    return odds / (1 + odds)


def overlap(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    aa, bb = int(a.sum()), int(b.sum())
    intersection = int(np.count_nonzero(a & b))
    return (intersection / (aa + bb - intersection),
            intersection / min(aa, bb), min(aa, bb) / max(aa, bb))


def box_iou(a: Region, b: Region) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def select_exemplars(nodes: dict[str, Node], config: Config) -> tuple[tuple[str, Region], ...]:
    """Select strong SAM3-grounded seed boxes; no VLM or ground truth input."""
    best_score = {n.node_id: max((score for score, _ in n.positives.values()), default=0.0)
                  for n in nodes.values()}
    eligible = [n for n in nodes.values() if best_score[n.node_id] >= config.exemplar_min_score]
    eligible.sort(key=lambda n: (-best_score[n.node_id], -n.belief, n.node_id))
    return tuple((n.node_id, n.box) for n in eligible[:config.exemplar_max_count])


def parse_batch(raw: Any, width: int, height: int, limit: int, *,
                allowed_region: Region | None = None) -> tuple[list[tuple[int, str, Region]], list[dict[str, Any]]]:
    """Parse a JSON object with `actions`, keeping valid entries in order."""
    rejected: list[dict[str, Any]] = []
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError) as exc:
        return [], [{"reason": "malformed_json", "detail": str(exc)}]
    if type(data) is not dict or set(data) != {"actions"} or type(data["actions"]) is not list:
        return [], [{"reason": "malformed_batch"}]
    actions = data["actions"]
    if len(actions) > limit:
        rejected.extend({"index": i, "reason": "batch_truncated"} for i in range(limit, len(actions)))
    accepted: list[tuple[int, str, Region]] = []
    for i, item in enumerate(actions[:limit]):
        if type(item) is not dict or set(item) != {"prompt", "region"}:
            rejected.append({"index": i, "reason": "invalid_action"})
            continue
        prompt, region = item["prompt"], item["region"]
        if type(prompt) is not str or not normalize_prompt(prompt):
            rejected.append({"index": i, "reason": "invalid_prompt"})
            continue
        if region is None:
            box = allowed_region or (0, 0, width, height)
        elif (type(region) is list and len(region) == 4 and
              all(type(v) is int for v in region)):
            box = tuple(region)
        else:
            rejected.append({"index": i, "reason": "invalid_region"})
            continue
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            rejected.append({"index": i, "reason": "invalid_region"})
            continue
        if allowed_region is not None and not (allowed_region[0] <= x1 < x2 <= allowed_region[2] and
                                               allowed_region[1] <= y1 < y2 <= allowed_region[3]):
            rejected.append({"index": i, "reason": "outside_roi"})
            continue
        accepted.append((i, prompt.strip(), box))
    return accepted, rejected


def parse_initial_plan(raw: Any, width: int, height: int, limit: int
                       ) -> tuple[Region, bool, list[tuple[int, str, Region]], list[dict[str, Any]]]:
    """Require an explicit VLM ROI and a first positive search batch."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError) as exc:
        raise ValueError(f"malformed initial VLM JSON: {exc}") from exc
    if type(data) is not dict or set(data) != {"roi", "tile_mode", "actions"}:
        raise ValueError("initial VLM plan requires roi, tile_mode, and actions")
    roi_raw = data["roi"]
    if roi_raw is None:
        roi = (0, 0, width, height)
    elif type(roi_raw) is list and _valid_region(roi_raw, width, height):
        roi = tuple(roi_raw)
    else:
        raise ValueError("invalid VLM ROI")
    if data["tile_mode"] not in ("auto", "force"):
        raise ValueError("invalid VLM tile mode")
    batch, rejected = parse_batch({"actions": data["actions"]}, width, height, limit,
                                  allowed_region=roi)
    if not batch:
        raise ValueError("initial VLM plan has no valid positive search")
    return roi, data["tile_mode"] == "force", batch, rejected


class Controller:
    def __init__(self, sam3: SAM3, vlm: VLM | None, config: Config = Config(), clock=time.monotonic):
        self.sam3, self.vlm, self.config, self.clock = sam3, vlm, config, clock

    def run(self, image: Any, target: str) -> Result:
        start = self.clock()
        width, height = image.size
        if width <= 0 or height <= 0 or not normalize_prompt(target):
            raise ValueError("an RGB image and nonempty target are required")
        if getattr(image, "mode", "RGB") != "RGB":
            raise ValueError("image must be RGB")
        for region in self.config.bootstrap_regions:
            if not _valid_region(region, width, height):
                raise ValueError(f"invalid bootstrap region: {region}")
        nodes: dict[str, Node] = {}
        actions: list[dict[str, Any]] = []
        proposals: list[dict[str, Any]] = []
        errors: list[str] = []
        applied: set[str] = set()
        completed_keys: set[tuple[str, Region]] = set()
        calls = {"sam3_attempted": 0, "sam3_successful": 0,
                 "vlm_attempted": 0, "vlm_successful": 0}
        model_seconds = {"sam3": 0.0, "vlm": 0.0}
        full_region = (0, 0, width, height)
        roi = full_region
        force_tiles = False
        unexecuted_bootstrap: list[Region] = []
        unexecuted_batch: list[dict[str, Any]] = []
        tiling: dict[str, Any] | None = None
        stop = ""

        def limit() -> str:
            if self.config.max_runtime_seconds is not None and self.clock() - start >= self.config.max_runtime_seconds:
                return "time_budget"
            if calls["sam3_attempted"] >= self.config.max_sam3_calls:
                return "sam3_budget"
            return ""

        def execute(prompt: str, region: Region, source: str,
                    exemplars: tuple[tuple[str, Region], ...] = ()) -> str:
            key = (normalize_prompt(prompt), region)
            if key in completed_keys and source != "exemplar_refinement":
                actions.append({"status": "duplicate", "source": source, "prompt": prompt, "region": region})
                return ""
            bounded = limit()
            if bounded:
                return bounded
            action_id = f"a{len(actions) + 1:04d}"
            boxes = tuple(box for _, box in exemplars)
            action = Action(action_id, source, prompt, region, boxes)
            entry: dict[str, Any] = {"action_id": action_id, "source": source,
                                      "prompt": prompt, "region": region, "status": "running", "events": [],
                                      "exemplar_node_ids": [node_id for node_id, _ in exemplars],
                                      "exemplar_boxes": boxes}
            actions.append(entry)
            calls["sam3_attempted"] += 1
            tick = self.clock()
            try:
                if boxes:
                    detections = self.sam3.search(image, prompt, region, boxes)
                else:
                    detections = self.sam3.search(image, prompt, region)
                model_seconds["sam3"] += max(0.0, self.clock() - tick)
                entry["status"] = "received"
                if not isinstance(detections, list):
                    raise ValueError("SAM3 adapter must return a detection list")
                # Invalid detections are discarded, but a successful response still counts.
                calls["sam3_successful"] += 1
                completed_keys.add(key)
                entry["status"] = "applying"
                self.apply_observation(action, detections, nodes, applied, entry["events"], (width, height))
                entry["status"] = "completed"
                entry["detection_count"] = len(detections)
                return ""
            except Exception as exc:
                if entry["status"] == "running":
                    model_seconds["sam3"] += max(0.0, self.clock() - tick)
                entry["status"] = "failed"
                entry["error"] = str(exc)
                errors.append(f"SAM3 {action_id}: {exc}")
                return "error"

        if self.config.vlm_first:
            stop = limit()
            if not stop:
                if self.vlm is None or not callable(getattr(self.vlm, "propose_initial", None)):
                    errors.append("VLM-first mode requires propose_initial")
                    stop = "error"
                else:
                    state = self._vlm_state(nodes, actions, calls, width, height, start)
                    calls["vlm_attempted"] += 1
                    tick = self.clock()
                    try:
                        raw = self.vlm.propose_initial(image, target, state)
                        model_seconds["vlm"] += max(0.0, self.clock() - tick)
                        calls["vlm_successful"] += 1
                        roi, force_tiles, initial_batch, rejected = parse_initial_plan(
                            raw, width, height, self.config.max_actions_per_vlm_call)
                    except Exception as exc:
                        if not calls["vlm_successful"]:
                            model_seconds["vlm"] += max(0.0, self.clock() - tick)
                        errors.append(f"initial VLM plan: {exc}")
                        stop = "error"
                    else:
                        proposal = {"call": 1, "raw": raw, "rejected": rejected, "accepted": [],
                                    "roi": roi, "tile_mode": "force" if force_tiles else "auto"}
                        proposals.append(proposal)
                        seen_initial: set[tuple[str, Region]] = set()
                        executable = []
                        for original_index, prompt, region in initial_batch:
                            key = (normalize_prompt(prompt), region)
                            if key in seen_initial:
                                proposal["rejected"].append({"index": original_index, "reason": "duplicate_batch"})
                                continue
                            seen_initial.add(key)
                            accepted = {"prompt": prompt, "region": region, "status": "pending"}
                            proposal["accepted"].append(accepted)
                            executable.append((accepted, prompt, region))
                        for j, (accepted, prompt, region) in enumerate(executable):
                            stop = execute(prompt, region, "vlm_initial")
                            accepted["status"] = "failed" if stop == "error" else "pending" if stop else "completed"
                            if stop:
                                unexecuted_batch = ([{"prompt": prompt, "region": region}] if stop != "error" else []) + [
                                    {"prompt": p, "region": r} for _, p, r in executable[j + 1:]]
                                break
        else:
            stop = execute(target, full_region, "bootstrap")
            if stop:
                unexecuted_bootstrap = ([] if stop == "error" else [full_region]) + list(self.config.bootstrap_regions)
        if not stop:
            stage1_boxes = [n.box for n in nodes.values() if
                            max((score for score, _ in n.positives.values()), default=0.0)
                            >= self.config.adaptive_density_seed_score]
            plan = plan_adaptive_tiles(stage1_boxes, width, height,
                                       density_threshold=self.config.adaptive_density_threshold,
                                       roi_override=roi if self.config.vlm_first else None,
                                       force=force_tiles and self.config.enable_adaptive_tiling)
            tiling = asdict(plan)
            tiling["enabled"] = self.config.enable_adaptive_tiling
            tiling["forced_by_vlm"] = force_tiles
            seed_exemplars = select_exemplars(nodes, self.config) if self.config.enable_exemplar_refinement else ()
            # A conditioned refinement is intentionally allowed to repeat the
            # full-image prompt. Its misses are not independent negative evidence.
            bootstrap_steps: list[tuple[str, Region, tuple[tuple[str, Region], ...]]] = []
            if seed_exemplars:
                bootstrap_steps.append(("exemplar_refinement", roi, seed_exemplars))
            if self.config.enable_adaptive_tiling and plan.trigger:
                for tile in plan.tiles:
                    bootstrap_steps.append(("adaptive_tile", tile, ()))
            bootstrap_steps.extend(("bootstrap", region, ()) for region in self.config.bootstrap_regions)
            tile_seed: tuple[tuple[str, Region], ...] | None = None
            for i, (source, region, exemplars) in enumerate(bootstrap_steps):
                if source == "adaptive_tile":
                    if tile_seed is None:
                        tile_seed = select_exemplars(nodes, self.config) if seed_exemplars else ()
                    exemplars = tuple((node_id, box) for node_id, box in tile_seed if
                                      box[0] >= region[0] and box[1] >= region[1] and
                                      box[2] <= region[2] and box[3] <= region[3])
                stop = execute(target, region, source, exemplars)
                if stop:
                    unexecuted_bootstrap = [step_region for _, step_region, _ in
                                            bootstrap_steps[i + (1 if stop == "error" else 0):]]
                    break
        while not stop:
            stop = limit()
            if stop:
                break
            if calls["vlm_attempted"] >= self.config.max_vlm_calls:
                stop = "qwen_budget"
                break
            if self.vlm is None:
                stop = "error"
                errors.append("VLM adapter required when max_vlm_calls > 0")
                break
            state = self._vlm_state(nodes, actions, calls, width, height, start)
            if self.config.vlm_first:
                state["roi"] = roi
            calls["vlm_attempted"] += 1
            tick = self.clock()
            try:
                raw = self.vlm.propose(image, target, state)
                model_seconds["vlm"] += self.clock() - tick
                calls["vlm_successful"] += 1
            except Exception as exc:
                model_seconds["vlm"] += self.clock() - tick
                errors.append(f"VLM request: {exc}")
                stop = "error"
                break
            batch, rejected = parse_batch(raw, width, height, self.config.max_actions_per_vlm_call,
                                          allowed_region=roi if self.config.vlm_first else None)
            proposal = {"call": calls["vlm_attempted"], "raw": raw, "rejected": rejected, "accepted": []}
            proposals.append(proposal)
            batch_keys: set[tuple[str, Region]] = set()
            executable: list[tuple[dict[str, Any], str, Region]] = []
            for original_index, prompt, region in batch:
                key = (normalize_prompt(prompt), region)
                if key in completed_keys or key in batch_keys:
                    proposal["rejected"].append({"index": original_index, "reason":
                                                 "duplicate_completed" if key in completed_keys else "duplicate_batch"})
                    continue
                batch_keys.add(key)
                accepted = {"prompt": prompt, "region": region, "status": "pending"}
                proposal["accepted"].append(accepted)
                executable.append((accepted, prompt, region))
            for j, (accepted, prompt, region) in enumerate(executable):
                stop = execute(prompt, region, "vlm")
                accepted["status"] = "failed" if stop == "error" else "pending" if stop else "completed"
                if stop:
                    unexecuted_batch = ([{"prompt": prompt, "region": region}] if stop != "error" else []) + [
                        {"prompt": p, "region": r} for _, p, r in executable[j + 1:]]
                    break
            if stop:
                break
            if not proposal["accepted"] and calls["vlm_attempted"] >= self.config.max_vlm_calls:
                stop = "no_valid_action"
                break
        countable = calls["sam3_successful"] > 0
        counts = {"soft": sum(n.belief for n in nodes.values()) if countable else None,
                  "hard": sum(n.belief >= self.config.hard_count_threshold for n in nodes.values()) if countable else None}
        return Result(nodes, actions, proposals, counts, calls, stop,
                      bool(errors or unexecuted_bootstrap or unexecuted_batch), errors,
                      unexecuted_bootstrap, unexecuted_batch, self.clock() - start, model_seconds, tiling,
                      roi if self.config.vlm_first else None)

    def _vlm_state(self, nodes: dict[str, Node], actions: list[dict[str, Any]],
                   calls: dict[str, int], width: int, height: int, start: float) -> dict[str, Any]:
        summary = [{"id": n.node_id, "box": n.box, "belief": n.belief} for n in nodes.values()]
        selected = sorted(nodes.values(), key=lambda n: (abs(n.belief - self.config.hard_count_threshold), n.node_id))
        history = [{"source": a.get("source"), "prompt": a.get("prompt"), "region": a.get("region"),
                    "status": a["status"], "detections": a.get("detection_count")}
                   for a in actions[-12:]]
        covered = [a["region"] for a in actions if a["status"] == "completed"]
        return {"width": width, "height": height, "nodes": summary, "history": history,
                "recent_successful_coverage": covered[-12:], "successful_search_count": len(covered),
                "candidate_boxes": [n.box for n in selected[:self.config.max_candidate_crops]],
                "remaining_vlm_calls": self.config.max_vlm_calls - calls["vlm_attempted"],
                "remaining_sam3_calls": self.config.max_sam3_calls - calls["sam3_attempted"],
                "remaining_seconds": None if self.config.max_runtime_seconds is None else
                max(0.0, self.config.max_runtime_seconds - (self.clock() - start)),
                "max_actions": self.config.max_actions_per_vlm_call}

    def apply_observation(self, action: Action, detections: list[Detection], nodes: dict[str, Node],
                          applied: set[str], events: list[dict[str, Any]], size: tuple[int, int]) -> None:
        if action.action_id in applied:
            return
        width, height = size
        valid: list[tuple[int, Detection, np.ndarray, Region]] = []
        for i, det in enumerate(detections):
            if (not isinstance(det, Detection) or isinstance(det.score, (bool, np.bool_))
                    or not isinstance(det.score, (int, float, np.integer, np.floating))):
                events.append({"type": "discarded_detection", "index": i})
                continue
            try:
                score = float(det.score)
            except (OverflowError, TypeError, ValueError):
                events.append({"type": "discarded_detection", "index": i})
                continue
            try:
                mask = np.asarray(det.mask)
            except (TypeError, ValueError):
                events.append({"type": "discarded_detection", "index": i})
                continue
            x1, y1, x2, y2 = action.region
            outside = mask.copy() if mask.shape == (height, width) and mask.dtype == np.bool_ else None
            if outside is not None:
                outside[y1:y2, x1:x2] = False
            if (mask.shape != (height, width) or mask.dtype != np.bool_ or not mask.any()
                    or (outside is not None and outside.any())
                    or not math.isfinite(score) or not 0 <= score <= 1
                    or score < self.config.sam3_score_threshold):
                events.append({"type": "discarded_detection", "index": i})
                continue
            valid.append((i, Detection(mask, score, det.detection_id), mask, mask_box(mask)))
        valid.sort(key=lambda row: (-row[1].score, -int(row[2].sum()), row[0]))
        previous_ids = set(nodes)
        retrieved: set[str] = set()
        ambiguous: set[str] = set()
        prompt_key = normalize_prompt(action.prompt)
        for i, det, mask, box in valid:
            matches: list[str] = []
            contained: list[str] = []
            for node in nodes.values():
                iou, iom, ratio = overlap(mask, node.mask)
                if iom >= self.config.match_iom_threshold and ratio < self.config.min_match_area_ratio:
                    contained.append(node.node_id)
                if iou >= self.config.match_iou_threshold or (iom >= self.config.match_iom_threshold and ratio >= self.config.min_match_area_ratio):
                    matches.append(node.node_id)
                elif (action.source == "adaptive_tile" and
                      box_iou(box, node.box) > self.config.tile_box_iou_threshold):
                    matches.append(node.node_id)
            if contained or len(matches) > 1:
                ambiguous.update(contained + matches)
                events.append({"type": "ambiguous_containment" if contained else "ambiguous_association",
                               "index": i, "nodes": sorted(set(contained + matches))})
                continue
            det_id = det.detection_id or f"{action.action_id}_d{i:03d}"
            if matches:
                node = nodes[matches[0]]
                retrieved.add(node.node_id)
                if prompt_key in node.negatives:
                    before = node.belief
                    del node.negatives[prompt_key]
                    node.belief = belief(node, self.config)
                    events.append({"type": "negative_cleared", "node": node.node_id,
                                   "prompt": prompt_key, "before": before, "after": node.belief})
                node.latest_at = action.action_id
                node.detection_ids.append(det_id)
                old = node.positives.get(prompt_key)
                if old is None or det.score > old[0]:
                    before = node.belief
                    node.positives[prompt_key] = (det.score, det_id)
                    node.belief = belief(node, self.config)
                    events.append({"type": "positive", "node": node.node_id,
                                   "before": before, "after": node.belief, "detection": det_id})
                if (int(mask.sum()), det.score, -i) > (int(node.mask.sum()), node.canonical_score, 0):
                    node.mask, node.box = mask.copy(), box
                    node.canonical_detection_id, node.canonical_score = det_id, det.score
                events.append({"type": "associated", "node": node.node_id, "detection": det_id})
            else:
                node_id = f"n{len(nodes) + 1:04d}"
                node = Node(node_id, mask.copy(), box, det_id, det.score,
                            action.action_id, action.action_id, [det_id], {prompt_key: (det.score, det_id)})
                node.belief = belief(node, self.config)
                nodes[node_id] = node
                events.append({"type": "new_node", "node": node_id, "detection": det_id,
                               "before": None, "after": node.belief})
        x1, y1, x2, y2 = action.region
        for node_id in sorted(previous_ids):
            if action.exemplar_boxes:
                continue
            node = nodes[node_id]
            coverage = float(node.mask[y1:y2, x1:x2].sum()) / float(node.mask.sum())
            if (coverage >= self.config.negative_coverage and node_id not in retrieved
                    and node_id not in ambiguous and prompt_key not in node.negatives):
                before = node.belief
                node.negatives[prompt_key] = action.action_id
                node.belief = belief(node, self.config)
                events.append({"type": "non_retrieval", "node": node_id, "prompt": prompt_key,
                               "coverage": coverage, "before": before, "after": node.belief})
        applied.add(action.action_id)


def _valid_region(region: Any, width: int, height: int) -> bool:
    return (type(region) in (tuple, list) and len(region) == 4
            and all(type(v) is int for v in region)
            and 0 <= region[0] < region[2] <= width
            and 0 <= region[1] < region[3] <= height)
