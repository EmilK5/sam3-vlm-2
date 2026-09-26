# Supported SAM3 + VLM counting MVP

This document is authoritative for the supported `sam3_vlm.mvp` workflow. The older M8 modules, configurations, experiments, and related documents remain historical research code. The previous specification is preserved as [HISTORICAL_M8_DESIGN_SPEC.md](HISTORICAL_M8_DESIGN_SPEC.md). This MVP does not run them.

## Scope and interface

One run takes one RGB image, a fixed target description, frozen `Config`, a SAM3 adapter, and optionally a VLM adapter. The controller owns a node dictionary and all state changes. Only SAM3 masks create nodes. A VLM proposes positive target searches; it neither detects nor counts. All persistent masks and integer, exclusive-end regions use original-image coordinates. The full-image target search runs first; fixed bootstrap regions run in configured order. Each prompt-region search is one attempted SAM3 request. No confounder actions, exemplars, automatic tiling, merging, action bank, saturation rule, or hidden retries are used.

The supported entry point is `python -m sam3_vlm.mvp.cli IMAGE TARGET --config configs/mvp.json --output result.json --masks masks.npz`. `--masks` is optional. The CLI initializes models before entering `Controller.run`; reported runtime measures the run, and adapter timings measure model requests. Qwen uses `QWEN_BASE_URL`, `QWEN_MODEL`, and optional `QWEN_API_KEY`. The OpenAI-compatible SDK has retries disabled. The Python API accepts compatible fake adapters for deterministic tests.

## VLM action contract

One VLM response is JSON with exactly one top-level `actions` list. Each action has exactly `prompt` (nonempty string) and `region` (`null` or four integer image coordinates). For example:

```json
{"actions":[{"prompt":"small green fruit","region":null},{"prompt":"green fruit","region":[0,0,512,512]}]}
```

`max_actions_per_vlm_call` sets the per-response maximum. Setting it to 1 gives the single-action ablation; setting it above 1 enables batches. Run both with the same images, model settings, SAM3 budget, VLM budget, prompts, and counting rule, then compare errors, calls, and elapsed time. Generated actions can differ, so repeat runs if model sampling is nondeterministic. No full dataset run is implied.

Malformed JSON or a malformed top-level object rejects the response. In a well-formed list, invalid entries are rejected individually; excess entries are rejected as `batch_truncated`. Exact duplicate means normalized prompt (trim, lowercase, collapse whitespace) plus exact region. Completed searches and earlier accepted entries in the same batch block that duplicate; the same prompt in another region is valid. Failed searches can be retried. Every proposal and rejection is traced. Accepted entries are recorded as pending before execution, then execute sequentially. Each observation is applied before the next action. A final permitted VLM response still executes all accepted actions allowed by remaining SAM3 and runtime budgets. Unexecuted accepted actions remain pending and are listed in `unexecuted_batch`.

The VLM receives the original image, target, nodes with boxes and beliefs, the last 12 search summaries, the last 12 successfully searched regions and total successful search count, remaining budgets, and at most `max_candidate_crops` node crops closest to the hard threshold (node ID breaks ties). Full coverage remains in controller history. Ground truth is never supplied.

## Detection, association, and belief

The SAM3 adapter crops one region, makes one model request, and places binary masks back on the original image. The controller discards empty or malformed masks, nonfinite/out-of-range scores, and scores below threshold. It processes remaining detections by descending score, descending mask area, then source order. A detection matches a node if mask IoU is at least `match_iou_threshold`, or mask IoM is at least `match_iom_threshold` with smaller/larger area ratio at least `min_match_area_ratio`. Extreme containment is discarded as ambiguous. A detection matching multiple nodes is also discarded. Accepted detections attach to one node or create one. The largest accepted mask is canonical; ties use higher score and then earlier detection. Nodes are never automatically merged or deleted.

Each node stores the maximum positive SAM3 score per normalized prompt. Sort these scores high to low and compute `E+ = sum(discount**j * score[j])`. For each normalized prompt there is at most one active non-retrieval penalty. Let `E- = sum(discount**j for j in range(number_of_active_penalties))`. Belief is `sigmoid(logit(prior) + log(evidence_base) * (E+ - negative_strength * E-))`. It is a heuristic support score, not a calibrated probability. Default prior is 0.25, base 9, discount 0.5, negative strength 0.5. Both bootstrap-only and adaptive runs use soft count `sum(beliefs)` and hard count `number of beliefs >= hard_count_threshold`.

A successful target search penalizes only nodes that existed before that observation, whose canonical mask is at least `negative_coverage` (default 0.9) inside the searched region, and that were neither retrieved nor touched by an ambiguous detection. A missed prompt adds one penalty at most, regardless of the number of eligible successful regions. Repeating the prompt and region after completion is rejected; repeating the prompt elsewhere may search, but does not stack another penalty. A later accepted retrieval of that node under the same prompt clears its penalty and may replace its strongest positive score. Different prompt penalties are geometrically discounted. Empty successful searches can therefore lower eligible existing beliefs. Unrelated regions, failed calls, replayed observations, discarded ambiguous matches, and newly created nodes in that observation do not receive a penalty. The action trace gives the source action and belief before/after for each update. Reapplying an action ID is a no-op.

Positive evidence can raise belief; non-retrieval can lower it. Counts can move in either direction and still omit undiscovered objects. No calibration or independent-evidence claim is made.

## Budgets, failures, and results

Attempted and successful SAM3/VLM calls are separate counters. A returned VLM response counts as a successful model request even if its JSON or proposals are invalid; parsing and action validation are separate trace outcomes. Budgets and deadline are checked before every model request and each batch item. One in-flight request may overrun the deadline. Boundary stop precedence is time, then SAM3, then VLM. Invalid final responses with no accepted action end as `no_valid_action`; exhausted VLM budget after executing its accepted batch ends as `qwen_budget`. A failed request ends as `error` without applying its partial output. Prior successful observations remain. Interrupted bootstrap or batch actions are listed, and the result is partial. Counts are null until one SAM3 search succeeds; a successful empty search gives zero counts. Runtime covers controller work from run entry, and model timings report request time separately. Models constructed before entry are outside that measured interval.

Evaluation is separate: `sam3_vlm.mvp.evaluate_count(result, ground_truth_count)` computes post-run errors without mutating the result. Keep the same evaluation protocol across action-count ablations. The JSON output includes boxes, beliefs, evidence, action trace, counters, stop reason, runtime, and metadata. Optional NPZ contains canonical masks.

## Limits

Mask matching can reject merged or severely occluded objects, and similar prompts remain correlated despite grouping and discounting. Search coverage records successful requests, not exhaustive visual coverage. Real-model quality, latency, and belief parameters require empirical study; local fake-adapter tests do not establish them.
