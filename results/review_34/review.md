# Review of the 34-image A–E pilot

Inputs: `review_bundle_34.zip` and `pilot_manifest.json`, supplied 7 September 2026.
All 170 run manifests use `green fruit`; all 34 manifest targets agree. The report's
metadata-level `green citrus` is the CLI default, not the per-image prompt used.
Ground-truth counts refer to fruit on trees, as specified by the user. Counts alone
do not provide object-level annotations for measuring detection precision/recall.

## Results and interpretation

| Variant | Rows used in original report | MAE | RMSE | Average Qwen calls | Average time (s) |
|---|---:|---:|---:|---:|---:|
| A | 34 | 13.88 | 18.44 | 0.00 | 0.38 |
| B | 34 | 7.79 | 9.79 | 0.00 | 1.46 |
| C | 34 | 7.54 | 9.16 | 1.00 | 5.30 |
| D | 33 | 7.95 | 9.28 | 1.67 | 7.94 |
| E | 34 | 7.61 | 9.13 | 5.82 | 24.76 |

B improves MAE by 43.86% over A. E takes 4.67 times C's runtime for very similar
aggregate error. E improves absolute error on 19 images versus C, worsens 12,
and ties three. On the common 33 successful images, C/D/E MAE is 7.69/7.95/7.76.
A/B use hard candidate counts; C/D/E use posterior counts with a 0.8 commitment
threshold. Comparisons from B to C therefore change both sensing and counting.

## Why E stops

All 34 E runs end at `NO_VALID_ACTIONS`, with 1–13 Qwen calls per image.
The terminal Qwen artifact identifies the cause in every case:

- 22 exact repeated prompts, despite the tried-prompt blacklist.
- 12 prompts longer than the allowed three words.
- No empty responses, no repair attempts, and no fallback actions in E.

Examples of rejected phrases: `small round green fruit`, `green fruit in shadow`,
and `green fruit under leaves`. These are correctly rejected by the requested
word limit. Eight runs stop at the first Qwen call without executing its action.
No run reaches the 100-call cap or combined numerical saturation.

All 198 E proposals suggest threshold 0.5, and the persisted configuration sets
the controller-owned Qwen threshold to 0.5. A further threshold increase is not
responsible for these rejections. Vocabulary remains open; no dictionary rejects
unfamiliar words.

## Discovery and counting

E makes 198 Qwen calls and executes 164 Qwen-generated SAM3 actions. Comparing
bootstrap graph snapshots, NODE_CREATED events, and final graphs shows **one new
node in total**, on image_11. The other 33 images gain no registered nodes after
bootstrap. This is a candidate count, not confirmation that the new node is a
true fruit. Association can also merge detections into existing nodes.

E's mean bootstrap posterior count is 20.3471, rising to 21.4042 after exploration;
the ground-truth mean is 28.2647. Within E, the posterior count increases on 22
images, decreases on four, and is unchanged on eight. The bootstrap posterior
MAE is 8.2545; final E MAE is 7.6148. Thus E provides some confidence refinement,
but almost no registered-node discovery.

| Image | Ground truth | Final candidate nodes | Final posterior count |
|---|---:|---:|---:|
| image_17 | 34 | 27 | 14.60 |
| image_25 | 49 | 52 | 29.91 |
| image_26 | 18 | 13 | 9.35 |
| image_27 | 40 | 51 | 29.52 |

For image_17/26, there are fewer nodes than the ground-truth objects, even before
posterior weighting. For image_25/27, the number of candidates exceeds ground
truth but posterior counts remain low. This does not prove that the candidate
set contains every real fruit: false positives and missed fruit can coexist.

162 of 164 executed E actions use positive pseudoexemplar boxes (151 use five).
An ablation should test whether these boxes favor rediscovering existing objects.
This pilot cannot establish that causal explanation. The fixed 0.5 threshold
on Qwen passes versus the 0.25 bootstrap threshold and IoM association are other
possible contributors; do not change all three together.

23 E runs report discovery plateau, but all 34 retain count variance above 0.5
(range 1.0285–15.4269). The current stopping test requires both discovery plateau
and low variance/aggregate entropy. More valid descriptions alone do not guarantee
that this combined criterion can be reached. Correlation discounting and target
non-retrieval penalties also affect posterior counts. No separate confounder
SAM3 experiments occur.

Some accepted phrases deserve semantic review, including `hairy green fruit`,
`hollowed green fruit`, and `ripe yellow fruit`. Passing lexical validation does
not establish that a phrase is visible in the image or preserves the requested
target. The remedy should preserve open vocabulary, not introduce a dictionary.

## D image_24: reproduced logging defect and implemented fix

Run: `pilot_D_Qwen_TwoRound_image_24_40e09c`.
The rejected first prompt is `green fruit hanging low` (four words). This is
independent of its replay failure.

The saved graph records node_000011 duplicate_risk = 0.13903794459765445, while
replay yields 0.0. Association updated diagnostics on an unmatched bootstrap node,
but bootstrap only emitted events for matched or newly created nodes. The saved
graph included the change; the event log did not. Probabilities and the reported
count are identical in the saved graph and replay.

The fix records changed diagnostics for previously existing, unmatched bootstrap
nodes with the responsible SAM3 action/call provenance. It creates no sensor
observation and makes no change to beliefs, association, thresholds, or experiment
stopping. Regression coverage checks both IoU and IoU/IoM association, exact replay,
and equality of logged and unlogged execution.

This fixes future logs. It does not retroactively repair or mark the supplied D
run as valid. The original report and archive remain untouched.

Local validation of all 102 C/D/E archives reproduced D's failure and found three
additional strict-equality differences in entropy only: C/image_23, E/image_6,
and E/image_24. Differences are approximately 5.6e-17 to 2.2e-16, consistent with
cross-platform floating-point recomputation; probabilities are identical. Those
three passed on the originating cluster. They are distinct from the missing
D diagnostic event. No validator comparison was weakened in this change.

## Recommended next phase

1. Verify the logging fix locally, then rerun image_24 on the originating cluster
   to check newly generated artifacts. Old archives do not acquire missing events.
2. Give E explicit feedback on a rejected proposal (the proposed phrase, rejection
   reason, and complete tried list), with a small documented correction allowance
   charged against the existing 100-call budget. Preserve the 1–3-word contract,
   target-only actions, full evidence, 0.5 threshold, and open vocabulary. This is
   a proposed policy change, not implemented as part of this logging fix.
3. Compare Qwen discovery with versus without positive exemplar boxes, keeping
   bootstrap, thresholds, and association identical. Record new nodes per action,
   bootstrap/final posterior counts, and rejection reasons. Add a zero-Qwen
   posterior-count control so that discovery and counting changes are separable.
4. Use object-level review on images 17, 25, 26, and 27 before tuning posterior
   penalties, count commitment, or deduplication. Total counts alone cannot
   identify which low-confidence nodes are true fruit.

See `e_diagnostics.json` for all 34 E images, including terminal rejections,
bootstrap/final counts, new-node counts, and remaining count variance.
