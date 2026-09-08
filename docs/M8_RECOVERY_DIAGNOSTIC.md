# M8: rejection correction and a small SAM3 discovery diagnostic

The current reference is D with the old prompt, negatives and soft counts.
Production uses that prompt/count policy with rejection correction enabled.
The comparison below includes an unchanged correction-disabled reference arm.
Old and V3 prompts remain versioned; V4 is the revised evidence-grounding prompt.

## Setup

Sync the updated V4 code to the GPU machine and activate the environment used
for the preceding pilot. Run from the V4 repository:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

Use fresh output directories for repeated trials, since reports and review ZIPs
at the same location are overwritten. Keep the existing completed manifest.

## 1. Check Qwen correction on three diagnostic images

The cases are image 4 (invalid first query), image 17 (large residual undercount),
and image 27 (prompt-dependent early termination). They are selected diagnostic
cases, not a representative accuracy sample.

```bash
python -m sam3_vlm.experiments.m8_smoke \
  --stage pilot --require-cuda \
  --pilot-suite recovery-ablation \
  --manifest pilot_manifest.json \
  --sample-ids image_4 image_17 image_27 --max-samples 3 \
  --output_dir runs/m8_recovery_probe
```

This runs nine image/variant combinations:

| Variant | Prompt | Rejection correction | Negatives | Qwen cap |
|---|---|---|---|---:|
| D_ReferenceOld | old | off | on | 2 |
| D_OldWithCorrection | old | on | on | 2 |
| D_V4WithCorrection | v4 | on | on | 2 |

Correction occurs only when no actions were accepted and there are explicit
rejections or an unsaturated empty-plan diagnostic. One correction is allowed,
within all existing hard budgets. A correction can use D's second Qwen call,
leaving no call for a later scene replan. It cannot trigger another JSON repair;
an initial JSON repair also prevents an extra rejection correction of that plan.
Partial rejections go into the next normal replan. Target identity, full evidence,
original image and contact-sheet inputs are retained. No dictionary or invented
fallback prompt is added.

V4 asks Qwen to separate direct observations, recorded sensor outcomes and
hypotheses. It explicitly rejects the unsupported assumption that earlier general
target queries searched only bright fruit, and asks for a final noun-last,
one-to-three-word, target-preserving, nonduplicate phrase check. Those are model
instructions; grammatical correctness is not guaranteed by a dictionary validator.

Send `runs/m8_recovery_probe/compact_review.zip`. The summary now includes
correction call counts and corrections that yielded accepted actions, in addition
to paired accuracy/cost changes, stop/rejection reasons and selected examples.

## 2. Isolate SAM3 discovery behavior on the same images

This runs no Qwen calls. The three phrases below are controlled probe inputs from
the preceding study; they are not a restriction on Qwen's vocabulary.

```bash
python -m sam3_vlm.experiments.discovery_diagnostic \
  --require-cuda \
  --manifest pilot_manifest.json \
  --sample-ids image_4 image_17 image_27 \
  --prompts "green fruit" "dark green fruit" "shadowed green fruit" \
  --max-sam3-calls 100 \
  --output_dir runs/m8_discovery_probe
```

Each image is bootstrapped once. Then each phrase is tested at thresholds 0.25
and 0.5, with and without the same fixed bootstrap exemplar boxes. The 12 probe
arms per image start from the same graph; they do not add candidates to one
another's baseline. No-exemplar cases are skipped explicitly if bootstrap found
no suitable seeds. The three-image command makes up to 36 probes plus bootstrap,
typically up to 48 total SAM3 calls, with a hard overall ceiling of 100.

Each detection set is then associated offline using both IoU-only and IoU+IoM
policies. This comparison takes no extra model calls and measures the combined
association/deduplication-policy effect. The report separates sensor detections,
matched detections and newly admitted candidates, with boxes for follow-up review.
The locked search region, tiling and bootstrap stay fixed across probe arms.
The production Qwen-generated-query threshold remains 0.5.

Send `runs/m8_discovery_probe/discovery_review.zip`. It contains the numeric
report, detection/new-candidate boxes and three reduced image previews. Full
bootstrap image assets remain local. Extra candidates are not necessarily fruit;
count-only labels cannot establish detection precision or recall.

Both commands support `--dry-run` for input validation without model calls.
Review the small probe results before repeating the 34-image study. If the
correction experiment is ready for a full evaluation, rerun the recovery pilot
without `--sample-ids`, with `--max-samples 34` and a fresh output directory.
