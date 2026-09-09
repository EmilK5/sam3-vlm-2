# M8: low-threshold positive discovery

This comparison is complete. Continue with [M8_FINAL_EVALUATION.md](M8_FINAL_EVALUATION.md).
New requests include shared scope/contract fixes; they are not exact historical prompt reruns.

Production now admits Qwen discovery detections at 0.20 and does not attach target
exemplar boxes to those queries. Negative queries remain at 0.5. Bootstrap keeps
its original threshold and exemplar refinement. Counting remains the pure sum of
target probabilities; lowering the sensor threshold does not automatically count
a new candidate as one fruit. Later positive or confounder evidence can raise or
lower target probability. The evidence model remains uncalibrated.

A rejected/missing target can now trigger one correction even when valid negatives
were accepted. Those negatives stay queued, retain their slot meanings and execute
after the corrected target. The correction consumes an ordinary Qwen call. D still
has only two calls total. A valid target plus an invalid negative does not trigger
immediate correction. Full Qwen evidence, images and open vocabulary are retained.

## Controlled comparison

`--pilot-suite discovery-ablation` runs:

| Variant | Positive threshold | Target exemplars in Qwen discovery |
|---|---:|---|
| D_Discovery_050_Exemplars | 0.50 | on |
| D_Discovery_025_NoExemplars | 0.25 | off |
| D_Discovery_020_NoExemplars | 0.20 | off |
| D_Discovery_015_NoExemplars | 0.15 | off |

All four use the old prompt, negatives at 0.5, soft counts, IoU+IoM, two Qwen calls,
one replan and the updated missing-target correction. The 0.5 reference changes
both threshold and exemplar policy relative to the lower arms. Comparisons among
0.25/0.20/0.15 isolate threshold. The prior SAM3-only diagnostic isolates exemplar
behavior. This reference is not an exact rerun of the previous correction policy.

## 1. Setup on the GPU machine

Sync this V4 code and activate the same environment used for the preceding pilot.
Run the following from the V4 repository directory containing `src`, `configs`
and your completed `pilot_manifest.json`:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q
```

Keep the existing Ollama service and alias `qwen3.5-9b-sam3`. Its 65536-token context
and current output budget are unchanged. Use the same model weights, manifest and
runtime setup as before. Ground truth counts only fruit on trees.

## 2. Diagnostic subset: 12 image/variant runs

```bash
python -m sam3_vlm.experiments.m8_smoke \
  --stage pilot --require-cuda \
  --pilot-suite discovery-ablation \
  --manifest pilot_manifest.json \
  --sample-ids image_4 image_17 image_27 --max-samples 3 \
  --output_dir runs/m8_discovery_low_probe
```

Send only `runs/m8_discovery_low_probe/compact_review.zip` for the first review.
These are selected diagnostic images, not a representative accuracy evaluation.

## 3. Full 34-image comparison: 136 image/variant runs

Once the subset has completed successfully, the full comparison uses:

```bash
python -m sam3_vlm.experiments.m8_smoke \
  --stage pilot --require-cuda \
  --pilot-suite discovery-ablation \
  --manifest pilot_manifest.json --max-samples 34 \
  --output_dir runs/m8_discovery_low_34
```

Send only `runs/m8_discovery_low_34/compact_review.zip`. Do not upload the individual
34-image run directories. Keep them locally in case a specific case needs inspection.
Use a fresh output directory if repeating either command.

Each ZIP includes:

- `review_summary.json`: accuracy/cost comparisons, failures, stopping reasons and
  correction calls, including corrections that accepted a target.
- `counts.json`: small count/error rows for all images and variants.
- `confidence_totals.json`: bootstrap and final target mass, candidate totals,
  target mass from newly created candidates, and later gains/losses on existing
  candidates for every image. Creation mass is measured when the node is born;
  subsequent changes to that node are included in existing-node changes.
- `selected_cases.json`: first/last Qwen outputs and per-action probability traces
  for at most three images. Traces identify NOT_RETRIEVED losses and matched gains,
  record the actual thresholds/exemplar counts and show the largest changes.
- `prompt_examples.json`, `prompt_yields.json`, and at most three image previews.

For each trace step:
`new soft count = previous soft count + new-node target mass + existing-node mass change - removed-node mass`.
The bootstrap entry combines bootstrap actions; subsequent entries are per SAM3
call. Diagnostics never alter probability updates. New nodes are candidate
hypotheses, not confirmed fruit. Compare MAE and signed error alongside candidate
growth to judge whether extra retrieval improves counts.

The historical `recovery-ablation`, `prompt-ablation`, `negative-ablation` and
`all` suites retain the measured 0.5/exemplar settings. Use `discovery-ablation`
for this comparison; the `standard` A–E suite uses the new configured defaults.
