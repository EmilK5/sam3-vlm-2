# Final M8 comparison and 34-image evaluation

This comparison is complete. The selected configuration and one-variant run command
are in [M8_FINAL_SELECTION.md](M8_FINAL_SELECTION.md).

Run one final two-arm comparison, then evaluate both arms on all 34 images.
No further threshold search or selection on the three diagnostic images is needed.
Both variants are D: two Qwen calls, at most one replan, positive threshold 0.20
without target exemplars, negatives at 0.50, soft counts and IoU+IoM association.
The bootstrap is unchanged.

| Variant | Confounder not retrieved | Matched confounder |
|---|---|---|
| D_CurrentNegativeEvidence | Existing penalty on the confounder class; target probability can rise | Existing negative evidence |
| D_NeutralNegativeMisses | Unit likelihood; probabilities unchanged apart from rounding | Same existing negative evidence |

`belief.neutral_confounder_misses` is the only configuration difference between
these arms. Observations remain logged in both, with the same existing observation
and correlation bookkeeping. Target non-retrieval is unchanged. Negative matches
can still reduce target probability; negatives do not create new countable nodes.
The general production default remains the current policy pending evaluation.

Both arms now receive the tree-only scope even with `prompt_version: old`.
The shared prompt reinforces color/maturity preservation and adjective-before-noun
word order. Correction asks Qwen to repair the rejected phrase instead of replacing
it with another invalid form. These instructions preserve full images, evidence,
65536-token Ollama context and open vocabulary; they do not guarantee that Qwen
will obey. No word dictionary, semantic fallback, extra model call or altered
output-token budget is introduced. Earlier reports used different instructions;
compare the two fresh arms for the final result.

## 1. Update the GPU code

Sync these V4 changes to the GPU repository and activate the same working
Python environment and Ollama service used for the preceding run. Run from the
repository directory containing `src`, `configs` and `pilot_manifest.json`:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q
```

The final suite is `final-ablation`; do not reuse `all` or `discovery-ablation`.
Keep the completed 34-image manifest, corrected targets and tree-only ground truth.
Use fresh output directories for repeats.

## 2. Last diagnostic check: 6 runs

```bash
python -m sam3_vlm.experiments.m8_smoke \
  --stage pilot --require-cuda \
  --pilot-suite final-ablation \
  --manifest pilot_manifest.json \
  --sample-ids image_4 image_17 image_27 --max-samples 3 \
  --output_dir runs/m8_final_probe
```

This exercises the actual GPU/model path, artifact validator, replay, both miss
policies and compact export. If it exits successfully, proceed immediately to the
full evaluation; no additional chat review is needed. An execution/validation
failure should be resolved before continuing. The diagnostic sample is not used
to select a winner or retune the settings.

## 3. Full evaluation: 34 images × 2 variants = 68 runs

```bash
python -m sam3_vlm.experiments.m8_smoke \
  --stage pilot --require-cuda \
  --pilot-suite final-ablation \
  --manifest pilot_manifest.json --max-samples 34 \
  --output_dir runs/m8_final_34
```

The earlier D runs averaged approximately 10–12 seconds per image/variant. At a
similar speed, budget roughly 12–15 minutes for 68 runs plus model startup; this
is an estimate, and hardware/model loading can change it.

## 4. Send two small files

Give the bundles distinct names before downloading:

```bash
cp runs/m8_final_probe/compact_review.zip runs/compact_review_final_probe.zip
cp runs/m8_final_34/compact_review.zip runs/compact_review_final_34.zip
```

Send `runs/compact_review_final_probe.zip` and `runs/compact_review_final_34.zip`.
Full scene graphs,
masks and per-image run folders stay on the GPU machine.
Each ZIP includes paired errors/cost, failures, Qwen correction diagnostics, prompt
examples, all-image confidence totals, and detailed traces for up to three images.
It also includes `mentor_summary.md`, which is separately written beside the ZIP:

```bash
cat runs/m8_final_34/mentor_summary.md
```

Use that summary's MAE, RMSE, signed error, paired wins/losses and runtime for the
presentation. It identifies incomplete comparisons instead of silently reporting
a winner from failed runs. Label this a 34-image development-set evaluation:
it includes the three diagnostic images and has already informed model settings.
It is not a held-out test. Both arms use an adaptive Qwen planner, so this estimates
the end-to-end policy difference rather than a replay of identical sensor actions.
