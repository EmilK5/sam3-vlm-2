# Final configuration selection

For the subsequently requested A–E final experiment, use [M8_FINAL_AE_RUN.md](M8_FINAL_AE_RUN.md).
The selection evidence below supplies the C/D/E sensing and belief policy.

Select `D_CurrentNegativeEvidence`, exactly as evaluated in the 34-image
`final-ablation` study. Retain the current negative non-retrieval penalty:
`belief.neutral_confounder_misses: false`. The proposed neutral-on-miss change
was a useful hypothesis, but it did not improve this development evaluation.
No additional threshold, prompt, posterior multiplier or call-budget tuning is
introduced for the final run.

## Evidence from the supplied reports

Both arms completed 34/34 runs, with no export warnings. The reported MAEs were
recomputed from the 68 per-image rows; sample IDs and ground truths are paired.
The scope fix is present in the recorded prompt examples. The three-image probe
is treated separately because it overlaps the full evaluation.

| Metric | Selected: current negative evidence | Neutral misses |
|---|---:|---:|
| MAE | 6.4555 | 7.6798 |
| RMSE | 7.8421 | 9.1033 |
| Mean relative error | 25.26% | 29.00% |
| Mean signed error | -4.3855 | -6.5381 |
| Median absolute error | 5.9915 | 7.7279 |
| Maximum absolute error | 15.7358 | 21.5523 |
| Mean runtime per image | 9.881 s | 9.656 s |

The selected policy wins on 23 images, loses on 10 and ties on one. Its MAE is
1.2243 fruit lower (15.94% lower relative to the neutral policy). Excluding
image_4, image_17 and image_27, the remaining 31 still favor the selected policy:
20 wins, 10 losses, one tie, mean absolute-error advantage 1.0372 fruit.
The separate three-image probe also favors current evidence (MAE 10.7975 vs
13.8263); it is not added to the 34-image sample size.

Bootstrap target mass is identical between full-evaluation arms (666.5569 in
aggregate). The selected policy finishes with 811.8920 target mass and 144 new
candidates; neutral finishes with 738.7042 and 84 new candidates. Both undercount
relative to 961 total ground-truth fruit. However, their Qwen proposals and action
histories differ: this is evidence for the complete adaptive configuration, not
proof that boosting confidence after every negative miss is well calibrated.
Selected examples include a failed duplicate correction in the neutral arm that
prevented positive discovery on image_17, whereas the current arm ran two useful
positive queries. Do not attribute that whole difference directly to the equation.

There were 9 `NO_VALID_ACTIONS` stops in the selected arm and 14 corrections, of
which 6 accepted a target. These are model/controller outcomes, not execution
crashes. Duplicate proposals remain a limitation. The final run preserves this
measured behavior rather than adding another unvalidated prompt or budget change.

## Frozen settings

The production config `configs/m8_real_smoke.json` and selected final suite match
the winning report's resolved V4 settings (excluding device/output/asset paths).
Keep the existing model weights and local Ollama alias used for the evaluation.

| Setting | Final value |
|---|---|
| Pilot variant | D_CurrentNegativeEvidence |
| Qwen positive SAM3 threshold | 0.20 |
| Positive exemplar boxes on Qwen discovery | off |
| Negative queries | on, threshold 0.50 |
| Neutral confounder misses | false |
| Count | sum of target probabilities; hard/commit thresholds null |
| Bootstrap | threshold 0.25; original exemplar refinement and tiled bootstrap |
| Context / target scope | tree canopy at 0.40; only fruit on trees |
| Prompt | old core plus the already-tested scope/qualifier/correction instructions |
| Qwen | max 2 calls; temperature 0.20; output 512 tokens; timeout 45 s |
| Ollama context | existing 65536-token profile |
| Replanning | max 1; missing-target correction enabled within the same 2-call cap |
| SAM3 budget | max 15 calls, 64 tiles; cleanup off |
| Other caps | 300 seconds/image; max 20 controller iterations |
| Association | existing IoU+IoM; IoU match 0.50, IoM match 0.90 |
| Tiling | 2×2, overlap 0.15, minimum size 256 |
| Belief | prior pseudocount 1.0, repeat discount 0.8, two confounder slots |
| Evaluation seed | existing 42 |

The 0.20 threshold is retained from the tested final configuration. This report
compares the two miss policies at that threshold; it does not establish a global
optimum over all possible thresholds or budgets.

## One final run: 34 images, one variant

Sync the updated V4 code to the GPU repository and use the same environment,
models and completed manifest as the previous run. From that repository directory:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python -m sam3_vlm.experiments.m8_smoke \
  --stage pilot --require-cuda \
  --pilot-suite final-ablation \
  --pilot-variant D_CurrentNegativeEvidence \
  --manifest pilot_manifest.json --max-samples 34 \
  --output_dir runs/m8_final_selected
```

This executes only 34 runs. At the preceding mean speed, sensing/planning time is
roughly 5.6 minutes total, plus startup and hardware variation. The new variant
filter changes orchestration only; it does not change the selected model policy.
Use a fresh output directory if repeating the command.

Send `runs/m8_final_selected/compact_review.zip`. Its `mentor_summary.md` is also
written beside the ZIP. A single-variant summary does not fabricate a paired
comparison. Label the result a 34-image development-set repeat, not a held-out
accuracy estimate. Qwen can generate different proposals across repetitions;
report the repeat as measured, rather than choosing whichever rerun looks best.

## Source fingerprints

- `compact_review_34f.zip`: SHA-256 `ddaab0eb4a08237385b602c2eac1a33d93086d70296a3b6918c1cb24fee9a3b3`
- `compact_review_fp.zip`: SHA-256 `1e5ecab520a66127b9a86a03d2099400d6b264a53893905a2b310db82dbfc153`
