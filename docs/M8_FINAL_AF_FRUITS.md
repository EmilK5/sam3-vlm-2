# Fruit experiment A–F

The existing final fruit experiment has five arms, A–E. `final-af` appends
`F_VLMFirst_AdaptiveROI` while keeping the original `final-ae` suite unchanged.
All six arms use the same `pilot_manifest.json` image paths, fruit-on-tree counts,
and model weights. F uses the FSCD-inspired [VLM-first MVP](V4_DESIGN_SPEC.md):
Qwen sees the image and tree-only scope before any SAM3 search, selects an ROI
and positive target prompt, then SAM3 may refine strong detections with
pseudoexemplars and search overlapping ROI-local tiles. There is no bootstrap
search in F. Its frozen settings come from `configs/fscd147.json`.
With the standard local Ollama endpoint (`:11434/v1`), F sends its VLM requests
through Ollama's native JSON chat API with thinking disabled, so the ROI plan
arrives in the answer field. Other OpenAI-compatible endpoints retain the
existing chat-completions transport.
The initial-plan parser accepts a JSON code fence and an action without a
`region` field, which uses the VLM-selected ROI. Invalid replies are saved in
the per-run Qwen artifacts for diagnosis.

F reports the number of nodes above its hard-belief threshold. A/B report hard
candidate counts; C/D/E report soft probability sums. The table labels each
count type, so interpret differences with this distinction in mind. F's plain
red box image shows exactly the nodes that contribute to its hard count. The
original M8 graph replay validator does not apply to F's different controller;
F saves `mvp_result.json`, `summary.json`, and Qwen proposal artifacts for review.

## Run on the cluster

Use the same GPU allocation, SAM3 access, Qwen endpoint, and fruit manifest as
the [A–E run](M8_FINAL_AE_RUN.md). From the repository root:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python -m sam3_vlm.experiments.m8_smoke \
  --stage pilot --require-cuda --pilot-suite final-af \
  --manifest pilot_manifest.json --max-samples 34 \
  --output_dir runs/m8_final_AF --dry-run

python -m sam3_vlm.experiments.m8_smoke \
  --stage pilot --require-cuda --pilot-suite final-af \
  --manifest pilot_manifest.json --max-samples 34 \
  --output_dir runs/m8_final_AF
```

The full run schedules 34 images × 6 variants = 204 image/arm runs. Use a fresh
output directory. For a one-image F smoke test before the full run, use a
different output directory with `--max-samples 1 --pilot-variant
F_VLMFirst_AdaptiveROI`. A single-variant run produces only F's tables; run the
full suite for within-run A–F comparisons.

The output includes `pilot_report.json`, `aggregate_results.csv`,
`per_image_results.csv`, `counts_by_image.csv`, `results_summary.md`,
`compact_review.zip`, and `bbox_images.zip`. F's per-image row includes its VLM
ROI and whether tiling was forced. Failed or budget-truncated F runs are shown
as failures and do not contribute invented count metrics. These 34 fruit images
are a development set, not a held-out accuracy estimate.
