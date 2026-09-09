# Final A–E experiment

This is the requested final study: five variants on the same completed 34-image
manifest, producing 170 image/variant results, plain bounding-box images and
final count-accuracy tables.

## Variant definitions

| Variant | Detection setup | Qwen call cap | Count |
|---|---|---:|---|
| A_SAM3_Global | One global target pass at 0.20 | 0 | Number of active candidates |
| B_SAM3_Bootstrap | Full bootstrap, target passes at 0.20 | 0 | Number of active candidates |
| C_Qwen_OneRound | Selected D policy | 1 | Sum of target probabilities |
| D_Qwen_TwoRound | Selected D policy | 2 | Sum of target probabilities |
| E_Qwen_UntilSaturation | Selected D sensing/belief policy with extended exploration | 100 | Sum of target probabilities |

C/D/E use the best validated D configuration: bootstrap target threshold 0.25,
Qwen positive threshold 0.20 without target exemplars, negatives at 0.50,
`neutral_confounder_misses: false`, IoU+IoM, old core prompt with the existing
scope/qualifier/correction improvements, temperature 0.20 and output limit 512.
Keep the current SAM3 weights and Qwen alias/context. No new Qwen wording is added.
Tree-only scope remains in every Qwen request and vocabulary stays open.

A/B target thresholds are lowered to 0.20 as requested. C/D/E bootstrap remains
0.25 to preserve the selected D configuration. B retains context locking,
exemplar refinement and tiled bootstrap. A remains a single global baseline
without those bootstrap stages. These are intentional experimental differences.

C has no replan; D has at most one. Both retain the 15-SAM3, 64-tile, 300-second
and 20-iteration caps. Corrections count toward the ordinary Qwen cap. E retains
its previous extended-exploration semantics: at most 100 Qwen calls, 1000 SAM3
calls and 99 replans, with no separate tile/iteration/total-runtime cap. It can
stop early on the controller's saturation or invalid/exhausted-action conditions.
Cleanup is disabled in all variants. E can take substantially longer than D.

## Run on the GPU machine

Sync the updated V4 code and activate the same Python environment and Ollama
service used for the preceding pilot. Run from the repository directory containing
`src`, `configs` and the completed `pilot_manifest.json`. All exports use existing
Pillow and Python standard-library CSV support; no new packages are required.

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python -m sam3_vlm.experiments.m8_smoke \
  --stage pilot --require-cuda \
  --pilot-suite final-ae \
  --manifest pilot_manifest.json --max-samples 34 \
  --output_dir runs/m8_final_AE
```

Use a fresh output directory for any repeat. No separate visualization or table
command is needed. To inspect the matrix/manifest before inference, run the same
command with `--dry-run`; it should report 34 samples × 5 variants = 170 runs.
Do not add `--pilot-variant` when running this complete A–E experiment.

The full Qwen input remains image dependent. Exact sampled system/user request
text is saved in `prompt_examples.json` within the compact ZIP; individual Qwen
artifacts retain each call's evidence and request text.

## Files produced

All paths below are under `runs/m8_final_AE/`:

- `aggregate_results.csv`: one row per variant with MAE, MSE, RMSE, MRE (%), signed
  error, median/max absolute error, all/successful-image GT totals, prediction
  totals, successful/failed counts, runtime and model-call averages. It also gives
  MAE over the common successful image set for comparable partial results.
- `per_image_results.csv`: one row per image and variant, with GT, prediction,
  count type, errors, runtime/calls, candidate/box counts, image path and status.
- `counts_by_image.csv`: one row per image with GT and five prediction columns.
- `results_summary.md`: a concise aggregate table for presentation.
- `bbox_images/<variant>/<sample_id>.png`: original-resolution images containing
  only red bounding-box rectangles. No text, score, ID, title, legend, mask or
  confidence color coding is drawn. Every final active candidate is shown, so
  rectangle count does not necessarily equal a soft predicted count.
- `bbox_manifest.json`: maps images/variants to the PNGs and records box counts.
- `bbox_images.zip`: all PNGs from this run, ready to download.
- `compact_review.zip`: the three CSV tables, Markdown summary and box manifest,
  plus the usual numerical results, configs, confidence traces, prompt examples
  and selected diagnostic cases. Full-resolution box images are kept in their
  separate ZIP so the review bundle stays small.
- `pilot_report.json`: complete local report. Full graphs/masks/events stay under
  the per-run artifact directories.

Send `compact_review.zip` for analysis. Download `bbox_images.zip` for your
presentation; there is no need to upload all 170 images unless visual review is
wanted. PNG filenames/directory names identify image and variant without putting
labels on the images themselves.

Counts are the only ground truth supplied, so the tables measure counting error;
they do not fabricate bounding-box precision, recall or IoU scores. MRE excludes
zero-GT images. Failed runs are shown explicitly and unavailable values are blank,
not replaced by zero. Preserve A/B hard counts versus C/D/E soft counts in reporting.
This is a development-set evaluation because these images informed configuration
selection.

## Execution prompt to paste into a GPU coding task

> Run the final A–E experiment using the existing V4 code and completed
> `pilot_manifest.json`. Use `--stage pilot --require-cuda --pilot-suite final-ae
> --max-samples 34 --output_dir runs/m8_final_AE`, with no single-variant filter.
> Keep the already configured SAM3/Qwen models and do not tune prompts, thresholds
> or belief parameters. A/B use target threshold 0.20; C/D/E use the selected D
> configuration with Qwen caps 1/2/100 and E's existing 1000-SAM3 safety cap.
> Let the built-in exporter produce rectangle-only images and the final CSV/Markdown
> tables. Check that the report contains all five variants, 34 rows per variant,
> and explicit success/failure counts. Return `compact_review.zip`, the aggregate
> results table and the location of `bbox_images.zip`. If a model/runtime failure
> occurs, report it rather than substituting detections or counts.
