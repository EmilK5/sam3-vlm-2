# M8: soft counts and positive-prompt comparison

This guide preserves the completed old/V3 comparison. The next small experiment
is in the [recovery and discovery diagnostic guide](M8_RECOVERY_DIAGNOSTIC.md).
This historical suite explicitly disables rejection correction.

Run these commands from the V4 repository on the model machine, after syncing
the updated V4 code and activating the same environment/model endpoint used for
the preceding pilot. Use the same completed 34-image manifest.

Make this checkout's source importable before running the commands:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

## What changes

Every C/D/E run reports `sum(P(target))` over active nodes. Both reporting
thresholds are null; probabilities are neither rounded nor committed to one.
SAM3's threshold for Qwen-generated positive and negative queries remains 0.5.
The existing model context and output limits are unchanged.

| Variant | Prompt | Negatives | Maximum Qwen calls |
|---|---|---|---:|
| C_OldPrompt | Old | Yes | 1 |
| C_NewPrompt | V3 | Yes | 1 |
| D_OldPrompt | Old | Yes | 2 |
| D_NewPrompt | V3 | Yes | 2 |
| E1_OldPrompt_NoNegatives | Old | No | 100 |
| E2_OldPrompt_WithNegatives | Old | Yes | 100 |
| E3_NewPrompt_NoNegatives | V3 | No | 100 |
| E4_NewPrompt_WithNegatives | V3 | Yes | 100 |

E retains its 1000 SAM3-call ceiling, without independent tile, iteration or
total-runtime caps. It can still stop for exhausted/invalid actions; inspect
stop reasons instead of assuming every completed E run saturated.

The new prompt adds image-grounded discovery guidance, tree-fruit scope,
and consistent E stopping instructions. The old prompt text is preserved.
Comparisons test this combined prompt revision, not each wording change in
isolation. Open vocabulary, short noun phrases, full evidence and image payloads
are retained. C/D both use negatives to match the preceding pilot's policy.

## 1. Optional two-image model smoke test

This runs all eight variants on two images (16 runs). It makes real model calls.

```bash
python -m sam3_vlm.experiments.m8_smoke \
  --stage pilot --require-cuda \
  --pilot-suite prompt-ablation \
  --manifest pilot_manifest.json --max-samples 2 \
  --output_dir runs/m8_prompt_v3_smoke
```

For manifest validation and a run-count preview without model calls, add
`--dry-run`; that does not validate model behavior or create a review bundle.

## 2. Run the full study in three batches

The batches run sequentially: 68 C runs, 68 D runs, then 136 E runs. Each batch
produces its own report and compact ZIP, so it can be reviewed independently.
The loop stops if a pilot returns failure. Keep output directories unique when
repeating the experiment, because reports at the same location are overwritten.

```bash
for family in C D E; do
  python -m sam3_vlm.experiments.m8_smoke \
    --stage pilot --require-cuda \
    --pilot-suite prompt-ablation --pilot-family "$family" \
    --manifest pilot_manifest.json --max-samples 34 \
    --output_dir "runs/m8_prompt_v3_${family}" || break
done
```

To run just one batch, use the same command with `--pilot-family C`, `D`, or `E`
and its corresponding output directory. To run all eight variants in one pilot,
omit `--pilot-family` and choose a fresh output directory.

## 3. Send just the compact ZIP after each batch

```text
runs/m8_prompt_v3_C/compact_review.zip
runs/m8_prompt_v3_D/compact_review.zip
runs/m8_prompt_v3_E/compact_review.zip
```

Each ZIP contains:

- `review_summary.json`: resolved configs, aggregates, paired improvements/costs,
  pair completeness, failures, stop reasons and Qwen rejection histograms.
- `counts.json`: a small numeric row per image/variant for independent comparison.
- `prompt_yields.json`: executions and candidate discovery yields per prompt,
  including bootstrap queries. New candidates are not ground-truth true positives.
- `prompt_examples.json`: one actual system/user text request per available
  variant, with its image and call identifiers.
- `selected_cases.json`: first/last Qwen outputs and rejection diagnostics for at
  most three images selected for improvement, regression, failure or large error.
- At most three reduced original-image previews, identified in the summary.

The selection is for diagnosis, not representative accuracy estimation. Full
graphs, masks, event logs, contact sheets and per-call evidence stay on your
machine. Keep them for a later focused request; no need to upload their folders
or the full `pilot_report.json` initially. Missing review artifacts are listed
under `export_warnings`.

The E batch reports four paired contrasts: old versus new with negatives off,
old versus new with negatives on, negatives off versus on under the old prompt,
and negatives off versus on under the new prompt. C and D each report one prompt
contrast. Positive error reduction means the right-hand variant improved MAE.
