# V4 single-image app

Use the same GPU environment, SAM3 weights and Qwen/Ollama service as the final
experiment. The app is a thin interface to that pipeline. It does not train a
model or replace Qwen's instructions.

## Start it

From the V4 repository directory containing `pyproject.toml`, `src` and `configs`:

```bash
python -m pip install -e '.[gui]'

# If these are already set for your pilot, keep your existing values.
export QWEN_BASE_URL="${QWEN_BASE_URL:-http://127.0.0.1:11434/v1}"
export QWEN_MODEL="${QWEN_MODEL:-qwen3.5-9b-sam3}"

python -m sam3_vlm.gui.app
```

Open **http://127.0.0.1:7860** in the browser on that machine. Keep the terminal
running. Stop the server with Ctrl+C. SAM3 loads once on the first submission;
Qwen's client is created only for a Qwen-enabled submission. The endpoint still
has to serve the existing Qwen model. No model is silently replaced by a mock.

If the GPU is remote, keep the app running there. On your laptop, replace
`USER@GPU_HOST` with the same SSH destination you already use:

```bash
ssh -N -L 7860:127.0.0.1:7860 USER@GPU_HOST
```

Then open **http://127.0.0.1:7860** on the laptop. This needs no public Gradio link.
For a hosted GPU service with its own port proxy, launch with `--host 0.0.0.0`
and open port 7860 through that service's existing access controls.

Model and output overrides are startup options:

```bash
python -m sam3_vlm.gui.app \
  --config configs/m8_real_smoke.json \
  --output-dir runs/gui \
  --sam3-model facebook/sam3 \
  --qwen-model qwen3.5-9b-sam3 \
  --qwen-base-url http://127.0.0.1:11434/v1 \
  --port 7860
```

Substitute your actual fine-tuned SAM3 model path if the preceding experiments
used one. The app uses the deployment config's device; production requires CUDA.
The Qwen API key, if needed, comes from `QWEN_API_KEY`, not a browser form.

## Use it

1. Upload an image and enter its target description, such as `green fruit`.
2. Optionally enter the true count. Blank means unknown; zero means no fruit.
3. Choose A, B, C, D or E and click **Apply preset**. D is loaded initially.
4. Adjust the controls. The form values are the settings used by **Run**; selecting
   a preset without applying it does not replace your edits.
5. Click **Run**. Read the count and call totals, view the plain-box image, and
   download the CSV or complete run ZIP.

| Preset | Bootstrap target threshold | Qwen calls | Count |
|---|---:|---:|---|
| A, global only | 0.20 | 0 | Active candidate count |
| B, full bootstrap | 0.20 | 0 | Active candidate count |
| C, one round | 0.25 | 1 maximum | Soft count |
| D, selected configuration | 0.25 | 2 maximum | Soft count |
| E, extended exploration | 0.25 | 100 maximum | Soft count |

C/D/E start with positive discovery at 0.20, negatives at 0.50, no exemplars for
Qwen positives, IoU+IoM association, and the current negative-miss policy. E has
a 1000-action SAM3 cap and no independent tile/runtime/iteration cap. Applying
another preset resets the controls, including nullable count cutoffs.

## What the controls mean

- **Enable Qwen:** off executes only the configured bootstrap and reports the
  candidate count. Negatives, posterior count cutoffs and cleanup are disabled in
  the effective configuration. Turning Qwen back on requires a call cap of at
  least one. A/B baselines do not need a running Qwen service.
- **Maximum Qwen calls:** includes repair and correction calls. `max_replans` is
  automatically set to this cap minus one, so changing 2 to 10 does not leave a
  hidden one-replan limit. It does not force the model to run ten times.
- **Continue exploration:** enables E's existing policy; does not automatically
  remove the other limits. Apply E to load all of its budget settings together.
- **Thresholds:** bootstrap, Qwen positives, negatives and context localization
  are separate SAM3 admission cutoffs. They are not posterior cutoffs. The SAM3
  mask binarization cutoff stays 0.50 inside the adapter.
- **Counting:** leave both confidence cutoffs blank for pure soft counting.
  Hard mode counts a node if `P(target) > cutoff`; commitment mode gives nodes
  with `P(target) >= cutoff` a contribution of one and keeps others soft. These
  two modes are mutually exclusive and apply only with Qwen enabled.
- **Context prompt:** a SAM3 localization request, normally `tree canopy`.
  Blank searches the full image. This does not edit Qwen's fixed tree-only scope.
- **Negatives:** separate object queries such as a Qwen-generated leaf description.
  They update existing candidates. They do not add negative detections as fruit.
  The neutral-miss switch controls whether an absent negative is neutral.
- **Tiling/exemplars:** control search crops and sensor-selected positive boxes.
  The bootstrap tile switch does not prohibit Qwen from proposing a tiled query.
- **Utility/cleanup:** expose the existing controller's research options. Cleanup
  remains off by default. With negatives enabled, accepted target/negative queries
  are executed in order instead of being discarded by a low utility score.
- **Seed:** seeds Python, NumPy and Torch locally. The remote Qwen request does
  not pass a seed; repeated real runs are not guaranteed identical.

Qwen's prompt version, system guidance, target-scope instruction, one-target-action
contract and reasoning mode are fixed by the selected production policy. The
existing request builder still inserts the uploaded image, new target text,
evidence/history and configured thresholds. The app does not truncate that context.
The Ollama context window remains a model/server setting; the output-token control
does not change it. Keep the same 65,536-token context configuration used in the pilot.

## Outputs and limits

Each submission writes `runs/gui/gui_<unique-id>/`. A successful directory contains
`settings.json`, `input.png`, `bboxes.png`, `result.json`, `result.csv`, and the usual
manifest, events, graph, masks and Qwen artifacts. `run_bundle.zip` contains that
directory's files, excluding the ZIP itself. Exact Qwen request text is in
`artifacts/qwen/*.json` under `input.request_text`. It is available for inspection,
not editing in the app. GT is used for reporting only and is not sent to Qwen.

Only rectangle outlines are drawn on the original-resolution output. All final
active candidates are shown, including low-probability ones. The number of boxes
need not equal the soft count. Errors in the single-image CSV are per-image errors,
not a dataset MAE; use the final-A–E runner for the full evaluation table.

Runs execute serially. Weight loading, queue waiting and final image/ZIP export
are excluded from the reported inference runtime. Limits are cooperative between
model calls, not a way to interrupt an in-flight GPU kernel or HTTP request.
The GUI adds a per-run adapter guard because bootstrap itself has no budget gate:
an insufficient bootstrap budget is reported as a failure with partial artifacts,
rather than silently exceeding the requested calls. Increase the limit or disable
bootstrap stages. Failed submissions clear prior results and retain `failure.json`
and the input/settings, so the previous image cannot be mistaken for a new result.

The app uses Gradio's [Blocks](https://www.gradio.app/docs/gradio/blocks) layout and
single-concurrency queue. It binds to localhost and disables public sharing by default.
The configured output directory is allowed for serving result files, including
when it is outside the repository. Use a dedicated output directory for this app.
