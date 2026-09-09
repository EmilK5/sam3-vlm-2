# SAM3-VLM V4 GPU Cluster Runbook (M8.9)

This is the definitive, from-scratch guide for executing real-model validation of the V4 controller on the GPU cluster.

For the final two-arm negative-evidence study, use the
[final evaluation guide](M8_FINAL_EVALUATION.md). Its sensing and
soft-count policy supersede older pilot examples below. The previous
[prompt comparison guide](M8_PROMPT_ABLATION.md) remains available for historical
comparisons; this runbook still covers environment and model setup.

## 1. Initial Setup

### 1.1 Clone the Repository
```bash
git clone https://github.com/EmilK5/sam3-vlm-2.git
cd sam3-vlm-2
```

### 1.2 Python Environment
Install a PyTorch build compatible with the cluster CUDA/toolchain first if the cluster requires a specific build. Then create your environment:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

### 1.3 Verify Transformers Version
The system requires `transformers>=5.0.0` for SAM3 access. Verify your installation:
```bash
python - <<'PY'
import transformers
from transformers import Sam3Model, Sam3Processor
print("transformers:", transformers.__version__)
print("SAM3 imports OK")
PY
```

### 1.4 Hugging Face & SAM3 Access
You MUST have accepted the model license for `facebook/sam3` on Hugging Face.
```bash
huggingface-cli login
# Or use the environment variable for read access:
export HF_TOKEN="your_hf_read_token_here"
```

### 1.5 Fast Qwen3.5-9B Ollama Profile

Create the bounded Qwen model on the machine that runs Ollama. The project
profile uses the official Q4_K_M quantization, a 16384-token context window,
and a 512-token generation ceiling. Keeping the context far below the model's
maximum is the main control on KV-cache memory.

If you start `ollama serve` manually, use one concurrent request and keep only
one model loaded. Apply the same environment variables to the Ollama service
configuration instead when Ollama is managed by systemd or another supervisor.
These are allocation controls, not an exact byte-level RAM cap.

```bash
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_FLASH_ATTENTION=1
ollama serve
```

Leave that process running and use a second shell for the remaining commands.

```bash
ollama pull qwen3.5:9b-q4_K_M
ollama create qwen3.5-9b-sam3 -f configs/ollama_qwen3_5_9b_fast.Modelfile
ollama show qwen3.5-9b-sam3
```

When upgrading an existing 8192-token alias, first run
`ollama stop qwen3.5-9b-sam3`, then repeat the `ollama create` command above and
verify `ollama show --modelfile qwen3.5-9b-sam3` reports `num_ctx 16384`.
Updating the repository alone does not update the server's existing alias.
The pilot exceeded the old context window with 8823–9074 input tokens; this
doubles input capacity while retaining the 512-token response limit. The client
sends the full formatted evidence and original image/contact-sheet bytes,
without additional text compaction or image resizing.

Point the experiment process at Ollama's OpenAI-compatible endpoint:

```bash
export QWEN_BASE_URL="http://127.0.0.1:11434/v1"
export QWEN_MODEL="qwen3.5-9b-sam3"
export QWEN_API_KEY="ollama"
```

Replace `127.0.0.1` with the Ollama host only when the model server runs on a
different machine. The Python client additionally requests non-thinking JSON,
limits each response to 512 tokens, applies a 45-second request timeout, and
disables hidden SDK retries. The experiment-level Qwen budget still controls
whether the pipeline makes one or two planner calls in C/D, or up to 100 in E.

After the first request, inspect the live allocation:

```bash
ollama ps
```

For the lowest latency, `PROCESSOR` should show the whole model on the GPU. To
release the loaded model after an experiment:

```bash
ollama stop qwen3.5-9b-sam3
```

---

## 2. Validation & Safety Checks

### 2.1 Release Check & Dry-Run
Before using the GPU, run the laptop-safe local readiness check. This automatically compiles the code, runs the test suite, parses the config, and executes a dry-run of the CLI.
```bash
bash scripts/check_m8_cluster_ready.sh
```
If this fails, stop before running either real model.

### 2.2 Gating Real Models
Enable the real model testing flag:
```bash
export RUN_REAL_MODELS=1
```
*Note: Unsetting this variable skips gated real-model unit tests.*

---

## 3. The Smoke Test (Fail-Fast)

The smoke test requires a real, representative image. It never fabricates fallback data.

Run the smoke sequence directly in the active shell:
```bash
export M8_IMAGE="/home/ekielar/sam3-vlm-2/assets/m8_test_img.jpg"
export M8_TARGET="green fruit"
export M8_OUTPUT_ROOT="runs/cluster_m8_smoke"

python -m sam3_vlm.experiments.m8_smoke \
  --stage preflight \
  --require-cuda \
  --image "$M8_IMAGE" \
  --target "$M8_TARGET" \
  --output_dir "$M8_OUTPUT_ROOT"

pytest -q -m real_models

python -m sam3_vlm.experiments.m8_smoke \
  --stage all \
  --require-cuda \
  --image "$M8_IMAGE" \
  --target "$M8_TARGET" \
  --output_dir "$M8_OUTPUT_ROOT"
```

**Important:** The `--stage all` command STOPS before the pilot. It will NOT run the pilot automatically.

---

## 4. The Pilot Experiment

### Complete comparison: A–D and both E variants

Use `--pilot-suite all` for A, B, C, D, E without negatives, and E with negatives:

```bash
python -m sam3_vlm.experiments.m8_smoke \
  --stage pilot --require-cuda \
  --pilot-suite all \
  --manifest pilot_manifest.json --max-samples 34 \
  --output_dir runs/m8_all_six_variants
```

This produces 204 runs for 34 images, one report with six aggregates, and a paired
comparison of the two E variants. C/D retain their current negative-prompt setting.
The two E variants replace the single standard E entry, avoiding a duplicate run.

### Negative prompts on/off comparison

Run only two E-based variants, with all configuration fields identical except
`planner.execute_confounder_prompts`:

- `E_NoNegativePrompts`: no separate confounder SAM3 queries.
- `E_WithNegativePrompts`: executes Qwen confounder labels as negative queries.

Both use posterior **> 0.5** hard counting, the same bootstrap and pseudoexemplars,
SAM3 threshold 0.5 for Qwen queries, and caps of 100 Qwen calls / 1000 SAM3 actions.
Both retain the full evidence context and frozen class vocabulary. This compares
the complete adaptive policies: enabling negatives changes evidence, planning
instructions, and potentially later target proposals/stopping. It does not force
identical model outputs or identical executed target sequences between runs.

```bash
python -m sam3_vlm.experiments.m8_smoke \
  --stage pilot --require-cuda \
  --pilot-suite negative-ablation \
  --manifest pilot_manifest.json --max-samples 34 \
  --output_dir runs/m8_negative_ablation
```

This produces **68 runs**, not the usual 170 A–E runs. Use `--max-samples 5` and a
separate output directory for a ten-run preliminary check. Omitting `--pilot-suite`
retains the standard A–E comparison. The visualization script reads the selected
variant names from report metadata and supports both suites.

The report includes `paired_comparison`, restricted to images that passed in
both variants. `mean_absolute_error_reduction` is MAE without negatives minus
MAE with negatives, so a positive value favors negatives. It also records better,
worse, and tied images, extra SAM3 calls/runtime, and whether all requested pairs
are complete. Individual aggregates, raw soft counts, and artifacts remain.

### Negative-evidence formula

The raw SAM3 score is never rewritten. A negative query updates the candidate's
class posterior. For queried confounder c, sensor score s, and correlation weight
w = 0.8^k (k counts other usable observations in the same semantic group), use:

| Observation relation | Multiplier L_c |
|---|---|
| Strong match | 1 + 3.0 s w |
| Weak match | 1 + 0.8 s w |
| Searched but not retrieved | max(0.1, 1 - 0.15 w) |
| Outside searched region / ambiguous association | 1 |

Other classes have multiplier 1. Normalize all classes together:
`p_new(j) = p_old(j) * L_j / sum(p_old(k) * L_k)`.
For the target specifically: `p_new(target) = p_old(target) / (1 + p_old(c)*(L_c-1))`.
Thus a confounder match lowers the target posterior. Confounder non-retrieval
slightly raises it; no observable evidence leaves it unchanged. These are hand-set,
uncalibrated proxy multipliers, not learned probabilities. Strong positive target
matches use coefficient 1.5, compared with 3.0 for strong confounder matches.

Example: target 0.70, queried confounder 0.20, other confounder 0.10; a first strong
negative match with score 0.80 gives L_c=3.4 and target posterior 0.70/1.48=0.4730.
That node changes from accepted to rejected under the strict >0.5 count rule.

### Current precision/counting revision

The current pilot enables `planner.execute_confounder_prompts`. Qwen still emits
one target action; its `likely_confounders` labels additionally become up to two
negative SAM3 queries at threshold 0.5. The controller freezes label-to-slot
meaning, skips previously tried labels, and runs target then negative queries
before replanning. Invalid phrases retain explicit rejection records. Negative
queries search the locked region without target exemplar boxes and cannot create
new countable nodes. They consume SAM3 calls and tiles within the existing caps.

C/D/E now count each active candidate as one only when its **target posterior is
strictly greater than 0.5**. This is thresholding of the existing evidence model,
not a trained linear classifier or thresholding of the latest SAM3 score. No
belief coefficients have been fitted or changed. `belief.target_count_hard_threshold`
is 0.5, and the old fractional commitment threshold is disabled. The true soft
posterior sum remains in `raw_soft_count`; uncertainty remains based on unrounded
posteriors. `summary.final_count` is the hard result; `final_soft_count` is a
legacy alias. The pilot reports `hard_posterior_count` for C/D/E and still uses
`hard_candidate_count` for A/B.

Use a fresh output directory, and compare raw soft and hard counts rather than
silently combining this revision's aggregates with older results.

After local tests pass, run a small pilot in the configured GPU environment:

```bash
python -m sam3_vlm.experiments.m8_smoke \
  --stage pilot --require-cuda \
  --manifest pilot_manifest.json --max-samples 5 \
  --output_dir runs/m8_negative_hard05_smoke
```

Check each C/D/E run's Qwen artifact for derived `CONFOUNDER` actions and rejection
reasons, SAM3 events for actual execution, and the summary for hard/soft count and
validation/replay results. Then run the 34-image comparison:

```bash
python -m sam3_vlm.experiments.m8_smoke \
  --stage pilot --require-cuda \
  --manifest pilot_manifest.json --max-samples 34 \
  --output_dir runs/m8_negative_hard05_full
```

Package `pilot_report.json`, `pilot/`, and the manifest used. This jointly changes
negative evidence and final reporting; the raw soft count supplies a same-run
comparison of counting rules, but attributing gains to negative sensing requires
a separate run with that setting disabled.


Once the smoke test passes cleanly, run the pilot separately. The pilot strictly requires a JSON manifest with a ground-truth count for every image.

The comparison uses five variants:

| Variant | SAM3 setup | Qwen |
|---|---|---|
| `A_SAM3_Global` | One global target prompt | None |
| `B_SAM3_Bootstrap` | Context lock, target refinement, and target tiling | None |
| `C_Qwen_OneRound` | Full SAM3 bootstrap plus one target prompt | One call, no replan |
| `D_Qwen_TwoRound` | Full SAM3 bootstrap plus adaptive target prompts | Up to two calls and one replan |
| `E_Qwen_UntilSaturation` | Full SAM3 bootstrap plus extended target discovery | Up to 100 calls / 99 replans, 1000 SAM3 actions |

With five images, this produces 25 runs. The two SAM3-only variants report the
hard number of registered candidate nodes. The Qwen variants report the
hard posterior count using the configured strict `> 0.5` rule.

All Qwen-generated SAM3 actions now execute at threshold **0.5**, including
C, D, and E. The controller overrides any different Qwen suggestion. Bootstrap
and context-lock thresholds are unchanged. Re-run C/D with this version to
compare against E at the same threshold.

E bypasses the earlier utility cutoff and has no separate iteration, tile, or
total-runtime limit. It stops when the existing numerical discovery **and**
uncertainty saturation test passes, or a hard call cap is reached. Specifically,
the last two target experiments must have a summed new-node count at most 0.05,
aggregate node entropy at most 1.0, and count variance at most 0.5. A discovery
plateau alone does not imply saturation while uncertainty remains high.
Invalid/empty/repeated proposals and model errors still terminate visibly;
reaching fewer than 100 calls does not by itself prove saturation. Check the
reported `stop_reason`. The 45-second request timeout and explicit repair policy
remain; repair requests count toward the 100-call cap.

`sam3_calls` counts sensor actions (including bootstrap), while `sam3_tiles`
counts tiles separately. E can therefore process more than 1000 individual
tiles. Each planning round can execute one target action plus new negative
labels; all these actions consume the same 1000-call safety budget.
Full evidence and original images remain supplied to Qwen, without compaction.
Long histories can exceed the server's 16384-token context and fail visibly.
Qwen confounder labels are executed as separate negative-evidence queries.

Example Manifest (`pilot_manifest.json`):
```json
[
  {
    "sample_id": "image_001",
    "image_path": "/data/image_001.jpg",
    "target": "green citrus",
    "gt_count": 17
  }
]
```

Run the pilot directly:
```bash
export M8_MANIFEST="/home/ekielar/sam3-vlm-2/pilot_manifest.json"
export M8_OUTPUT_ROOT="runs/cluster_m8_pilot-test-full"

python -m sam3_vlm.experiments.m8_smoke \
    --stage pilot \
    --require-cuda \
    --manifest "$M8_MANIFEST" \
    --max-samples 5 \
    --output_dir "$M8_OUTPUT_ROOT"
```

Inspect the aggregate comparison:

```bash
python - <<'PY'
import json
import os

path = os.path.join(os.environ["M8_OUTPUT_ROOT"], "pilot_report.json")
with open(path) as file:
    report = json.load(file)

for variant, metrics in report["aggregates"].items():
    print(
        variant,
        "MAE=", round(metrics["MAE"], 3),
        "SAM3=", round(metrics["avg_sam3_calls"], 2),
        "tiles=", round(metrics["avg_sam3_tiles"], 2),
        "Qwen=", round(metrics["avg_qwen_calls"], 2),
        "runtime_s=", round(metrics["avg_runtime_ms"] / 1000, 1),
    )
PY
```

---

## 5. Artifact Inspection & Diagnostics

### Artifact Locations
- **Aggregated Pilot Results:** `runs/cluster_m8_pilot/pilot_report.json`. This contains exact JSON schema fields for `.metadata`, `.samples`, and `.aggregates`.
- **E2E Runs:** Located in `runs/cluster_m8_smoke/M8.3/` or `runs/cluster_m8_pilot/pilot/<variant>/<run_id>/`.
  - `run.json` (Full manifest & config)
  - `events.jsonl` (State machine transitions)
  - `final_graph.json` (Detected semantics)
  - `summary.json` (E2E metrics)

The M8 config uses `belief.target_count_hard_threshold: 0.5`. An active node
contributes one only when its target posterior is strictly above 0.5; all others
contribute zero. Stored posteriors and variance are unchanged. The posterior sum
is available as `discovery_statistics.raw_soft_count`, alongside the hard threshold
and accepted-node count. `summary.final_count` is the reported count.

M8 executes novel target prompts and Qwen's confounder labels as separate negative
SAM3 queries. Confounder detections update existing candidates only.
Qwen target prompts, confounder labels, and missing-appearance labels must be
one noun alone or one/two basic adjectives followed by a noun (1–3 words).
Simple wording and adjective/noun roles are guided by the Qwen instructions.
Vocabulary remains open; no dictionary filters object names or descriptive
labels. Executable prompts retain length, lexical, and method/prose checks.
Qwen is instructed to preserve the user's target object category in both
executable prompts and missing-appearance labels. Confounder labels must not
become target actions. The instruction blocks omit concrete example phrase
lists; full evidence and tried-prompt history remain available.
Each Qwen round contributes at most one target experiment, with at most one
replan and two Qwen calls total in C/D; E uses the extended limits above.

Qwen must propose exactly one novel target `DISCOVERY` experiment unless the
controller's evidence explicitly reports saturated discovery (C/D). In E,
every requested plan must contain one novel action, even during a discovery-only
plateau. Convincing
current candidates are not permission to abstain. An empty unsaturated plan
persists `metadata.contract_diagnostic: EMPTY_UNSATURATED_PLAN` in the Qwen
artifact; if no target or negative query is accepted, execution ends under
`NO_VALID_ACTIONS`, without an invented target action or an extra repair call. Inspect that field alongside `output.proposed_actions`,
`metadata.rejections`, `repair_attempted`, `fallback_used`, and
`qwen_runtime_ms`.

Empty or whitespace-only `--output_dir` / config `output_root` values fail
explicitly. Relative paths are resolved against the active shell's working
directory, and `~` is expanded. M8.3 logs its absolute artifact directory before
model loading and its absolute `summary.json` path after validation and replay
succeed. Pilot completion likewise logs the absolute `pilot_report.json` path.
M8.2 only exercises the planner and intentionally creates no `summary.json`.

For the Qwen3.5 contract patch, first rerun M8.2 and M8.3 on the same difficult
image, directly from an interactive GPU shell:

```bash
cd /home/ekielar/sam3-vlm-2
git pull --ff-only
source .venv/bin/activate
python -m pip install -e .

export QWEN_BASE_URL="http://127.0.0.1:11434/v1"
export QWEN_MODEL="qwen3.5-9b-sam3"
export QWEN_API_KEY="ollama"
export M8_TARGET="green citrus"
export M8_IMAGE="/absolute/path/to/the/same/test/image.jpg"
export M8_OUTPUT_ROOT="$(pwd)/runs/qwen35_9b_smoke"

bash scripts/check_m8_cluster_ready.sh

time python -m sam3_vlm.experiments.m8_smoke \
  --stage M8.2 --require-cuda --target "$M8_TARGET" \
  --qwen-base-url "$QWEN_BASE_URL" --qwen-model "$QWEN_MODEL" \
  --output_dir "${M8_OUTPUT_ROOT:?Set M8_OUTPUT_ROOT}"

time python -m sam3_vlm.experiments.m8_smoke \
  --stage M8.3 --require-cuda --image "$M8_IMAGE" --target "$M8_TARGET" \
  --qwen-base-url "$QWEN_BASE_URL" --qwen-model "$QWEN_MODEL" \
  --output_dir "${M8_OUTPUT_ROOT:?Set M8_OUTPUT_ROOT}"
```

Use the logged absolute summary path to inspect that exact run. If setting a
shell variable for inspection, guard it before building child paths:

```bash
export LATEST_M8_RUN="/absolute/run/directory/from/the/M8.3/log"
cat "${LATEST_M8_RUN:?Set the logged M8.3 run directory}/summary.json"
ls "${LATEST_M8_RUN:?Set the logged M8.3 run directory}/artifacts/qwen"
```

Return the new summary and each Qwen artifact's proposed actions, rejections,
contract diagnostic, repair/fallback flags, and runtime. Do not start the
five-image A/B/C/D pilot until this same-image M8.3 run actually executes a
valid Qwen-derived target action and passes validator/canonical replay. Then
continue with all five variants in section 4.

### Diagnosing Pilot Failures
Open `pilot_report.json`. Look in `.samples` for any sample where `"success": false`.
- If `failure_category` is present, look at `failure_message` for infrastructure crashes (e.g., CUDA OOM or Qwen payload errors).
- If `validator_status` is FAIL, the state machine produced corrupted semantic memory.
- If `replay_status` is FAIL, the runtime execution diverged from canonical replay constraints.
