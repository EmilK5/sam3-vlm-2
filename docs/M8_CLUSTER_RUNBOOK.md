# SAM3-VLM V4 GPU Cluster Runbook (M8.9)

This is the definitive, from-scratch guide for executing real-model validation of the V4 controller on the GPU cluster.

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
whether the pipeline makes one or two planner calls.

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
export M8_IMAGE="/path/to/representative_image.jpg"
export M8_TARGET="green citrus"
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

Once the smoke test passes cleanly, run the pilot separately. The pilot strictly requires a JSON manifest with a ground-truth count for every image.

The first comparison uses four variants:

| Variant | SAM3 setup | Qwen |
|---|---|---|
| `A_SAM3_Global` | One global target prompt | None |
| `B_SAM3_Bootstrap` | Context lock, target refinement, and target tiling | None |
| `C_Qwen_OneRound` | Full SAM3 bootstrap plus one target prompt | One call, no replan |
| `D_Qwen_TwoRound` | Full SAM3 bootstrap plus adaptive target prompts | Up to two calls and one replan |

With five images, this produces 20 runs. The two SAM3-only variants report the
hard number of registered candidate nodes. The Qwen variants report the
posterior count using the configured `0.8` commitment rule.

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
export M8_MANIFEST="/path/to/pilot_manifest.json"
export M8_OUTPUT_ROOT="runs/cluster_m8_pilot"

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

The M8 config uses `belief.target_count_commit_threshold: 0.8`. The final
reported count therefore commits target posteriors at or above `0.8` to a
per-node contribution of `1.0`, without changing the stored posterior. The
unmodified posterior sum is available as
`discovery_statistics.raw_soft_count`, alongside the threshold and number of
committed nodes.

M8 executes only novel target prompts. Qwen may describe likely confounders as
context, but the controller does not issue separate confounder SAM3 queries.
Qwen target prompts, confounder labels, and missing-appearance labels must be
one noun alone or one/two basic adjectives followed by a noun (1–3 words).
Simple wording and adjective/noun roles are guided by the Qwen instructions.
Vocabulary remains open; no dictionary filters object names or descriptive
labels. Executable prompts retain length, lexical, and method/prose checks.
Each Qwen round contributes at most one target experiment, with at most one
replan and two Qwen calls total.

Qwen must propose exactly one novel target `DISCOVERY` experiment unless the
controller's evidence explicitly reports saturated discovery. Convincing
current candidates are not permission to abstain. An empty unsaturated plan
persists `metadata.contract_diagnostic: EMPTY_UNSATURATED_PLAN` in the Qwen
artifact and ends under `NO_VALID_ACTIONS`, without an invented action or an
extra repair call. Inspect that field alongside `output.proposed_actions`,
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
continue with all four variants in section 4.

### Diagnosing Pilot Failures
Open `pilot_report.json`. Look in `.samples` for any sample where `"success": false`.
- If `failure_category` is present, look at `failure_message` for infrastructure crashes (e.g., CUDA OOM or Qwen payload errors).
- If `validator_status` is FAIL, the state machine produced corrupted semantic memory.
- If `replay_status` is FAIL, the runtime execution diverged from canonical replay constraints.
