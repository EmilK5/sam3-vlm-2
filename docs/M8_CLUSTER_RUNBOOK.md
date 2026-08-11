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

### 1.5 Qwen Endpoints
Export exactly the following variables for your local cluster vLLM deployment:
```bash
export QWEN_BASE_URL="http://your.cluster.ip:8000/v1"
export QWEN_MODEL="qwen2.5-vl-72b-instruct"
export QWEN_API_KEY="your_api_key_or_EMPTY_if_vllm"
```

---

## 2. Validation & Safety Checks

### 2.1 Release Check & Dry-Run
Before you submit jobs to the GPU, run the laptop-safe local readiness check. This automatically compiles the code, runs the test suite, parses the config, and executes a dry-run of the CLI.
```bash
bash scripts/check_m8_cluster_ready.sh
```
If this fails, STOP. Do not submit jobs.

### 2.2 Gating Real Models
Enable the real model testing flag:
```bash
export RUN_REAL_MODELS=1
```
*Note: Unsetting this variable skips gated real-model unit tests.*

---

## 3. The Smoke Test (Fail-Fast)

The smoke test requires a real, representative image. It never fabricates fallback data.

Submit the smoke job:
```bash
export M8_IMAGE="/path/to/representative_image.jpg"
export M8_TARGET="green citrus"
export M8_OUTPUT_ROOT="runs/cluster_m8_smoke"

sbatch scripts/m8_cluster_smoke.slurm
```

This Slurm script strictly executes:
1. `preflight`:
   ```bash
   python -m sam3_vlm.experiments.m8_smoke --stage preflight --require-cuda --image "$M8_IMAGE" --target "$M8_TARGET" --output_dir "$M8_OUTPUT_ROOT"
   ```
2. Real Models Pytest Suite:
   ```bash
   pytest -q -m real_models
   ```
3. Bounded Sequence (SAM3 Smoke, Qwen Smoke, Bounded E2E):
   ```bash
   python -m sam3_vlm.experiments.m8_smoke --stage all --require-cuda --image "$M8_IMAGE" --target "$M8_TARGET" --output_dir "$M8_OUTPUT_ROOT"
   ```

**Important:** The `--stage all` command STOPS before the pilot. It will NOT run the pilot automatically.

---

## 4. The Pilot Experiment

Once the smoke test passes cleanly, submit the pilot job separately. The pilot strictly requires a JSON manifest.

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

Submit the pilot job:
```bash
export M8_MANIFEST="/path/to/pilot_manifest.json"
export M8_OUTPUT_ROOT="runs/cluster_m8_pilot"

sbatch scripts/m8_cluster_pilot.slurm
```

This executes:
```bash
python -m sam3_vlm.experiments.m8_smoke \
    --stage pilot \
    --require-cuda \
    --manifest "$M8_MANIFEST" \
    --output_dir "$M8_OUTPUT_ROOT"
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

### Diagnosing Pilot Failures
Open `pilot_report.json`. Look in `.samples` for any sample where `"success": false`.
- If `failure_category` is present, look at `failure_message` for infrastructure crashes (e.g., CUDA OOM or Qwen payload errors).
- If `validator_status` is FAIL, the state machine produced corrupted semantic memory.
- If `replay_status` is FAIL, the runtime execution diverged from canonical replay constraints.
