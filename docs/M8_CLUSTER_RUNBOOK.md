# SAM3-VLM V4 Cluster Validation (M8)

This guide defines the explicit sequence of operations to perform real-model validation of the V4 controller on a Slurm/GPU instance.

## 1. Environment Setup

Before attempting to execute real models, you must inject the necessary runtime variables. **Never check in API keys or passwords to version control.**

```bash
# 1. Hugging Face Authentication for SAM3 access
# Note: You MUST have accepted the model license for facebook/sam3 on HF.
export HF_TOKEN="your_hf_read_token_here"

# 2. Qwen Inference Endpoint
export QWEN_BASE_URL="http://your.cluster.ip:8000/v1"
export QWEN_MODEL="qwen2.5-vl-72b-instruct"
export QWEN_API_KEY="your_api_key_or_EMPTY_if_vllm"

# 3. Enable Strict Real Model Testing
export RUN_REAL_MODELS=1
```

## 2. Cluster Execution Order

Execute the following commands sequentially. **Stop immediately if any stage returns a non-zero exit code.** 
Each command leverages the strict testing logic defined in `m8_smoke.py`.

### Step 1: Preflight
Check that PyTorch sees your GPU, dependencies exist, and output directories are writable.
```bash
python -m sam3_vlm.experiments.m8_smoke --stage preflight
```

### Step 2: Gated Regression Verification
Run the standard validation suite explicitly checking real dependencies. **Ensure no tests are unexpectedly skipped.**
```bash
pytest -q -m real_models
```

### Step 3: SAM3 Global Smoke
Validates the fundamental `RealSAM3Sensor` GLOBAL invocation without engaging Qwen.
```bash
python -m sam3_vlm.experiments.m8_smoke --stage M8.1 --image /path/to/test.jpg
```

### Step 4: Qwen Multimodal Smoke
Verifies that the `RealQwenPlanner` correctly assembles the multimodal prompt (text, scene image, contact sheet) and receives a strictly valid JSON response.
```bash
python -m sam3_vlm.experiments.m8_smoke --stage M8.2
```

### Step 5: M8.3 One Full Real Image
Executes one complete pass of the V4 dynamic pipeline, forcing `RunRecorder` to trace the state and instantly following up with `RunValidator` and `ReplayEngine` fidelity checks.
```bash
python -m sam3_vlm.experiments.m8_smoke --stage M8.3 --image /path/to/test.jpg --target "green citrus"
```

### Step 6: Multi-Image Pilot (A/B/C)
Runs all three validation variants on a sequence of images or a provided JSON manifest, outputting the results into `pilot_report.json`.
```bash
# Using a directory of images
python -m sam3_vlm.experiments.m8_smoke --stage pilot --image /path/to/images/ --target "green citrus"

# Or using a pilot manifest
python -m sam3_vlm.experiments.m8_smoke --stage pilot --manifest pilot_manifest.json
```

## 3. Artifact Inspection

Artifacts are written sequentially into `runs/m8_real_smoke/`. If an error occurs, inspect:
1. `events.jsonl` to pinpoint the last emitted state-machine transition.
2. `pilot_report.json` for structured error string bubbling.
3. If SAM3 initialization fails, review your HF permissions.
4. If Qwen fails with `Raw output unparseable`, verify your backend endpoint respects strictly constrained JSON schema prompting.
