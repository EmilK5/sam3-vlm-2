# SAM3-VLM V4 GPU Cluster Runbook (M8.8)

This is the definitive guide for executing real-model validation of the V4 controller on the GPU cluster.

**PREREQUISITE:** The repository is fully configured for cluster execution. You DO NOT need to write any new tests or change any source code before running this validation sequence.

## 1. Environment and Configuration

You must export your runtime credentials. **Never check API keys into version control.**

```bash
# 1. Hugging Face Authentication for SAM3 weights (requires accepted license)
export HF_TOKEN="your_hf_read_token_here"

# 2. Qwen Inference Endpoint
export QWEN_BASE_URL="http://your.cluster.ip:8000/v1"
export QWEN_MODEL="qwen2.5-vl-72b-instruct"
export QWEN_API_KEY="your_api_key_or_EMPTY_if_vllm"

# 3. Enable Strict Real Model Testing
export RUN_REAL_MODELS=1
```

Configuration precedence is strict:
`CLI Overrides` > `Environment Variables` > `configs/m8_real_smoke.json` > `Code Defaults`.

## 2. Stage 1: Smoke Testing (Fail-Fast)

Before running a long pilot, you must prove the system works structurally. We provide a single Slurm script that executes preflight checks, real adapter unit tests, and a single-image end-to-end run.

```bash
sbatch scripts/m8_cluster_smoke.slurm
```

This script will run:
1. `pytest -m real_models` (Verifies adapters)
2. `python -m sam3_vlm.experiments.m8_smoke --stage all` (Runs Preflight -> M8.0 -> M8.1 -> M8.2 -> M8.3)

**Crucial Behavior:** `--stage all` STOPS BEFORE the pilot. It is strictly bounded. If any stage fails, the script will abort immediately with a non-zero exit code to prevent wasting GPU allocation.

Check `logs/m8_smoke_*.out`. If this succeeds, you are ready for the pilot.

## 3. Stage 2: The Pilot

Once the smoke test succeeds, run the pilot experiment.

```bash
sbatch scripts/m8_cluster_pilot.slurm
```

This script will run:
```bash
python -m sam3_vlm.experiments.m8_smoke --stage pilot --image /path/to/pilot/dataset --target "green citrus"
```

The pilot executes three variants (A_OneShot, B_FixedBank, C_V4_NoExemplarCleanup).
Results are written to `runs/cluster_m8_pilot/pilot_report.json`.

## 4. Diagnostics

If something fails:
- Review `pilot_report.json` for per-sample error strings.
- The `m8_smoke.py` script automatically verifies that `canonical_scene_state` matches during ReplayEngine serialization checks. If this fails, there is a divergence between runtime and disk state.
- `UnsupportedRealSAM3ActionError` is currently mitigated by disabling `cleanup` (`max_cleanup_calls = 0`) via `configs/m8_real_smoke.json`. Do not re-enable it until visual prompting is implemented.
