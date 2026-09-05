#!/usr/bin/env bash
set -euo pipefail

echo "Checking M8 Cluster Readiness..."

# Block accidental live model calls
unset RUN_REAL_MODELS || true

# 1. Ensure Python files compile
echo "--- Compiling source and tests ---"
python -m compileall src tests

# 2. Run laptop-safe test suite
echo "--- Running pytest (mocked adapters) ---"
pytest -q tests/

# 3. Validate committed configuration parses successfully
echo "--- Validating M8 Config Parsing ---"
python - <<'PY'
import sys
from sam3_vlm.experiments.m8_smoke import load_m8_config

class DummyArgs:
    require_cuda = False
    output_dir = 'runs/test'

try:
    c = load_m8_config(DummyArgs(), config_path="configs/m8_real_smoke.json")
    v4 = c.v4_config
    assert v4.budget.max_cleanup_calls == 0, "Cleanup must be disabled for M8"
    assert v4.budget.max_qwen_calls == 2, "M8 must allow at most two Qwen calls"
    assert v4.planner.max_actions_per_prompt == 1, "M8 must admit one target action per round"
    assert v4.planner.max_output_tokens == 512, "M8 must bound Qwen output"
    assert v4.planner.request_timeout_seconds == 45.0, "M8 must bound each Qwen request"
    assert v4.planner.reasoning_effort == "none", "M8 must disable Qwen thinking"
    assert v4.replanning.max_replans == 1, "M8 must allow at most one replan"
    assert v4.belief.target_count_commit_threshold == 0.8, "M8 count commitment must use 0.8"
    print("M8 Config parsed successfully, constraints verified.")
except Exception as e:
    print(f"Config parsing failed: {e}", file=sys.stderr)
    sys.exit(1)
PY

# 4. Dry-run the M8 smoke script (preflight -> M8.3 bounding)
echo "--- Dry-running M8 Smoke (stage: all) ---"
# We create a tiny /tmp file to satisfy the image path check without leaving a repo trace
DRY_RUN_IMG="/tmp/m8_dry_run_image.jpg"
python -c "from PIL import Image; Image.new('RGB', (10, 10), color='green').save('$DRY_RUN_IMG')"

python -m sam3_vlm.experiments.m8_smoke \
    --dry-run \
    --stage all \
    --image "$DRY_RUN_IMG" \
    --target "dry-run target" \
    --output_dir "runs/cluster_m8_smoke_dry" \
    --allow-cpu \
    --qwen-base-url "http://fake"

rm "$DRY_RUN_IMG"

# 5. Check real model test suite discoverability
echo "--- Checking real test collection ---"
# We enable RUN_REAL_MODELS strictly for collection, no execution
RUN_REAL_MODELS=1 pytest -q -m real_models --collect-only

echo "Readiness check passed! Repository is ready for the GPU cluster."
