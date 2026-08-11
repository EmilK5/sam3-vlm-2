#!/bin/bash
set -e

echo "Checking M8 Cluster Readiness..."

# Check Transformers Version
TRANSFORMERS_VER=$(python -c "import transformers; print(transformers.__version__)")
echo "Transformers version: $TRANSFORMERS_VER"

# Check pytest -m real_models with dry-run env vars
export RUN_REAL_MODELS=1
echo "Dry running pytest..."
# We can't actually run it because models would load. We just collect to ensure syntactical validity.
pytest -q -m real_models --collect-only

# Check configuration loads cleanly
python -c "
from sam3_vlm.experiments.m8_smoke import load_m8_config
import os

class DummyArgs:
    require_cuda = False
    output_dir = 'runs/test'

c = load_m8_config(DummyArgs())
assert c.v4_config.budget.max_cleanup_calls == 0, 'Cleanup must be disabled in config'
"
echo "Config parsed correctly, cleanup disabled."

echo "Readiness check passed!"
