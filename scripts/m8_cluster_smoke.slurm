#!/bin/bash
#SBATCH --job-name=sam3-vlm-m8-smoke
#SBATCH --output=logs/m8_smoke_%j.out
#SBATCH --error=logs/m8_smoke_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

echo "Starting M8 Smoke Test Sequence..."
echo "Node: $(hostname)"
echo "Date: $(date)"

: "${M8_IMAGE:?Set M8_IMAGE to a real representative image}"
: "${M8_TARGET:?Set M8_TARGET}"
: "${M8_OUTPUT_ROOT:?Set M8_OUTPUT_ROOT}"

if [ ! -f "$M8_IMAGE" ]; then
    echo "M8_IMAGE does not exist: $M8_IMAGE" >&2
    exit 1
fi

export RUN_REAL_MODELS=1

echo "--- 1. Running Preflight ---"
python -m sam3_vlm.experiments.m8_smoke \
    --stage preflight \
    --require-cuda \
    --image "$M8_IMAGE" \
    --target "$M8_TARGET" \
    --output_dir "$M8_OUTPUT_ROOT"

echo "--- 2. Running Real Models Pytest Suite ---"
pytest -q -m real_models

echo "--- 3. Running Bounded Smoke Sequence ---"
python -m sam3_vlm.experiments.m8_smoke \
    --stage all \
    --require-cuda \
    --image "$M8_IMAGE" \
    --target "$M8_TARGET" \
    --output_dir "$M8_OUTPUT_ROOT"

echo "Smoke test complete and successful! Ready for Pilot."
