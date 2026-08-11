#!/bin/bash
#SBATCH --job-name=sam3-vlm-m8-smoke
#SBATCH --output=logs/m8_smoke_%j.out
#SBATCH --error=logs/m8_smoke_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# Exit on error immediately
set -e

echo "Starting M8 Smoke Test Sequence..."
echo "Node: $(hostname)"
echo "Date: $(date)"

if [ -z "$QWEN_BASE_URL" ] || [ -z "$QWEN_MODEL" ]; then
    echo "ERROR: QWEN_BASE_URL and QWEN_MODEL must be set in the environment"
    exit 1
fi

export RUN_REAL_MODELS=1

# 1. Run real model unit tests first
echo "--- Running pytest -m real_models ---"
pytest -q -m real_models

# 2. Run the bounded smoke sequence (preflight -> M8.3)
echo "--- Running bounded smoke sequence ---"
# Create a dummy image for smoke testing
python -c "from PIL import Image; Image.new('RGB', (1024, 1024), color='green').save('smoke_test.jpg')"

python -m sam3_vlm.experiments.m8_smoke --stage all --require-cuda --image smoke_test.jpg --target "green citrus" --output_dir "runs/cluster_m8_smoke"

echo "Smoke test complete and successful! Ready for Pilot."
