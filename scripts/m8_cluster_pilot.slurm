#!/bin/bash
#SBATCH --job-name=sam3-vlm-m8-pilot
#SBATCH --output=logs/m8_pilot_%j.out
#SBATCH --error=logs/m8_pilot_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Exit on error
set -e

echo "Starting M8 Pilot..."
echo "Node: $(hostname)"
echo "Date: $(date)"

if [ -z "$QWEN_BASE_URL" ] || [ -z "$QWEN_MODEL" ]; then
    echo "ERROR: QWEN_BASE_URL and QWEN_MODEL must be set in the environment"
    exit 1
fi

export RUN_REAL_MODELS=1

# Change /path/to/dataset to the actual path of the cluster dataset, or pass via argument
DATASET_PATH=${1:-"test_images/"}
TARGET_PROMPT=${2:-"green citrus"}

if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset path $DATASET_PATH does not exist."
    # Create a dummy dir for testing if needed
    mkdir -p "$DATASET_PATH"
    python -c "from PIL import Image; Image.new('RGB', (1024, 1024), color='green').save('$DATASET_PATH/pilot_1.jpg')"
fi

echo "--- Running Pilot on $DATASET_PATH ---"
python -m sam3_vlm.experiments.m8_smoke --stage pilot --require-cuda --image "$DATASET_PATH" --target "$TARGET_PROMPT" --output_dir "runs/cluster_m8_pilot"

echo "Pilot complete. Check runs/cluster_m8_pilot/pilot_report.json"
