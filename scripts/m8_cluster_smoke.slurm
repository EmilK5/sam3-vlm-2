#!/bin/bash
#SBATCH --job-name=sam3-vlm-m8-smoke
#SBATCH --output=logs/m8_smoke_%j.out
#SBATCH --error=logs/m8_smoke_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Exit on error
set -e

# Load necessary modules (adjust to your cluster environment)
# module load cuda/12.1
# module load anaconda3

# Activate your environment
# source activate sam3-vlm

echo "Starting M8.7 Pre-Cluster Smoke Test..."
echo "Node: $(hostname)"
echo "Date: $(date)"

# Export Qwen endpoint config (must be set by user or environment)
if [ -z "$QWEN_BASE_URL" ]; then
    echo "ERROR: QWEN_BASE_URL is not set"
    exit 1
fi

if [ -z "$QWEN_MODEL" ]; then
    echo "ERROR: QWEN_MODEL is not set"
    exit 1
fi

# Run the complete sequence
python -m sam3_vlm.experiments.m8_smoke --stage all --require-cuda --target "green citrus" --output_dir "runs/cluster_m8_smoke"

echo "Smoke test complete. Check pilot_report.json in runs/cluster_m8_smoke/"
