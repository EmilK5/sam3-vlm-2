#!/bin/bash
#SBATCH --job-name=sam3-vlm-m8-pilot
#SBATCH --output=logs/m8_pilot_%j.out
#SBATCH --error=logs/m8_pilot_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

echo "Starting M8 Pilot..."
echo "Node: $(hostname)"
echo "Date: $(date)"

: "${M8_MANIFEST:?Set M8_MANIFEST}"
: "${M8_OUTPUT_ROOT:?Set M8_OUTPUT_ROOT}"

if [ ! -f "$M8_MANIFEST" ]; then
    echo "Manifest does not exist: $M8_MANIFEST" >&2
    exit 1
fi

export RUN_REAL_MODELS=1

echo "--- Running Pilot on $M8_MANIFEST ---"
python -m sam3_vlm.experiments.m8_smoke \
    --stage pilot \
    --require-cuda \
    --manifest "$M8_MANIFEST" \
    --output_dir "$M8_OUTPUT_ROOT"

echo "Pilot complete. Check runs/cluster_m8_pilot/pilot_report.json"
