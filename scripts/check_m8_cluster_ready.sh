#!/bin/bash
set -e

echo "=== M8.7 Pre-Cluster Readiness Check ==="

echo "1. Checking Python Syntax (compileall)..."
python -m compileall src tests > /dev/null
echo "  [OK] Syntax is clean."

echo ""
echo "2. Running Static Pytest Suite (Mocks only)..."
# Unset RUN_REAL_MODELS to ensure we test static logic only
unset RUN_REAL_MODELS
pytest -q tests/
echo "  [OK] Pytest static suite passed."

echo ""
echo "3. Testing Orchestration Dry-Run..."
touch test.jpg
python -m sam3_vlm.experiments.m8_smoke --dry-run --allow-cpu --output_dir "runs/dry_run_test" --qwen-base-url "http://fake" --image test.jpg
rm test.jpg
echo "  [OK] Dry-run complete. Orchestration constructs cleanly."

echo ""
echo "=== READY FOR CLUSTER ==="
echo "You can now clone this branch to the GPU cluster and run: sbatch scripts/m8_cluster_smoke.slurm"
