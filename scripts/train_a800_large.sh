#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

CONDA_ENV="${CONDA_ENV:-lhs}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda is unavailable; activate ${CONDA_ENV} before running this script" >&2
    exit 1
  fi
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "warning: SLURM_JOB_ID is unset; training will run outside a Slurm allocation" >&2
fi

VALIDATION_MANIFEST="${VALIDATION_MANIFEST:-datasets/validation/manifest.json}"
TEST_MANIFEST="${TEST_MANIFEST:-datasets/test/manifest.json}"
for manifest in "$VALIDATION_MANIFEST" "$TEST_MANIFEST"; do
  if [[ ! -f "$manifest" ]]; then
    echo "manifest not found: $manifest" >&2
    exit 1
  fi
done

TOTAL_STEPS="${TOTAL_STEPS:-1000000}"
RUN_ID="${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-runs/generator_a800_large_${RUN_ID}}"

export PYTHONUNBUFFERED=1

echo "Starting A800 generator training"
echo "  job        : ${SLURM_JOB_ID:-none}"
echo "  host       : $(hostname)"
echo "  CPUs       : $(nproc)"
echo "  run_dir    : $RUN_DIR"
echo "  total_steps: $TOTAL_STEPS"

exec python -u -m cluster_rl.train \
  --train-mode generator \
  --num-envs 16 \
  --cpu-workers 0 \
  --generator-seed 42 \
  --generator-max-attempts 64 \
  --validation-manifest "$VALIDATION_MANIFEST" \
  --test-manifest "$TEST_MANIFEST" \
  --device cuda \
  --run-dir "$RUN_DIR" \
  --total-steps "$TOTAL_STEPS" \
  --rollout-steps 128 \
  --epochs 4 \
  --minibatch-size 512 \
  --model-dim 128 \
  --num-heads 8 \
  --hgt-layers 3 \
  --num-layers 3 \
  --feedforward-dim 512 \
  --learning-rate 1e-4 \
  --gae-lambda 0.99 \
  --target-kl 0.02 \
  --seed 42 \
  --log-interval 1 \
  --checkpoint-interval 5
