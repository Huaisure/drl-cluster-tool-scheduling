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

# Prefer the adjacent Toolkit checkout during local development. On a training
# machine without that checkout, the v0.3.2 package installed in the Conda
# environment is used instead.
TOOLKIT_DIR="${TOOLKIT_DIR:-$PROJECT_DIR/../cluster-tool-validator}"
if [[ -d "$TOOLKIT_DIR" ]]; then
  export PYTHONPATH="$TOOLKIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

if ! python -c \
  'from cluster_engine import ClusterEngine; assert hasattr(ClusterEngine, "load_lock_observation")'
then
  echo "cluster-tool-toolkit v0.3.2 or newer is required" >&2
  echo "install it with: python -m pip install -r requirements.txt" >&2
  exit 1
fi

TRAIN_MANIFEST="${TRAIN_MANIFEST:-datasets/train/manifest.json}"
VALIDATION_MANIFEST="${VALIDATION_MANIFEST:-datasets/validation/manifest.json}"
TEST_MANIFEST="${TEST_MANIFEST:-datasets/test/manifest.json}"

if [[ ! -f "$TRAIN_MANIFEST" || ! -f "$VALIDATION_MANIFEST" || ! -f "$TEST_MANIFEST" ]]; then
  echo "Dataset manifests are missing; generating train/validation/test datasets first"
  ./scripts/generate_datasets.sh
fi

for manifest in "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" "$TEST_MANIFEST"; do
  if [[ ! -f "$manifest" ]]; then
    echo "manifest not found after generation: $manifest" >&2
    exit 1
  fi
done

TOTAL_STEPS="${TOTAL_STEPS:-1000000}"
NUM_ENVS="${NUM_ENVS:-16}"
DEVICE="${DEVICE:-auto}"
EVALUATION_INTERVAL="${EVALUATION_INTERVAL:-25}"
VALIDATION_CASES="${VALIDATION_CASES:-20}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-runs/ppo_dataset_${RUN_ID}}"

export PYTHONUNBUFFERED=1

echo "Starting dataset PPO training"
echo "  environment : $CONDA_ENV"
echo "  device      : $DEVICE"
echo "  run_dir     : $RUN_DIR"
echo "  total_steps : $TOTAL_STEPS"
echo "  num_envs    : $NUM_ENVS"

exec python -u -m cluster_rl.train \
  --train-mode dataset \
  --num-envs "$NUM_ENVS" \
  --cpu-workers 0 \
  --train-manifest "$TRAIN_MANIFEST" \
  --validation-manifest "$VALIDATION_MANIFEST" \
  --test-manifest "$TEST_MANIFEST" \
  --device "$DEVICE" \
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
  --checkpoint-interval 5 \
  --evaluation-interval "$EVALUATION_INTERVAL" \
  --validation-cases "$VALIDATION_CASES"
