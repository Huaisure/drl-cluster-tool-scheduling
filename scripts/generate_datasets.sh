#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

DATASET_SEED="${DATASET_SEED:-42}"
TRAIN_COUNT="${TRAIN_COUNT:-1000}"
VALIDATION_COUNT="${VALIDATION_COUNT:-100}"
TEST_COUNT="${TEST_COUNT:-100}"

generate_split() {
  local split="$1"
  local count="$2"
  python -m run_rl_generation "datasets/${split}" \
    --split "$split" \
    --count "$count" \
    --seed "$DATASET_SEED" \
    --without-reference-actions \
    --overwrite
}

generate_split train "$TRAIN_COUNT"
generate_split validation "$VALIDATION_COUNT"
generate_split test "$TEST_COUNT"

echo "Generated materialized PPO datasets"
echo "  train      : datasets/train/manifest.json (${TRAIN_COUNT})"
echo "  validation : datasets/validation/manifest.json (${VALIDATION_COUNT})"
echo "  test       : datasets/test/manifest.json (${TEST_COUNT})"
