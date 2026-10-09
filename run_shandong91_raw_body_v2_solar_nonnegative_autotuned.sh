#!/usr/bin/env bash
set -Eeuo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8

BASE_CONFIG="configs/shandong91/raw_body_v2_faithful24_solar_nonnegative.yaml"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to a new isolated optimized run directory}"
TUNING_ROOT="${TUNING_ROOT:-${OUTPUT_ROOT}_throughput_tuning}"

[[ ! -e "$OUTPUT_ROOT" ]] || { echo "Refusing existing OUTPUT_ROOT: $OUTPUT_ROOT"; exit 2; }
[[ ! -e "$TUNING_ROOT" ]] || { echo "Refusing existing TUNING_ROOT: $TUNING_ROOT"; exit 2; }
[[ -f reliable_channel_training_v2_solar_nonnegative/projection_verification.json ]] || {
  echo "Missing verified Solar-nonnegative data"; exit 2;
}

python -u tools/tune_shandong91_v2_cuda_throughput.py \
  --config "$BASE_CONFIG" --output-dir "$TUNING_ROOT"

export CONFIG="$TUNING_ROOT/resolved_config.yaml"
export OUTPUT_ROOT
bash run_shandong91_raw_body_v2_faithful24_pipeline.sh
