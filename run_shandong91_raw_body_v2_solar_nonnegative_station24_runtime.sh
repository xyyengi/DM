#!/usr/bin/env bash
set -Eeuo pipefail

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CONFIG="configs/shandong91/raw_body_v2_faithful24_solar_nonnegative_station24_runtime.yaml"
DATA_ROOT="reliable_channel_training_v2_solar_nonnegative"

[[ -f "$DATA_ROOT/projection_audit.json" ]] || {
  echo "Missing projected data. Build it first with tools/build_shandong91_solar_nonnegative.py"
  exit 2
}
python tools/build_shandong91_solar_nonnegative.py \
  --source reliable_channel_training_v1 --output "$DATA_ROOT" --verify-only

bash run_shandong91_raw_body_v2_faithful24_pipeline.sh
