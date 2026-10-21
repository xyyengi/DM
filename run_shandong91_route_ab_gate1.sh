#!/usr/bin/env bash
set -euo pipefail

: "${ROUTE:?set ROUTE=A or ROUTE=B}"
: "${OUTPUT_ROOT:?set an isolated OUTPUT_ROOT}"
: "${SHANDONG91_DATA:=./reliable_channel_training_v2_solar_nonnegative}"
export SHANDONG91_DATA
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [[ "$ROUTE" == "A" ]]; then
  CONFIG=configs/shandong91/route_a_system_load_short_full.yaml
elif [[ "$ROUTE" == "B" ]]; then
  CONFIG=configs/shandong91/route_b_multiscale_graph_short_full.yaml
else
  echo "ROUTE must be A or B" >&2; exit 2
fi

python -u -m unittest tests.test_shandong91_route_ab -v
python -u train_shandong91_route_ab.py \
  --route "$ROUTE" --config "$CONFIG" --output-dir "$OUTPUT_ROOT" \
  --device cuda --max-epochs "${MAX_EPOCHS:-5}"

