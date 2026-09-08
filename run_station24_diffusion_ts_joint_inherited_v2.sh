#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME:-dm_env}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
MODE=${1:-run}
ROOT=${2:-outputs_shandong/station24/diffusion_ts_joint_inherited_v2_$(date +%Y%m%d_%H%M%S)}
BASELINE=${3:-outputs_shandong/station24/body_tail_moe_raw_inference_20260824_224151/validation_results/geo_history_actual_body_tail_moe_raw_val_n500_seed424242}
CONFIG=configs/station24_diffusion_ts_joint_inherited_v2.yaml
if [[ "${TS_INHERITED_WORKER:-0}" != 1 ]]; then
  [[ "$MODE" == run || "$MODE" == resume ]] || { echo 'mode must be run or resume'; exit 1; }
  git diff --quiet && git diff --cached --quiet || { echo 'tracked changes must be reviewed'; exit 1; }
  [[ "$(git branch --show-current)" == experiment/24site-diffusion-ts-joint-v1 ]] || { echo 'wrong branch'; exit 1; }
  [[ -f "$BASELINE/metrics.json" ]] || { echo 'Raw baseline missing'; exit 1; }
  [[ "$MODE" == resume || ! -e "$ROOT" ]] || { echo 'output exists; refusing overwrite'; exit 1; }
  [[ "$MODE" == run || -f "$ROOT/training/training_summary.json" ]] || { echo 'resume requires completed training'; exit 1; }
  mkdir -p logs/station24
  LOG="logs/station24/station24_diffusion_ts_joint_inherited_v2_$(date +%Y%m%d_%H%M%S).log"
  STATUS="${LOG%.log}.status"
  nohup setsid env TS_INHERITED_WORKER=1 STATUS="$STATUS" bash "$0" "$MODE" "$ROOT" "$BASELINE" > "$LOG" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "${LOG%.log}.pid"
  echo "Monitor: tail -f '$LOG'"; echo "Status: $STATUS"; echo "ROOT=$ROOT"
  exit 0
fi
finish() {
  code=$?; trap - EXIT
  if [[ $code == 0 ]]; then state=completed; else state=failed; fi
  printf 'state=%s\nexit_code=%s\nroot=%s\n' "$state" "$code" "$ROOT" > "$STATUS"
  exit "$code"
}
trap finish EXIT
printf 'state=running\nroot=%s\n' "$ROOT" > "$STATUS"
python -c 'import torch,einops; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name(0))'
if [[ "$MODE" == run ]]; then
  python -m unittest tests.test_station24_diffusion_ts tests.test_station24_diffusion_ts_inherited
  python -m tools.station24_diffusion_ts_experiment preflight --config "$CONFIG" --output "$ROOT/preflight" --device cuda
  python -m tools.station24_diffusion_ts_experiment train --config "$CONFIG" --output "$ROOT/training" --gate "$ROOT/preflight/preflight.json" --device cuda
fi
AUDIT="$ROOT/component_audit_$(date +%Y%m%d_%H%M%S)"
python -m tools.audit_station24_diffusion_ts_components --run "$ROOT/training" --output "$AUDIT" --device cuda
if [[ ! -f "$ROOT/validation_results/generation_metadata.json" ]]; then
  [[ ! -e "$ROOT/validation_results" ]] || { echo 'Partial generation retained; review before retry, no overwrite.'; exit 1; }
  python -m tools.station24_diffusion_ts_experiment generate --config "$CONFIG" --run "$ROOT/training" --output "$ROOT/validation_results" --device cuda
fi
POST="$ROOT/postprocess_$(date +%Y%m%d_%H%M%S)"
python -m tools.evaluate_station24_diffusion_ts --baseline "$BASELINE" --candidate "$ROOT/validation_results" --output "$POST/joint" --candidate-label 'TS inherited V2'
python -m tools.evaluate_station24_jstd_events --baseline "$BASELINE" --candidate "$ROOT/validation_results" --candidate-run "$ROOT/training" --output-dir "$POST/events" --candidate-label 'TS inherited V2'
python -m tools.plot_station24_extreme_tail --baseline "$BASELINE" --candidate "$ROOT/validation_results" --data-path diffusion_input_station --output-dir "$POST/extreme" --top-issues 5 --baseline-label 'Raw body-tail' --candidate-label 'TS inherited V2'
python -m tools.diagnose_station24_wind_event_timing "$BASELINE" "$ROOT/validation_results" --output-dir "$POST/wind_timing" --baseline-variant geo_history_actual_body_tail_moe_raw --candidate-variant station24_diffusion_ts_joint_inherited_v2 --baseline-label 'Raw body-tail' --candidate-label 'TS inherited V2'
ARCHIVE="${ROOT}_completed_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$ARCHIVE" -C "$(dirname "$ROOT")" "$(basename "$ROOT")"
echo "DIFFUSION_TS_INHERITED_V2_COMPLETE ARCHIVE=$ARCHIVE"
