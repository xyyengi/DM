#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=${1:?'usage: run_shandong91_raw_body_finalize.sh PIPELINE_ROOT'}
CONFIG=${CONFIG:-configs/shandong91/raw_body_heterogeneous_formal_v1.yaml}
JOB=$(date +%Y%m%d_%H%M%S)
GEN="$ROOT/generation/test_n500_seed424242"
EVAL="$ROOT/evaluation"
cd "$(dirname "$0")"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
test -f "$ROOT/checkpoints/best.pt"
printf '%s\n' "$ROOT" > "$ROOT/train_run.txt"
if [[ ! -f "$GEN/metadata.json" || ! -f "$GEN/actual_scenarios_mw.npy" ]]; then
  [[ ! -e "$GEN" ]] || GEN="$ROOT/generation/test_n500_seed424242_retry_$JOB"
  python generate_shandong91.py --run-dir "$ROOT" --output-dir "$GEN" --config "$CONFIG" --split test --n-samples 500 --seed 424242 --member-chunk 5
fi
if [[ ! -f "$EVAL/RESULT_SUMMARY.md" || ! -f "$EVAL/metrics.json" ]]; then
  [[ ! -e "$EVAL" ]] || EVAL="$ROOT/evaluation_retry_$JOB"
  python tools/evaluate_shandong91.py --result "$GEN" --output-dir "$EVAL"
fi
python - "$ROOT/finalize_artifacts.json" "$GEN" "$EVAL" <<'PY'
import json,pathlib,sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({'generation':sys.argv[2],'evaluation':sys.argv[3],'summary':sys.argv[3]+'/RESULT_SUMMARY.md'},indent=2),encoding='utf-8')
PY
ARCHIVE="${ROOT}_completed_${JOB}.tar.gz"
tar -czf "$ARCHIVE" -C "$(dirname "$ROOT")" "$(basename "$ROOT")"
test -s "$ARCHIVE"
echo "FINALIZE_COMPLETE generation=$GEN evaluation=$EVAL archive=$ARCHIVE"
