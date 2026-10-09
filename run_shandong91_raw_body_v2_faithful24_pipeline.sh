#!/usr/bin/env bash
set -Eeuo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CONFIG="${CONFIG:-configs/shandong91/raw_body_v2_faithful24.yaml}"
DATA_ROOT="$(python - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["data"]["data_path"])
PY
)"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to a new isolated V2 run directory}"
PREFLIGHT_ROOT="${OUTPUT_ROOT}_cuda_preflight"
STATUS_FILE="${OUTPUT_ROOT}_status.json"
PHASE="startup"

record_status() {
  python - "$STATUS_FILE" "$1" "$PHASE" "$OUTPUT_ROOT" <<'PY'
import json, pathlib, sys, time
pathlib.Path(sys.argv[1]).write_text(json.dumps({"state":sys.argv[2],"phase":sys.argv[3],"root":sys.argv[4],"timestamp":time.time()}, indent=2))
PY
}
trap 'code=$?; record_status "failed"; exit "$code"' ERR

[[ ! -e "$OUTPUT_ROOT" ]] || { echo "Refusing existing OUTPUT_ROOT: $OUTPUT_ROOT"; exit 2; }
[[ ! -e "$PREFLIGHT_ROOT" ]] || { echo "Refusing existing PREFLIGHT_ROOT: $PREFLIGHT_ROOT"; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG"; exit 2; }
[[ -f "$DATA_ROOT/quality_report.json" ]] || { echo "Missing configured data root: $DATA_ROOT"; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "Tracked working tree is dirty"; exit 2; }
[[ "${CONDA_DEFAULT_ENV:-}" == "dm_env" ]] || { echo "Activate dm_env before launch"; exit 2; }

PHASE="cuda_amp_preflight"; record_status "running"
python tools/preflight_shandong91_raw_body_v2.py \
  --config "$CONFIG" --output "$PREFLIGHT_ROOT" --device cuda --amp --full-model
python - "$PREFLIGHT_ROOT/report.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1], encoding="utf-8"))
assert r["status"] == "PASS" and r["device"].startswith("cuda") and r["amp"] is True
assert r["full_model"] is True
assert r["checks"]["formal_microbatch_cuda_amp"]["status"] == "PASS"
PY

PHASE="formal_training"; record_status "running"
python -u train_shandong91_v2.py --config "$CONFIG" --output-dir "$OUTPUT_ROOT" --device cuda

PHASE="generation_evaluation_archive"; record_status "running"
CONFIG="$CONFIG" OUTPUT_ROOT="$OUTPUT_ROOT" \
  bash run_shandong91_raw_body_v2_faithful24_finalize.sh

PHASE="complete"; record_status "complete"
echo "V2_PIPELINE_COMPLETE=$OUTPUT_ROOT"
