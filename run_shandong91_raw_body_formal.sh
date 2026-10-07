#!/usr/bin/env bash
set -Eeuo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CONFIG="${CONFIG:-configs/shandong91/raw_body_heterogeneous_formal_v1.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to a new formal run directory}"
PREFLIGHT_ROOT="${OUTPUT_ROOT}_launch_preflight"
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
[[ -f reliable_channel_training_v1/quality_report.json ]] || { echo "Missing reliable_channel_training_v1"; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "Tracked working tree is dirty"; exit 2; }
[[ "${CONDA_DEFAULT_ENV:-}" == "dm_env" ]] || { echo "Activate dm_env before launch"; exit 2; }

PHASE="cuda_e2e_preflight"; record_status "running"
python tools/preflight_shandong91_formal_e2e.py --device cuda --amp --config "$CONFIG" --output "$PREFLIGHT_ROOT"
python - "$PREFLIGHT_ROOT/e2e_report.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1], encoding="utf-8"))
assert r["status"] == "PASS", r
PY

PHASE="formal_training"; record_status "running"
python -u train_shandong91.py --device cuda --config "$CONFIG" --output-dir "$OUTPUT_ROOT"
PHASE="training_complete"; record_status "complete"
