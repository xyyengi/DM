#!/usr/bin/env bash
# CUDA/AMP gate only. This script never starts formal generation.
set -Eeuo pipefail

CONFIG=${1:?"usage: $0 CONFIG OUTPUT_DIR"}
OUTPUT=${2:?"usage: $0 CONFIG OUTPUT_DIR"}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
PY=${PYTHON_BIN:-python}
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/station24/s5_crn_gpu_gate_${STAMP}.log"
STATUS="${OUTPUT}.status"
PHASE=initializing
mkdir -p logs/station24 "$(dirname "$OUTPUT")"
exec > >(tee -a "$LOG") 2>&1

write_status() {
  printf 'state=%s\nphase=%s\noutput=%s\nlog=%s\nupdated_at=%s\n' \
    "$1" "$PHASE" "$OUTPUT" "$LOG" "$(date --iso-8601=seconds)" > "$STATUS"
}
trap 'code=$?; write_status failed; echo "S5_CRN_GPU_GATE_FAILED phase=$PHASE code=$code"; exit $code' ERR
write_status running

test ! -e "$OUTPUT"
PHASE=config_and_environment
bash -n run_station24_s5_crn_gpu_gate.sh \
  run_station24_s5_crn_single_seed_generate.sh \
  run_station24_s5_crn_single_seed_finalize.sh
"$PY" -m tools.audit_station24_s5_crn_phase1 --help >/dev/null
"$PY" -m tools.validate_station24_s5_crn_stage2 --help >/dev/null
"$PY" -m tools.materialize_station24_s5_crn_single_seed --help >/dev/null
"$PY" -m tools.summarize_station24_s5_crn_single_seed --help >/dev/null
"$PY" -m tools.validate_station24_s5_crn_stage2 --config "$CONFIG" --dry-run
"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("CUDA", torch.__version__, torch.cuda.get_device_name(0))'

RAW_RUN=$("$PY" -m tools.validate_station24_s5_crn_stage2 --config "$CONFIG" --field paths.raw_run)
TAIL_RUN=$("$PY" -m tools.validate_station24_s5_crn_stage2 --config "$CONFIG" --field paths.tail_run)
DATA=$("$PY" -m tools.validate_station24_s5_crn_stage2 --config "$CONFIG" --field paths.data_path)
SEED=$("$PY" -m tools.validate_station24_s5_crn_stage2 --config "$CONFIG" --field experiment.generation_seed)

PHASE=cuda_amp_one_issue_four_pairs
"$PY" -m tools.audit_station24_s5_crn_phase1 \
  --raw-run "$RAW_RUN" --tail-run "$TAIL_RUN" --data-path "$DATA" \
  --output-dir "$OUTPUT" --seed "$SEED" --member-offset 400 --amp

PHASE=gate_report_validation
"$PY" -m tools.validate_station24_s5_crn_stage2 \
  --config "$CONFIG" --gate-report "$OUTPUT/phase1_audit.json"
cp "$CONFIG" "$OUTPUT/config_used.yaml"
PHASE=complete
write_status complete
cp "$STATUS" "$OUTPUT/STATUS.env"
echo "S5_CRN_GPU_GATE_COMPLETE output=$OUTPUT"
echo "FORMAL_GENERATION=NOT_RUN"

