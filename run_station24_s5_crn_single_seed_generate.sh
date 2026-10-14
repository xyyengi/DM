#!/usr/bin/env bash
# Formal single-seed Raw500 + Tail100 generation only. Evaluation is separate.
set -Eeuo pipefail

CONFIG=${1:?"usage: $0 CONFIG OUTPUT_ROOT GPU_GATE_REPORT"}
ROOT=${2:?"usage: $0 CONFIG OUTPUT_ROOT GPU_GATE_REPORT"}
GATE_REPORT=${3:?"usage: $0 CONFIG OUTPUT_ROOT GPU_GATE_REPORT"}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
PY=${PYTHON_BIN:-python}
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/station24/s5_crn_single_seed_generate_${STAMP}.log"
STATUS="$ROOT/STATUS.env"
PHASE=initializing
mkdir -p logs/station24

"$PY" -m tools.validate_station24_s5_crn_stage2 \
  --config "$CONFIG" --gate-report "$GATE_REPORT"

if [[ -e "$ROOT" && ! -s "$ROOT/EXPERIMENT_MANIFEST.json" ]]; then
  echo "existing root lacks CRN manifest: $ROOT" >&2
  exit 1
fi
mkdir -p "$ROOT"
exec > >(tee -a "$LOG") 2>&1
write_status() {
  printf 'state=%s\nphase=%s\nroot=%s\nlog=%s\nupdated_at=%s\n' \
    "$1" "$PHASE" "$ROOT" "$LOG" "$(date --iso-8601=seconds)" > "$STATUS"
}
trap 'code=$?; write_status failed; echo "S5_CRN_GENERATION_FAILED phase=$PHASE code=$code"; exit $code' ERR
write_status running

PHASE=gate_and_config_validation
"$PY" -m tools.validate_station24_s5_crn_stage2 \
  --config "$CONFIG" --gate-report "$GATE_REPORT"
"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
RAW_RUN=$("$PY" -m tools.validate_station24_s5_crn_stage2 --config "$CONFIG" --field paths.raw_run)
TAIL_RUN=$("$PY" -m tools.validate_station24_s5_crn_stage2 --config "$CONFIG" --field paths.tail_run)
DATA=$("$PY" -m tools.validate_station24_s5_crn_stage2 --config "$CONFIG" --field paths.data_path)
SEED=$("$PY" -m tools.validate_station24_s5_crn_stage2 --config "$CONFIG" --field experiment.generation_seed)
ISSUE_BATCH=$("$PY" -m tools.validate_station24_s5_crn_stage2 --config "$CONFIG" --field runtime.issue_batch_size)
MEMBER_CHUNK=$("$PY" -m tools.validate_station24_s5_crn_stage2 --config "$CONFIG" --field runtime.member_chunk_size)
CHECKPOINT_STATE=$("$PY" -m tools.validate_station24_s5_crn_stage2 --config "$CONFIG" --field runtime.checkpoint_state)
ENERGY_LIMIT=$("$PY" -m tools.validate_station24_s5_crn_stage2 --config "$CONFIG" --field experiment.energy_score_member_limit)

AVAILABLE_KB=$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')
REQUIRED_KB=$((12 * 1024 * 1024))
echo "DISK_PREFLIGHT available_kb=$AVAILABLE_KB required_kb=$REQUIRED_KB"
(( AVAILABLE_KB >= REQUIRED_KB ))

if [[ ! -s "$ROOT/EXPERIMENT_MANIFEST.json" ]]; then
  cp "$CONFIG" "$ROOT/config_used.yaml"
  "$PY" - "$ROOT/EXPERIMENT_MANIFEST.json" "$CONFIG" "$GATE_REPORT" "$SEED" <<'PY'
import json, pathlib, subprocess, sys
out, config, gate, seed = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
payload={"family":"station24_s5_crn_single_seed_v1","git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"config":config,"gpu_gate_report":gate,"generation_seed":seed,"generation_seed_count":1,"training":"NOT RUN","test":"LOCKED","formal_generation_amp":False,"evaluation":"NOT RUN"}
out.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
PY
fi

generate_one() {
  local NAME=$1 RUN=$2 COUNT=$3 OFFSET=$4 VARIANT=$5
  local OUT="$ROOT/$NAME"
  if [[ -s "$OUT/generation_metadata.json" && -s "$OUT/actual_scenarios_normalized.npy" ]]; then
    echo "GENERATION_REUSED output=$OUT"
    return
  fi
  if [[ -e "$OUT" ]]; then
    mv "$OUT" "${OUT}_incomplete_$(date +%Y%m%d_%H%M%S)"
  fi
  "$PY" generate_station24.py \
    --run-dir "$RUN" --data-path "$DATA" --output-dir "$OUT" \
    --split val --n-samples "$COUNT" --seed "$SEED" \
    --issue-batch-size "$ISSUE_BATCH" --member-chunk-size "$MEMBER_CHUNK" \
    --no-auto-tune-member-chunk --checkpoint-state "$CHECKPOINT_STATE" \
    --result-variant "$VARIANT" --energy-score-member-limit "$ENERGY_LIMIT" \
    --ancestral-noise-mode keyed_full_path --ancestral-noise-seed "$SEED" \
    --ancestral-member-offset "$OFFSET"
}

PHASE=raw500_generation
generate_one raw_n500 "$RAW_RUN" 500 0 "raw_full_path_crn_seed_${SEED}"
PHASE=tail100_generation
generate_one tail_n100 "$TAIL_RUN" 100 400 "full_v2_tail_full_path_crn_seed_${SEED}"

PHASE=generation_integrity
"$PY" - "$ROOT" "$SEED" "$MEMBER_CHUNK" <<'PY'
import json, pathlib, sys
root, seed, chunk = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
for name,n,offset in (("raw_n500",500,0),("tail_n100",100,400)):
    p=json.load(open(root/name/"generation_metadata.json",encoding="utf-8"))
    assert p["split"]=="val" and not p["test_used"]
    assert p["generation_seed"]==seed and p["ancestral_noise_seed"]==seed
    assert p["ancestral_noise_mode"]=="keyed_full_path"
    assert p["ancestral_noise_controls_initial_and_all_reverse_steps"] is True
    assert p["n_samples"]==n and p["member_chunk_size"]==chunk
    assert p["ancestral_member_offset"]==offset
print("SINGLE_SEED_GENERATION_INTEGRITY_PASS")
PY

PHASE=generation_complete_evaluation_not_run
write_status generation_complete
echo "S5_CRN_SINGLE_SEED_GENERATION_COMPLETE root=$ROOT"
echo "EVALUATION=NOT_RUN; run finalize manually after inspecting STATUS.env"

