#!/usr/bin/env bash
# Checkpoint-frozen generation-seed stability: preflight -> generate -> merge -> evaluate -> summarize -> archive.
set -Eeuo pipefail
cd /root/autodl-tmp/DM
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dm_env
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

ROOT=${1:-outputs_shandong/station24/generation_seed_stability_$(date +%Y%m%d_%H%M%S)}
RAW_ROOT=${RAW_ROOT:-outputs_shandong/station24/body_tail_moe_20260824_191036}
FULL_ROOT=${FULL_ROOT:-outputs_shandong/station24/independent_joint_tail_v2_event_balanced_20260910_200653}
LIGHT_ROOT=${LIGHT_ROOT:-outputs_shandong/station24/lightweight_joint_tail_v2_fair_20260923_194256}
RAW_424242=${RAW_424242:-outputs_shandong/station24/body_tail_moe_raw_inference_20260824_224151/validation_results/geo_history_actual_body_tail_moe_raw_val_n500_seed424242}
FULL_424242=${FULL_424242:-$FULL_ROOT/mixture_body400_tail100_n500}
LIGHT_424242=${LIGHT_424242:-$LIGHT_ROOT/mixture_body400_tail100_n500}
NEW_SEEDS=(${GENERATION_SEEDS:-271828 314159})
JOB=$(date +%Y%m%d_%H%M%S)
LOG=${STABILITY_LOG:-logs/station24/generation_seed_stability_${JOB}.log}
STATUS=${STABILITY_STATUS:-logs/station24/generation_seed_stability_${JOB}.status}
PHASE=initializing
mkdir -p logs/station24 "$ROOT"
if [[ "${STABILITY_LOG_ATTACHED:-0}" != 1 ]]; then
  exec > >(tee -a "$LOG") 2>&1
fi
write_status() {
  printf 'state=%s\nphase=%s\nroot=%s\nlog=%s\nupdated_at=%s\n' \
    "$1" "$PHASE" "$ROOT" "$LOG" "$(date --iso-8601=seconds)" > "$STATUS"
}
on_error() {
  code=$?
  write_status failed
  echo "GENERATION_SEED_STABILITY_FAILED phase=$PHASE exit_code=$code root=$ROOT log=$LOG" >&2
  exit "$code"
}
trap on_error ERR
write_status running

find_run() {
  local root=$1
  if [[ -f "$root/train_run.txt" ]]; then
    cat "$root/train_run.txt"
    return
  fi
  local matches=()
  mapfile -t matches < <(find "$root" -type f -path '*/checkpoints/model_best.pt' -printf '%h\n' | sed 's#/checkpoints$##' | sort)
  [[ ${#matches[@]} -eq 1 ]] || { echo "expected one run under $root, found ${#matches[@]}" >&2; return 1; }
  printf '%s\n' "${matches[0]}"
}

RAW_RUN=$(find_run "$RAW_ROOT/training")
FULL_RUN=$(find_run "$FULL_ROOT")
LIGHT_RUN=$(find_run "$LIGHT_ROOT")
for required in \
  "$RAW_RUN/checkpoints/model_best.pt" "$FULL_RUN/checkpoints/model_best.pt" "$LIGHT_RUN/checkpoints/model_best.pt" \
  "$RAW_424242/metrics.json" "$FULL_424242/metrics.json" "$LIGHT_424242/metrics.json"; do
  test -f "$required" || { echo "missing required artifact: $required" >&2; false; }
done
AVAILABLE_KB=$(df -Pk "$(dirname "$ROOT")" | awk 'NR==2 {print $4}')
REQUIRED_KB=$((14 * 1024 * 1024))
echo "DISK_PREFLIGHT available_kb=$AVAILABLE_KB required_kb=$REQUIRED_KB"
[[ "$AVAILABLE_KB" -ge "$REQUIRED_KB" ]] || {
  echo "at least 14 GiB free space is required for two seeds plus the complete archive" >&2
  false
}
printf '%s\n' "$RAW_RUN" > "$ROOT/raw_run.txt"
printf '%s\n' "$FULL_RUN" > "$ROOT/full_v2_run.txt"
printf '%s\n' "$LIGHT_RUN" > "$ROOT/lightweight_run.txt"

PHASE=protocol_preflight
write_status running
python -m tools.audit_station24_generation_seed_stability \
  --raw-run "$RAW_RUN" --full-run "$FULL_RUN" --lightweight-run "$LIGHT_RUN" \
  --raw-424242 "$RAW_424242" --full-424242 "$FULL_424242" --lightweight-424242 "$LIGHT_424242" \
  --seeds "${NEW_SEEDS[@]}" --output "$ROOT/PREFLIGHT.json"
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for formal generation")
x=torch.randn(1024,1024,device="cuda")
y=x@x.T
if not torch.isfinite(y).all():
    raise SystemExit("nonfinite CUDA probe")
print("CUDA_GENERATION_GATE_PASS",torch.__version__,torch.cuda.get_device_name(0),flush=True)
PY

PHASE=generation_smoke
write_status running
SMOKE="$ROOT/generation_smoke"
SMOKE_EXISTING=$(find "$ROOT" -maxdepth 1 -type d -name 'generation_smoke*' -exec test -f '{}/COMPLETE' \; -print 2>/dev/null | sort | tail -n 1)
[[ -z "$SMOKE_EXISTING" ]] || SMOKE="$SMOKE_EXISTING"
if [[ ! -f "$SMOKE/COMPLETE" ]]; then
  if [[ -e "$SMOKE" ]]; then
    SMOKE="$ROOT/generation_smoke_retry_${JOB}"
  fi
  mkdir -p "$SMOKE"
  python generate_station24.py --run-dir "$RAW_RUN" --data-path diffusion_input_station \
    --output-dir "$SMOKE/raw_n2" --split val --n-samples 2 --seed 99001 \
    --issue-batch-size 1 --member-chunk-size 2 --no-auto-tune-member-chunk \
    --checkpoint-state raw --result-variant geo_history_actual_body_tail_moe_raw
  python generate_station24.py --run-dir "$FULL_RUN" --data-path diffusion_input_station \
    --output-dir "$SMOKE/full_n2" --split val --n-samples 2 --seed 99001 \
    --issue-batch-size 1 --member-chunk-size 2 --no-auto-tune-member-chunk \
    --checkpoint-state raw --result-variant independent_joint_tail_v2_event_balanced_raw
  python generate_station24.py --run-dir "$LIGHT_RUN" --data-path diffusion_input_station \
    --output-dir "$SMOKE/light_n2" --split val --n-samples 2 --seed 99001 \
    --issue-batch-size 1 --member-chunk-size 2 --no-auto-tune-member-chunk \
    --checkpoint-state raw --result-variant lightweight_joint_tail_v2_fair
  touch "$SMOKE/COMPLETE"
fi

choose_output() {
  local preferred=$1 sentinel=$2
  local parent base existing
  parent=$(dirname "$preferred")
  base=$(basename "$preferred")
  existing=$(find "$parent" -maxdepth 1 -type d -name "${base}*" -exec test -f "{}/$sentinel" \; -print 2>/dev/null | sort | tail -n 1)
  if [[ -n "$existing" ]]; then
    printf '%s\n' "$existing"
  elif [[ ! -e "$preferred" ]]; then
    printf '%s\n' "$preferred"
  else
    printf '%s_retry_%s\n' "$preferred" "$JOB"
  fi
}

evaluate_pair() {
  local seed=$1 baseline=$2 candidate=$3 candidate_run=$4 slug=$5 label=$6
  local base_variant candidate_variant evaluation
  evaluation="$ROOT/seed_${seed}/evaluation_${slug}"
  mkdir -p "$evaluation"
  base_variant=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]+"/metrics.json"))["run"]["condition_variant"])' "$baseline")
  candidate_variant=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]+"/metrics.json"))["run"]["condition_variant"])' "$candidate")
  if ! find "$evaluation" -maxdepth 1 -type d -name 'comparisons*' -exec test -f '{}/comparison_report.md' \; -print 2>/dev/null | grep -q .; then
    local out
    out=$(choose_output "$evaluation/comparisons" comparison_report.md)
    python tools/compare_station24_multiscale_2a.py "$baseline" "$candidate" --output-dir "$out" \
      --baseline-variant "$base_variant" --candidate-variant "$candidate_variant" \
      --baseline-parallel-levels encoder_0 --candidate-parallel-levels encoder_0 \
      --baseline-label "Raw body-tail" --candidate-label "$label" \
      --candidate-spatial-levels bottleneck --title "Raw body-tail vs $label"
  fi
  if ! find "$evaluation" -maxdepth 1 -type d -name 'continuous_event_evaluation*' -exec test -f '{}/continuous_event_three_standard_summary.csv' \; -print 2>/dev/null | grep -q .; then
    local out
    out=$(choose_output "$evaluation/continuous_event_evaluation" continuous_event_three_standard_summary.csv)
    python tools/evaluate_station24_jstd_events.py --baseline "$baseline" --candidate "$candidate" \
      --candidate-run "$candidate_run" --data-path diffusion_input_station --output-dir "$out" \
      --baseline-label "Raw body-tail" --candidate-label "$label"
  fi
  local event_dir
  event_dir=$(find "$evaluation" -maxdepth 1 -type d -name 'continuous_event_evaluation*' -exec test -f '{}/continuous_event_three_standard_summary.csv' ';' -print | sort | tail -n 1)
  test -n "$event_dir"
  if ! find "$evaluation" -maxdepth 1 -type d -name 'joint_wind_solar_evaluation*' -exec test -f '{}/ordinary_comparison.csv' \; -print 2>/dev/null | grep -q .; then
    local out
    out=$(choose_output "$evaluation/joint_wind_solar_evaluation" ordinary_comparison.csv)
    python tools/evaluate_station24_diffusion_ts.py --baseline "$baseline" --candidate "$candidate" \
      --data diffusion_input_station --output "$out" --candidate-label "$label"
  fi
  local joint_dir
  joint_dir=$(find "$evaluation" -maxdepth 1 -type d -name 'joint_wind_solar_evaluation*' -exec test -f '{}/ordinary_comparison.csv' ';' -print | sort | tail -n 1)
  test -n "$joint_dir"
  if ! find "$evaluation" -maxdepth 1 -type d -name 'extreme_wind_tail*' -exec test -f '{}/extreme_wind_tail_summary.json' \; -print 2>/dev/null | grep -q .; then
    local out
    out=$(choose_output "$evaluation/extreme_wind_tail" extreme_wind_tail_summary.json)
    python tools/plot_station24_extreme_tail.py --baseline "$baseline" --candidate "$candidate" \
      --data-path diffusion_input_station --output-dir "$out" --top-issues 5 \
      --baseline-label "Raw body-tail" --candidate-label "$label"
  fi
  if ! find "$evaluation" -maxdepth 1 -type d -name 'wind_event_timing*' -exec test -f '{}/timing_diagnostics.md' \; -print 2>/dev/null | grep -q .; then
    local out
    out=$(choose_output "$evaluation/wind_event_timing" timing_diagnostics.md)
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2 \
      python tools/diagnose_station24_wind_event_timing.py "$baseline" "$candidate" \
        --data-path diffusion_input_station --output-dir "$out" \
        --baseline-variant "$base_variant" --candidate-variant "$candidate_variant" \
        --baseline-label "Raw body-tail" --candidate-label "$label"
  fi
  if ! find "$evaluation/representative_joint_plots" -maxdepth 1 -type f -name '*.png' -print -quit 2>/dev/null | grep -q .; then
    python tools/plot_station24_independent_tail_mixture.py --result "$candidate" \
      --data-path diffusion_input_station --output-dir "$evaluation/representative_joint_plots" --top-issues 5
  fi
  python tools/summarize_station24_independent_tail_v2.py --baseline "$baseline" --candidate "$candidate" \
    --event-dir "$event_dir" --joint-dir "$joint_dir" --output "$evaluation/RESULT_SUMMARY.md" \
    --candidate-label "$label"
  EVAL_EVENT_DIR="$event_dir"
  EVAL_JOINT_DIR="$joint_dir"
}

MANIFEST="$ROOT/generation_seed_manifest.json"
printf '{"protocol":"station24_generation_seed_stability_v1","historical_seed":424242,"new_seeds":[' > "$MANIFEST"
printf '%s' "$(IFS=,; echo "${NEW_SEEDS[*]}")" >> "$MANIFEST"
printf '],"runs":[]}\n' >> "$MANIFEST"

for seed in 424242 "${NEW_SEEDS[@]}"; do
  PHASE="seed_${seed}_generation"
  write_status running
  SEED_ROOT="$ROOT/seed_${seed}"
  mkdir -p "$SEED_ROOT"
  if [[ "$seed" == 424242 ]]; then
    RAW_RESULT="$RAW_424242"
    FULL_MIXTURE="$FULL_424242"
    LIGHT_MIXTURE="$LIGHT_424242"
  else
    RAW_RESULT=$(choose_output "$SEED_ROOT/raw_n500" metrics.json)
    if [[ ! -f "$RAW_RESULT/metrics.json" ]]; then
      python generate_station24.py --run-dir "$RAW_RUN" --data-path diffusion_input_station \
        --output-dir "$RAW_RESULT" --split val --n-samples 500 --seed "$seed" \
        --issue-batch-size 1 --member-chunk-size 500 --no-auto-tune-member-chunk \
        --checkpoint-state raw --result-variant geo_history_actual_body_tail_moe_raw \
        --energy-score-member-limit 80
    fi
    FULL_TAIL=$(choose_output "$SEED_ROOT/full_tail_n100" metrics.json)
    if [[ ! -f "$FULL_TAIL/metrics.json" ]]; then
      python generate_station24.py --run-dir "$FULL_RUN" --data-path diffusion_input_station \
        --output-dir "$FULL_TAIL" --split val --n-samples 100 --seed "$seed" \
        --issue-batch-size 2 --member-chunk-size 100 --no-auto-tune-member-chunk \
        --checkpoint-state raw --result-variant independent_joint_tail_v2_event_balanced_raw \
        --energy-score-member-limit 80
    fi
    LIGHT_TAIL=$(choose_output "$SEED_ROOT/lightweight_tail_n100" metrics.json)
    if [[ ! -f "$LIGHT_TAIL/metrics.json" ]]; then
      python generate_station24.py --run-dir "$LIGHT_RUN" --data-path diffusion_input_station \
        --output-dir "$LIGHT_TAIL" --split val --n-samples 100 --seed "$seed" \
        --issue-batch-size 2 --member-chunk-size 100 --no-auto-tune-member-chunk \
        --checkpoint-state raw --result-variant lightweight_joint_tail_v2_fair \
        --energy-score-member-limit 80
    fi
    FULL_MIXTURE=$(choose_output "$SEED_ROOT/full_mixture_body400_tail100_n500" metrics.json)
    if [[ ! -f "$FULL_MIXTURE/metrics.json" ]]; then
      python tools/merge_station24_independent_tail_members.py --body-results "$RAW_RESULT" --body-member-limit 400 \
        --tail-results "$FULL_TAIL" --tail-member-limit 100 --output-dir "$FULL_MIXTURE" \
        --data-path diffusion_input_station --energy-score-member-limit 80 \
        --condition-variant independent_joint_tail_v2_event_balanced_mixture \
        --family fixed_quota_independent_joint_tail_v2_event_balanced
    fi
    LIGHT_MIXTURE=$(choose_output "$SEED_ROOT/lightweight_mixture_body400_tail100_n500" metrics.json)
    if [[ ! -f "$LIGHT_MIXTURE/metrics.json" ]]; then
      python tools/merge_station24_independent_tail_members.py --body-results "$RAW_RESULT" --body-member-limit 400 \
        --tail-results "$LIGHT_TAIL" --tail-member-limit 100 --output-dir "$LIGHT_MIXTURE" \
        --data-path diffusion_input_station --energy-score-member-limit 80 \
        --condition-variant lightweight_joint_tail_v2_fair_mixture \
        --family fixed_quota_lightweight_joint_tail_v2
    fi
  fi
  PHASE="seed_${seed}_full_v2_evaluation"
  write_status running
  evaluate_pair "$seed" "$RAW_RESULT" "$FULL_MIXTURE" "$FULL_RUN" full_v2 "Full Independent V2"
  FULL_EVENT_DIR="$EVAL_EVENT_DIR"
  FULL_JOINT_DIR="$EVAL_JOINT_DIR"
  PHASE="seed_${seed}_lightweight_evaluation"
  write_status running
  evaluate_pair "$seed" "$RAW_RESULT" "$LIGHT_MIXTURE" "$LIGHT_RUN" lightweight "Lightweight Joint Tail"
  LIGHT_EVENT_DIR="$EVAL_EVENT_DIR"
  LIGHT_JOINT_DIR="$EVAL_JOINT_DIR"
  python - "$MANIFEST" "$seed" "$RAW_RESULT" "$FULL_MIXTURE" "$LIGHT_MIXTURE" \
    "$FULL_EVENT_DIR" "$FULL_JOINT_DIR" "$LIGHT_EVENT_DIR" "$LIGHT_JOINT_DIR" <<'PY'
import json,sys
path,seed,raw,full,light,fe,fj,le,lj=sys.argv[1:]
data=json.load(open(path,encoding="utf-8"))
data["runs"].append({"seed":int(seed),"models":{
 "Raw":{"result_dir":raw,"event_dir":fe,"joint_dir":fj,"event_label":"Raw body-tail","joint_label":"Raw body-tail"},
 "Full Independent V2":{"result_dir":full,"event_dir":fe,"joint_dir":fj,"event_label":"Full Independent V2","joint_label":"Full Independent V2"},
 "Lightweight Joint Tail":{"result_dir":light,"event_dir":le,"joint_dir":lj,"event_label":"Lightweight Joint Tail","joint_label":"Lightweight Joint Tail"}}})
open(path,"w",encoding="utf-8").write(json.dumps(data,ensure_ascii=False,indent=2))
PY
done

PHASE=cross_seed_summary
write_status running
SUMMARY="$ROOT/cross_seed_summary"
SUMMARY=$(choose_output "$SUMMARY" GENERATION_SEED_STABILITY_REPORT.md)
if [[ ! -f "$SUMMARY/GENERATION_SEED_STABILITY_REPORT.md" ]]; then
  python -m tools.summarize_station24_generation_seed_stability --manifest "$MANIFEST" --output-dir "$SUMMARY"
fi

PHASE=archive
write_status running
ARCHIVE="${ROOT}_completed_${JOB}.tar.gz"
REPORT_ARCHIVE="${ROOT}_reports_${JOB}.tar.gz"
tar -czf "$ARCHIVE" -C "$(dirname "$ROOT")" "$(basename "$ROOT")"
SUMMARY_RELATIVE=${SUMMARY#"$ROOT"/}
tar -czf "$REPORT_ARCHIVE" -C "$ROOT" PREFLIGHT.json generation_seed_manifest.json "$SUMMARY_RELATIVE" \
  $(find "$ROOT" -type f \( -name '*.md' -o -name '*.csv' -o -name '*.json' -o -name '*.png' \) -printf '%P\n')
test -s "$ARCHIVE"
test -s "$REPORT_ARCHIVE"
PHASE=complete
write_status complete
echo "GENERATION_SEED_STABILITY_COMPLETE root=$ROOT"
echo "ARCHIVE=$ARCHIVE"
echo "REPORT_ARCHIVE=$REPORT_ARCHIVE"
