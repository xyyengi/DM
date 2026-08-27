#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
EXPECTED_BRANCH=${EXPECTED_BRANCH:-experiment/24site-retrieval-conditioned-dual-tail}
PYTHON_BIN=${PYTHON_BIN:-python}
DATA=${DATA:-diffusion_input_station}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_shandong/station24}
LOG_ROOT=${LOG_ROOT:-logs/station24}
SOURCE_PIPELINE_ROOT=${SOURCE_PIPELINE_ROOT:-${1:-}}
BASELINE_RESULT=${BASELINE_RESULT:-${2:-}}
RESUME_PIPELINE_ROOT=${RESUME_PIPELINE_ROOT:-${3:-}}
GEN_SEED=${GEN_SEED:-424242}
SCREEN_MEMBERS=${SCREEN_MEMBERS:-100}
FORMAL_MEMBERS=${FORMAL_MEMBERS:-500}
ENERGY_MEMBERS=${ENERGY_MEMBERS:-80}
ISSUE_BATCH=${ISSUE_BATCH:-2}
MEMBER_CHUNK=${MEMBER_CHUNK:-500}

die() { echo "ERROR: $*" >&2; exit 1; }

record_exit() {
  local code=$?
  trap - EXIT
  local state=completed
  [[ ${code} -eq 0 ]] || state=failed
  printf 'state=%s\npid=%s\nfinished_at=%s\nexit_code=%s\n' \
    "${state}" "${BASHPID}" "$(date --iso-8601=seconds)" "${code}" \
    > "${STATUS_FILE}"
  exit "${code}"
}

launch_background() {
  cd "${REPO_ROOT}"
  mkdir -p "${LOG_ROOT}"
  local stamp log_file pid_file status_file
  stamp=$(date +%Y%m%d_%H%M%S)
  log_file="${LOG_ROOT}/station24_retrieval_dual_tail_${stamp}.log"
  pid_file="${LOG_ROOT}/station24_retrieval_dual_tail_${stamp}.pid"
  status_file="${LOG_ROOT}/station24_retrieval_dual_tail_${stamp}.status"
  nohup setsid env PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    STATION24_RETRIEVAL_DUAL_TAIL_WORKER=1 JOB_STAMP="${stamp}" \
    LOG_FILE="${log_file}" PID_FILE="${pid_file}" STATUS_FILE="${status_file}" \
    EXPECTED_BRANCH="${EXPECTED_BRANCH}" PYTHON_BIN="${PYTHON_BIN}" \
    DATA="${DATA}" OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
    SOURCE_PIPELINE_ROOT="${SOURCE_PIPELINE_ROOT}" \
    BASELINE_RESULT="${BASELINE_RESULT}" RESUME_PIPELINE_ROOT="${RESUME_PIPELINE_ROOT}" \
    GEN_SEED="${GEN_SEED}" SCREEN_MEMBERS="${SCREEN_MEMBERS}" \
    FORMAL_MEMBERS="${FORMAL_MEMBERS}" ENERGY_MEMBERS="${ENERGY_MEMBERS}" \
    ISSUE_BATCH="${ISSUE_BATCH}" MEMBER_CHUNK="${MEMBER_CHUNK}" \
    bash "$0" > "${log_file}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_file}"
  printf 'state=running\npid=%s\nstarted_at=%s\n' \
    "${pid}" "$(date --iso-8601=seconds)" > "${status_file}"
  echo "Started retrieval-conditioned dual-tail experiment"
  echo "PID: ${pid}"
  echo "Log: ${log_file}"
  echo "Status: ${status_file}"
  echo "Monitor: tail -f '${log_file}'"
  echo "Stop entire pipeline: kill -- -\$(cat '${pid_file}')"
}

if [[ "${STATION24_RETRIEVAL_DUAL_TAIL_WORKER:-0}" != "1" ]]; then
  launch_background
  exit 0
fi

trap record_exit EXIT
cd "${REPO_ROOT}"
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] \
  || die "expected branch ${EXPECTED_BRANCH}, got $(git branch --show-current)"
git diff --quiet || die "tracked working-tree changes are present; commit/pull first"
git diff --cached --quiet || die "staged changes are present; commit/pull first"
[[ "${SCREEN_MEMBERS}" -eq 100 ]] || die "screening protocol requires 100 members"
[[ "${FORMAL_MEMBERS}" -eq 500 ]] || die "formal protocol requires 500 members"
[[ "${GEN_SEED}" -eq 424242 ]] || die "generation seed must remain 424242"

for required in train_forecast.npy train_actual.npy train_residual.npy train_fill_mask.npy \
  train_issue_dates.csv val_forecast.npy val_actual.npy val_residual.npy val_fill_mask.npy \
  val_issue_dates.csv station_features.npy station_adjacency.npy station_order.csv \
  export_metadata.json; do
  [[ -f "${DATA}/${required}" ]] || die "missing data artifact ${DATA}/${required}"
done

"${PYTHON_BIN}" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

echo "RETRIEVAL_DUAL_TAIL_PREFLIGHT_START"
"${PYTHON_BIN}" -m py_compile station_retrieval_memory.py station_dataset.py \
  train_station24.py generate_station24.py \
  src/models/station_conditioned_diffusion.py \
  tools/audit_station24_retrieval_dual_tail.py
"${PYTHON_BIN}" -m unittest \
  tests.test_station24_pipeline.Station24ModelTests.test_retrieval_dual_tail_preserves_inherited_paths_and_routes_members

if [[ -z "${SOURCE_PIPELINE_ROOT}" ]]; then
  while IFS= read -r -d '' candidate; do
    if [[ -z "${SOURCE_PIPELINE_ROOT}" || "${candidate}" -nt "${SOURCE_PIPELINE_ROOT}" ]]; then
      SOURCE_PIPELINE_ROOT=${candidate}
    fi
  done < <(find "${OUTPUT_ROOT}" -maxdepth 1 -type d -name 'body_tail_moe_20*' -print0)
fi
[[ -n "${SOURCE_PIPELINE_ROOT}" && -d "${SOURCE_PIPELINE_ROOT}" ]] \
  || die "source body-tail pipeline not found; pass argument 1"
shopt -s nullglob
source_runs=("${SOURCE_PIPELINE_ROOT}"/training/*_station24_body_tail_moe_*_seed2027)
shopt -u nullglob
[[ ${#source_runs[@]} -eq 1 ]] || die "expected one source body-tail run"
SOURCE_RUN=${source_runs[0]}
SOURCE_CHECKPOINT="${SOURCE_RUN}/checkpoints/model_best.pt"
SECONDARY_ADJACENCY="${SOURCE_RUN}/graphs/secondary_adjacency.npy"
[[ -f "${SOURCE_CHECKPOINT}" ]] || die "missing source Raw checkpoint"
[[ -f "${SECONDARY_ADJACENCY}" ]] || die "missing historical secondary graph"

if [[ -z "${BASELINE_RESULT}" ]]; then
  while IFS= read -r -d '' candidate; do
    if [[ -z "${BASELINE_RESULT}" || "${candidate}" -nt "${BASELINE_RESULT}" ]]; then
      BASELINE_RESULT=${candidate}
    fi
  done < <(find "${OUTPUT_ROOT}" -type d -path \
    '*/validation_results/geo_history_actual_body_tail_moe_raw_val_n500_seed424242' -print0)
fi
[[ -n "${BASELINE_RESULT}" && -f "${BASELINE_RESULT}/metrics.json" ]] \
  || die "Raw Body-tail 500-member baseline not found; pass argument 2"

"${PYTHON_BIN}" - "${SOURCE_CHECKPOINT}" "${BASELINE_RESULT}" <<'PY'
import json,sys,torch
source=torch.load(sys.argv[1],map_location="cpu",weights_only=False)
if source.get("condition_variant") != "geo_history_actual_body_tail_moe":
    raise SystemExit("source checkpoint is not Raw Body-tail MoE")
meta=json.load(open(sys.argv[2]+"/generation_metadata.json",encoding="utf-8"))
expected={"condition_variant":"geo_history_actual_body_tail_moe_raw","split":"val","n_samples":500,"generation_seed":424242,"test_used":False}
for key,value in expected.items():
    if meta.get(key) != value: raise SystemExit(f"baseline protocol mismatch {key}")
print("SOURCE_AND_BASELINE_AUDIT_PASSED")
PY

PIPELINE_ROOT=${RESUME_PIPELINE_ROOT:-"${OUTPUT_ROOT}/retrieval_dual_tail_${JOB_STAMP}"}
TRAIN_ROOT="${PIPELINE_ROOT}/training"
RESULT_ROOT="${PIPELINE_ROOT}/validation_results"
mkdir -p "${TRAIN_ROOT}" "${RESULT_ROOT}"

shopt -s nullglob
candidate_runs=("${TRAIN_ROOT}"/*_station24_retrieval_dual_tail_*_seed2027)
shopt -u nullglob
if [[ ${#candidate_runs[@]} -eq 1 && ! -f "${candidate_runs[0]}/training_summary.json" ]]; then
  mkdir -p "${PIPELINE_ROOT}/incomplete_training"
  mv "${candidate_runs[0]}" \
    "${PIPELINE_ROOT}/incomplete_training/$(basename "${candidate_runs[0]}")_${JOB_STAMP}"
  candidate_runs=()
  echo "INCOMPLETE_TRAINING_PRESERVED_AND_RESTARTED"
fi
if [[ ${#candidate_runs[@]} -eq 0 ]]; then
  echo "RETRIEVAL_DUAL_TAIL_TRAINING_START source_state=raw"
  "${PYTHON_BIN}" train_station24.py \
    --config configs/station24_retrieval_conditioned_dual_tail_moe_168h.yaml \
    --data-path "${DATA}" --output-root "${TRAIN_ROOT}" \
    --exp-name "station24_retrieval_dual_tail_${JOB_STAMP}" \
    --secondary-adjacency "${SECONDARY_ADJACENCY}" \
    --initialize-checkpoint "${SOURCE_CHECKPOINT}"
  shopt -s nullglob
  candidate_runs=("${TRAIN_ROOT}"/*_station24_retrieval_dual_tail_*_seed2027)
  shopt -u nullglob
else
  echo "TRAINING_SKIP existing=${candidate_runs[0]}"
fi
[[ ${#candidate_runs[@]} -eq 1 ]] || die "expected exactly one candidate run"
CANDIDATE_RUN=${candidate_runs[0]}
[[ -f "${CANDIDATE_RUN}/checkpoints/model_best.pt" ]] || die "candidate checkpoint missing"

generate_if_missing() {
  local members=$1 output=$2 variant=$3
  if [[ -f "${output}/metrics.json" ]]; then
    echo "GENERATION_SKIP members=${members} output=${output}"
    return
  fi
  if [[ -d "${output}" ]]; then
    mv "${output}" "${output}.incomplete_${JOB_STAMP}"
  fi
  echo "GENERATION_START members=${members} checkpoint=raw"
  "${PYTHON_BIN}" generate_station24.py \
    --run-dir "${CANDIDATE_RUN}" --data-path "${DATA}" \
    --output-dir "${output}" --split val --n-samples "${members}" \
    --seed "${GEN_SEED}" --issue-batch-size "${ISSUE_BATCH}" \
    --member-chunk-size "${MEMBER_CHUNK}" --auto-tune-member-chunk \
    --energy-score-member-limit "${ENERGY_MEMBERS}" --checkpoint-state raw \
    --result-variant "${variant}"
}

SCREEN_RESULT="${RESULT_ROOT}/retrieval_dual_tail_raw_val_n${SCREEN_MEMBERS}_seed${GEN_SEED}"
FORMAL_RESULT="${RESULT_ROOT}/retrieval_dual_tail_raw_val_n${FORMAL_MEMBERS}_seed${GEN_SEED}"
generate_if_missing "${SCREEN_MEMBERS}" "${SCREEN_RESULT}" retrieval_dual_tail_raw_screen

echo "SCREENING_GATE_START"
"${PYTHON_BIN}" - "${BASELINE_RESULT}" "${SCREEN_RESULT}" \
  "${PIPELINE_ROOT}/screening_gate.json" <<'PY'
import json,sys
from pathlib import Path
b=json.load(open(Path(sys.argv[1])/"metrics.json",encoding="utf-8"))
c=json.load(open(Path(sys.argv[2])/"metrics.json",encoding="utf-8"))
def find(d,*paths):
    for path in paths:
        x=d
        try:
            for key in path: x=x[key]
            return float(x)
        except (KeyError,TypeError): pass
    raise KeyError(paths)
bcrps=find(b,("aggregate_mw","wind","crps"))
ccrps=find(c,("aggregate_mw","wind","crps"))
bcov=find(b,("aggregate_mw","wind","coverage_90"))
ccov=find(c,("aggregate_mw","wind","coverage_90"))
route=float(c["run"]["mismatch_member_fraction"])
checks={"aggregate_wind_crps_not_catastrophic":ccrps<=1.10*bcrps,
        "aggregate_wind_coverage_not_catastrophic":ccov>=bcov-0.05,
        "mismatch_route_nonzero_and_bounded":0.005<=route<=0.35}
record={"baseline_crps":bcrps,"screen_crps":ccrps,"baseline_coverage90":bcov,
        "screen_coverage90":ccov,"mismatch_member_fraction":route,"checks":checks,
        "passed":all(checks.values())}
Path(sys.argv[3]).write_text(json.dumps(record,indent=2),encoding="utf-8")
print(json.dumps(record))
PY

SCREEN_PASSED=$("${PYTHON_BIN}" - "${PIPELINE_ROOT}/screening_gate.json" <<'PY'
import json,sys
print("1" if json.load(open(sys.argv[1],encoding="utf-8"))["passed"] else "0")
PY
)
if [[ "${SCREEN_PASSED}" != "1" ]]; then
  ARCHIVE="${OUTPUT_ROOT}/station24_$(basename "${PIPELINE_ROOT}")_screen_rejected.tar.gz"
  tar -czf "${ARCHIVE}" -C "$(dirname "${PIPELINE_ROOT}")" "$(basename "${PIPELINE_ROOT}")"
  RESULT_FILE="${LOG_FILE%.log}.results.env"
  {
    echo "finished_at=$(date --iso-8601=seconds)"
    echo "SCREEN_RESULT=${SCREEN_RESULT}"
    echo "SCREENING_REJECTED=true"
    echo "ARCHIVE=${ARCHIVE}"
  } > "${RESULT_FILE}"
  cat "${RESULT_FILE}"
  exit 0
fi

generate_if_missing "${FORMAL_MEMBERS}" "${FORMAL_RESULT}" retrieval_conditioned_dual_tail_moe_raw

preserve_partial_dir() {
  local directory=$1 sentinel=$2
  if [[ -d "${directory}" && ! -f "${directory}/${sentinel}" ]]; then
    mv "${directory}" "${directory}.incomplete_${JOB_STAMP}"
  fi
}

COMPARISON="${PIPELINE_ROOT}/comparisons/body_tail_raw_vs_retrieval_dual_tail"
preserve_partial_dir "${COMPARISON}" comparison_summary.csv
if [[ ! -f "${COMPARISON}/comparison_summary.csv" ]]; then
  "${PYTHON_BIN}" tools/compare_station24_multiscale_2a.py \
    "${BASELINE_RESULT}" "${FORMAL_RESULT}" --data-path "${DATA}" \
    --output-dir "${COMPARISON}" \
    --baseline-variant geo_history_actual_body_tail_moe_raw \
    --candidate-variant retrieval_conditioned_dual_tail_moe_raw \
    --baseline-label "Raw Body-tail" --candidate-label "Retrieval Dual-tail" \
    --baseline-spatial-levels bottleneck --candidate-spatial-levels bottleneck \
    --baseline-parallel-levels encoder_0 --candidate-parallel-levels encoder_0 \
    --baseline-parallel-adjacency fixed --candidate-parallel-adjacency fixed \
    --title "Raw Body-tail versus retrieval-conditioned dual-tail" \
    --figure-prefix body_tail_vs_retrieval_dual_tail
fi

AUDIT="${PIPELINE_ROOT}/retrieval_dual_tail_audit"
preserve_partial_dir "${AUDIT}" retrieval_dual_tail_audit.json
if [[ ! -f "${AUDIT}/retrieval_dual_tail_audit.json" ]]; then
  "${PYTHON_BIN}" tools/audit_station24_retrieval_dual_tail.py \
    --result-dir "${FORMAL_RESULT}" --data-path "${DATA}" --output-dir "${AUDIT}"
fi
TAIL="${PIPELINE_ROOT}/extreme_wind_tail/body_tail_vs_retrieval_dual_tail"
preserve_partial_dir "${TAIL}" extreme_wind_tail_summary.json
if [[ ! -f "${TAIL}/extreme_wind_tail_summary.json" ]]; then
  "${PYTHON_BIN}" tools/plot_station24_extreme_tail.py \
    --baseline "${BASELINE_RESULT}" --candidate "${FORMAL_RESULT}" \
    --data-path "${DATA}" --output-dir "${TAIL}" --top-issues 5 \
    --baseline-label "Raw Body-tail" --candidate-label "Retrieval Dual-tail"
fi
TIMING="${PIPELINE_ROOT}/wind_event_timing/body_tail_vs_retrieval_dual_tail"
preserve_partial_dir "${TIMING}" timing_diagnostics.md
if [[ ! -f "${TIMING}/timing_diagnostics.md" ]]; then
  "${PYTHON_BIN}" tools/diagnose_station24_wind_event_timing.py \
    "${BASELINE_RESULT}" "${FORMAL_RESULT}" --data-path "${DATA}" \
    --output-dir "${TIMING}" --baseline-variant geo_history_actual_body_tail_moe_raw \
    --candidate-variant retrieval_conditioned_dual_tail_moe_raw \
    --baseline-label "Raw Body-tail" --candidate-label "Retrieval Dual-tail"
fi

ARCHIVE="${OUTPUT_ROOT}/station24_$(basename "${PIPELINE_ROOT}").tar.gz"
tar -czf "${ARCHIVE}" -C "$(dirname "${PIPELINE_ROOT}")" "$(basename "${PIPELINE_ROOT}")"
RESULT_FILE="${LOG_FILE%.log}.results.env"
{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT}"
  echo "BASELINE_RESULT=${BASELINE_RESULT}"
  echo "CANDIDATE_RUN=${CANDIDATE_RUN}"
  echo "SCREEN_RESULT=${SCREEN_RESULT}"
  echo "FORMAL_RESULT=${FORMAL_RESULT}"
  echo "ARCHIVE=${ARCHIVE}"
  echo "ALL_RETRIEVAL_DUAL_TAIL_COMPLETED"
} > "${RESULT_FILE}"
cat "${RESULT_FILE}"
