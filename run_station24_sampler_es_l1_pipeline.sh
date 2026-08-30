#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
EXPECTED_BRANCH=${EXPECTED_BRANCH:-experiment/24site-sampler-proper-score-tail-ft}
PYTHON_BIN=${PYTHON_BIN:-python}
DATA=${DATA:-diffusion_input_station}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_shandong/station24}
LOG_ROOT=${LOG_ROOT:-logs/station24}
SOURCE_RAW_RESULT=${SOURCE_RAW_RESULT:-}
SEED=${SEED:-2027}
NSAMPLES=${NSAMPLES:-500}
GEN_SEED=${GEN_SEED:-424242}
ENERGY_MEMBERS=${ENERGY_MEMBERS:-80}
ISSUE_BATCH=${ISSUE_BATCH:-1}
MEMBER_CHUNK=${MEMBER_CHUNK:-10}

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
  log_file="${LOG_ROOT}/station24_sampler_es_l1_${stamp}.log"
  pid_file="${LOG_ROOT}/station24_sampler_es_l1_${stamp}.pid"
  status_file="${LOG_ROOT}/station24_sampler_es_l1_${stamp}.status"
  nohup setsid env \
    PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 \
    STATION24_SAMPLER_ES_L1_INTERNAL_WORKER=1 JOB_STAMP="${stamp}" \
    LOG_FILE="${log_file}" PID_FILE="${pid_file}" STATUS_FILE="${status_file}" \
    EXPECTED_BRANCH="${EXPECTED_BRANCH}" PYTHON_BIN="${PYTHON_BIN}" \
    DATA="${DATA}" OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
    SOURCE_RAW_RESULT="${SOURCE_RAW_RESULT}" SEED="${SEED}" \
    NSAMPLES="${NSAMPLES}" GEN_SEED="${GEN_SEED}" \
    ENERGY_MEMBERS="${ENERGY_MEMBERS}" ISSUE_BATCH="${ISSUE_BATCH}" \
    MEMBER_CHUNK="${MEMBER_CHUNK}" \
    bash "$0" > "${log_file}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_file}"
  printf 'state=running\npid=%s\nstarted_at=%s\n' \
    "${pid}" "$(date --iso-8601=seconds)" > "${status_file}"
  echo "Started Station24 sampler Energy Score L1"
  echo "PID: ${pid}"
  echo "Log: ${log_file}"
  echo "Status: ${status_file}"
  echo "Monitor: tail -f '${log_file}'"
  echo "Stop entire pipeline: kill -- -\$(cat '${pid_file}')"
}

if [[ "${STATION24_SAMPLER_ES_L1_INTERNAL_WORKER:-0}" != "1" ]]; then
  launch_background
  exit 0
fi

trap record_exit EXIT
cd "${REPO_ROOT}"
command -v git >/dev/null || die "git is unavailable"
command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
command -v "${PYTHON_BIN}" >/dev/null || die "${PYTHON_BIN} is unavailable"
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] \
  || die "expected branch ${EXPECTED_BRANCH}, got $(git branch --show-current)"
git diff --quiet || die "tracked working-tree changes are present; commit/pull first"
git diff --cached --quiet || die "staged changes are present; commit/pull first"
[[ "${NSAMPLES}" -eq 500 ]] || die "formal comparison requires 500 members"
[[ "${GEN_SEED}" -eq 424242 ]] || die "generation seed must remain 424242"

for required in \
  train_forecast.npy train_actual.npy train_residual.npy train_time_mark.npy \
  train_lead_mark.npy train_fill_mask.npy train_issue_dates.csv \
  val_forecast.npy val_actual.npy val_residual.npy val_time_mark.npy \
  val_lead_mark.npy val_fill_mask.npy val_issue_dates.csv \
  station_features.npy station_adjacency.npy station_order.csv export_metadata.json; do
  [[ -f "${DATA}/${required}" ]] || die "missing data artifact ${DATA}/${required}"
done

CONFIG=configs/station24_body_tail_sampler_es_l1_168h.yaml
[[ -f "${CONFIG}" ]] || die "missing L1 configuration ${CONFIG}"

"${PYTHON_BIN}" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

echo "SAMPLER_ES_L1_UNIT_TEST_START"
"${PYTHON_BIN}" -m unittest discover -s tests \
  -p 'test_station24_pipeline.py' -k sampler_energy_score

if [[ -z "${SOURCE_RAW_RESULT}" ]]; then
  while IFS= read -r -d '' candidate; do
    if [[ -z "${SOURCE_RAW_RESULT}" || "${candidate}" -nt "${SOURCE_RAW_RESULT}" ]]; then
      SOURCE_RAW_RESULT=${candidate}
    fi
  done < <(find "${OUTPUT_ROOT}" -type d \
    -path '*/body_tail_moe_raw_inference_*/validation_results/geo_history_actual_body_tail_moe_raw_val_n500_seed424242' \
    -print0)
fi
[[ -n "${SOURCE_RAW_RESULT}" && -d "${SOURCE_RAW_RESULT}" ]] \
  || die "Raw body-tail 500-member result not found; set SOURCE_RAW_RESULT"

readarray -t source_info < <("${PYTHON_BIN}" - \
  "${SOURCE_RAW_RESULT}" "${NSAMPLES}" "${GEN_SEED}" <<'PY'
import json
import sys
from pathlib import Path

result = Path(sys.argv[1])
meta = json.loads((result / "generation_metadata.json").read_text(encoding="utf-8"))
expected = {
    "condition_variant": "geo_history_actual_body_tail_moe_raw",
    "split": "val",
    "n_samples": int(sys.argv[2]),
    "generation_seed": int(sys.argv[3]),
    "test_used": False,
    "checkpoint_state_source": "raw",
}
for key, value in expected.items():
    if meta.get(key) != value:
        raise SystemExit(f"source mismatch {key}: {meta.get(key)!r} != {value!r}")
print(meta["run_dir"])
PY
)
SOURCE_RUN=${source_info[0]}
SOURCE_CHECKPOINT="${SOURCE_RUN}/checkpoints/model_best.pt"
SECONDARY_GRAPH="${SOURCE_RUN}/graphs/secondary_adjacency.npy"
[[ -f "${SOURCE_CHECKPOINT}" ]] || die "missing source checkpoint ${SOURCE_CHECKPOINT}"
[[ -f "${SECONDARY_GRAPH}" ]] || die "missing source secondary graph ${SECONDARY_GRAPH}"

echo "SAMPLER_ES_L1_PREFLIGHT_START"
"${PYTHON_BIN}" - "${SOURCE_RUN}/config_used.yaml" "${CONFIG}" \
  "${SOURCE_CHECKPOINT}" <<'PY'
import copy
import sys
import torch
import yaml

source = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
candidate = yaml.safe_load(open(sys.argv[2], encoding="utf-8"))
if source["target"] != candidate["target"] or source["data"] != candidate["data"]:
    raise SystemExit("L1 changed the data or residual target")
allowed = {
    "train_sampler_energy_score_only", "sampler_energy_score_weight",
    "sampler_energy_score_members", "sampler_energy_score_steps",
    "sampler_energy_score_backprop_steps",
    "sampler_energy_score_max_issues_per_batch",
    "sampler_energy_score_route_temperature",
    "event_x0_magnitude_loss_weight", "event_x0_timing_loss_weight",
    "event_x0_sync_loss_weight",
}
source_model = copy.deepcopy(source["model"])
candidate_model = copy.deepcopy(candidate["model"])
for key in allowed:
    source_model.pop(key, None)
    candidate_model.pop(key, None)
if source_model != candidate_model:
    raise SystemExit("L1 changed model fields outside its proper-score settings")
checkpoint = torch.load(sys.argv[3], map_location="cpu", weights_only=False)
if checkpoint.get("condition_variant") != "geo_history_actual_body_tail_moe":
    raise SystemExit("source checkpoint variant mismatch")
if "model_state_dict" not in checkpoint:
    raise SystemExit("source checkpoint lacks Raw parameters")
print("SAMPLER_ES_L1_PREFLIGHT_PASSED body_source=raw third_expert=false")
PY

PIPELINE_ROOT="${OUTPUT_ROOT}/sampler_es_l1_${JOB_STAMP}"
TRAIN_ROOT="${PIPELINE_ROOT}/training"
RESULT_ROOT="${PIPELINE_ROOT}/validation_results"
RESULT="${RESULT_ROOT}/geo_history_actual_body_tail_sampler_es_l1_val_n${NSAMPLES}_seed${GEN_SEED}"
COMPARISON="${PIPELINE_ROOT}/comparisons/raw_body_tail_vs_sampler_es_l1"
TIMING="${PIPELINE_ROOT}/wind_event_timing/raw_body_tail_vs_sampler_es_l1"
ATTRIBUTION="${PIPELINE_ROOT}/forecast_event_attribution/raw_body_tail_vs_sampler_es_l1"
TAIL="${PIPELINE_ROOT}/extreme_wind_tail/raw_body_tail_vs_sampler_es_l1"
AUDIT="${PIPELINE_ROOT}/sampler_es_l1_audit"
mkdir -p "${TRAIN_ROOT}" "${RESULT_ROOT}" \
  "$(dirname "${COMPARISON}")" "$(dirname "${TIMING}")" \
  "$(dirname "${ATTRIBUTION}")" "$(dirname "${TAIL}")"

EXP_NAME="station24_sampler_es_l1_${JOB_STAMP}"
echo "TRAIN_START variant=geo_history_actual_body_tail_sampler_es_l1"
"${PYTHON_BIN}" train_station24.py \
  --config "${CONFIG}" --data-path "${DATA}" \
  --secondary-adjacency "${SECONDARY_GRAPH}" \
  --initialize-checkpoint "${SOURCE_CHECKPOINT}" \
  --output-root "${TRAIN_ROOT}" --exp-name "${EXP_NAME}" --seed "${SEED}"
shopt -s nullglob
matches=("${TRAIN_ROOT}"/*_"${EXP_NAME}"_seed"${SEED}")
shopt -u nullglob
[[ ${#matches[@]} -eq 1 ]] || die "expected one L1 run, found ${#matches[@]}"
RUN=${matches[0]}

echo "GENERATION_START variant=geo_history_actual_body_tail_sampler_es_l1 members=${NSAMPLES} state=raw"
"${PYTHON_BIN}" generate_station24.py \
  --run-dir "${RUN}" --data-path "${DATA}" \
  --output-dir "${RESULT}" --split val \
  --n-samples "${NSAMPLES}" --seed "${GEN_SEED}" \
  --issue-batch-size "${ISSUE_BATCH}" --member-chunk-size "${MEMBER_CHUNK}" \
  --energy-score-member-limit "${ENERGY_MEMBERS}" \
  --checkpoint-state raw \
  --result-variant geo_history_actual_body_tail_sampler_es_l1

echo "SAMPLER_ES_L1_AUDIT_START"
"${PYTHON_BIN}" tools/audit_station24_sampler_es_l1.py \
  --source-run "${SOURCE_RUN}" --candidate-run "${RUN}" \
  --source-result "${SOURCE_RAW_RESULT}" --candidate-result "${RESULT}" \
  --output-dir "${AUDIT}"

echo "COMPARISON_START"
"${PYTHON_BIN}" tools/compare_station24_multiscale_2a.py \
  "${SOURCE_RAW_RESULT}" "${RESULT}" --data-path "${DATA}" \
  --output-dir "${COMPARISON}" \
  --baseline-variant geo_history_actual_body_tail_moe_raw \
  --candidate-variant geo_history_actual_body_tail_sampler_es_l1 \
  --baseline-label "Raw body-tail" --candidate-label "Sampler Energy Score L1" \
  --baseline-spatial-levels bottleneck --candidate-spatial-levels bottleneck \
  --baseline-parallel-levels encoder_0 --candidate-parallel-levels encoder_0 \
  --baseline-parallel-adjacency fixed --candidate-parallel-adjacency fixed \
  --title "Raw body-tail versus final-member Energy Score L1" \
  --figure-prefix raw_body_tail_vs_sampler_es_l1

echo "WIND_EVENT_TIMING_START"
"${PYTHON_BIN}" tools/diagnose_station24_wind_event_timing.py \
  "${SOURCE_RAW_RESULT}" "${RESULT}" --data-path "${DATA}" \
  --output-dir "${TIMING}" \
  --baseline-variant geo_history_actual_body_tail_moe_raw \
  --candidate-variant geo_history_actual_body_tail_sampler_es_l1 \
  --baseline-label "Raw body-tail" --candidate-label "Sampler Energy Score L1"

echo "FORECAST_EVENT_ATTRIBUTION_START"
"${PYTHON_BIN}" tools/diagnose_station24_forecast_event_attribution.py \
  "${SOURCE_RAW_RESULT}" "${RESULT}" \
  --event-records "${TIMING}/event_records.csv" --data-path "${DATA}" \
  --output-dir "${ATTRIBUTION}"

echo "SUSTAINED_TAIL_AUDIT_START"
"${PYTHON_BIN}" tools/plot_station24_extreme_tail.py \
  --baseline "${SOURCE_RAW_RESULT}" --candidate "${RESULT}" \
  --data-path "${DATA}" --output-dir "${TAIL}" --top-issues 5 \
  --baseline-label "Raw body-tail" --candidate-label "Sampler Energy Score L1"

RESULT_FILE="${LOG_FILE%.log}.results.env"
ARCHIVE="${OUTPUT_ROOT}/station24_sampler_es_l1_${JOB_STAMP}.tar.gz"
tar -czf "${ARCHIVE}" -C "$(dirname "${PIPELINE_ROOT}")" \
  "$(basename "${PIPELINE_ROOT}")"
{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "SOURCE_RAW_RESULT=${SOURCE_RAW_RESULT}"
  echo "SOURCE_RUN=${SOURCE_RUN}"
  echo "RUN=${RUN}"
  echo "RESULT=${RESULT}"
  echo "COMPARISON=${COMPARISON}"
  echo "ARCHIVE=${ARCHIVE}"
  echo "SAMPLER_ES_L1_PIPELINE_COMPLETED"
} > "${RESULT_FILE}"
cat "${RESULT_FILE}"
