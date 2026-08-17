#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
EXPECTED_BRANCH=${EXPECTED_BRANCH:-experiment/24site-historical-spatial-prior}
PYTHON_BIN=${PYTHON_BIN:-python}
DATA=${DATA:-diffusion_input_station}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_shandong/station24}
LOG_ROOT=${LOG_ROOT:-logs/station24}
REFERENCE500_RESULT=${REFERENCE500_RESULT:-}
SEED=${SEED:-2027}
NSAMPLES=${NSAMPLES:-500}
GEN_SEED=${GEN_SEED:-424242}
ENERGY_MEMBERS=${ENERGY_MEMBERS:-80}
ISSUE_BATCH=${ISSUE_BATCH:-1}
MEMBER_CHUNK=${MEMBER_CHUNK:-10}
BOOTSTRAP_REPETITIONS=${BOOTSTRAP_REPETITIONS:-200}

die() { echo "ERROR: $*" >&2; exit 1; }

record_exit() {
  local code=$?
  trap - EXIT
  if [[ -n "${STATUS_FILE:-}" ]]; then
    local state=completed
    [[ ${code} -eq 0 ]] || state=failed
    printf 'state=%s\npid=%s\nfinished_at=%s\nexit_code=%s\n' \
      "${state}" "${BASHPID}" "$(date --iso-8601=seconds)" "${code}" \
      > "${STATUS_FILE}"
  fi
  exit "${code}"
}

launch_background() {
  cd "${REPO_ROOT}"
  mkdir -p "${LOG_ROOT}"
  local stamp log_file pid_file status_file
  stamp=$(date +%Y%m%d_%H%M%S)
  log_file="${LOG_ROOT}/station24_historical_dual_graph_${stamp}.log"
  pid_file="${LOG_ROOT}/station24_historical_dual_graph_${stamp}.pid"
  status_file="${LOG_ROOT}/station24_historical_dual_graph_${stamp}.status"
  nohup setsid env \
    PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
    STATION24_HISTORICAL_DUAL_INTERNAL_WORKER=1 JOB_STAMP="${stamp}" \
    LOG_FILE="${log_file}" PID_FILE="${pid_file}" STATUS_FILE="${status_file}" \
    EXPECTED_BRANCH="${EXPECTED_BRANCH}" PYTHON_BIN="${PYTHON_BIN}" \
    DATA="${DATA}" OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
    REFERENCE500_RESULT="${REFERENCE500_RESULT}" SEED="${SEED}" \
    NSAMPLES="${NSAMPLES}" GEN_SEED="${GEN_SEED}" \
    ENERGY_MEMBERS="${ENERGY_MEMBERS}" ISSUE_BATCH="${ISSUE_BATCH}" \
    MEMBER_CHUNK="${MEMBER_CHUNK}" \
    BOOTSTRAP_REPETITIONS="${BOOTSTRAP_REPETITIONS}" \
    bash "$0" > "${log_file}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_file}"
  printf 'state=running\npid=%s\nstarted_at=%s\n' \
    "${pid}" "$(date --iso-8601=seconds)" > "${status_file}"
  echo "Started Station24 historical dual-graph pipeline"
  echo "PID: ${pid}"
  echo "Log: ${log_file}"
  echo "Status: ${status_file}"
  echo "Monitor: tail -f '${log_file}'"
  echo "Stop entire pipeline: kill -- -\$(cat '${pid_file}')"
}

if [[ "${STATION24_HISTORICAL_DUAL_INTERNAL_WORKER:-0}" != "1" ]]; then
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
[[ "${NSAMPLES}" -eq 500 ]] || die "formal historical graph comparison requires 500 members"
[[ "${GEN_SEED}" -eq 424242 ]] || die "generation seed must remain 424242"

for required in \
  train_forecast.npy train_actual.npy train_residual.npy train_time_mark.npy \
  train_lead_mark.npy train_fill_mask.npy train_issue_dates.csv \
  val_forecast.npy val_actual.npy val_residual.npy val_time_mark.npy \
  val_lead_mark.npy val_fill_mask.npy val_issue_dates.csv \
  station_features.npy station_adjacency.npy station_distance_km.npy \
  station_order.csv export_metadata.json; do
  [[ -f "${DATA}/${required}" ]] || die "missing data artifact ${DATA}/${required}"
done

CONFIG_ACTUAL=configs/station24_geo_history_actual_dual_168h.yaml
CONFIG_RESIDUAL=configs/station24_geo_history_residual_dual_168h.yaml
CONFIG_REFERENCE=configs/station24_state_v1_cdsg_2d_conditional_scale_168h.yaml
DIAGNOSTIC_CONFIG=configs/station24_historical_spatial_prior_diagnostic.yaml
for required in \
  "${CONFIG_ACTUAL}" "${CONFIG_RESIDUAL}" "${CONFIG_REFERENCE}" \
  "${DIAGNOSTIC_CONFIG}"; do
  [[ -f "${required}" ]] || die "missing configuration ${required}"
done

"${PYTHON_BIN}" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

if [[ -z "${REFERENCE500_RESULT}" ]]; then
  while IFS= read -r -d '' candidate; do
    if [[ -z "${REFERENCE500_RESULT}" || "${candidate}" -nt "${REFERENCE500_RESULT}" ]]; then
      REFERENCE500_RESULT=${candidate}
    fi
  done < <(find "${OUTPUT_ROOT}" -type d -path '*/reference500_*/validation_result' -print0)
fi
[[ -n "${REFERENCE500_RESULT}" && -d "${REFERENCE500_RESULT}" ]] \
  || die "500-member reference not found; set REFERENCE500_RESULT=/path/to/validation_result"

REFERENCE_RUN=$("${PYTHON_BIN}" - "${REFERENCE500_RESULT}" "${NSAMPLES}" "${GEN_SEED}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
meta = json.loads((path / "generation_metadata.json").read_text(encoding="utf-8"))
expected = {
    "condition_variant": "state_v1_cdsg_2d_conditional_scale",
    "split": "val",
    "n_samples": int(sys.argv[2]),
    "generation_seed": int(sys.argv[3]),
    "test_used": False,
}
for key, value in expected.items():
    if meta.get(key) != value:
        raise SystemExit(f"reference metadata mismatch {key}: {meta.get(key)!r} != {value!r}")
print(meta["run_dir"])
PY
)
[[ -f "${REFERENCE_RUN}/residual_scale.json" ]] \
  || die "reference residual scale missing: ${REFERENCE_RUN}/residual_scale.json"

PIPELINE_ROOT="${OUTPUT_ROOT}/historical_dual_graph_${JOB_STAMP}"
PRIOR_ROOT="${PIPELINE_ROOT}/train_only_priors"
TRAIN_ROOT="${PIPELINE_ROOT}/training"
RESULT_ROOT="${PIPELINE_ROOT}/validation_results"
COMPARISON_ROOT="${PIPELINE_ROOT}/comparisons"
TAIL_AUDIT_ROOT="${PIPELINE_ROOT}/spatial_tail_audit"
TIMING_ROOT="${PIPELINE_ROOT}/wind_event_timing"
mkdir -p \
  "${PIPELINE_ROOT}" "${TRAIN_ROOT}" "${RESULT_ROOT}" \
  "${COMPARISON_ROOT}" "${TIMING_ROOT}"

echo "PRIOR_DIAGNOSTIC_START bootstrap=${BOOTSTRAP_REPETITIONS}"
"${PYTHON_BIN}" tools/diagnose_station24_historical_spatial_priors.py \
  --config "${DIAGNOSTIC_CONFIG}" --data-path "${DATA}" \
  --residual-scale-path "${REFERENCE_RUN}/residual_scale.json" \
  --output-dir "${PRIOR_ROOT}" \
  --bootstrap-repetitions "${BOOTSTRAP_REPETITIONS}"

"${PYTHON_BIN}" - "${PRIOR_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
audit = json.loads((root / "historical_spatial_prior_audit.json").read_text(encoding="utf-8"))
if not audit["reference_reproduction"]["passed"]:
    raise SystemExit("historical prior reference audit failed")
if not audit["train_only"] or audit["validation_actual_used"] or audit["test_actual_used"]:
    raise SystemExit("historical prior split audit failed")
for name in ["adjacency_actual.npy", "adjacency_residual_std.npy"]:
    if not (root / name).is_file():
        raise SystemExit(f"missing diagnosed graph {name}")
print("HISTORICAL_PRIOR_GATE_PASSED")
PY

"${PYTHON_BIN}" - \
  "${CONFIG_ACTUAL}" "${CONFIG_RESIDUAL}" "${CONFIG_REFERENCE}" \
  "${PRIOR_ROOT}/adjacency_actual.npy" \
  "${PRIOR_ROOT}/adjacency_residual_std.npy" "${DATA}" <<'PY'
import copy
import sys
import numpy as np
import torch
import yaml

from src.models.station_conditioned_diffusion import Station24DiffusionModel
from station_dataset import load_station_static_data

with open(sys.argv[1], encoding="utf-8") as handle:
    actual = yaml.safe_load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    residual = yaml.safe_load(handle)
with open(sys.argv[3], encoding="utf-8") as handle:
    reference = yaml.safe_load(handle)
actual_core = copy.deepcopy(actual)
residual_core = copy.deepcopy(residual)
actual_core.pop("experiment")
residual_core.pop("experiment")
for config in [actual_core, residual_core]:
    config["model"].pop("secondary_adjacency_path", None)
    config["model"].pop("secondary_adjacency_role", None)
if actual_core != residual_core:
    raise SystemExit("historical dual configs differ outside experiment identity/source")

reference_model_core = copy.deepcopy(reference["model"])
candidate_model_core = copy.deepcopy(actual["model"])
for key in [
    "use_dual_fixed_graph",
    "secondary_adjacency_path",
    "secondary_adjacency_role",
    "secondary_adjacency_fit_split",
    "dual_graph_primary_logit_init",
    "dual_graph_secondary_logit_init",
]:
    candidate_model_core.pop(key, None)
if candidate_model_core != reference_model_core:
    raise SystemExit("dual candidate changed the reference model outside dual-graph fields")
if actual["target"] != reference["target"] or actual["train"] != reference["train"]:
    raise SystemExit("dual candidate changed target or training hyperparameters")

static = load_station_static_data(sys.argv[6])
secondary_actual = np.load(sys.argv[4]).astype(np.float32)
secondary_residual = np.load(sys.argv[5]).astype(np.float32)
reference_model = Station24DiffusionModel(
    reference["model"], static["station_features"], static["station_adjacency"],
    static["station_capacities"],
)
models = [
    Station24DiffusionModel(
        config["model"], static["station_features"], static["station_adjacency"],
        static["station_capacities"], secondary,
    )
    for config, secondary in [
        (actual, torch.from_numpy(secondary_actual)),
        (residual, torch.from_numpy(secondary_residual)),
    ]
]
counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models]
if counts[0] != counts[1]:
    raise SystemExit(f"candidate parameter counts differ: {counts}")
dual_parameters = sum(
    parameter.numel()
    for name, parameter in models[0].named_parameters()
    if "dual_graph_logits" in name
)
if dual_parameters != 4:
    raise SystemExit(f"expected exactly four dual-graph logits, got {dual_parameters}")
reference_count = sum(parameter.numel() for parameter in reference_model.parameters())
if counts[0] - reference_count != 4:
    raise SystemExit(
        f"candidate must add exactly four parameters: reference={reference_count} candidate={counts[0]}"
    )
ratio = 100.0 * (counts[0] - reference_count) / reference_count
if ratio >= 1.0:
    raise SystemExit(f"dual graph parameter increment exceeds 1%: {ratio:.6f}%")
print(
    "HISTORICAL_DUAL_PARITY_OK "
    f"reference={reference_count} candidate={counts[0]} "
    f"increment={counts[0] - reference_count} ratio={ratio:.6f}%"
)
PY

ENVIRONMENT_FILE="${LOG_FILE%.log}.environment.txt"
{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse HEAD)"
  echo "data=${DATA}"
  echo "reference500_result=${REFERENCE500_RESULT}"
  echo "seed=${SEED}"
  echo "generation_seed=${GEN_SEED}"
  echo "ensemble_members=${NSAMPLES}"
  echo "energy_score_members=${ENERGY_MEMBERS}"
  echo "test_used=false"
  nvidia-smi
} > "${ENVIRONMENT_FILE}"

train_and_generate() {
  local variant=$1 config=$2 secondary=$3
  local experiment_name="station24_${variant}_${JOB_STAMP}"
  echo "TRAIN_START variant=${variant} secondary=${secondary}"
  "${PYTHON_BIN}" train_station24.py \
    --config "${config}" --data-path "${DATA}" \
    --secondary-adjacency "${secondary}" \
    --output-root "${TRAIN_ROOT}" --exp-name "${experiment_name}" \
    --seed "${SEED}"
  shopt -s nullglob
  local matches=("${TRAIN_ROOT}"/*_"${experiment_name}"_seed"${SEED}")
  shopt -u nullglob
  [[ ${#matches[@]} -eq 1 ]] || die "expected one ${variant} run, found ${#matches[@]}"
  LAST_RUN=${matches[0]}
  cp "${ENVIRONMENT_FILE}" "${LAST_RUN}/logs/server_environment.txt"
  LAST_RESULT="${RESULT_ROOT}/${variant}_val_n${NSAMPLES}_seed${GEN_SEED}"
  echo "GENERATION_START variant=${variant} members=${NSAMPLES}"
  "${PYTHON_BIN}" generate_station24.py \
    --run-dir "${LAST_RUN}" --data-path "${DATA}" \
    --output-dir "${LAST_RESULT}" --split val \
    --n-samples "${NSAMPLES}" --seed "${GEN_SEED}" \
    --issue-batch-size "${ISSUE_BATCH}" --member-chunk-size "${MEMBER_CHUNK}" \
    --energy-score-member-limit "${ENERGY_MEMBERS}"
  echo "EXPERIMENT_COMPLETE variant=${variant} run_dir=${LAST_RUN} result_dir=${LAST_RESULT}"
}

train_and_generate geo_history_actual_dual "${CONFIG_ACTUAL}" \
  "${PRIOR_ROOT}/adjacency_actual.npy"
ACTUAL_RUN=${LAST_RUN}
ACTUAL_RESULT=${LAST_RESULT}

train_and_generate geo_history_residual_dual "${CONFIG_RESIDUAL}" \
  "${PRIOR_ROOT}/adjacency_residual_std.npy"
RESIDUAL_RUN=${LAST_RUN}
RESIDUAL_RESULT=${LAST_RESULT}

compare_pair() {
  local left=$1 right=$2 output=$3 left_variant=$4 right_variant=$5
  local left_label=$6 right_label=$7 prefix=$8
  "${PYTHON_BIN}" tools/compare_station24_multiscale_2a.py \
    "${left}" "${right}" --data-path "${DATA}" --output-dir "${output}" \
    --baseline-variant "${left_variant}" --candidate-variant "${right_variant}" \
    --baseline-label "${left_label}" --candidate-label "${right_label}" \
    --baseline-spatial-levels bottleneck --candidate-spatial-levels bottleneck \
    --baseline-parallel-levels encoder_0 --candidate-parallel-levels encoder_0 \
    --baseline-parallel-adjacency fixed --candidate-parallel-adjacency fixed \
    --title "${left_label} versus ${right_label}" --figure-prefix "${prefix}"
}

echo "COMPARISON_START"
compare_pair "${REFERENCE500_RESULT}" "${ACTUAL_RESULT}" \
  "${COMPARISON_ROOT}/reference_vs_geo_history_actual_dual" \
  state_v1_cdsg_2d_conditional_scale geo_history_actual_dual \
  "500-member geographic reference" "Geographic + historical actual" \
  reference_vs_geo_history_actual_dual
compare_pair "${REFERENCE500_RESULT}" "${RESIDUAL_RESULT}" \
  "${COMPARISON_ROOT}/reference_vs_geo_history_residual_dual" \
  state_v1_cdsg_2d_conditional_scale geo_history_residual_dual \
  "500-member geographic reference" "Geographic + standardized residual" \
  reference_vs_geo_history_residual_dual
compare_pair "${ACTUAL_RESULT}" "${RESIDUAL_RESULT}" \
  "${COMPARISON_ROOT}/actual_vs_residual" \
  geo_history_actual_dual geo_history_residual_dual \
  "Geographic + historical actual" "Geographic + standardized residual" \
  actual_vs_residual

echo "SPATIAL_TAIL_AUDIT_START"
"${PYTHON_BIN}" tools/analyze_station24_historical_dual_graphs.py \
  "${REFERENCE500_RESULT}" "${ACTUAL_RESULT}" "${RESIDUAL_RESULT}" \
  --data-path "${DATA}" --prior-dir "${PRIOR_ROOT}" \
  --output-dir "${TAIL_AUDIT_ROOT}"

echo "WIND_EVENT_TIMING_START"
"${PYTHON_BIN}" tools/diagnose_station24_wind_event_timing.py \
  "${REFERENCE500_RESULT}" "${ACTUAL_RESULT}" \
  --data-path "${DATA}" \
  --output-dir "${TIMING_ROOT}/reference_vs_geo_history_actual_dual" \
  --baseline-variant state_v1_cdsg_2d_conditional_scale \
  --candidate-variant geo_history_actual_dual \
  --baseline-label "500-member geographic reference" \
  --candidate-label "Geographic + historical actual"
"${PYTHON_BIN}" tools/diagnose_station24_wind_event_timing.py \
  "${REFERENCE500_RESULT}" "${RESIDUAL_RESULT}" \
  --data-path "${DATA}" \
  --output-dir "${TIMING_ROOT}/reference_vs_geo_history_residual_dual" \
  --baseline-variant state_v1_cdsg_2d_conditional_scale \
  --candidate-variant geo_history_residual_dual \
  --baseline-label "500-member geographic reference" \
  --candidate-label "Geographic + standardized residual"

RESULT_FILE="${LOG_FILE%.log}.results.env"
ARCHIVE="${OUTPUT_ROOT}/station24_historical_dual_graph_${JOB_STAMP}.tar.gz"
tar -czf "${ARCHIVE}" -C "$(dirname "${PIPELINE_ROOT}")" "$(basename "${PIPELINE_ROOT}")"
{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "REFERENCE500_RESULT=${REFERENCE500_RESULT}"
  echo "PRIOR_DIR=${PRIOR_ROOT}"
  echo "GEO_HISTORY_ACTUAL_RUN=${ACTUAL_RUN}"
  echo "GEO_HISTORY_ACTUAL_RESULT=${ACTUAL_RESULT}"
  echo "GEO_HISTORY_RESIDUAL_RUN=${RESIDUAL_RUN}"
  echo "GEO_HISTORY_RESIDUAL_RESULT=${RESIDUAL_RESULT}"
  echo "COMPARISON_DIR=${COMPARISON_ROOT}"
  echo "SPATIAL_TAIL_AUDIT_DIR=${TAIL_AUDIT_ROOT}"
  echo "TIMING_DIR=${TIMING_ROOT}"
  echo "ARCHIVE=${ARCHIVE}"
  echo "ALL_HISTORICAL_DUAL_GRAPH_EXPERIMENTS_COMPLETED"
} > "${RESULT_FILE}"
cat "${RESULT_FILE}"
