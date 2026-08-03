#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
EXPECTED_BRANCH=${EXPECTED_BRANCH:-experiment/24site-wind-solar-168h}
PYTHON_BIN=${PYTHON_BIN:-python}
DATA=${DATA:-diffusion_input_station}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_shandong/station24}
LOG_ROOT=${LOG_ROOT:-logs/station24}
REFERENCE_2B_RESULT=${REFERENCE_2B_RESULT:-}
SEED=${SEED:-2027}
NSAMPLES=${NSAMPLES:-80}
GEN_SEED=${GEN_SEED:-424242}
ISSUE_BATCH=${ISSUE_BATCH:-1}
MEMBER_CHUNK=${MEMBER_CHUNK:-10}

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
  log_file="${LOG_ROOT}/station24_cdsg_2d_2e_${stamp}.log"
  pid_file="${LOG_ROOT}/station24_cdsg_2d_2e_${stamp}.pid"
  status_file="${LOG_ROOT}/station24_cdsg_2d_2e_${stamp}.status"
  nohup setsid env \
    PYTHONUNBUFFERED=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    OMP_NUM_THREADS=1 \
    STATION24_CDSG_2D_2E_INTERNAL_WORKER=1 \
    JOB_STAMP="${stamp}" LOG_FILE="${log_file}" \
    PID_FILE="${pid_file}" STATUS_FILE="${status_file}" \
    EXPECTED_BRANCH="${EXPECTED_BRANCH}" PYTHON_BIN="${PYTHON_BIN}" \
    DATA="${DATA}" OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
    REFERENCE_2B_RESULT="${REFERENCE_2B_RESULT}" \
    SEED="${SEED}" NSAMPLES="${NSAMPLES}" GEN_SEED="${GEN_SEED}" \
    ISSUE_BATCH="${ISSUE_BATCH}" MEMBER_CHUNK="${MEMBER_CHUNK}" \
    bash "$0" > "${log_file}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_file}"
  printf 'state=running\npid=%s\nstarted_at=%s\n' \
    "${pid}" "$(date --iso-8601=seconds)" > "${status_file}"
  echo "Started Station24 Experiments 2D -> 2E pipeline"
  echo "PID: ${pid}"
  echo "Log: ${log_file}"
  echo "Status: ${status_file}"
  echo "Monitor: tail -f '${log_file}'"
  echo "Stop entire pipeline: kill -- -\$(cat '${pid_file}')"
}

if [[ "${STATION24_CDSG_2D_2E_INTERNAL_WORKER:-0}" != "1" ]]; then
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

for required in \
  train_forecast.npy train_actual.npy train_residual.npy train_time_mark.npy \
  train_lead_mark.npy train_fill_mask.npy train_issue_dates.csv \
  val_forecast.npy val_actual.npy val_residual.npy val_time_mark.npy \
  val_lead_mark.npy val_fill_mask.npy val_issue_dates.csv \
  station_features.npy station_adjacency.npy station_order.csv export_metadata.json; do
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

CONFIG_2B="configs/station24_state_v1_cdsg_lite_parallel_168h.yaml"
CONFIG_2D="configs/station24_state_v1_cdsg_2d_conditional_scale_168h.yaml"
CONFIG_2E="configs/station24_state_v1_cdsg_2e_ramp_aux_168h.yaml"
for config in "${CONFIG_2B}" "${CONFIG_2D}" "${CONFIG_2E}"; do
  [[ -f "${config}" ]] || die "missing config ${config}"
done

"${PYTHON_BIN}" - "${CONFIG_2B}" "${CONFIG_2D}" "${CONFIG_2E}" "${DATA}" <<'PY'
import copy
import sys
import yaml

from src.models.station_conditioned_diffusion import Station24DiffusionModel
from station_dataset import load_station_static_data

configs = []
for path in sys.argv[1:4]:
    with open(path, "r", encoding="utf-8") as handle:
        configs.append(yaml.safe_load(handle))
base, exp2d, exp2e = configs

if base["target"]["residual_scaling"]["method"] != "per_station_std":
    raise SystemExit("2B reference config must use per-station scaling")
for name, config in [("2D", exp2d), ("2E", exp2e)]:
    if config["target"]["residual_scaling"]["method"] != "wind_factorized_condition_std":
        raise SystemExit(f"{name} must use wind conditional residual scaling")
if exp2d["target"] != exp2e["target"]:
    raise SystemExit("2D and 2E residual target configurations must be identical")
if float(exp2d["model"].get("ramp_auxiliary_loss_weight", 0.0)) != 0.0:
    raise SystemExit("2D must not use ramp auxiliary loss")
if float(exp2e["model"].get("ramp_auxiliary_loss_weight", 0.0)) <= 0.0:
    raise SystemExit("2E must use ramp auxiliary loss")

approved_model = {
    "ramp_auxiliary_loss_weight",
    "ramp_auxiliary_lags",
    "ramp_auxiliary_lag_weights",
}
cores = []
for config in configs:
    core = copy.deepcopy(config["model"])
    for key in approved_model:
        core.pop(key, None)
    cores.append(core)
if not (cores[0] == cores[1] == cores[2]):
    raise SystemExit("2B/2D/2E model backbones differ outside ramp-loss fields")

static = load_station_static_data(sys.argv[4])
models = [
    Station24DiffusionModel(
        config["model"],
        static["station_features"],
        static["station_adjacency"],
        static["station_capacities"],
    )
    for config in configs
]
counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models]
if len(set(counts)) != 1:
    raise SystemExit(f"2D/2E must not add model parameters: {counts}")
print(
    "CDSG_2D_2E_PARITY_OK "
    f"parameters={counts[0]} ramp_aux_weight={exp2e['model']['ramp_auxiliary_loss_weight']}"
)
PY

if [[ -z "${REFERENCE_2B_RESULT}" ]]; then
  while IFS= read -r -d '' candidate; do
    if [[ -z "${REFERENCE_2B_RESULT}" || "${candidate}" -nt "${REFERENCE_2B_RESULT}" ]]; then
      REFERENCE_2B_RESULT=${candidate}
    fi
  done < <(
    find "${OUTPUT_ROOT}" -type d \
      -name "state_v1_cdsg_lite_parallel_val_n${NSAMPLES}_seed${GEN_SEED}" -print0
  )
fi
[[ -n "${REFERENCE_2B_RESULT}" && -d "${REFERENCE_2B_RESULT}" ]] \
  || die "2B result not found; set REFERENCE_2B_RESULT=/path/to/result"

"${PYTHON_BIN}" - "${REFERENCE_2B_RESULT}" "${NSAMPLES}" "${GEN_SEED}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
metadata = json.loads((path / "generation_metadata.json").read_text(encoding="utf-8"))
expected = {
    "condition_variant": "state_v1_cdsg_lite_parallel",
    "spatial_mode": "fixed_graph",
    "split": "val",
    "n_samples": int(sys.argv[2]),
    "generation_seed": int(sys.argv[3]),
    "test_used": False,
}
for key, value in expected.items():
    if metadata.get(key) != value:
        raise SystemExit(f"2B metadata mismatch {key}: {metadata.get(key)!r} != {value!r}")
if metadata.get("parallel_spatial_fusion_levels") != ["encoder_0"]:
    raise SystemExit("reference 2B must use the encoder_0 parallel branch")
print(f"REFERENCE_2B_RESULT_OK path={path}")
PY

PIPELINE_ROOT="${OUTPUT_ROOT}/cdsg_2d_2e_${JOB_STAMP}"
TRAIN_ROOT="${PIPELINE_ROOT}/training"
RESULT_ROOT="${PIPELINE_ROOT}/validation_results"
COMPARISON_ROOT="${PIPELINE_ROOT}/comparisons"
TIMING_ROOT="${PIPELINE_ROOT}/wind_event_timing"
mkdir -p "${TRAIN_ROOT}" "${RESULT_ROOT}" "${COMPARISON_ROOT}" "${TIMING_ROOT}"
printf '%s\n' "${REFERENCE_2B_RESULT}" > "${PIPELINE_ROOT}/reference_2b_result.txt"

ENVIRONMENT_FILE="${LOG_FILE%.log}.environment.txt"
{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse HEAD)"
  echo "data=${DATA}"
  echo "reference_2b_result=${REFERENCE_2B_RESULT}"
  echo "seed=${SEED}"
  echo "generation_seed=${GEN_SEED}"
  echo "ensemble_members=${NSAMPLES}"
  echo "experiment_family=station24_cdsg_2d_2e"
  echo "test_used=false"
  nvidia-smi
} > "${ENVIRONMENT_FILE}"

train_and_generate() {
  local variant=$1
  local config=$2
  local experiment_name=$3
  local result_name=$4
  echo "TRAIN_START variant=${variant}"
  "${PYTHON_BIN}" train_station24.py \
    --config "${config}" --data-path "${DATA}" \
    --output-root "${TRAIN_ROOT}" --exp-name "${experiment_name}" \
    --seed "${SEED}"

  shopt -s nullglob
  local matches=("${TRAIN_ROOT}"/*_"${experiment_name}"_seed"${SEED}")
  shopt -u nullglob
  [[ ${#matches[@]} -eq 1 ]] \
    || die "expected one ${variant} run, found ${#matches[@]}"
  LAST_RUN=${matches[0]}
  cp "${ENVIRONMENT_FILE}" "${LAST_RUN}/logs/server_environment.txt"
  LAST_RESULT="${RESULT_ROOT}/${result_name}"
  echo "GENERATION_START variant=${variant} members=${NSAMPLES}"
  "${PYTHON_BIN}" generate_station24.py \
    --run-dir "${LAST_RUN}" --data-path "${DATA}" \
    --output-dir "${LAST_RESULT}" --split val \
    --n-samples "${NSAMPLES}" --seed "${GEN_SEED}" \
    --issue-batch-size "${ISSUE_BATCH}" \
    --member-chunk-size "${MEMBER_CHUNK}"
  echo "EXPERIMENT_COMPLETE variant=${variant} run_dir=${LAST_RUN} result_dir=${LAST_RESULT}"
}

EXP2D_NAME="station24_cdsg_2d_conditional_scale_${JOB_STAMP}"
train_and_generate \
  state_v1_cdsg_2d_conditional_scale "${CONFIG_2D}" "${EXP2D_NAME}" \
  "state_v1_cdsg_2d_conditional_scale_val_n${NSAMPLES}_seed${GEN_SEED}"
EXP2D_RUN=${LAST_RUN}
EXP2D_RESULT=${LAST_RESULT}

EXP2E_NAME="station24_cdsg_2e_conditional_scale_ramp_aux_${JOB_STAMP}"
train_and_generate \
  state_v1_cdsg_2e_conditional_scale_ramp_aux "${CONFIG_2E}" "${EXP2E_NAME}" \
  "state_v1_cdsg_2e_conditional_scale_ramp_aux_val_n${NSAMPLES}_seed${GEN_SEED}"
EXP2E_RUN=${LAST_RUN}
EXP2E_RESULT=${LAST_RESULT}

compare_pair() {
  local left_result=$1 right_result=$2 output=$3
  local left_variant=$4 right_variant=$5 left_label=$6 right_label=$7 prefix=$8
  "${PYTHON_BIN}" tools/compare_station24_multiscale_2a.py \
    "${left_result}" "${right_result}" \
    --data-path "${DATA}" --output-dir "${output}" \
    --baseline-variant "${left_variant}" \
    --candidate-variant "${right_variant}" \
    --baseline-label "${left_label}" --candidate-label "${right_label}" \
    --baseline-spatial-levels bottleneck --candidate-spatial-levels bottleneck \
    --baseline-parallel-levels encoder_0 --candidate-parallel-levels encoder_0 \
    --baseline-parallel-adjacency fixed --candidate-parallel-adjacency fixed \
    --title "Station24 ${left_label} versus ${right_label}" \
    --figure-prefix "${prefix}"
}

timing_pair() {
  local left_result=$1 right_result=$2 output=$3
  local left_variant=$4 right_variant=$5 left_label=$6 right_label=$7
  "${PYTHON_BIN}" tools/diagnose_station24_wind_event_timing.py \
    "${left_result}" "${right_result}" \
    --data-path "${DATA}" --output-dir "${output}" \
    --baseline-variant "${left_variant}" \
    --candidate-variant "${right_variant}" \
    --baseline-label "${left_label}" --candidate-label "${right_label}"
}

echo "COMPARISON_START"
compare_pair "${REFERENCE_2B_RESULT}" "${EXP2D_RESULT}" \
  "${COMPARISON_ROOT}/2b_vs_2d" \
  state_v1_cdsg_lite_parallel state_v1_cdsg_2d_conditional_scale \
  "2B fixed parallel" "2D conditional scale" 2b_vs_2d
compare_pair "${EXP2D_RESULT}" "${EXP2E_RESULT}" \
  "${COMPARISON_ROOT}/2d_vs_2e" \
  state_v1_cdsg_2d_conditional_scale state_v1_cdsg_2e_conditional_scale_ramp_aux \
  "2D conditional scale" "2E ramp auxiliary" 2d_vs_2e
compare_pair "${REFERENCE_2B_RESULT}" "${EXP2E_RESULT}" \
  "${COMPARISON_ROOT}/2b_vs_2e" \
  state_v1_cdsg_lite_parallel state_v1_cdsg_2e_conditional_scale_ramp_aux \
  "2B fixed parallel" "2E conditional scale + ramp auxiliary" 2b_vs_2e

echo "TIMING_DIAGNOSTICS_START"
timing_pair "${REFERENCE_2B_RESULT}" "${EXP2D_RESULT}" \
  "${TIMING_ROOT}/2b_vs_2d" \
  state_v1_cdsg_lite_parallel state_v1_cdsg_2d_conditional_scale \
  "2B fixed parallel" "2D conditional scale"
timing_pair "${EXP2D_RESULT}" "${EXP2E_RESULT}" \
  "${TIMING_ROOT}/2d_vs_2e" \
  state_v1_cdsg_2d_conditional_scale state_v1_cdsg_2e_conditional_scale_ramp_aux \
  "2D conditional scale" "2E ramp auxiliary"
timing_pair "${REFERENCE_2B_RESULT}" "${EXP2E_RESULT}" \
  "${TIMING_ROOT}/2b_vs_2e" \
  state_v1_cdsg_lite_parallel state_v1_cdsg_2e_conditional_scale_ramp_aux \
  "2B fixed parallel" "2E conditional scale + ramp auxiliary"

"${PYTHON_BIN}" - \
  "${REFERENCE_2B_RESULT}" "${EXP2D_RESULT}" "${EXP2E_RESULT}" \
  "${COMPARISON_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

paths = [Path(value) for value in sys.argv[1:4]]
output = Path(sys.argv[4])

def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten(child, name)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, value

long_rows = []
manifest = []
for path in paths:
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    run = metrics["run"]
    variant = run["condition_variant"]
    manifest.append(
        {
            "variant": variant,
            "result_dir": str(path),
            "parameter_count": run["parameter_count"],
            "checkpoint_validation_objective": run["checkpoint_validation_mse"],
            "residual_scaling_method": run.get("residual_scaling_method", "per_station_std"),
            "ramp_auxiliary_loss_weight": run.get("ramp_auxiliary_loss_weight", 0.0),
            "n_samples": run["n_samples"],
            "generation_seed": run["generation_seed"],
            "test_used": run.get("test_used", False),
        }
    )
    for metric, value in flatten(metrics):
        long_rows.append({"variant": variant, "metric": metric, "value": value})

with (output / "experiment_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
    writer.writeheader()
    writer.writerows(manifest)
with (output / "all_numeric_metrics_long.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["variant", "metric", "value"])
    writer.writeheader()
    writer.writerows(long_rows)
print(f"UNIFIED_METRICS_COMPLETE output={output}")
PY

RESULT_FILE="${LOG_FILE%.log}.results.env"
ARCHIVE="${OUTPUT_ROOT}/station24_cdsg_2d_2e_${JOB_STAMP}.tar.gz"
tar -czf "${ARCHIVE}" -C "$(dirname "${PIPELINE_ROOT}")" "$(basename "${PIPELINE_ROOT}")"
{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "REFERENCE_2B_RESULT=${REFERENCE_2B_RESULT}"
  echo "EXP2D_RUN=${EXP2D_RUN}"
  echo "EXP2D_RESULT=${EXP2D_RESULT}"
  echo "EXP2E_RUN=${EXP2E_RUN}"
  echo "EXP2E_RESULT=${EXP2E_RESULT}"
  echo "COMPARISON_DIR=${COMPARISON_ROOT}"
  echo "TIMING_DIAGNOSTIC_DIR=${TIMING_ROOT}"
  echo "ARCHIVE=${ARCHIVE}"
  echo "ALL_STATION24_CDSG_2D_2E_EXPERIMENTS_COMPLETED"
} > "${RESULT_FILE}"
cat "${RESULT_FILE}"
