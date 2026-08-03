#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
EXPECTED_BRANCH=${EXPECTED_BRANCH:-experiment/24site-wind-solar-168h}
PYTHON_BIN=${PYTHON_BIN:-python}
DATA=${DATA:-diffusion_input_station}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_shandong/station24}
LOG_ROOT=${LOG_ROOT:-logs/station24}
BASELINE_RESULT=${BASELINE_RESULT:-}
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
  log_file="${LOG_ROOT}/station24_cdsg_2b_2c_${stamp}.log"
  pid_file="${LOG_ROOT}/station24_cdsg_2b_2c_${stamp}.pid"
  status_file="${LOG_ROOT}/station24_cdsg_2b_2c_${stamp}.status"
  nohup setsid env \
    PYTHONUNBUFFERED=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    OMP_NUM_THREADS=1 \
    STATION24_CDSG_2B_2C_INTERNAL_WORKER=1 \
    JOB_STAMP="${stamp}" LOG_FILE="${log_file}" \
    PID_FILE="${pid_file}" STATUS_FILE="${status_file}" \
    EXPECTED_BRANCH="${EXPECTED_BRANCH}" PYTHON_BIN="${PYTHON_BIN}" \
    DATA="${DATA}" OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
    BASELINE_RESULT="${BASELINE_RESULT}" \
    SEED="${SEED}" NSAMPLES="${NSAMPLES}" GEN_SEED="${GEN_SEED}" \
    ISSUE_BATCH="${ISSUE_BATCH}" MEMBER_CHUNK="${MEMBER_CHUNK}" \
    bash "$0" > "${log_file}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_file}"
  printf 'state=running\npid=%s\nstarted_at=%s\n' \
    "${pid}" "$(date --iso-8601=seconds)" > "${status_file}"
  echo "Started Station24 Experiments 2B + 2C pipeline"
  echo "PID: ${pid}"
  echo "Log: ${log_file}"
  echo "Status: ${status_file}"
  echo "Monitor: tail -f '${log_file}'"
  echo "Stop entire pipeline: kill -- -\$(cat '${pid_file}')"
}

if [[ "${STATION24_CDSG_2B_2C_INTERNAL_WORKER:-0}" != "1" ]]; then
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

CONFIG_BASELINE="configs/station24_state_v1_fixed_graph_168h.yaml"
CONFIG_2B="configs/station24_state_v1_cdsg_lite_parallel_168h.yaml"
CONFIG_2C="configs/station24_state_v1_cdsg_lite_hybrid_dynamic_168h.yaml"
for config in "${CONFIG_BASELINE}" "${CONFIG_2B}" "${CONFIG_2C}"; do
  [[ -f "${config}" ]] || die "missing config ${config}"
done

"${PYTHON_BIN}" - "${CONFIG_BASELINE}" "${CONFIG_2B}" "${CONFIG_2C}" "${DATA}" <<'PY'
import copy
import sys
import yaml

from src.models.station_conditioned_diffusion import Station24DiffusionModel
from station_dataset import load_station_static_data

configs = []
for path in sys.argv[1:4]:
    with open(path, "r", encoding="utf-8") as handle:
        configs.append(yaml.safe_load(handle))

approved = {
    "spatial_mix_levels",
    "parallel_spatial_fusion_levels",
    "parallel_spatial_gate_init",
    "parallel_spatial_adjacency_mode",
    "dynamic_graph_embedding_dim",
    "dynamic_graph_top_k",
    "dynamic_graph_temperature",
    "dynamic_graph_mix_gate_init",
}
cores = []
for config in configs:
    core = copy.deepcopy(config)
    core.pop("experiment", None)
    for key in approved:
        core["model"].pop(key, None)
    cores.append(core)
if not (cores[0] == cores[1] == cores[2]):
    raise SystemExit("2B/2C configs differ outside approved graph/fusion fields")

baseline, exp2b, exp2c = configs
if baseline["model"].get("spatial_mix_levels", ["bottleneck"]) != ["bottleneck"]:
    raise SystemExit("baseline must use the bottleneck graph only")
for name, config in [("2B", exp2b), ("2C", exp2c)]:
    if config["model"].get("spatial_mix_levels") != ["bottleneck"]:
        raise SystemExit(f"{name} must preserve the baseline bottleneck graph")
    if config["model"].get("parallel_spatial_fusion_levels") != ["encoder_0"]:
        raise SystemExit(f"{name} must use one 168 h parallel branch")
if exp2b["model"].get("parallel_spatial_adjacency_mode") != "fixed":
    raise SystemExit("2B parallel adjacency must remain fixed")
if exp2c["model"].get("parallel_spatial_adjacency_mode") != "hybrid_dynamic":
    raise SystemExit("2C parallel adjacency must be hybrid_dynamic")
if any(config["model"].get("state_feature_dim") != 4 for config in configs):
    raise SystemExit("all variants must keep the four State V1 indicators")

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
if not counts[0] < counts[1] < counts[2]:
    raise SystemExit(f"unexpected parameter ordering: {counts}")
if counts[2] - counts[1] >= 5000:
    raise SystemExit(f"2C dynamic graph is not lightweight: delta={counts[2] - counts[1]}")
print(
    "CDSG_2B_2C_PARITY_OK "
    f"baseline_params={counts[0]} exp2b_params={counts[1]} "
    f"exp2c_params={counts[2]} dynamic_delta={counts[2] - counts[1]}"
)
PY

if [[ -z "${BASELINE_RESULT}" ]]; then
  while IFS= read -r -d '' candidate; do
    if [[ -z "${BASELINE_RESULT}" || "${candidate}" -nt "${BASELINE_RESULT}" ]]; then
      BASELINE_RESULT=${candidate}
    fi
  done < <(
    find "${OUTPUT_ROOT}" -type d \
      -name "state_v1_fixed_graph_val_n${NSAMPLES}_seed${GEN_SEED}" -print0
  )
fi
[[ -n "${BASELINE_RESULT}" && -d "${BASELINE_RESULT}" ]] \
  || die "State V1 baseline result not found; set BASELINE_RESULT=/path/to/result"

"${PYTHON_BIN}" - "${BASELINE_RESULT}" "${NSAMPLES}" "${GEN_SEED}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
metadata = json.loads((path / "generation_metadata.json").read_text(encoding="utf-8"))
expected = {
    "condition_variant": "state_v1_fixed_graph",
    "spatial_mode": "fixed_graph",
    "split": "val",
    "n_samples": int(sys.argv[2]),
    "generation_seed": int(sys.argv[3]),
    "test_used": False,
}
for key, value in expected.items():
    if metadata.get(key) != value:
        raise SystemExit(f"baseline metadata mismatch {key}: {metadata.get(key)!r} != {value!r}")
if metadata.get("spatial_mix_levels", ["bottleneck"]) != ["bottleneck"]:
    raise SystemExit("baseline is not bottleneck-only graph mixing")
print(f"BASELINE_RESULT_OK path={path}")
PY

PIPELINE_ROOT="${OUTPUT_ROOT}/cdsg_2b_2c_${JOB_STAMP}"
TRAIN_ROOT="${PIPELINE_ROOT}/training"
RESULT_ROOT="${PIPELINE_ROOT}/validation_results"
COMPARISON_ROOT="${PIPELINE_ROOT}/comparisons"
TIMING_ROOT="${PIPELINE_ROOT}/wind_event_timing"
mkdir -p "${TRAIN_ROOT}" "${RESULT_ROOT}" "${COMPARISON_ROOT}" "${TIMING_ROOT}"
printf '%s\n' "${BASELINE_RESULT}" > "${PIPELINE_ROOT}/baseline_result_reference.txt"

ENVIRONMENT_FILE="${LOG_FILE%.log}.environment.txt"
{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse HEAD)"
  echo "data=${DATA}"
  echo "baseline_result=${BASELINE_RESULT}"
  echo "seed=${SEED}"
  echo "generation_seed=${GEN_SEED}"
  echo "ensemble_members=${NSAMPLES}"
  echo "experiment_family=station24_cdsg_2b_2c"
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
}

EXP2B_NAME="station24_state_v1_cdsg_lite_parallel_${JOB_STAMP}"
train_and_generate \
  state_v1_cdsg_lite_parallel "${CONFIG_2B}" "${EXP2B_NAME}" \
  "state_v1_cdsg_lite_parallel_val_n${NSAMPLES}_seed${GEN_SEED}"
EXP2B_RUN=${LAST_RUN}
EXP2B_RESULT=${LAST_RESULT}

EXP2C_NAME="station24_state_v1_cdsg_lite_hybrid_dynamic_${JOB_STAMP}"
train_and_generate \
  state_v1_cdsg_lite_hybrid_dynamic "${CONFIG_2C}" "${EXP2C_NAME}" \
  "state_v1_cdsg_lite_hybrid_dynamic_val_n${NSAMPLES}_seed${GEN_SEED}"
EXP2C_RUN=${LAST_RUN}
EXP2C_RESULT=${LAST_RESULT}

compare_pair() {
  local left_result=$1
  local right_result=$2
  local output=$3
  local left_variant=$4
  local right_variant=$5
  local left_label=$6
  local right_label=$7
  local left_parallel=$8
  local right_parallel=$9
  local left_adjacency=${10}
  local right_adjacency=${11}
  local prefix=${12}
  local parallel_args=()
  [[ -n "${left_parallel}" ]] && parallel_args+=(--baseline-parallel-levels "${left_parallel}")
  [[ -n "${right_parallel}" ]] && parallel_args+=(--candidate-parallel-levels "${right_parallel}")
  "${PYTHON_BIN}" tools/compare_station24_multiscale_2a.py \
    "${left_result}" "${right_result}" \
    --data-path "${DATA}" --output-dir "${output}" \
    --baseline-variant "${left_variant}" \
    --candidate-variant "${right_variant}" \
    --baseline-label "${left_label}" \
    --candidate-label "${right_label}" \
    --baseline-spatial-levels bottleneck \
    --candidate-spatial-levels bottleneck \
    "${parallel_args[@]}" \
    --baseline-parallel-adjacency "${left_adjacency}" \
    --candidate-parallel-adjacency "${right_adjacency}" \
    --title "Station24 ${left_label} versus ${right_label}" \
    --figure-prefix "${prefix}"
}

echo "COMPARISON_START baseline_vs_2b"
compare_pair \
  "${BASELINE_RESULT}" "${EXP2B_RESULT}" \
  "${COMPARISON_ROOT}/baseline_vs_2b" \
  state_v1_fixed_graph state_v1_cdsg_lite_parallel \
  "State V1 baseline" "Experiment 2B fixed parallel" \
  "" encoder_0 fixed fixed baseline_vs_2b

echo "COMPARISON_START baseline_vs_2c"
compare_pair \
  "${BASELINE_RESULT}" "${EXP2C_RESULT}" \
  "${COMPARISON_ROOT}/baseline_vs_2c" \
  state_v1_fixed_graph state_v1_cdsg_lite_hybrid_dynamic \
  "State V1 baseline" "Experiment 2C hybrid dynamic" \
  "" encoder_0 fixed hybrid_dynamic baseline_vs_2c

echo "COMPARISON_START 2b_vs_2c"
compare_pair \
  "${EXP2B_RESULT}" "${EXP2C_RESULT}" \
  "${COMPARISON_ROOT}/2b_vs_2c" \
  state_v1_cdsg_lite_parallel state_v1_cdsg_lite_hybrid_dynamic \
  "Experiment 2B fixed parallel" "Experiment 2C hybrid dynamic" \
  encoder_0 encoder_0 fixed hybrid_dynamic 2b_vs_2c

timing_pair() {
  local left_result=$1
  local right_result=$2
  local output=$3
  local left_variant=$4
  local right_variant=$5
  local left_label=$6
  local right_label=$7
  "${PYTHON_BIN}" tools/diagnose_station24_wind_event_timing.py \
    "${left_result}" "${right_result}" \
    --data-path "${DATA}" --output-dir "${output}" \
    --baseline-variant "${left_variant}" \
    --candidate-variant "${right_variant}" \
    --baseline-label "${left_label}" \
    --candidate-label "${right_label}"
}

echo "TIMING_DIAGNOSTICS_START"
timing_pair \
  "${BASELINE_RESULT}" "${EXP2B_RESULT}" "${TIMING_ROOT}/baseline_vs_2b" \
  state_v1_fixed_graph state_v1_cdsg_lite_parallel \
  "State V1 baseline" "Experiment 2B fixed parallel"
timing_pair \
  "${BASELINE_RESULT}" "${EXP2C_RESULT}" "${TIMING_ROOT}/baseline_vs_2c" \
  state_v1_fixed_graph state_v1_cdsg_lite_hybrid_dynamic \
  "State V1 baseline" "Experiment 2C hybrid dynamic"
timing_pair \
  "${EXP2B_RESULT}" "${EXP2C_RESULT}" "${TIMING_ROOT}/2b_vs_2c" \
  state_v1_cdsg_lite_parallel state_v1_cdsg_lite_hybrid_dynamic \
  "Experiment 2B fixed parallel" "Experiment 2C hybrid dynamic"

"${PYTHON_BIN}" - \
  "${BASELINE_RESULT}" "${EXP2B_RESULT}" "${EXP2C_RESULT}" \
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
            "checkpoint_validation_mse": run["checkpoint_validation_mse"],
            "parallel_levels": ",".join(run.get("parallel_spatial_fusion_levels", [])),
            "parallel_adjacency": run.get("parallel_spatial_adjacency_mode", "fixed"),
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
ARCHIVE="${OUTPUT_ROOT}/station24_cdsg_2b_2c_${JOB_STAMP}.tar.gz"
tar -czf "${ARCHIVE}" -C "$(dirname "${PIPELINE_ROOT}")" "$(basename "${PIPELINE_ROOT}")"
{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "BASELINE_RESULT=${BASELINE_RESULT}"
  echo "EXP2B_RUN=${EXP2B_RUN}"
  echo "EXP2B_RESULT=${EXP2B_RESULT}"
  echo "EXP2C_RUN=${EXP2C_RUN}"
  echo "EXP2C_RESULT=${EXP2C_RESULT}"
  echo "COMPARISON_DIR=${COMPARISON_ROOT}"
  echo "TIMING_DIAGNOSTIC_DIR=${TIMING_ROOT}"
  echo "ARCHIVE=${ARCHIVE}"
  echo "ALL_STATION24_CDSG_2B_2C_EXPERIMENTS_COMPLETED"
} > "${RESULT_FILE}"
cat "${RESULT_FILE}"
