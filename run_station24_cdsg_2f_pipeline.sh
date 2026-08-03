#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
EXPECTED_BRANCH=${EXPECTED_BRANCH:-experiment/24site-wind-solar-168h}
PYTHON_BIN=${PYTHON_BIN:-python}
DATA=${DATA:-diffusion_input_station}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_shandong/station24}
LOG_ROOT=${LOG_ROOT:-logs/station24}
REFERENCE_2D_RUN=${REFERENCE_2D_RUN:-}
SEED=${SEED:-2027}
NSAMPLES=${NSAMPLES:-160}
GEN_SEED=${GEN_SEED:-424242}
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
  log_file="${LOG_ROOT}/station24_cdsg_2f_${stamp}.log"
  pid_file="${LOG_ROOT}/station24_cdsg_2f_${stamp}.pid"
  status_file="${LOG_ROOT}/station24_cdsg_2f_${stamp}.status"
  nohup setsid env \
    PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
    STATION24_CDSG_2F_INTERNAL_WORKER=1 JOB_STAMP="${stamp}" \
    LOG_FILE="${log_file}" PID_FILE="${pid_file}" STATUS_FILE="${status_file}" \
    EXPECTED_BRANCH="${EXPECTED_BRANCH}" PYTHON_BIN="${PYTHON_BIN}" \
    DATA="${DATA}" OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
    REFERENCE_2D_RUN="${REFERENCE_2D_RUN}" SEED="${SEED}" \
    NSAMPLES="${NSAMPLES}" GEN_SEED="${GEN_SEED}" \
    ISSUE_BATCH="${ISSUE_BATCH}" MEMBER_CHUNK="${MEMBER_CHUNK}" \
    bash "$0" > "${log_file}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_file}"
  printf 'state=running\npid=%s\nstarted_at=%s\n' \
    "${pid}" "$(date --iso-8601=seconds)" > "${status_file}"
  echo "Started Station24 2F pipeline (2D regenerate -> 2F train/generate -> audit)"
  echo "PID: ${pid}"
  echo "Log: ${log_file}"
  echo "Status: ${status_file}"
  echo "Monitor: tail -f '${log_file}'"
  echo "Stop entire pipeline: kill -- -\$(cat '${pid_file}')"
}

if [[ "${STATION24_CDSG_2F_INTERNAL_WORKER:-0}" != "1" ]]; then
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

CONFIG_2D="configs/station24_state_v1_cdsg_2d_conditional_scale_168h.yaml"
CONFIG_2F="configs/station24_state_v1_cdsg_2f_common_event_168h.yaml"
[[ -f "${CONFIG_2D}" && -f "${CONFIG_2F}" ]] || die "2D/2F config missing"
for required in train_forecast.npy train_actual.npy train_residual.npy \
  train_fill_mask.npy train_issue_dates.csv val_forecast.npy val_actual.npy \
  val_residual.npy val_fill_mask.npy val_issue_dates.csv station_features.npy \
  station_adjacency.npy station_order.csv export_metadata.json; do
  [[ -f "${DATA}/${required}" ]] || die "missing data artifact ${DATA}/${required}"
done

"${PYTHON_BIN}" - <<'PY'
import torch
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

if [[ -z "${REFERENCE_2D_RUN}" ]]; then
  REFERENCE_2D_RUN=$("${PYTHON_BIN}" - "${OUTPUT_ROOT}" <<'PY'
import sys, torch
from pathlib import Path
candidates = []
for checkpoint_path in Path(sys.argv[1]).glob("**/checkpoints/model_best.pt"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("condition_variant") == "state_v1_cdsg_2d_conditional_scale":
        candidates.append((checkpoint_path.stat().st_mtime, checkpoint_path.parent.parent))
if not candidates:
    raise SystemExit("no 2D checkpoint found; set REFERENCE_2D_RUN explicitly")
print(max(candidates)[1])
PY
  )
fi
[[ -f "${REFERENCE_2D_RUN}/checkpoints/model_best.pt" ]] \
  || die "REFERENCE_2D_RUN has no model_best.pt: ${REFERENCE_2D_RUN}"

"${PYTHON_BIN}" - "${REFERENCE_2D_RUN}" "${DATA}" <<'PY'
import sys, torch, yaml
from pathlib import Path
from src.models.station_conditioned_diffusion import Station24DiffusionModel
from station_dataset import load_station_static_data
run = Path(sys.argv[1])
checkpoint = torch.load(run / "checkpoints/model_best.pt", map_location="cpu", weights_only=False)
if checkpoint.get("condition_variant") != "state_v1_cdsg_2d_conditional_scale":
    raise SystemExit(
        f"reference is not 2D: {checkpoint.get('condition_variant')!r}; "
        "set REFERENCE_2D_RUN explicitly"
    )
config = yaml.safe_load((run / "config_used.yaml").read_text(encoding="utf-8"))
static = load_station_static_data(sys.argv[2])
model = Station24DiffusionModel(config["model"], static["station_features"], static["station_adjacency"], static["station_capacities"])
model.load_state_dict(checkpoint.get("ema_model_state_dict", checkpoint["model_state_dict"]), strict=True)
print(f"REFERENCE_2D_RUN_OK run={run} parameters={checkpoint['parameter_count']}")
PY

"${PYTHON_BIN}" - "${CONFIG_2D}" "${CONFIG_2F}" "${DATA}" <<'PY'
import sys, yaml
from src.models.station_conditioned_diffusion import Station24DiffusionModel
from station_dataset import load_station_static_data
configs = [yaml.safe_load(open(path, encoding="utf-8")) for path in sys.argv[1:3]]
base, candidate = configs
if not candidate["model"].get("use_forecast_revision"):
    raise SystemExit("2F signed forecast revision must be enabled")
if not candidate["model"].get("use_wind_common_residual_head"):
    raise SystemExit("2F common wind head must be enabled")
if not candidate["model"].get("use_extreme_event_weighting"):
    raise SystemExit("2F train-only event weighting must be enabled")
static = load_station_static_data(sys.argv[3])
models = [Station24DiffusionModel(c["model"], static["station_features"], static["station_adjacency"], static["station_capacities"]) for c in configs]
counts = [sum(p.numel() for p in model.parameters()) for model in models]
increment = counts[1] - counts[0]
if increment <= 0 or increment / counts[0] >= 0.05:
    raise SystemExit(f"2F parameter increment is not lightweight: {counts}")
print(f"2F_PARITY_OK baseline={counts[0]} candidate={counts[1]} increment={increment} ratio={increment/counts[0]:.4%}")
PY

PIPELINE_ROOT="${OUTPUT_ROOT}/cdsg_2f_${JOB_STAMP}"
TRAIN_ROOT="${PIPELINE_ROOT}/training"
RESULT_ROOT="${PIPELINE_ROOT}/validation_results"
COMPARISON_ROOT="${PIPELINE_ROOT}/comparisons/2d_vs_2f"
TIMING_ROOT="${PIPELINE_ROOT}/wind_event_timing/2d_vs_2f"
TAIL_ROOT="${PIPELINE_ROOT}/extreme_wind_tail/2d_vs_2f"
mkdir -p "${TRAIN_ROOT}" "${RESULT_ROOT}" "${COMPARISON_ROOT}" "${TIMING_ROOT}" "${TAIL_ROOT}"
printf '%s\n' "${REFERENCE_2D_RUN}" > "${PIPELINE_ROOT}/reference_2d_run.txt"

ENVIRONMENT_FILE="${LOG_FILE%.log}.environment.txt"
{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse HEAD)"
  echo "data=${DATA}"
  echo "reference_2d_run=${REFERENCE_2D_RUN}"
  echo "training_seed=${SEED}"
  echo "generation_seed=${GEN_SEED}"
  echo "ensemble_members=${NSAMPLES}"
  echo "test_used=false"
  nvidia-smi
} > "${ENVIRONMENT_FILE}"

BASELINE_RESULT="${RESULT_ROOT}/state_v1_cdsg_2d_conditional_scale_val_n${NSAMPLES}_seed${GEN_SEED}"
echo "BASELINE_GENERATION_START members=${NSAMPLES} run=${REFERENCE_2D_RUN}"
"${PYTHON_BIN}" generate_station24.py \
  --run-dir "${REFERENCE_2D_RUN}" --data-path "${DATA}" \
  --output-dir "${BASELINE_RESULT}" --split val --n-samples "${NSAMPLES}" \
  --seed "${GEN_SEED}" --issue-batch-size "${ISSUE_BATCH}" \
  --member-chunk-size "${MEMBER_CHUNK}"

EXP2F_NAME="station24_cdsg_2f_common_event_${JOB_STAMP}"
echo "TRAIN_START variant=state_v1_cdsg_2f_common_event"
"${PYTHON_BIN}" train_station24.py \
  --config "${CONFIG_2F}" --data-path "${DATA}" \
  --output-root "${TRAIN_ROOT}" --exp-name "${EXP2F_NAME}" --seed "${SEED}"
shopt -s nullglob
matches=("${TRAIN_ROOT}"/*_"${EXP2F_NAME}"_seed"${SEED}")
shopt -u nullglob
[[ ${#matches[@]} -eq 1 ]] || die "expected one 2F run, found ${#matches[@]}"
EXP2F_RUN=${matches[0]}
cp "${ENVIRONMENT_FILE}" "${EXP2F_RUN}/logs/server_environment.txt"
EXP2F_RESULT="${RESULT_ROOT}/state_v1_cdsg_2f_common_event_val_n${NSAMPLES}_seed${GEN_SEED}"
echo "GENERATION_START variant=state_v1_cdsg_2f_common_event members=${NSAMPLES}"
"${PYTHON_BIN}" generate_station24.py \
  --run-dir "${EXP2F_RUN}" --data-path "${DATA}" \
  --output-dir "${EXP2F_RESULT}" --split val --n-samples "${NSAMPLES}" \
  --seed "${GEN_SEED}" --issue-batch-size "${ISSUE_BATCH}" \
  --member-chunk-size "${MEMBER_CHUNK}"

"${PYTHON_BIN}" tools/compare_station24_multiscale_2a.py \
  "${BASELINE_RESULT}" "${EXP2F_RESULT}" --data-path "${DATA}" \
  --output-dir "${COMPARISON_ROOT}" \
  --baseline-variant state_v1_cdsg_2d_conditional_scale \
  --candidate-variant state_v1_cdsg_2f_common_event \
  --baseline-label "2D conditional scale" --candidate-label "2F common-event" \
  --baseline-spatial-levels bottleneck --candidate-spatial-levels bottleneck \
  --baseline-parallel-levels encoder_0 --candidate-parallel-levels encoder_0 \
  --baseline-parallel-adjacency fixed --candidate-parallel-adjacency fixed \
  --title "Station24 2D versus 2F common-event" --figure-prefix 2d_vs_2f

"${PYTHON_BIN}" tools/diagnose_station24_wind_event_timing.py \
  "${BASELINE_RESULT}" "${EXP2F_RESULT}" --data-path "${DATA}" \
  --output-dir "${TIMING_ROOT}" \
  --baseline-variant state_v1_cdsg_2d_conditional_scale \
  --candidate-variant state_v1_cdsg_2f_common_event \
  --baseline-label "2D conditional scale" --candidate-label "2F common-event"

"${PYTHON_BIN}" tools/plot_station24_extreme_tail.py \
  --baseline "${BASELINE_RESULT}" --candidate "${EXP2F_RESULT}" \
  --data-path "${DATA}" --output-dir "${TAIL_ROOT}" --top-issues 3

RESULT_FILE="${LOG_FILE%.log}.results.env"
ARCHIVE="${OUTPUT_ROOT}/station24_cdsg_2f_${JOB_STAMP}.tar.gz"
tar -czf "${ARCHIVE}" -C "$(dirname "${PIPELINE_ROOT}")" "$(basename "${PIPELINE_ROOT}")"
{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "REFERENCE_2D_RUN=${REFERENCE_2D_RUN}"
  echo "BASELINE_2D_RESULT=${BASELINE_RESULT}"
  echo "EXP2F_RUN=${EXP2F_RUN}"
  echo "EXP2F_RESULT=${EXP2F_RESULT}"
  echo "COMPARISON_DIR=${COMPARISON_ROOT}"
  echo "TIMING_DIAGNOSTIC_DIR=${TIMING_ROOT}"
  echo "EXTREME_TAIL_DIR=${TAIL_ROOT}"
  echo "ARCHIVE=${ARCHIVE}"
  echo "ALL_STATION24_CDSG_2F_PIPELINE_COMPLETED"
} > "${RESULT_FILE}"
cat "${RESULT_FILE}"
