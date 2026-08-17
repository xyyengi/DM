#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
EXPECTED_BRANCH=${EXPECTED_BRANCH:-experiment/24site-historical-spatial-prior}
PYTHON_BIN=${PYTHON_BIN:-python}
DATA=${DATA:-diffusion_input_station}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_shandong/station24}
LOG_ROOT=${LOG_ROOT:-logs/station24}
REFERENCE_RUN=${REFERENCE_RUN:-}
NSAMPLES=${NSAMPLES:-500}
GEN_SEED=${GEN_SEED:-424242}
ISSUE_BATCH=${ISSUE_BATCH:-1}
MEMBER_CHUNK=${MEMBER_CHUNK:-10}
ENERGY_SCORE_MEMBERS=${ENERGY_SCORE_MEMBERS:-80}
CONVERGENCE_RESAMPLES=${CONVERGENCE_RESAMPLES:-20}
CONVERGENCE_SEED=${CONVERGENCE_SEED:-20260817}

die() { echo "ERROR: $*" >&2; exit 1; }

record_exit() {
  local code=$?
  trap - EXIT
  if [[ -n "${STATUS_FILE:-}" ]]; then
    local state=completed
    [[ ${code} -eq 0 ]] || state=failed
    {
      printf 'state=%s\n' "${state}"
      printf 'pid=%s\n' "${BASHPID}"
      printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
      printf 'exit_code=%s\n' "${code}"
      [[ -n "${REFERENCE_RUN:-}" ]] && printf 'reference_run=%s\n' "${REFERENCE_RUN}"
      [[ -n "${RESULT_DIR:-}" ]] && printf 'result_dir=%s\n' "${RESULT_DIR}"
      [[ -n "${ANALYSIS_DIR:-}" ]] && printf 'analysis_dir=%s\n' "${ANALYSIS_DIR}"
      [[ -n "${ARCHIVE_FILE:-}" ]] && printf 'archive=%s\n' "${ARCHIVE_FILE}"
    } > "${STATUS_FILE}"
  fi
  exit "${code}"
}

launch_background() {
  cd "${REPO_ROOT}"
  mkdir -p "${LOG_ROOT}"
  local stamp log_file pid_file status_file
  stamp=$(date +%Y%m%d_%H%M%S)
  log_file="${LOG_ROOT}/station24_reference500_${stamp}.log"
  pid_file="${LOG_ROOT}/station24_reference500_${stamp}.pid"
  status_file="${LOG_ROOT}/station24_reference500_${stamp}.status"
  nohup setsid env \
    PYTHONUNBUFFERED=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    OMP_NUM_THREADS=1 \
    STATION24_REFERENCE500_INTERNAL_WORKER=1 \
    JOB_STAMP="${stamp}" LOG_FILE="${log_file}" \
    PID_FILE="${pid_file}" STATUS_FILE="${status_file}" \
    EXPECTED_BRANCH="${EXPECTED_BRANCH}" PYTHON_BIN="${PYTHON_BIN}" \
    DATA="${DATA}" OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
    REFERENCE_RUN="${REFERENCE_RUN}" NSAMPLES="${NSAMPLES}" \
    GEN_SEED="${GEN_SEED}" ISSUE_BATCH="${ISSUE_BATCH}" \
    MEMBER_CHUNK="${MEMBER_CHUNK}" \
    ENERGY_SCORE_MEMBERS="${ENERGY_SCORE_MEMBERS}" \
    CONVERGENCE_RESAMPLES="${CONVERGENCE_RESAMPLES}" \
    CONVERGENCE_SEED="${CONVERGENCE_SEED}" \
    bash "$0" > "${log_file}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_file}"
  printf 'state=running\npid=%s\nstarted_at=%s\n' \
    "${pid}" "$(date --iso-8601=seconds)" > "${status_file}"
  echo "Started Station24 500-member validation reference"
  echo "PID: ${pid}"
  echo "Log: ${log_file}"
  echo "Status: ${status_file}"
  echo "Monitor: tail -f '${log_file}'"
  echo "Stop entire job: kill -- -\$(cat '${pid_file}')"
}

if [[ "${STATION24_REFERENCE500_INTERNAL_WORKER:-0}" != "1" ]]; then
  launch_background
  exit 0
fi

trap record_exit EXIT
cd "${REPO_ROOT}"
command -v git >/dev/null || die "git is unavailable"
command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
command -v "${PYTHON_BIN}" >/dev/null || die "${PYTHON_BIN} is unavailable"
command -v tar >/dev/null || die "tar is unavailable"
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] \
  || die "expected branch ${EXPECTED_BRANCH}, got $(git branch --show-current)"
git diff --quiet || die "tracked working-tree changes are present; commit/pull first"
git diff --cached --quiet || die "staged changes are present; commit/pull first"
[[ "${NSAMPLES}" -eq 500 ]] || die "phase-0 reference requires exactly 500 members"
[[ "${ENERGY_SCORE_MEMBERS}" -ge 2 && "${ENERGY_SCORE_MEMBERS}" -le 500 ]] \
  || die "ENERGY_SCORE_MEMBERS must be in [2,500]"

for required in \
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

if [[ -z "${REFERENCE_RUN}" ]]; then
  REFERENCE_RUN=$("${PYTHON_BIN}" - "${OUTPUT_ROOT}" <<'PY'
import sys
from pathlib import Path

import yaml

root = Path(sys.argv[1])
matches = []
for config_path in root.rglob("config_used.yaml"):
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if config.get("experiment", {}).get("variant") != "state_v1_cdsg_2d_conditional_scale":
        continue
    run_dir = config_path.parent
    checkpoint = run_dir / "checkpoints" / "model_best.pt"
    if checkpoint.is_file():
        matches.append((checkpoint.stat().st_mtime, run_dir))
if not matches:
    raise SystemExit(
        "no state_v1_cdsg_2d_conditional_scale checkpoint found; set REFERENCE_RUN"
    )
print(max(matches, key=lambda item: item[0])[1])
PY
  )
fi
[[ -f "${REFERENCE_RUN}/config_used.yaml" ]] \
  || die "missing ${REFERENCE_RUN}/config_used.yaml"
[[ -f "${REFERENCE_RUN}/checkpoints/model_best.pt" ]] \
  || die "missing best checkpoint under ${REFERENCE_RUN}"
[[ -f "${REFERENCE_RUN}/residual_scale.json" ]] \
  || die "missing ${REFERENCE_RUN}/residual_scale.json"

"${PYTHON_BIN}" - "${REFERENCE_RUN}" <<'PY'
import sys
from pathlib import Path

import yaml

run_dir = Path(sys.argv[1])
config = yaml.safe_load((run_dir / "config_used.yaml").read_text(encoding="utf-8"))
variant = config.get("experiment", {}).get("variant")
if variant != "state_v1_cdsg_2d_conditional_scale":
    raise SystemExit(f"unexpected reference variant: {variant}")
if config.get("data", {}).get("sequence_length") != 168:
    raise SystemExit("reference run is not a 168-hour model")
print(f"REFERENCE_RUN_OK variant={variant} path={run_dir}")
PY

PIPELINE_ROOT="${OUTPUT_ROOT}/reference500_${JOB_STAMP}"
RESULT_DIR="${PIPELINE_ROOT}/validation_result"
ANALYSIS_DIR="${PIPELINE_ROOT}/member_convergence"
ARCHIVE_FILE="${PIPELINE_ROOT}.tar.gz"
mkdir -p "${PIPELINE_ROOT}"

ENVIRONMENT_FILE="${PIPELINE_ROOT}/server_environment.txt"
{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse HEAD)"
  echo "data=${DATA}"
  echo "reference_run=${REFERENCE_RUN}"
  echo "split=val"
  echo "test_used=false"
  echo "generation_seed=${GEN_SEED}"
  echo "ensemble_members=${NSAMPLES}"
  echo "energy_score_members=${ENERGY_SCORE_MEMBERS}"
  echo "convergence_resamples=${CONVERGENCE_RESAMPLES}"
  nvidia-smi
} > "${ENVIRONMENT_FILE}"
cp "${REFERENCE_RUN}/config_used.yaml" "${PIPELINE_ROOT}/reference_config_used.yaml"

echo "GENERATION_START members=${NSAMPLES} split=val reference_run=${REFERENCE_RUN}"
"${PYTHON_BIN}" generate_station24.py \
  --run-dir "${REFERENCE_RUN}" --data-path "${DATA}" \
  --output-dir "${RESULT_DIR}" --split val \
  --n-samples "${NSAMPLES}" --seed "${GEN_SEED}" \
  --issue-batch-size "${ISSUE_BATCH}" \
  --member-chunk-size "${MEMBER_CHUNK}" \
  --energy-score-member-limit "${ENERGY_SCORE_MEMBERS}"

"${PYTHON_BIN}" - "${RESULT_DIR}" "${NSAMPLES}" "${GEN_SEED}" \
  "${ENERGY_SCORE_MEMBERS}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
metadata = json.loads((path / "generation_metadata.json").read_text(encoding="utf-8"))
expected = {
    "condition_variant": "state_v1_cdsg_2d_conditional_scale",
    "split": "val",
    "n_samples": int(sys.argv[2]),
    "generation_seed": int(sys.argv[3]),
    "evaluation_member_count": int(sys.argv[2]),
    "energy_score_member_count": int(sys.argv[4]),
    "test_used": False,
}
for key, value in expected.items():
    if metadata.get(key) != value:
        raise SystemExit(f"metadata mismatch {key}: {metadata.get(key)!r} != {value!r}")
print(f"GENERATION_AUDIT_OK result_dir={path}")
PY

echo "CONVERGENCE_START sizes=20,40,80,160,300,500"
"${PYTHON_BIN}" tools/analyze_station24_member_convergence.py \
  "${RESULT_DIR}" --data-path "${DATA}" --output-dir "${ANALYSIS_DIR}" \
  --member-sizes 20 40 80 160 300 500 \
  --resamples "${CONVERGENCE_RESAMPLES}" --seed "${CONVERGENCE_SEED}"

cp "${LOG_FILE}" "${PIPELINE_ROOT}/pipeline.log"
printf '%s\n' "${REFERENCE_RUN}" > "${PIPELINE_ROOT}/reference_run.txt"
printf '%s\n' "${RESULT_DIR}" > "${PIPELINE_ROOT}/result_dir.txt"

echo "ARCHIVE_START path=${ARCHIVE_FILE}"
tar -czf "${ARCHIVE_FILE}" -C "$(dirname "${PIPELINE_ROOT}")" \
  "$(basename "${PIPELINE_ROOT}")"
if command -v sha256sum >/dev/null; then
  sha256sum "${ARCHIVE_FILE}" > "${ARCHIVE_FILE}.sha256"
fi

echo "REFERENCE500_COMPLETE"
echo "REFERENCE_RUN=${REFERENCE_RUN}"
echo "RESULT_DIR=${RESULT_DIR}"
echo "ANALYSIS_DIR=${ANALYSIS_DIR}"
echo "ARCHIVE=${ARCHIVE_FILE}"
