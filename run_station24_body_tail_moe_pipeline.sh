#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
EXPECTED_BRANCH=${EXPECTED_BRANCH:-experiment/24site-body-tail-moe}
PYTHON_BIN=${PYTHON_BIN:-python}
DATA=${DATA:-diffusion_input_station}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_shandong/station24}
LOG_ROOT=${LOG_ROOT:-logs/station24}
REFERENCE_HISTORY_RESULT=${REFERENCE_HISTORY_RESULT:-}
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
  log_file="${LOG_ROOT}/station24_body_tail_moe_${stamp}.log"
  pid_file="${LOG_ROOT}/station24_body_tail_moe_${stamp}.pid"
  status_file="${LOG_ROOT}/station24_body_tail_moe_${stamp}.status"
  nohup setsid env \
    PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
    STATION24_BODY_TAIL_MOE_INTERNAL_WORKER=1 JOB_STAMP="${stamp}" \
    LOG_FILE="${log_file}" PID_FILE="${pid_file}" STATUS_FILE="${status_file}" \
    EXPECTED_BRANCH="${EXPECTED_BRANCH}" PYTHON_BIN="${PYTHON_BIN}" \
    DATA="${DATA}" OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
    REFERENCE_HISTORY_RESULT="${REFERENCE_HISTORY_RESULT}" SEED="${SEED}" \
    NSAMPLES="${NSAMPLES}" GEN_SEED="${GEN_SEED}" \
    ENERGY_MEMBERS="${ENERGY_MEMBERS}" ISSUE_BATCH="${ISSUE_BATCH}" \
    MEMBER_CHUNK="${MEMBER_CHUNK}" \
    bash "$0" > "${log_file}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_file}"
  printf 'state=running\npid=%s\nstarted_at=%s\n' \
    "${pid}" "$(date --iso-8601=seconds)" > "${status_file}"
  echo "Started Station24 body-tail MoE pipeline"
  echo "PID: ${pid}"
  echo "Log: ${log_file}"
  echo "Status: ${status_file}"
  echo "Monitor: tail -f '${log_file}'"
  echo "Stop entire pipeline: kill -- -\$(cat '${pid_file}')"
}

if [[ "${STATION24_BODY_TAIL_MOE_INTERNAL_WORKER:-0}" != "1" ]]; then
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

CONFIG_BASE=configs/station24_geo_history_actual_dual_168h.yaml
CONFIG_MOE=configs/station24_geo_history_actual_body_tail_moe_168h.yaml
[[ -f "${CONFIG_BASE}" && -f "${CONFIG_MOE}" ]] \
  || die "body-tail MoE configuration is missing"

"${PYTHON_BIN}" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

echo "BODY_TAIL_UNIT_TEST_START"
"${PYTHON_BIN}" -m unittest discover -s tests \
  -p 'test_station24_pipeline.py' -k body_tail

if [[ -z "${REFERENCE_HISTORY_RESULT}" ]]; then
  while IFS= read -r -d '' candidate; do
    if [[ -z "${REFERENCE_HISTORY_RESULT}" || "${candidate}" -nt "${REFERENCE_HISTORY_RESULT}" ]]; then
      REFERENCE_HISTORY_RESULT=${candidate}
    fi
  done < <(find "${OUTPUT_ROOT}" -type d -path '*/historical_dual_graph_*/validation_results/geo_history_actual_dual_val_n500_seed424242' -print0)
fi
[[ -n "${REFERENCE_HISTORY_RESULT}" && -d "${REFERENCE_HISTORY_RESULT}" ]] \
  || die "historical-spatial 500-member result not found; set REFERENCE_HISTORY_RESULT"

REFERENCE_HISTORY_RUN=$("${PYTHON_BIN}" - "${REFERENCE_HISTORY_RESULT}" "${NSAMPLES}" "${GEN_SEED}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
meta = json.loads((path / "generation_metadata.json").read_text(encoding="utf-8"))
expected = {
    "condition_variant": "geo_history_actual_dual",
    "split": "val",
    "n_samples": int(sys.argv[2]),
    "generation_seed": int(sys.argv[3]),
    "test_used": False,
}
for key, value in expected.items():
    if meta.get(key) != value:
        raise SystemExit(f"reference mismatch {key}: {meta.get(key)!r} != {value!r}")
print(meta["run_dir"])
PY
)
BODY_CHECKPOINT="${REFERENCE_HISTORY_RUN}/checkpoints/model_best.pt"
SECONDARY_GRAPH="${REFERENCE_HISTORY_RUN}/graphs/secondary_adjacency.npy"
[[ -f "${BODY_CHECKPOINT}" ]] || die "reference checkpoint missing: ${BODY_CHECKPOINT}"
[[ -f "${SECONDARY_GRAPH}" ]] || die "reference graph missing: ${SECONDARY_GRAPH}"

"${PYTHON_BIN}" - "${CONFIG_BASE}" "${CONFIG_MOE}" "${DATA}" "${SECONDARY_GRAPH}" "${BODY_CHECKPOINT}" <<'PY'
import copy
import sys
import numpy as np
import torch
import yaml

from src.models.station_conditioned_diffusion import Station24DiffusionModel
from station_dataset import fit_station_event_replay, load_station_static_data

base = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
candidate = yaml.safe_load(open(sys.argv[2], encoding="utf-8"))
allowed = {
    "use_body_tail_experts", "tail_expert_channels", "tail_epsilon_context_hours",
    "tail_common_gate_init",
    "tail_gate_channels", "tail_gate_prior_probability", "tail_gate_loss_weight",
    "use_extreme_event_weighting", "use_event_replay_x0",
    "event_replay_window_hours", "event_replay_merge_gap_hours",
    "event_replay_quantiles", "event_replay_weights",
    "event_x0_magnitude_loss_weight", "event_x0_timing_loss_weight",
    "event_x0_sync_loss_weight", "event_x0_window_hours",
    "event_x0_error_scale", "event_x0_timing_temperature",
}
base_core = copy.deepcopy(base)
candidate_core = copy.deepcopy(candidate)
base_core.pop("experiment")
candidate_core.pop("experiment")
candidate_core["train"] = copy.deepcopy(base_core["train"])
for key in allowed:
    candidate_core["model"].pop(key, None)
if base_core != candidate_core:
    raise SystemExit("body-tail experiment changed fields outside its identity and expert settings")

spec = fit_station_event_replay(sys.argv[3], candidate["model"])
if spec["independent_event_count"] <= 0:
    raise SystemExit("body-tail experiment found no independent events")
static = load_station_static_data(sys.argv[3])
secondary = torch.from_numpy(np.load(sys.argv[4]).astype(np.float32))
body = Station24DiffusionModel(
    base["model"], static["station_features"], static["station_adjacency"],
    static["station_capacities"], secondary,
)
moe = Station24DiffusionModel(
    candidate["model"], static["station_features"], static["station_adjacency"],
    static["station_capacities"], secondary,
)
checkpoint = torch.load(sys.argv[5], map_location="cpu", weights_only=False)
state = checkpoint.get("ema_model_state_dict", checkpoint["model_state_dict"])
body.load_state_dict(state, strict=True)
missing = moe.load_state_dict(state, strict=False)
if set(missing.missing_keys) != set(moe.body_tail_state_dict_keys) or missing.unexpected_keys:
    raise SystemExit("body-tail initialization is not isolated to expert parameters")
trainable = moe.configure_body_tail_training()
total = sum(p.numel() for p in moe.parameters())
active = sum(p.numel() for p in moe.parameters() if p.requires_grad)
if active / total >= 0.15:
    raise SystemExit(f"tail expert is too large: {active}/{total}")
print(
    "BODY_TAIL_PARITY_OK "
    f"body_parameters={sum(p.numel() for p in body.parameters())} "
    f"candidate_parameters={total} trainable={active} ratio={active/total:.4%} "
    f"events={spec['independent_event_count']} trainable_tensors={len(trainable)}"
)
PY

PIPELINE_ROOT="${OUTPUT_ROOT}/body_tail_moe_${JOB_STAMP}"
TRAIN_ROOT="${PIPELINE_ROOT}/training"
RESULT_ROOT="${PIPELINE_ROOT}/validation_results"
COMPARISON_ROOT="${PIPELINE_ROOT}/comparisons/history_vs_body_tail_moe"
TIMING_ROOT="${PIPELINE_ROOT}/wind_event_timing/history_vs_body_tail_moe"
ATTRIBUTION_ROOT="${PIPELINE_ROOT}/forecast_event_attribution/history_vs_body_tail_moe"
TAIL_ROOT="${PIPELINE_ROOT}/extreme_wind_tail/history_vs_body_tail_moe"
AUDIT_ROOT="${PIPELINE_ROOT}/body_tail_moe_audit"
SPECIALIZATION_ROOT="${PIPELINE_ROOT}/body_tail_specialization/history_vs_body_tail_moe"
mkdir -p "${TRAIN_ROOT}" "${RESULT_ROOT}" \
  "$(dirname "${COMPARISON_ROOT}")" "$(dirname "${TIMING_ROOT}")" \
  "$(dirname "${ATTRIBUTION_ROOT}")" "$(dirname "${TAIL_ROOT}")" \
  "$(dirname "${SPECIALIZATION_ROOT}")"

EXP_NAME="station24_body_tail_moe_${JOB_STAMP}"
echo "TRAIN_START variant=geo_history_actual_body_tail_moe"
"${PYTHON_BIN}" train_station24.py \
  --config "${CONFIG_MOE}" --data-path "${DATA}" \
  --secondary-adjacency "${SECONDARY_GRAPH}" \
  --initialize-checkpoint "${BODY_CHECKPOINT}" \
  --output-root "${TRAIN_ROOT}" --exp-name "${EXP_NAME}" --seed "${SEED}"
shopt -s nullglob
matches=("${TRAIN_ROOT}"/*_"${EXP_NAME}"_seed"${SEED}")
shopt -u nullglob
[[ ${#matches[@]} -eq 1 ]] || die "expected one body-tail run, found ${#matches[@]}"
MOE_RUN=${matches[0]}
MOE_RESULT="${RESULT_ROOT}/geo_history_actual_body_tail_moe_val_n${NSAMPLES}_seed${GEN_SEED}"

echo "GENERATION_START variant=geo_history_actual_body_tail_moe members=${NSAMPLES}"
"${PYTHON_BIN}" generate_station24.py \
  --run-dir "${MOE_RUN}" --data-path "${DATA}" \
  --output-dir "${MOE_RESULT}" --split val \
  --n-samples "${NSAMPLES}" --seed "${GEN_SEED}" \
  --issue-batch-size "${ISSUE_BATCH}" --member-chunk-size "${MEMBER_CHUNK}" \
  --energy-score-member-limit "${ENERGY_MEMBERS}"

echo "BODY_TAIL_AUDIT_START"
"${PYTHON_BIN}" tools/audit_station24_body_tail_moe.py \
  --body-run "${REFERENCE_HISTORY_RUN}" --body-result "${REFERENCE_HISTORY_RESULT}" \
  --candidate-run "${MOE_RUN}" --candidate-result "${MOE_RESULT}" \
  --output-dir "${AUDIT_ROOT}"

echo "BODY_TAIL_SPECIALIZATION_START"
"${PYTHON_BIN}" tools/analyze_station24_body_tail_specialization.py \
  --run-dir "${MOE_RUN}" --result-dir "${MOE_RESULT}" \
  --data-path "${DATA}" --output-dir "${SPECIALIZATION_ROOT}" --top-issues 5

echo "COMPARISON_START"
"${PYTHON_BIN}" tools/compare_station24_multiscale_2a.py \
  "${REFERENCE_HISTORY_RESULT}" "${MOE_RESULT}" \
  --data-path "${DATA}" --output-dir "${COMPARISON_ROOT}" \
  --baseline-variant geo_history_actual_dual \
  --candidate-variant geo_history_actual_body_tail_moe \
  --baseline-label "Historical-spatial body" \
  --candidate-label "Body-tail MoE" \
  --baseline-spatial-levels bottleneck --candidate-spatial-levels bottleneck \
  --baseline-parallel-levels encoder_0 --candidate-parallel-levels encoder_0 \
  --baseline-parallel-adjacency fixed --candidate-parallel-adjacency fixed \
  --title "Historical-spatial body versus body-tail MoE" \
  --figure-prefix history_vs_body_tail_moe

echo "WIND_EVENT_TIMING_START"
"${PYTHON_BIN}" tools/diagnose_station24_wind_event_timing.py \
  "${REFERENCE_HISTORY_RESULT}" "${MOE_RESULT}" --data-path "${DATA}" \
  --output-dir "${TIMING_ROOT}" \
  --baseline-variant geo_history_actual_dual \
  --candidate-variant geo_history_actual_body_tail_moe \
  --baseline-label "Historical-spatial body" --candidate-label "Body-tail MoE"

echo "FORECAST_EVENT_ATTRIBUTION_START"
"${PYTHON_BIN}" tools/diagnose_station24_forecast_event_attribution.py \
  "${REFERENCE_HISTORY_RESULT}" "${MOE_RESULT}" \
  --event-records "${TIMING_ROOT}/event_records.csv" --data-path "${DATA}" \
  --output-dir "${ATTRIBUTION_ROOT}"

echo "SUSTAINED_TAIL_AUDIT_START"
"${PYTHON_BIN}" tools/plot_station24_extreme_tail.py \
  --baseline "${REFERENCE_HISTORY_RESULT}" --candidate "${MOE_RESULT}" \
  --data-path "${DATA}" --output-dir "${TAIL_ROOT}" --top-issues 5 \
  --baseline-label "Historical-spatial body" --candidate-label "Body-tail MoE"

RESULT_FILE="${LOG_FILE%.log}.results.env"
ARCHIVE="${OUTPUT_ROOT}/station24_body_tail_moe_${JOB_STAMP}.tar.gz"
tar -czf "${ARCHIVE}" -C "$(dirname "${PIPELINE_ROOT}")" "$(basename "${PIPELINE_ROOT}")"
{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "REFERENCE_HISTORY_RUN=${REFERENCE_HISTORY_RUN}"
  echo "REFERENCE_HISTORY_RESULT=${REFERENCE_HISTORY_RESULT}"
  echo "MOE_RUN=${MOE_RUN}"
  echo "MOE_RESULT=${MOE_RESULT}"
  echo "COMPARISON_DIR=${COMPARISON_ROOT}"
  echo "TIMING_DIAGNOSTIC_DIR=${TIMING_ROOT}"
  echo "FORECAST_ATTRIBUTION_DIR=${ATTRIBUTION_ROOT}"
  echo "EXTREME_TAIL_DIR=${TAIL_ROOT}"
  echo "BODY_TAIL_AUDIT_DIR=${AUDIT_ROOT}"
  echo "BODY_TAIL_SPECIALIZATION_DIR=${SPECIALIZATION_ROOT}"
  echo "ARCHIVE=${ARCHIVE}"
  echo "ALL_BODY_TAIL_MOE_EXPERIMENTS_COMPLETED"
} > "${RESULT_FILE}"
cat "${RESULT_FILE}"
