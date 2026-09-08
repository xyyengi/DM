#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME:-dm_env}"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN=${PYTHON_BIN:-python}
DATA=${DATA:-diffusion_input_station}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_shandong/station24}
SOURCE_H1_ROOT=${1:-${SOURCE_H1_ROOT:-}}

die() { echo "ERROR: $*" >&2; exit 1; }

git diff --quiet || die "tracked working-tree changes are present; pull/commit first"
git diff --cached --quiet || die "staged working-tree changes are present; pull/commit first"
"${PYTHON_BIN}" -c "import torch; assert torch.cuda.is_available(); print('torch=', torch.__version__, 'gpu=', torch.cuda.get_device_name(0))"

if [[ -z "${SOURCE_H1_ROOT}" ]]; then
  while IFS= read -r -d '' candidate; do
    if [[ -z "${SOURCE_H1_ROOT}" || "${candidate}" -nt "${SOURCE_H1_ROOT}" ]]; then
      SOURCE_H1_ROOT=${candidate}
    fi
  done < <(find "${OUTPUT_ROOT}" -maxdepth 1 -type d -name 'jstd_event_hypothesis_h1_20*' -print0)
fi
[[ -n "${SOURCE_H1_ROOT}" && -d "${SOURCE_H1_ROOT}" ]] \
  || die "H1 pipeline root not found; pass it as argument 1"
shopt -s nullglob
source_runs=("${SOURCE_H1_ROOT}"/training/*_station24_jstd_event_hypothesis_h1_*_seed2027)
shopt -u nullglob
[[ ${#source_runs[@]} -eq 1 ]] || die "expected exactly one H1 training run"
SOURCE_RUN=${source_runs[0]}
SOURCE_CHECKPOINT="${SOURCE_RUN}/checkpoints/model_best.pt"
[[ -f "${SOURCE_CHECKPOINT}" ]] || die "missing H1 checkpoint ${SOURCE_CHECKPOINT}"

STAMP=$(date +%Y%m%d_%H%M%S)
PREFLIGHT="${OUTPUT_ROOT}/jstd_msep_v2_preflight_${STAMP}"

echo "JSTD_MSEP_V2_STATIC_AND_UNIT_TEST_START"
"${PYTHON_BIN}" -m py_compile \
  station_jstd_targets.py station_dataset.py train_station24.py generate_station24.py \
  src/models/station_joint_decomposed_tail.py \
  src/models/station_conditioned_diffusion.py \
  tools/audit_station24_jstd_msep_preflight.py
"${PYTHON_BIN}" -m unittest \
  tests.test_station24_jstd_targets tests.test_station24_jstd_tail \
  tests.test_station24_jstd_h1 tests.test_station24_jstd_msep

echo "JSTD_MSEP_V2_CUDA_PREFLIGHT_START"
"${PYTHON_BIN}" -m tools.audit_station24_jstd_msep_preflight \
  --config configs/station24_jstd_msep_v2_prior_only_168h.yaml \
  --checkpoint "${SOURCE_CHECKPOINT}" \
  --data-path "${DATA}" --output-dir "${PREFLIGHT}" --device cuda

echo "JSTD_MSEP_V2_PREFLIGHT_COMPLETE output=${PREFLIGHT}"
echo "NO_FORMAL_TRAINING_WAS_STARTED"
