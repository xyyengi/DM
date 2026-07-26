#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
STAGE1_LOG_DIR=${STAGE1_LOG_DIR:-logs/v5_stage2}
OUTPUTS_DIR=${OUTPUTS_DIR:-outputs_shandong/v5_stage2}
export STAGE1_LOG_DIR OUTPUTS_DIR
source "${REPO_ROOT}/server_stage1_common.sh"

if [[ "${STAGE1_INTERNAL_WORKER:-0}" != "1" ]]; then
  stage1_launch_background "$0" "v5_tf_va_stage2" "$@"
  exit 0
fi

stage1_install_status_trap
stage1_preflight "v5_tf_va_stage2"
stage1_run_training \
  "configs/v5_tf_va_stage2_168h.yaml" \
  "v5_tf_va" \
  "v5_tf_va_stage2"
