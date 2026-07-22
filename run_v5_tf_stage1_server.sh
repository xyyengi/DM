#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
source "${REPO_ROOT}/server_stage1_common.sh"

if [[ "${STAGE1_INTERNAL_WORKER:-0}" != "1" ]]; then
  stage1_launch_background "$0" "v5_tf_stage1" "$@"
  exit 0
fi

stage1_install_status_trap
stage1_preflight "v5_tf_stage1"
stage1_run_training \
  "configs/v5_tf_stage1_168h.yaml" \
  "v5_tf" \
  "v5_tf_stage1"
