#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
source "${REPO_ROOT}/server_stage1_common.sh"

if [[ "${STAGE1_INTERNAL_WORKER:-0}" != "1" ]]; then
  stage1_launch_background "$0" "v4rs_repro_stage1" "$@"
  exit 0
fi

stage1_install_status_trap
stage1_preflight "v4rs_repro_stage1"
stage1_run_training \
  "configs/v4rs_reproducible_no_guidance_168h.yaml" \
  "v4_legacy" \
  "v4rs_repro_stage1"
