#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=${1:?'usage: run_shandong91_raw_body_full_pipeline.sh PIPELINE_ROOT [RESUME_CHECKPOINT]'}
RESUME=${2:-}
CONFIG=${CONFIG:-configs/shandong91/raw_body_heterogeneous_formal_v1.yaml}
cd "$(dirname "$0")"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
[[ "${CONDA_DEFAULT_ENV:-}" == dm_env ]] || { echo 'Activate dm_env first'; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo 'Tracked working tree is dirty'; exit 2; }
[[ -f reliable_channel_training_v1/quality_report.json ]] || { echo 'Dataset missing'; exit 2; }
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
STATUS="${ROOT}_status.json"; JOB=$(date +%Y%m%d_%H%M%S); LOG="${ROOT}_resume_${JOB}.log"; PHASE=startup
write_status(){ python - "$STATUS" "$1" "$PHASE" "$ROOT" "$LOG" <<'PY'
import json,pathlib,sys,time
pathlib.Path(sys.argv[1]).write_text(json.dumps({'state':sys.argv[2],'phase':sys.argv[3],'root':sys.argv[4],'log':sys.argv[5],'timestamp':time.time()},indent=2),encoding='utf-8')
PY
}
on_error(){ code=$?; write_status failed; exit "$code"; }; trap on_error ERR
exec > >(tee -a "$LOG") 2>&1
PHASE=preflight; write_status running
PREFLIGHT="${ROOT}_resume_preflight_$JOB"
python tools/preflight_shandong91_formal_e2e.py --device cuda --amp --config "$CONFIG" --output "$PREFLIGHT"
python -c "import json; assert json.load(open('$PREFLIGHT/e2e_report.json'))['status']=='PASS'"
PHASE=training; write_status running
if [[ -n "$RESUME" ]]; then
  test -f "$RESUME"
  python -u train_shandong91.py --device cuda --config "$CONFIG" --output-dir "$ROOT" --resume "$RESUME"
else
  [[ ! -e "$ROOT" ]] || { echo 'New ROOT already exists'; exit 2; }
  python -u train_shandong91.py --device cuda --config "$CONFIG" --output-dir "$ROOT"
fi
PHASE=generation_evaluation_archive; write_status running
bash run_shandong91_raw_body_finalize.sh "$ROOT"
PHASE=complete; write_status complete
echo "SHANDONG91_FULL_PIPELINE_COMPLETE root=$ROOT"
