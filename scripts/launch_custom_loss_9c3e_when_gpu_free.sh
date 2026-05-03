#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/huangyanjie/miniconda3/bin/python}"
QUEUE_PID="${QUEUE_PID:-4027420}"
GPU_IDS="${GPU_IDS:-3 4 5 6 7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/bindcraft_v2_tcr_cases_custom_loss}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-${PROJECT_ROOT}/outputs/protenix_runtime_tcr_cases_custom_loss_9C3E_parallel}"
WATCHER_LOG="${LOG_DIR}/custom_loss_9C3E_gpu_free_watcher.log"
mkdir -p "${LOG_DIR}"

timestamp() {
  date '+%F %T'
}

log() {
  echo "[$(timestamp)] $*" | tee -a "${WATCHER_LOG}"
}

CUSTOM_LOSS_ARGS=(
  --w-plddt 0.1
  --w-pae 0.4
  --w-i-pae 0.2
  --w-con 0.2
  --w-i-con 3.0
  --w-iptm 0.3
  --w-rg 0.3
  --w-helix -0.3
  --w-clash 0.1
  --w-i-clash 0.1
  --w-pde 0.05
  --w-i-pde 0.03
)

idle_gpu() {
  local ids=" ${GPU_IDS} "
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F',' -v ids="${ids}" '
      {
        idx=$1; mem=$2; util=$3;
        gsub(/ /, "", idx); gsub(/ /, "", mem); gsub(/ /, "", util);
        if (index(ids, " " idx " ") && mem < 2000 && util < 20) {
          print idx;
          exit;
        }
      }'
}

if [[ -f "${OUTPUT_ROOT}/9C3E_alpha_39_43/result.json" ]]; then
  log "9C3E result already exists; nothing to launch."
  exit 0
fi

log "Watching queue PID=${QUEUE_PID}; will stop queue after custom-loss 3QDJ starts, then launch 9C3E on first idle GPU among: ${GPU_IDS}"

while true; do
  if pgrep -P "${QUEUE_PID}" -f 'run_bindcraft_tcr_case.py.*--pdb-id 3QDJ' >/dev/null 2>&1; then
    log "Detected custom-loss 3QDJ running under queue PID=${QUEUE_PID}."
    break
  fi
  if [[ -f "${OUTPUT_ROOT}/3QDJ_alpha_18_23/result.json" ]]; then
    log "3QDJ result already exists."
    break
  fi
  if ! kill -0 "${QUEUE_PID}" 2>/dev/null; then
    log "Queue PID=${QUEUE_PID} is not alive; continuing to wait for an idle GPU for 9C3E."
    break
  fi
  sleep 10
done

if kill -0 "${QUEUE_PID}" 2>/dev/null; then
  log "Stopping queue parent PID=${QUEUE_PID} to prevent duplicate queued 9C3E. Existing child process continues."
  kill -TERM "${QUEUE_PID}" || true
fi

while true; do
  gpu="$(idle_gpu || true)"
  if [[ -n "${gpu}" ]]; then
    break
  fi
  sleep 30
done

log "Launching custom-loss 9C3E_alpha_39_43 on GPU ${gpu}."
CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_bindcraft_tcr_case.py" \
  --pdb-id 9C3E \
  --hotspot-start 39 \
  --hotspot-end 43 \
  --n-steps 50 \
  --output-root "${OUTPUT_ROOT}" \
  --protenix-root-dir "${PROTENIX_ROOT_DIR}" \
  --log-file "${LOG_DIR}/9C3E_alpha_39_43.log" \
  "${CUSTOM_LOSS_ARGS[@]}" \
  --alpha-chain A \
  --beta-chain B

log "Finished custom-loss 9C3E_alpha_39_43."
