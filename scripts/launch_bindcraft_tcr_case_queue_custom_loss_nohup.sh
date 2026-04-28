#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/bindcraft_v2_tcr_cases_custom_loss}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-${PROJECT_ROOT}/outputs/protenix_runtime_tcr_cases_custom_loss}"
GPU_ID="${GPU_ID:-2}"
PYTHON_BIN="${PYTHON_BIN:-/home/huangyanjie/miniconda3/bin/python}"
WAIT_PID="${WAIT_PID:-}"
N_STEPS="${N_STEPS:-50}"

mkdir -p "${LOG_DIR}"

queue_log="${LOG_DIR}/queue_custom_loss_${RUN_ID}.nohup.log"
queue_pid_file="${LOG_DIR}/queue_custom_loss_${RUN_ID}.pid"

nohup setsid -f env \
  PYTHON_BIN="${PYTHON_BIN}" \
  GPU_ID="${GPU_ID}" \
  WAIT_PID="${WAIT_PID}" \
  N_STEPS="${N_STEPS}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  LOG_DIR="${LOG_DIR}" \
  PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR}" \
  QUEUE_PID_FILE="${queue_pid_file}" \
  bash "${PROJECT_ROOT}/scripts/run_bindcraft_tcr_case_queue_custom_loss.sh" \
  > "${queue_log}" 2>&1 < /dev/null

for _ in {1..20}; do
  if [[ -s "${queue_pid_file}" ]]; then
    break
  fi
  sleep 0.2
done

pid="$(cat "${queue_pid_file}" 2>/dev/null || true)"

echo "Launched BindCraft-v2 TCR custom-loss case queue with nohup."
echo "  Queue PID: ${pid:-unknown}"
echo "  WAIT_PID: ${WAIT_PID:-none}"
echo "  GPU_ID:   ${GPU_ID}"
echo "  N_STEPS:  ${N_STEPS}"
echo "  Output:   ${OUTPUT_ROOT}"
echo "  Log:      ${queue_log}"
echo "  PID file: ${queue_pid_file}"
