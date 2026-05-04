#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/huangyanjie/miniconda3/bin/python}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_ROOT}/benchmark/yaml/cdr_tcr/cdr_tcr_domain_redesign.yaml}"
RUN_ID="${RUN_ID:-9nmw_cdr_gpu01_$(date '+%Y%m%d_%H%M%S')}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/benchmark/outputs/cdr_tcr/logs/${RUN_ID}}"

CASES=(9NMW_alpha 9NMW_beta)
GPUS=(0 1)

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

foreground=0
passthrough=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fg|--foreground)
      foreground=1
      shift
      ;;
    *)
      passthrough+=("$1")
      shift
      ;;
  esac
done

if [[ "${foreground}" -eq 0 ]]; then
  launcher_log="${LOG_DIR}/launcher.log"
  nohup env \
    PYTHON_BIN="${PYTHON_BIN}" \
    CONFIG_FILE="${CONFIG_FILE}" \
    RUN_ID="${RUN_ID}" \
    LOG_DIR="${LOG_DIR}" \
    bash "$0" --fg "${passthrough[@]}" \
    > "${launcher_log}" 2>&1 < /dev/null &
  pid="$!"
  echo "${pid}" > "${LOG_DIR}/launcher.pid"
  echo "Launched 9NMW CDR jobs in background."
  echo "  RUN_ID: ${RUN_ID}"
  echo "  PID: ${pid}"
  echo "  Log dir: ${LOG_DIR}"
  exit 0
fi

echo "[$(date '+%F %T')] Run 9NMW CDR jobs"
echo "[$(date '+%F %T')] Python: ${PYTHON_BIN}"
echo "[$(date '+%F %T')] Config: ${CONFIG_FILE}"
echo "[$(date '+%F %T')] Log dir: ${LOG_DIR}"

pids=()
status=0

for idx in "${!CASES[@]}"; do
  case_id="${CASES[$idx]}"
  gpu_id="${GPUS[$idx]}"
  worker_log="${LOG_DIR}/${case_id}_gpu${gpu_id}.stdout.log"
  launcher_log="${LOG_DIR}/${case_id}.launcher.log"

  echo "[$(date '+%F %T')] Launch ${case_id} on GPU ${gpu_id}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" benchmark/script/run_cdr_tcr_domain_redesign.py \
    --config "${CONFIG_FILE}" \
    --case "${case_id}" \
    --device cuda:0 \
    --log-file "${launcher_log}" \
    "${passthrough[@]}" \
    > "${worker_log}" 2>&1 &

  pids+=("$!")
  echo "${pids[-1]}" > "${LOG_DIR}/${case_id}_gpu${gpu_id}.pid"
done

for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

echo "[$(date '+%F %T')] 9NMW CDR jobs finished with status=${status}"
exit "${status}"
