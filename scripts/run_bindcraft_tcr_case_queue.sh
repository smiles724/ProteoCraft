#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/huangyanjie/miniconda3/bin/python}"
GPU_ID="${GPU_ID:-2}"
WAIT_PID="${WAIT_PID:-}"
N_STEPS="${N_STEPS:-50}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/bindcraft_v2_tcr_cases}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
QUEUE_PID_FILE="${QUEUE_PID_FILE:-}"
mkdir -p "${LOG_DIR}"

if [[ -n "${QUEUE_PID_FILE}" ]]; then
  mkdir -p "$(dirname "${QUEUE_PID_FILE}")"
  echo "$$" > "${QUEUE_PID_FILE}"
fi

timestamp() {
  date '+%F %T'
}

if [[ -n "${WAIT_PID}" ]]; then
  echo "[$(timestamp)] Waiting for existing run PID=${WAIT_PID}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 60
  done
  echo "[$(timestamp)] PID=${WAIT_PID} finished; starting queued cases"
fi

run_case() {
  local pdb_id="$1"
  local hs_start="$2"
  local hs_end="$3"
  local alpha_chain="${4:-}"
  local beta_chain="${5:-}"

  if [[ "${hs_start}" == "TODO" || "${hs_end}" == "TODO" ]]; then
    echo "[$(timestamp)] SKIP ${pdb_id}: hotspot is TODO"
    return 0
  fi

  local case_name="${pdb_id}_alpha_${hs_start}_${hs_end}"
  local log_file="${LOG_DIR}/${case_name}.log"
  echo "[$(timestamp)] START ${case_name}; log=${log_file}"
  extra_args=()
  if [[ -n "${alpha_chain}" ]]; then
    extra_args+=(--alpha-chain "${alpha_chain}")
  fi
  if [[ -n "${beta_chain}" ]]; then
    extra_args+=(--beta-chain "${beta_chain}")
  fi

  if CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_bindcraft_tcr_case.py" \
      --pdb-id "${pdb_id}" \
      --hotspot-start "${hs_start}" \
      --hotspot-end "${hs_end}" \
      --n-steps "${N_STEPS}" \
      --output-root "${OUTPUT_ROOT}" \
      --log-file "${log_file}" \
      "${extra_args[@]}"; then
    echo "[$(timestamp)] DONE ${case_name}"
  else
    echo "[$(timestamp)] FAILED ${case_name}; see ${log_file}"
  fi
}

run_case "6BJ8" "20" "24" "D" "H"
run_case "5BS0" "20" "25" "D" "E"
run_case "3QDJ" "18" "23" "D" "E"
run_case "9C3E" "39" "43" "A" "B"

echo "[$(timestamp)] Queue finished"
