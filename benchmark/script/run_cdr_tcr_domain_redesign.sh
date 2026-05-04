#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/huangyanjie/miniconda3/bin/python}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_ROOT}/benchmark/yaml/cdr_tcr/cdr_tcr_domain_redesign.yaml}"
GPU_IDS=(${GPU_IDS:-0 1 2 3})
RUN_ID="${RUN_ID:-cdr_tcr_$(date '+%Y%m%d_%H%M%S')}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/benchmark/outputs/cdr_tcr/logs/${RUN_ID}}"

foreground=0
prepare_only=0
passthrough=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --config FILE       YAML config file
  --fg                Run in foreground
  --prepare-only      Validate config and case masks only
  --case ID           Run one case id
  -h, --help          Show help

Environment:
  GPU_IDS="0 1 2 3"   GPU IDs for the four parallel workers
  PYTHON_BIN=python    Python executable
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --config=*)
      CONFIG_FILE="${1#--config=}"
      shift
      ;;
    --fg|--foreground)
      foreground=1
      shift
      ;;
    --prepare-only)
      prepare_only=1
      passthrough+=("$1")
      shift
      ;;
    *)
      passthrough+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config not found: ${CONFIG_FILE}" >&2
  exit 2
fi
CONFIG_FILE="$(cd "$(dirname "${CONFIG_FILE}")" && pwd)/$(basename "${CONFIG_FILE}")"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

if [[ "${foreground}" -eq 0 && "${prepare_only}" -eq 0 ]]; then
  launcher_log="${LOG_DIR}/launcher.log"
  nohup env \
    CONFIG_FILE="${CONFIG_FILE}" \
    GPU_IDS="${GPU_IDS[*]}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    RUN_ID="${RUN_ID}" \
    LOG_DIR="${LOG_DIR}" \
    bash "$0" --fg "${passthrough[@]}" \
    > "${launcher_log}" 2>&1 < /dev/null &
  pid="$!"
  echo "${pid}" > "${LOG_DIR}/launcher.pid"
  echo "Launched CDR TCR redesign in background."
  echo "  RUN_ID: ${RUN_ID}"
  echo "  PID: ${pid}"
  echo "  Config: ${CONFIG_FILE}"
  echo "  Log dir: ${LOG_DIR}"
  exit 0
fi

echo "[$(date '+%F %T')] CDR TCR launcher"
echo "[$(date '+%F %T')] Python: ${PYTHON_BIN}"
echo "[$(date '+%F %T')] Config: ${CONFIG_FILE}"
echo "[$(date '+%F %T')] Log dir: ${LOG_DIR}"

if [[ "${prepare_only}" -eq 1 ]]; then
  "${PYTHON_BIN}" benchmark/script/run_cdr_tcr_domain_redesign.py \
    --config "${CONFIG_FILE}" \
    "${passthrough[@]}"
  exit 0
fi

case_ids=($("${PYTHON_BIN}" - "${CONFIG_FILE}" <<'PY'
import sys
import yaml
with open(sys.argv[1], encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}
print(" ".join(str(case["id"]) for case in cfg.get("cases", [])))
PY
))

if [[ "${#case_ids[@]}" -eq 0 ]]; then
  echo "No cases in config." >&2
  exit 2
fi
if [[ "${#GPU_IDS[@]}" -lt "${#case_ids[@]}" ]]; then
  echo "Need at least ${#case_ids[@]} GPU_IDS, got ${#GPU_IDS[@]}" >&2
  exit 2
fi

pids=()
status=0
for idx in "${!case_ids[@]}"; do
  case_id="${case_ids[$idx]}"
  gpu_id="${GPU_IDS[$idx]}"
  worker_log="${LOG_DIR}/${case_id}_gpu${gpu_id}.stdout.log"
  echo "[$(date '+%F %T')] Launch ${case_id} on GPU ${gpu_id}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" benchmark/script/run_cdr_tcr_domain_redesign.py \
    --config "${CONFIG_FILE}" \
    --case "${case_id}" \
    --device cuda:0 \
    --log-file "${LOG_DIR}/${case_id}.launcher.log" \
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

echo "[$(date '+%F %T')] All CDR TCR jobs finished with status=${status}"
exit "${status}"
