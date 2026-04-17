#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCHMARK_DIR="${PROJECT_ROOT}/benchmark"
RESULTS_DIR="${BENCHMARK_DIR}/results"
CONFIG_FILE="${BENCHMARK_DIR}/yaml/default.yaml"
LOG_DIR="${BENCHMARK_DIR}/log"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
PYTHON_BIN="${PYTHON_BIN:-/home/huangyanjie/miniconda3/bin/python}"
GPU_IDS=(0 1 2 3 4 5 6 7)

mkdir -p "${RESULTS_DIR}"
mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

foreground=0
passthrough=()
for arg in "$@"; do
  case "${arg}" in
    --fg|--foreground) foreground=1 ;;
    --bg|--background) foreground=0 ;;
    *) passthrough+=("${arg}") ;;
  esac
done

if [[ "${foreground}" -eq 0 ]]; then
  nohup env RUN_ID="${RUN_ID}" \
    bash "${BASH_SOURCE[0]}" --foreground "${passthrough[@]}" \
    > /dev/null 2>&1 &
  child_pid="$!"
  disown "${child_pid}" || true
  echo "Launched in background."
  echo "  RUN_ID: ${RUN_ID}"
  echo "  PID:    ${child_pid}"
  echo "  Logs:   ${LOG_DIR}/worker_*.log"
  exit 0
fi

echo "[$(date '+%F %T')] Starting benchmark launcher"
echo "[$(date '+%F %T')] Python: ${PYTHON_BIN}"
echo "[$(date '+%F %T')] Config: ${CONFIG_FILE}"
echo "[$(date '+%F %T')] GPUs: ${GPU_IDS[*]}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON_BIN}" benchmark/run_benchmark.py \
  --config "${CONFIG_FILE}" \
  --prepare-only \
  "${passthrough[@]}"

pids=()
num_workers="${#GPU_IDS[@]}"

for i in "${!GPU_IDS[@]}"; do
  gpu_id="${GPU_IDS[$i]}"
  echo "[$(date '+%F %T')] Launch worker ${i}/${num_workers} on GPU ${gpu_id}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" benchmark/run_benchmark.py \
    --config "${CONFIG_FILE}" \
    --device cuda:0 \
    --skip-input-write \
    --worker-index "${i}" \
    --num-workers "${num_workers}" \
    "${passthrough[@]}" \
    > /dev/null 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

echo "[$(date '+%F %T')] All workers finished with status=${status}"
exit "${status}"
