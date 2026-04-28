#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCHMARK_DIR="${PROJECT_ROOT}/benchmark"
RESULTS_DIR="${BENCHMARK_DIR}/results"
CONFIG_FILE="${CONFIG_FILE:-${BENCHMARK_DIR}/yaml/default.yaml}"
LOG_DIR_ENV="${LOG_DIR:-}"
LOG_DIR="${LOG_DIR_ENV:-${BENCHMARK_DIR}/log}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
PYTHON_BIN="${PYTHON_BIN:-/home/huangyanjie/miniconda3/bin/python}"
GPU_IDS=(${GPU_IDS:-0 1 2 3 4 5 6 7})

foreground=0
prepare_only_requested=0
passthrough=()
config_seen=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fg|--foreground)
      foreground=1
      shift
      ;;
    --bg|--background)
      foreground=0
      shift
      ;;
    --prepare-only)
      prepare_only_requested=1
      passthrough+=("$1")
      shift
      ;;
    --config)
      if [[ $# -lt 2 ]]; then
        echo "Error: --config requires a YAML path" >&2
        exit 2
      fi
      CONFIG_FILE="$2"
      config_seen=1
      shift 2
      ;;
    --config=*)
      CONFIG_FILE="${1#--config=}"
      config_seen=1
      shift
      ;;
    *.yaml|*.yml)
      if [[ "${config_seen}" -eq 0 ]]; then
        CONFIG_FILE="$1"
        config_seen=1
      else
        passthrough+=("$1")
      fi
      shift
      ;;
    *)
      passthrough+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Error: config YAML not found: ${CONFIG_FILE}" >&2
  exit 2
fi
CONFIG_FILE="$(cd "$(dirname "${CONFIG_FILE}")" && pwd)/$(basename "${CONFIG_FILE}")"

if [[ -z "${LOG_DIR_ENV}" ]]; then
  config_stem="$(basename "${CONFIG_FILE}")"
  config_stem="${config_stem%.*}"
  LOG_DIR="${BENCHMARK_DIR}/log/${config_stem}"
fi

mkdir -p "${RESULTS_DIR}"
mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

launcher_log="${LOG_DIR}/launcher_${RUN_ID}.log"
pid_file="${LOG_DIR}/launcher_${RUN_ID}.pid"

if [[ "${foreground}" -eq 0 ]]; then
  nohup env RUN_ID="${RUN_ID}" CONFIG_FILE="${CONFIG_FILE}" LOG_DIR="${LOG_DIR}" PYTHON_BIN="${PYTHON_BIN}" GPU_IDS="${GPU_IDS[*]}" \
    bash "${BASH_SOURCE[0]}" --foreground "${CONFIG_FILE}" "${passthrough[@]}" \
    > "${launcher_log}" 2>&1 < /dev/null &
  child_pid="$!"
  echo "${child_pid}" > "${pid_file}"
  disown "${child_pid}" || true
  echo "Launched in background."
  echo "  RUN_ID:      ${RUN_ID}"
  echo "  PID:         ${child_pid}"
  echo "  Config:      ${CONFIG_FILE}"
  echo "  Launcher:    ${launcher_log}"
  echo "  PID file:    ${pid_file}"
  echo "  Worker logs: ${LOG_DIR}/worker_*.log"
  exit 0
fi

echo "[$(date '+%F %T')] Starting benchmark launcher"
echo "[$(date '+%F %T')] Python: ${PYTHON_BIN}"
echo "[$(date '+%F %T')] Config: ${CONFIG_FILE}"
echo "[$(date '+%F %T')] Log dir: ${LOG_DIR}"
echo "[$(date '+%F %T')] GPUs: ${GPU_IDS[*]}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON_BIN}" benchmark/run_benchmark.py \
  --config "${CONFIG_FILE}" \
  --prepare-only \
  "${passthrough[@]}"

if [[ "${prepare_only_requested}" -eq 1 ]]; then
  echo "[$(date '+%F %T')] prepare-only requested; not launching workers"
  exit 0
fi

pids=()
num_workers="${#GPU_IDS[@]}"

for i in "${!GPU_IDS[@]}"; do
  gpu_id="${GPU_IDS[$i]}"
  worker_stdout="${LOG_DIR}/worker_${i}.stdout.log"
  echo "[$(date '+%F %T')] Launch worker ${i}/${num_workers} on GPU ${gpu_id}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" benchmark/run_benchmark.py \
    --config "${CONFIG_FILE}" \
    --device cuda:0 \
    --skip-input-write \
    --worker-index "${i}" \
    --num-workers "${num_workers}" \
    "${passthrough[@]}" \
    > "${worker_stdout}" 2>&1 &
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
