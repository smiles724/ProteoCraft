#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_ROOT}/configs/tcr5_hotspot_v1like_loss.yaml}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
PYTHON_BIN="${PYTHON_BIN:-/home/huangyanjie/miniconda3/bin/python}"
GPU_IDS="${GPU_IDS:-}"

log_dir="$("${PYTHON_BIN}" - "${CONFIG_FILE}" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}
print(cfg.get("log_dir") or (str(cfg.get("output_root", "outputs")) + "/logs"))
PY
)"
mkdir -p "${log_dir}"

launcher_log="${log_dir}/yaml_parallel_${RUN_ID}.nohup.log"
pid_file="${log_dir}/yaml_parallel_${RUN_ID}.pid"

cmd=("${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_bindcraft_tcr_case_yaml_parallel.py" --config "${CONFIG_FILE}" --python-bin "${PYTHON_BIN}")
if [[ -n "${GPU_IDS}" ]]; then
  cmd+=(--gpu-ids "${GPU_IDS}")
fi

nohup setsid bash -c 'echo "$$" > "$1"; shift; exec "$@"' _ "${pid_file}" "${cmd[@]}" \
  > "${launcher_log}" 2>&1 < /dev/null &

for _ in {1..20}; do
  if [[ -s "${pid_file}" ]]; then
    break
  fi
  sleep 0.2
done

pid="$(cat "${pid_file}" 2>/dev/null || true)"

echo "Launched BindCraft-v2 TCR YAML parallel queue with nohup."
echo "  PID:      ${pid:-unknown}"
echo "  Config:   ${CONFIG_FILE}"
echo "  GPU_IDS:  ${GPU_IDS:-from YAML}"
echo "  Log:      ${launcher_log}"
echo "  PID file: ${pid_file}"
