#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/benchmark/log"

dry_run=0
force=0

usage() {
  cat <<'EOF'
Usage:
  bash benchmark/script/kill_benchmark.sh [--dry-run] [--force]

What it kills:
  - Background launcher PIDs recorded in benchmark/log/run_*.pid
    (kills their whole process group if possible)
  - Any remaining python processes running benchmark/run_benchmark.py (fallback)

Options:
  --dry-run   Print what would be killed, do nothing
  --force     Escalate to SIGKILL (-9) after SIGTERM
EOF
}

args=()
for a in "$@"; do
  case "$a" in
    -h|--help) usage; exit 0 ;;
    --dry-run) dry_run=1 ;;
    --force) force=1 ;;
    *) args+=("$a") ;;
  esac
done
if ((${#args[@]} > 0)); then
  echo "Unknown args: ${args[*]}" >&2
  usage >&2
  exit 2
fi

do_kill() {
  local sig="$1"
  shift
  local pids=("$@")
  if ((${#pids[@]} == 0)); then
    return 0
  fi
  if ((dry_run)); then
    echo "[dry-run] kill -${sig} ${pids[*]}"
    return 0
  fi
  kill "-${sig}" "${pids[@]}" 2>/dev/null || true
}

do_kill_pgrp() {
  # Kill process group for a given PID (best effort).
  local sig="$1"
  local pid="$2"
  if ((dry_run)); then
    echo "[dry-run] kill -${sig} -- -${pid}   # process group"
    return 0
  fi
  kill "-${sig}" "--" "-${pid}" 2>/dev/null || true
}

echo "LOG_DIR=${LOG_DIR}"

launcher_pids=()
if [[ -d "${LOG_DIR}" ]]; then
  shopt -s nullglob
  for f in "${LOG_DIR}"/run_*.pid; do
    pid="$(tr -d ' \t\r\n' < "${f}" || true)"
    if [[ -n "${pid}" && "${pid}" =~ ^[0-9]+$ ]]; then
      launcher_pids+=("${pid}")
    fi
  done
  shopt -u nullglob
fi

if ((${#launcher_pids[@]} > 0)); then
  echo "Found launcher pid files: ${#launcher_pids[@]}"
  echo "Sending SIGTERM to launcher process groups..."
  for pid in "${launcher_pids[@]}"; do
    do_kill_pgrp "TERM" "${pid}"
  done
else
  echo "No launcher pid files found in ${LOG_DIR} (run_*.pid)."
fi

echo "Sending SIGTERM to any remaining run_benchmark.py processes..."
if ((dry_run)); then
  echo "[dry-run] pkill -f 'benchmark/run_benchmark.py'"
else
  pkill -f "benchmark/run_benchmark.py" 2>/dev/null || true
fi

if ((force)); then
  # Give processes a moment to exit gracefully.
  if ((dry_run)); then
    echo "[dry-run] sleep 1"
  else
    sleep 1
  fi
  if ((${#launcher_pids[@]} > 0)); then
    echo "Escalating to SIGKILL for launcher process groups..."
    for pid in "${launcher_pids[@]}"; do
      do_kill_pgrp "KILL" "${pid}"
    done
  fi
  echo "Escalating to SIGKILL for any remaining run_benchmark.py processes..."
  if ((dry_run)); then
    echo "[dry-run] pkill -9 -f 'benchmark/run_benchmark.py'"
  else
    pkill -9 -f "benchmark/run_benchmark.py" 2>/dev/null || true
  fi
fi

echo "Done."

