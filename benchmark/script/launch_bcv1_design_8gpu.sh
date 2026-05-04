#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-bcv1_pxd_rf3_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/8_bcv1/pxd_rf3/bcv1_run_logs/$RUN_ID}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
METHOD="${METHOD:-all}"
CASE="${CASE:-all}"
CONDITION="${CONDITION:-all}"
LENGTH="${LENGTH:-all}"
MANIFEST="${MANIFEST:-$ROOT/outputs/8_bcv1/pxd_rf3/bcv1_design_manifest.csv}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

mkdir -p "$LOG_DIR"

read -r -a GPUS <<< "$GPU_IDS"
SHARD_COUNT="${SHARD_COUNT:-${#GPUS[@]}}"
if [[ "$SHARD_COUNT" -ne "${#GPUS[@]}" ]]; then
  echo "SHARD_COUNT must match the number of GPU_IDS for this launcher." >&2
  exit 2
fi

cat > "$LOG_DIR/run.env" <<EOF
RUN_ID=$RUN_ID
LOG_DIR=$LOG_DIR
GPU_IDS=$GPU_IDS
SHARD_COUNT=$SHARD_COUNT
METHOD=$METHOD
CASE=$CASE
CONDITION=$CONDITION
LENGTH=$LENGTH
MANIFEST=$MANIFEST
CONTINUE_ON_ERROR=$CONTINUE_ON_ERROR
EOF

for shard in "${!GPUS[@]}"; do
  gpu="${GPUS[$shard]}"
  log="$LOG_DIR/worker_${shard}_gpu${gpu}.log"
  fail="$LOG_DIR/worker_${shard}_gpu${gpu}.failures.tsv"
  env \
    METHOD="$METHOD" \
    CASE="$CASE" \
    CONDITION="$CONDITION" \
    LENGTH="$LENGTH" \
    MANIFEST="$MANIFEST" \
    SHARD_INDEX="$shard" \
    SHARD_COUNT="$SHARD_COUNT" \
    PX_GPU_ID="$gpu" \
    RF3_GPU_ID="$gpu" \
    CONTINUE_ON_ERROR="$CONTINUE_ON_ERROR" \
    FAILURES_FILE="$fail" \
    nohup bash -lc \
      "cd '$ROOT' && echo '[start] shard=$shard gpu=$gpu pid=\$\$ time='\"\$(date '+%F %T')\" && exec bash '$ROOT/script/run_bcv1_design_batch.sh'" \
      > "$log" 2>&1 < /dev/null &
  pid="$!"
  echo "$pid" > "$LOG_DIR/worker_${shard}_gpu${gpu}.pid"
  echo "started shard=$shard gpu=$gpu pid=$pid log=$log"
done

echo "$LOG_DIR"
