#!/usr/bin/env bash
# Slurm entry point for the paired generation pilot or its blind rescore.
set -euo pipefail

REPO_ROOT="${PROTEOCRAFT_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
: "${ESMOPT_PYTHON:?Set ESMOPT_PYTHON to the isolated Python executable}"
: "${PROTEOCRAFT_UPLIFT_ROOT:?Set PROTEOCRAFT_UPLIFT_ROOT}"
: "${HF_HOME:?Set HF_HOME to the model cache}"
: "${ESMCFOLD_CCD_PATH:?Set ESMCFOLD_CCD_PATH}"
: "${ESMOPT_CAMPAIGN:?Set ESMOPT_CAMPAIGN to campaign.json}"
ESMOPT_MODE="${ESMOPT_MODE:-run}"
ESMOPT_SHARD_INDEX="${ESMOPT_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
ESMOPT_OUTPUT="${ESMOPT_OUTPUT:-$(dirname "$ESMOPT_CAMPAIGN")/slurm_${ESMOPT_MODE}_${ESMOPT_SHARD_INDEX}}"

mkdir -p "$ESMOPT_OUTPUT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TRITON_CACHE_DIR="$ESMOPT_OUTPUT/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$ESMOPT_OUTPUT/inductor_cache"
cd "$REPO_ROOT"

if [[ "$ESMOPT_MODE" == run ]]; then
  "$ESMOPT_PYTHON" benchmark/script/run_esm_optimization.py run \
    --manifest "$ESMOPT_CAMPAIGN" \
    --num-shards "${ESMOPT_NUM_SHARDS:-1}" \
    --shard-index "$ESMOPT_SHARD_INDEX" \
    --skip-existing
elif [[ "$ESMOPT_MODE" == rescore ]]; then
  RESCORE_DIR="${ESMOPT_RESCORE_DIR:-$(dirname "$ESMOPT_CAMPAIGN")/rescore}"
  REPORT_CSV="${ESMOPT_REPORT_CSV:-$(dirname "$ESMOPT_CAMPAIGN")/pilot_report.csv}"
  read -r -a RESCORE_SEEDS <<< "${ESMOPT_RESCORING_SEEDS:-1043 1044}"
  "$ESMOPT_PYTHON" benchmark/script/run_esm_optimization.py rescore \
    --manifest "$ESMOPT_CAMPAIGN" --output "$RESCORE_DIR" \
    --top-k "${ESMOPT_TOP_K:-5}" --seeds "${RESCORE_SEEDS[@]}" \
    --loops "${ESMOPT_RESCORING_LOOPS:-4}" \
    --sampling-steps "${ESMOPT_RESCORING_STEPS:-50}"
  "$ESMOPT_PYTHON" benchmark/script/run_esm_optimization.py report \
    --manifest "$ESMOPT_CAMPAIGN" --scores "$RESCORE_DIR" \
    --output "$REPORT_CSV" --success-threshold "${ESMOPT_SUCCESS_THRESHOLD:-0.8}"
else
  echo "Unknown ESMOPT_MODE: $ESMOPT_MODE (expected run or rescore)" >&2
  exit 2
fi
