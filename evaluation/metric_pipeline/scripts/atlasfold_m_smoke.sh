#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 MANIFEST OUT_DIR PAPER_CASE SUB_RUN [DEVICE]" >&2
  exit 2
fi

MANIFEST="$1"
OUT_DIR="$2"
PAPER_CASE="$3"
SUB_RUN="$4"
DEVICE="${5:-cuda:0}"
PROTEOCRAFT_PYTHON="${PROTEOCRAFT_PYTHON:-python}"

for name in \
  ATLASFOLD_PYTHON \
  ATLASFOLD_SOURCE_ROOT \
  ATLASFOLD_CACHE_DIR \
  ATLASFOLD_MODEL_PATH \
  ATLASFOLD_LM_PATH \
  ATLASFOLD_WEIGHTS_RECEIPT; do
  if [[ -z "${!name:-}" ]]; then
    echo "Required environment variable is unset: $name" >&2
    exit 2
  fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

"$PROTEOCRAFT_PYTHON" -m eval_pipeline.cli \
  --manifest "$MANIFEST" \
  --out-dir "$OUT_DIR" \
  --evaluator atlasfold_m \
  --paper-case "$PAPER_CASE" \
  --sub-run "$SUB_RUN" \
  --device "$DEVICE" \
  --limit 1 \
  --atlasfold-num-samples 1 \
  --atlasfold-seed 1 \
  --atlasfold-num-steps 10 \
  --no-skip-existing
