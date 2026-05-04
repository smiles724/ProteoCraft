#!/bin/bash
# Usage: bash animate.sh <run_dir|checkpoints_dir|phase_dir> [fps] [--all-steps]
# Default: only accepted steps. Pass --all-steps to include rejected steps.

set -e

INPUT_DIR="${1:?Usage: $0 <run_dir|checkpoints_dir|phase_dir> [fps] [--all-steps]}"
INPUT_DIR="${INPUT_DIR%/}"
FPS="${2:-4}"
EXTRA="${3:-}"

if [ -d "$INPUT_DIR/checkpoints" ]; then
    RUN_DIR="$INPUT_DIR"
elif [ "$(basename "$INPUT_DIR")" = "checkpoints" ]; then
    RUN_DIR="$(dirname "$INPUT_DIR")"
else
    RUN_DIR="$INPUT_DIR"
fi

# Use the parent folder name as part of the output filename to avoid collisions
RUN_NAME="$(basename "$RUN_DIR")"
OUTPUT="$RUN_DIR/${RUN_NAME}_animation.html"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python3 "$SCRIPT_DIR/animate_structure_steps.py" \
    --run_dir "$RUN_DIR" \
    --output "$OUTPUT" \
    --fps "$FPS" \
    $EXTRA
