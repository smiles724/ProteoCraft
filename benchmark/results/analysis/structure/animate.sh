#!/bin/bash
# Usage: bash animate.sh <checkpoints_dir> [fps] [--all-steps]
# Default: only accepted steps. Pass --all-steps to include rejected steps.

set -e

CHECKPOINTS_DIR="${1:?Usage: $0 <checkpoints_dir> [fps] [--all-steps]}"
FPS="${2:-4}"
EXTRA="${3:-}"

RUN_DIR="$(dirname "$CHECKPOINTS_DIR")"
OUTPUT="$RUN_DIR/animation.html"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python3 "$SCRIPT_DIR/animate_structure_steps.py" \
    --run_dir "$RUN_DIR" \
    --output "$OUTPUT" \
    --fps "$FPS" \
    $EXTRA
