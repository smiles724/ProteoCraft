#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${1:-}"
BASE_PYTHON="${2:-python3}"
VENV_DIR="${3:-$HOME/.venvs/proteocraft_atlasfold_m_py312}"
KERNEL_EXTRA="${4:-torch}"
EXPECTED_COMMIT="444f376d85b9954a5f2f5f3f8b3cbcae1201ebb1"

if [[ -z "$SOURCE_ROOT" ]]; then
  echo "Usage: $0 SOURCE_ROOT [BASE_PYTHON] [VENV_DIR] [torch|cuequiv]" >&2
  exit 2
fi
SOURCE_ROOT="$(cd "$SOURCE_ROOT" && pwd)"

if [[ ! -d "$SOURCE_ROOT/.git" || ! -d "$SOURCE_ROOT/src/atlasfold" ]]; then
  echo "Not a full AtlasFold source checkout: $SOURCE_ROOT" >&2
  exit 1
fi
ACTUAL_COMMIT="$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "AtlasFold commit mismatch: $ACTUAL_COMMIT (expected $EXPECTED_COMMIT)" >&2
  exit 1
fi
if ! git -C "$SOURCE_ROOT" diff --quiet || ! git -C "$SOURCE_ROOT" diff --cached --quiet; then
  echo "AtlasFold source has tracked modifications: $SOURCE_ROOT" >&2
  exit 1
fi
if [[ "$KERNEL_EXTRA" != "torch" && "$KERNEL_EXTRA" != "cuequiv" ]]; then
  echo "Kernel extra must be torch or cuequiv" >&2
  exit 2
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$VENV_DIR"
fi
VENV_PYTHON="$VENV_DIR/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
if [[ "$KERNEL_EXTRA" == "cuequiv" ]]; then
  "$VENV_PYTHON" -m pip install --editable "${SOURCE_ROOT}[fold,cuequiv]"
else
  "$VENV_PYTHON" -m pip install --editable "${SOURCE_ROOT}[fold]"
fi
"$VENV_PYTHON" "$SCRIPT_DIR/check_environment.py" \
  --source-root "$SOURCE_ROOT" \
  --require-cuda

cat <<EOF

AtlasFold-M venv installed: $VENV_DIR
Pinned source: $SOURCE_ROOT@$EXPECTED_COMMIT
export ATLASFOLD_PYTHON=$VENV_PYTHON
EOF
