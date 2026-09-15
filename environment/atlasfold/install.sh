#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${1:-}"
ENV_NAME="${2:-proteocraft_atlasfold_m_py312}"
KERNEL_EXTRA="${3:-torch}"
EXPECTED_COMMIT="444f376d85b9954a5f2f5f3f8b3cbcae1201ebb1"

if [[ -z "$SOURCE_ROOT" ]]; then
  echo "Usage: bash environment/atlasfold/install.sh /path/to/atlasfold-v1.0.2 [env-name] [torch|cuequiv]" >&2
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
if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required. Install Miniforge or Anaconda first." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Using existing conda environment: $ENV_NAME"
else
  conda env create --name "$ENV_NAME" --file "$SCRIPT_DIR/environment.yml"
fi

run_in_env() {
  conda run --no-capture-output --name "$ENV_NAME" "$@"
}

run_in_env python -m pip install --upgrade pip setuptools wheel
if [[ "$KERNEL_EXTRA" == "cuequiv" ]]; then
  run_in_env python -m pip install --editable "${SOURCE_ROOT}[fold,cuequiv]"
else
  run_in_env python -m pip install --editable "${SOURCE_ROOT}[fold]"
fi
run_in_env python "$SCRIPT_DIR/check_environment.py" \
  --source-root "$SOURCE_ROOT" \
  --require-cuda

cat <<EOF

AtlasFold-M environment installed: $ENV_NAME
Pinned source: $SOURCE_ROOT@$EXPECTED_COMMIT

Next:
  conda activate $ENV_NAME
  python environment/atlasfold/download_models.py --cache-dir /path/to/cache

Use the two local weight paths printed by download_models.py for formal runs.
EOF
