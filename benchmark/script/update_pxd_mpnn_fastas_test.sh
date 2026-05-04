#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash benchmark/script/update_pxd_mpnn_fastas_test.sh [--gpu GPU] [--out-dir DIR] [--seed SEED] [--cpu]

Defaults:
  --gpu      0
  --out-dir  /data/huangyanjie/binder_design/BindCraft-v2/benchmark/outputs/pxd_rf3
  --seed     0

Optional environment overrides:
  CONDA_BIN=/home/huangyanjie/miniconda3/bin/conda
  CONDA_ENV=pxdesign_20260416_cu121
  PXDESIGN_ROOT=/data/huangyanjie/binder_design/PXDesign
  PXDESIGN_RUNNER=/data/huangyanjie/binder_design/PXDesign/scripts/run_pxdesign_gpu.sh
  NUM_SEQS=1
  TEMPERATURE=0.0001
  RM_AA=C
  KEEP_TEMP=1
  FORCE_CPU=1
EOF
}

REPO_ROOT="/data/huangyanjie/binder_design"
PXDESIGN_ROOT="${PXDESIGN_ROOT:-${REPO_ROOT}/PXDesign}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/BindCraft-v2/benchmark/outputs/pxd_rf3}"
PY_SCRIPT="${PY_SCRIPT:-${REPO_ROOT}/BindCraft-v2/benchmark/script/update_pxd_mpnn_fastas.py}"
CONDA_BIN="${CONDA_BIN:-/home/huangyanjie/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-pxdesign_20260416_cu121}"
PXDESIGN_RUNNER="${PXDESIGN_RUNNER:-${PXDESIGN_ROOT}/scripts/run_pxdesign_gpu.sh}"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
NUM_SEQS="${NUM_SEQS:-1}"
TEMPERATURE="${TEMPERATURE:-0.0001}"
RM_AA="${RM_AA:-C}"
KEEP_TEMP="${KEEP_TEMP:-0}"
FORCE_CPU="${FORCE_CPU:-0}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/pxdesign-xdg-cache}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/pxdesign-torch-extensions}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --num-seqs)
      NUM_SEQS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --rm-aa)
      RM_AA="$2"
      shift 2
      ;;
    --keep-temp)
      KEEP_TEMP=1
      shift
      ;;
    --cpu)
      FORCE_CPU=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "Conda executable not found: $CONDA_BIN" >&2
  exit 1
fi

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "Python updater not found: $PY_SCRIPT" >&2
  exit 1
fi

if [[ ! -d "$OUT_DIR" ]]; then
  echo "Output directory not found: $OUT_DIR" >&2
  exit 1
fi

if [[ "$FORCE_CPU" != "1" && ! -x "$PXDESIGN_RUNNER" ]]; then
  echo "PXDesign GPU runner not found or not executable: $PXDESIGN_RUNNER" >&2
  exit 1
fi

mkdir -p "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR"

args=(
  "$PY_SCRIPT"
  --out-dir "$OUT_DIR"
  --gpu "$GPU"
  --seed "$SEED"
  --num-seqs "$NUM_SEQS"
  --temperature "$TEMPERATURE"
  --rm-aa "$RM_AA"
)

if [[ "$KEEP_TEMP" == "1" ]]; then
  args+=(--keep-temp)
fi

echo "PXDesign root: $PXDESIGN_ROOT"
echo "Output dir:    $OUT_DIR"
echo "GPU:           $GPU"
echo "Conda env:     $CONDA_ENV"
echo "Force CPU:     $FORCE_CPU"
if [[ "$FORCE_CPU" != "1" ]]; then
  echo "PXDesign run:  $PXDESIGN_RUNNER"
fi

cd "$PXDESIGN_ROOT"

if [[ "$FORCE_CPU" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="" \
  JAX_PLATFORMS=cpu \
  JAX_PLATFORM_NAME=cpu \
  MPLCONFIGDIR="$MPLCONFIGDIR" \
  TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python "${args[@]}"
else
  PX_GPU_ID="$GPU" \
  MPLCONFIGDIR="$MPLCONFIGDIR" \
  TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
  "$PXDESIGN_RUNNER" python "${args[@]}"
fi
