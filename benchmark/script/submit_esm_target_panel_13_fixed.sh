#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${PROTEOCRAFT_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
PROTEOCRAFT_REPO_ROOT="$REPO_ROOT"
: "${ESMOPT_PYTHON:?Set ESMOPT_PYTHON to the Fast environment Python executable}"
: "${PROTEOCRAFT_UPLIFT_ROOT:?Set PROTEOCRAFT_UPLIFT_ROOT}"
: "${HF_HOME:?Set HF_HOME to the pinned model cache}"
ESMCFOLD_CCD_PATH="${ESMCFOLD_CCD_PATH:-$HF_HOME/hub/models--biohub--ESMFold2/snapshots/1ebf0e3481a5184eb6171d40615c79e384b48796/ccd.pkl}"
ESM_TARGET_PANEL_OUTPUT="${ESM_TARGET_PANEL_OUTPUT:-$REPO_ROOT/outputs/esm_optimization/target_panel_13_reporting_fixed}"

export PROTEOCRAFT_REPO_ROOT ESMOPT_PYTHON PROTEOCRAFT_UPLIFT_ROOT HF_HOME
export ESMCFOLD_CCD_PATH ESM_TARGET_PANEL_OUTPUT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "$REPO_ROOT"

RUNNER="$REPO_ROOT/benchmark/script/run_esm_target_panel_13_fixed.py"

if [[ "${ESM_TARGET_PANEL_FINALIZE:-0}" == 1 ]]; then
  "$ESMOPT_PYTHON" "$RUNNER" summarize --output "$ESM_TARGET_PANEL_OUTPUT"
  exit 0
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  : "${SLURM_ARRAY_TASK_ID:?This worker must run as a Slurm array task}"
  "$ESMOPT_PYTHON" "$RUNNER" run-target \
    --output "$ESM_TARGET_PANEL_OUTPUT" \
    --index "$SLURM_ARRAY_TASK_ID"
  exit 0
fi

mkdir -p "$ESM_TARGET_PANEL_OUTPUT"
"$ESMOPT_PYTHON" "$RUNNER" prepare \
  --output "$ESM_TARGET_PANEL_OUTPUT" \
  --kit-root "$PROTEOCRAFT_UPLIFT_ROOT"

: "${ESM_SLURM_ACCOUNT:?Set ESM_SLURM_ACCOUNT for this cluster}"
: "${ESM_SLURM_PARTITION:?Set ESM_SLURM_PARTITION for this cluster}"
array_job_id="$(sbatch --parsable \
  --job-name=pc-esm-target13-fixed \
  --account="$ESM_SLURM_ACCOUNT" \
  --partition="$ESM_SLURM_PARTITION" \
  --gres="${ESM_SLURM_GRES:-gpu:h200:2}" \
  --cpus-per-task="${ESM_SLURM_CPUS:-16}" \
  --mem="${ESM_SLURM_MEMORY:-192G}" \
  --time="${ESM_SLURM_TIME:-24:00:00}" \
  --chdir="$REPO_ROOT" \
  --array=0-12%13 \
  --output="$ESM_TARGET_PANEL_OUTPUT/slurm-%A_%a.log" \
  "$0")"

export ESM_TARGET_PANEL_FINALIZE=1
final_job_id="$(sbatch --parsable \
  --job-name=pc-esm-target13-fixed-final \
  --account="$ESM_SLURM_ACCOUNT" \
  --partition="$ESM_SLURM_PARTITION" \
  --cpus-per-task="${ESM_SLURM_FINAL_CPUS:-2}" \
  --mem="${ESM_SLURM_FINAL_MEMORY:-16G}" \
  --time="${ESM_SLURM_FINAL_TIME:-00:30:00}" \
  --chdir="$REPO_ROOT" \
  --dependency="afterany:$array_job_id" \
  --output="$ESM_TARGET_PANEL_OUTPUT/slurm-final-%j.log" \
  "$0")"

cat <<EOF
Submitted 13-target GPU array: $array_job_id (indices 0-12, concurrency cap 13)
Submitted dependent analysis job: $final_job_id
Output: $ESM_TARGET_PANEL_OUTPUT
Logs: $ESM_TARGET_PANEL_OUTPUT/slurm-${array_job_id}_<index>.log
EOF
