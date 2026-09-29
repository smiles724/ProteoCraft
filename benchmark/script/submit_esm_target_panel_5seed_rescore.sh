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

RUNNER="$REPO_ROOT/benchmark/script/run_esm_target_panel_5seed_rescore.py"

case "${ESM_5SEED_MODE:-submit}" in
  worker)
    : "${SLURM_ARRAY_TASK_ID:?Five-seed worker must run as a Slurm array task}"
    "$ESMOPT_PYTHON" "$RUNNER" rescore-target \
      --output "$ESM_TARGET_PANEL_OUTPUT" \
      --index "$SLURM_ARRAY_TASK_ID"
    ;;
  worker_all)
    IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
    if (( ${#visible_devices[@]} < 2 )); then
      echo "Five-seed worker_all requires two visible GPUs" >&2
      exit 2
    fi
    run_lane() {
      local lane="$1"
      local index
      export CUDA_VISIBLE_DEVICES="${visible_devices[$lane]}"
      export OMP_NUM_THREADS=8
      for ((index=lane; index<13; index+=2)); do
        "$ESMOPT_PYTHON" "$RUNNER" rescore-target \
          --output "$ESM_TARGET_PANEL_OUTPUT" \
          --index "$index"
      done
    }
    run_lane 0 &
    lane_zero_pid="$!"
    run_lane 1 &
    lane_one_pid="$!"
    failed=0
    wait "$lane_zero_pid" || failed=1
    wait "$lane_one_pid" || failed=1
    if (( failed != 0 )); then
      echo "At least one five-seed rescore lane failed" >&2
      exit 1
    fi
    "$ESMOPT_PYTHON" "$RUNNER" summarize --output "$ESM_TARGET_PANEL_OUTPUT"
    ;;
  worker_completed)
    IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
    if (( ${#visible_devices[@]} < 2 )); then
      echo "Five-seed worker_completed requires two visible GPUs" >&2
      exit 2
    fi
    run_completed_lane() {
      local lane="$1"
      local index
      export CUDA_VISIBLE_DEVICES="${visible_devices[$lane]}"
      export OMP_NUM_THREADS=8
      for ((index=lane; index<7; index+=2)); do
        "$ESMOPT_PYTHON" "$RUNNER" rescore-target \
          --output "$ESM_TARGET_PANEL_OUTPUT" \
          --index "$index"
      done
    }
    run_completed_lane 0 &
    lane_zero_pid="$!"
    run_completed_lane 1 &
    lane_one_pid="$!"
    failed=0
    wait "$lane_zero_pid" || failed=1
    wait "$lane_one_pid" || failed=1
    if (( failed != 0 )); then
      echo "At least one completed-target rescore lane failed" >&2
      exit 1
    fi
    "$ESMOPT_PYTHON" "$RUNNER" summarize \
      --output "$ESM_TARGET_PANEL_OUTPUT" \
      --allow-partial
    ;;
  worker_remaining)
    IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
    if (( ${#visible_devices[@]} < 2 )); then
      echo "Five-seed worker_remaining requires two visible GPUs" >&2
      exit 2
    fi
    run_remaining_lane() {
      local lane="$1"
      local index
      export CUDA_VISIBLE_DEVICES="${visible_devices[$lane]}"
      export OMP_NUM_THREADS=8
      for ((index=7+lane; index<13; index+=2)); do
        "$ESMOPT_PYTHON" "$RUNNER" rescore-target \
          --output "$ESM_TARGET_PANEL_OUTPUT" \
          --index "$index"
      done
    }
    run_remaining_lane 0 &
    lane_zero_pid="$!"
    run_remaining_lane 1 &
    lane_one_pid="$!"
    failed=0
    wait "$lane_zero_pid" || failed=1
    wait "$lane_one_pid" || failed=1
    if (( failed != 0 )); then
      echo "At least one remaining-target rescore lane failed" >&2
      exit 1
    fi
    "$ESMOPT_PYTHON" "$RUNNER" summarize --output "$ESM_TARGET_PANEL_OUTPUT"
    ;;
  finalize)
    "$ESMOPT_PYTHON" "$RUNNER" summarize --output "$ESM_TARGET_PANEL_OUTPUT"
    ;;
  submit)
    : "${ESM_TARGET_PANEL_SEARCH_JOB_ID:?Set ESM_TARGET_PANEL_SEARCH_JOB_ID to the search array job ID}"
    : "${ESM_SLURM_ACCOUNT:?Set ESM_SLURM_ACCOUNT for this cluster}"
    : "${ESM_SLURM_PARTITION:?Set ESM_SLURM_PARTITION for this cluster}"
    "$ESMOPT_PYTHON" "$RUNNER" prepare --output "$ESM_TARGET_PANEL_OUTPUT"
    mkdir -p "$ESM_TARGET_PANEL_OUTPUT/rescore_5seed"
    rescore_job_id="$(sbatch --parsable \
      --job-name=pc-esm-5seed \
      --account="$ESM_SLURM_ACCOUNT" \
      --partition="$ESM_SLURM_PARTITION" \
      --gres="${ESM_SLURM_GRES:-gpu:h200:2}" \
      --cpus-per-task="${ESM_SLURM_CPUS:-16}" \
      --mem="${ESM_SLURM_MEMORY:-192G}" \
      --time="${ESM_SLURM_TIME:-04:00:00}" \
      --chdir="$REPO_ROOT" \
      --dependency="afterok:$ESM_TARGET_PANEL_SEARCH_JOB_ID" \
      --export=ALL,ESM_5SEED_MODE=worker_all \
      --output="$ESM_TARGET_PANEL_OUTPUT/rescore_5seed/slurm-%j.log" \
      "$0")"
    cat <<EOF
Submitted five-seed ESMFold2 rescore and report job: $rescore_job_id
Search dependency: $ESM_TARGET_PANEL_SEARCH_JOB_ID
Resources: 2 H200 GPUs; two target lanes
Expected score files: 1300
Output: $ESM_TARGET_PANEL_OUTPUT/rescore_5seed
EOF
    ;;
  *)
    echo "Unknown ESM_5SEED_MODE=${ESM_5SEED_MODE}" >&2
    exit 2
    ;;
esac
