#!/bin/bash

#SBATCH --job-name=affibody_T002_T005
#SBATCH --account=yejin
#SBATCH --partition=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --time=23:59:00
#SBATCH --array=0-11%12
#SBATCH --no-requeue
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail

SEED_START="${SEED_START:-46}"
SEED_COUNT=100
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/affibody_design_T002_T005}"
LIBRARY="${LIBRARY:-libb}"
TARGET_IDS=(T002 T003 T004 T005)
RUNS_PER_TASK=4
SLURM_LOCAL_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required.}"
SLURM_TASK_COUNT="${SLURM_ARRAY_TASK_COUNT:?SLURM_ARRAY_TASK_COUNT is required.}"
LANE_COUNT="${LANE_COUNT:-${SLURM_TASK_COUNT}}"
LANE_INDEX="${LANE_INDEX:-${SLURM_LOCAL_INDEX}}"
CHUNK_INDEX="${CHUNK_INDEX:-${SLURM_LOCAL_INDEX}}"
CAMPAIGN_ID="${CAMPAIGN_ID:-${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is required.}}"
SCRIPT_PATH=/afs/ir/users/r/y/rycwang/ProteoCraft/scripts/tep_eval/step2_design.sh

if ! [[ "${SEED_START}" =~ ^[0-9]+$ && "${SEED_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SEED_START must be a non-negative integer and SEED_COUNT must be positive." >&2
  exit 2
fi

if [[ "${LIBRARY}" != "liba" && "${LIBRARY}" != "libb" ]]; then
  echo "LIBRARY must be liba or libb." >&2
  exit 2
fi

if ! [[ "${LANE_COUNT}" =~ ^[1-9][0-9]*$ && \
        "${LANE_INDEX}" =~ ^[0-9]+$ && \
        "${CHUNK_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "LANE_COUNT must be positive; LANE_INDEX and CHUNK_INDEX must be non-negative integers." >&2
  exit 2
fi

SEED_END=$((SEED_START + SEED_COUNT - 1))
TARGET_COUNT=${#TARGET_IDS[@]}
TOTAL_RUNS=$((TARGET_COUNT * SEED_COUNT))
CHUNKS_PER_TARGET=$(((SEED_COUNT + RUNS_PER_TASK - 1) / RUNS_PER_TASK))
TOTAL_CHUNKS=$((TARGET_COUNT * CHUNKS_PER_TARGET))

if (( LANE_COUNT > 12 )); then
  echo "LANE_COUNT cannot exceed the yejin QOS running-job limit of 12." >&2
  exit 2
fi

if (( LANE_INDEX >= LANE_COUNT )); then
  echo "LANE_INDEX ${LANE_INDEX} must be smaller than LANE_COUNT ${LANE_COUNT}." >&2
  exit 2
fi

if (( CHUNK_INDEX >= TOTAL_CHUNKS )); then
  echo "Chunk ${CHUNK_INDEX} is outside the valid range 0-$((TOTAL_CHUNKS - 1))." >&2
  exit 2
fi

if (( CHUNK_INDEX % LANE_COUNT != LANE_INDEX )); then
  echo "Chunk ${CHUNK_INDEX} does not belong to lane ${LANE_INDEX}/${LANE_COUNT}." >&2
  exit 2
fi

if ! [[ "${CAMPAIGN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "CAMPAIGN_ID may contain only letters, digits, dots, underscores, and hyphens." >&2
  exit 2
fi

MODEL_ROOT=/hai/scratch/rycwang/models
export HF_HOME="${MODEL_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Campaign ID: ${CAMPAIGN_ID}"
echo "Lane: ${LANE_INDEX}/$((LANE_COUNT - 1))"
echo "Chunk: ${CHUNK_INDEX}/$((TOTAL_CHUNKS - 1))"
echo "Node: $(hostname)"
echo "Hugging Face cache: ${HF_HUB_CACHE}"
echo "Targets: ${TARGET_IDS[*]}"
echo "Library: ${LIBRARY}"
echo "Seed range: ${SEED_START}-${SEED_END}"
echo "Design count: $((TARGET_COUNT * SEED_COUNT)) (${SEED_COUNT} per target)"

nvidia-smi

cd /afs/ir/users/r/y/rycwang/ProteoCraft

source /afs/ir/users/r/y/rycwang/miniconda3/etc/profile.d/conda.sh
conda activate proteocraft_esmfold2_py312

# Fail before GPU inference if any ESMFold2-workflow model cannot be resolved
# entirely from the local Hugging Face cache.
python - <<'PY'
from huggingface_hub import snapshot_download

model_ids = (
    "biohub/ESMFold2",
    "biohub/ESMFold2-Experimental-Fast",
    "biohub/ESMC-6B",
)
for model_id in model_ids:
    local_path = snapshot_download(model_id, local_files_only=True)
    print(f"Local model ready: {model_id} -> {local_path}")
PY

FAILURE_DIR="${OUTPUT_ROOT}/failures/${CAMPAIGN_ID}"
FAILURE_FILE="${FAILURE_DIR}/chunk_${CHUNK_INDEX}.failed_runs"
STATE_DIR="${OUTPUT_ROOT}/.step2_scheduler/${CAMPAIGN_ID}"
CHUNK_STATE_DIR="${STATE_DIR}/chunks"
SUCCESSOR_DIR="${STATE_DIR}/successors"
LANE_DONE_DIR="${STATE_DIR}/lanes_done"
mkdir -p "${FAILURE_DIR}"
mkdir -p "${CHUNK_STATE_DIR}" "${SUCCESSOR_DIR}" "${LANE_DONE_DIR}"

# Interleave target-specific chunks: chunks 0-3 run seeds 46-49 for
# T002-T005, chunks 4-7 run seeds 50-53, and so on.
TARGET_INDEX=$((CHUNK_INDEX % TARGET_COUNT))
TARGET_CHUNK_INDEX=$((CHUNK_INDEX / TARGET_COUNT))
TARGET_ID="${TARGET_IDS[TARGET_INDEX]}"
FIRST_SEED_OFFSET=$((TARGET_CHUNK_INDEX * RUNS_PER_TASK))
FAILURE_COUNT=0

for ((slot = 0; slot < RUNS_PER_TASK; slot++)); do
  SEED_OFFSET=$((FIRST_SEED_OFFSET + slot))
  if (( SEED_OFFSET >= SEED_COUNT )); then
    break
  fi

  SEED=$((SEED_START + SEED_OFFSET))
  RUN_INDEX=$((TARGET_INDEX * SEED_COUNT + SEED_OFFSET + 1))
  echo "Run ${RUN_INDEX}/${TOTAL_RUNS}: target=${TARGET_ID}, seed=${SEED}"

  if python scripts/run_affibody_design.py \
      --target-manifest data_prepared/affibody/targets.csv \
      --target-id "${TARGET_ID}" \
      --library "${LIBRARY}" \
      --replicates 1 \
      --seed "${SEED}" \
      --output-root "${OUTPUT_ROOT}/${TARGET_ID}/seed_${SEED}" \
      --execute; then
    echo "Completed target ${TARGET_ID}, seed ${SEED}."
  else
    printf '%s:%s\n' "${TARGET_ID}" "${SEED}" >> "${FAILURE_FILE}"
    FAILURE_COUNT=$((FAILURE_COUNT + 1))
    echo "Target ${TARGET_ID}, seed ${SEED} failed; recorded in ${FAILURE_FILE}." >&2
  fi
done

# Record completion before extending this lane. Each lane advances by
# LANE_COUNT chunks, so no two lanes can design the same target/seed pair.
DONE_MARKER="${CHUNK_STATE_DIR}/chunk_${CHUNK_INDEX}.done"
printf 'job=%s chunk=%s failures=%s\n' \
  "${SLURM_JOB_ID:-NA}" "${CHUNK_INDEX}" "${FAILURE_COUNT}" > "${DONE_MARKER}"

CHAIN_FAILURE=0
NEXT_CHUNK=$((CHUNK_INDEX + LANE_COUNT))

if (( NEXT_CHUNK < TOTAL_CHUNKS )); then
  SUCCESSOR_FILE="${SUCCESSOR_DIR}/chunk_${CHUNK_INDEX}.next_job_id"
  SUBMITTED=0
  exec 9>"${STATE_DIR}/submit.lock"

  # A finishing task is still counted against MaxSubmitJobs until it exits.
  # Serialize short submission attempts, but release the lock while waiting so
  # other completed lanes can also make progress as QOS slots become free.
  for ((submit_attempt = 1; submit_attempt <= 960; submit_attempt++)); do
    flock 9
    if [[ -s "${SUCCESSOR_FILE}" ]]; then
      SUBMIT_OUTPUT="$(<"${SUCCESSOR_FILE}")"
      SUBMITTED=1
    elif SUBMIT_OUTPUT="$(sbatch --parsable \
        --array=0-0 \
        --export="ALL,CAMPAIGN_ID=${CAMPAIGN_ID},LANE_COUNT=${LANE_COUNT},LANE_INDEX=${LANE_INDEX},CHUNK_INDEX=${NEXT_CHUNK}" \
        "${SCRIPT_PATH}" 2>&1)"; then
      printf '%s\n' "${SUBMIT_OUTPUT}" > "${SUCCESSOR_FILE}"
      SUBMITTED=1
    fi
    flock -u 9

    if (( SUBMITTED == 1 )); then
      echo "Submitted lane ${LANE_INDEX} successor: job=${SUBMIT_OUTPUT}, chunk=${NEXT_CHUNK}."
      break
    fi

    if (( submit_attempt == 1 || submit_attempt % 20 == 0 )); then
      echo "Successor submission attempt ${submit_attempt} failed: ${SUBMIT_OUTPUT}" >&2
    fi
    sleep 15
  done

  if (( SUBMITTED == 0 )); then
    CHAIN_FAILURE=1
    echo "Could not extend lane ${LANE_INDEX}; resume it at CHUNK_INDEX=${NEXT_CHUNK}." >&2
  fi
else
  # This lane has no more chunks. The final lane to finish creates the campaign
  # marker; it does not delay or gate any earlier successor submissions.
  LANE_DONE_MARKER="${LANE_DONE_DIR}/lane_${LANE_INDEX}.done"
  printf 'job=%s final_chunk=%s failures=%s\n' \
    "${SLURM_JOB_ID:-NA}" "${CHUNK_INDEX}" "${FAILURE_COUNT}" > "${LANE_DONE_MARKER}"

  exec 8>"${STATE_DIR}/completion.lock"
  flock 8
  shopt -s nullglob
  LANE_DONE_MARKERS=("${LANE_DONE_DIR}"/lane_*.done)
  shopt -u nullglob
  LANE_DONE_COUNT=${#LANE_DONE_MARKERS[@]}
  if (( LANE_DONE_COUNT == LANE_COUNT )); then
    printf 'completed_by_job=%s\n' "${SLURM_JOB_ID:-NA}" > "${STATE_DIR}/campaign.complete"
    echo "Campaign ${CAMPAIGN_ID} completed all ${TOTAL_RUNS} target/seed runs."
  fi
  flock -u 8
fi

if (( FAILURE_COUNT > 0 || CHAIN_FAILURE > 0 )); then
  exit 1
fi
