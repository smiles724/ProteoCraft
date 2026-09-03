#!/bin/bash

#SBATCH --job-name=affibody_preflight
#SBATCH --account=yejin
#SBATCH --partition=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

MODEL_ROOT=/hai/scratch/rycwang/models
export HF_HOME="${MODEL_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Node: $(hostname)"
echo "Hugging Face cache: ${HF_HUB_CACHE}"

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

python scripts/score_affibody_complexes.py \
  --complexes data_prepared/affibody/known_complexes_blind.csv \
  --reference-pdb datafolder/nyeso_xx133_complex.pdb \
  --output validation/known_pair_scores_blind.jsonl \
  --verifier-loss-mode aligned_common \
  --seeds 101,102,103 \
  --device cuda \
  --execute