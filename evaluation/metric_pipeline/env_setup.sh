#!/usr/bin/env bash
# Create + populate conda env esmfold-eval (idempotent).
# v1.1: requires accelerate for transformers low_cpu_mem_usage=True.
set -euo pipefail

ENV_NAME=esmfold-eval
CONDA_BASE=/mnt/shared-storage-user/liyanchao/conda/miniconda3
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "Env ${ENV_NAME} already exists; reusing"
else
    conda create -y -n "${ENV_NAME}" python=3.10
fi
conda activate "${ENV_NAME}"

pip install --no-cache-dir -r "${HERE}/requirements.txt"

python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "from transformers import EsmForProteinFolding; print('transformers OK (EsmForProteinFolding importable)')"
python -c "import accelerate; print('accelerate', accelerate.__version__)"

echo "env ${ENV_NAME} ready"
