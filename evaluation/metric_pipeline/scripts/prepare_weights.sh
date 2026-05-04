#!/usr/bin/env bash
# Pre-download ESMFold v1 weights to gpfs2 shared storage.
# Runnable on CPU node — no GPU / no CUDA needed (just network + disk).
# Idempotent: re-running is a fast no-op once cache is populated.
set -euo pipefail

source /mnt/shared-storage-user/liyanchao/conda/miniconda3/etc/profile.d/conda.sh
conda activate esmfold-eval

export HF_HOME="/mnt/shared-storage-gpfs2/liyanchao2/model/huggingface"
mkdir -p "${HF_HOME}"

echo "HF_HOME = ${HF_HOME}"
echo "downloading facebook/esmfold_v1 (~7GB) — first run takes a few minutes..."

python <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = "facebook/esmfold_v1"
# Fetch model + tokenizer; allow both safetensors and pytorch_model.bin formats
local_path = snapshot_download(
    repo_id=repo_id,
    # Empty allow_patterns = fetch all files in the repo
)
print(f"\nrepo cached at: {local_path}")

# List downloaded files + sizes
print("\nfiles:")
total = 0
for p in sorted(Path(local_path).rglob("*")):
    if p.is_file() and not p.is_symlink():
        size_mb = p.stat().st_size / (1024 * 1024)
        total += size_mb
        if size_mb > 1.0:
            print(f"  {size_mb:>8.1f} MB  {p.relative_to(local_path)}")
print(f"\ntotal: {total:.1f} MB ({total/1024:.2f} GB)")

# Sanity: load tokenizer (no GPU needed) — confirms the cache is functional
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(repo_id)
print(f"\ntokenizer OK: {type(tokenizer).__name__}")
PY

echo ""
echo "✓ ESMFold weights ready at ${HF_HOME}"
echo "  GPU runs will hit cache instantly (no download)"
