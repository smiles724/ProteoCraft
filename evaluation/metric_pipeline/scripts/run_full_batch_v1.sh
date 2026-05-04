#!/usr/bin/env bash
# M5 full batch: 13 cases sequentially on a single GPU.
# User will adapt to rjob array job for parallelism (1 case per rjob).
set -euo pipefail

ROOT="/mnt/shared-storage-user/liyanchao/af3/BCv2/eval0501"
cd "${ROOT}"

source /mnt/shared-storage-user/liyanchao/conda/miniconda3/etc/profile.d/conda.sh
conda activate esmfold-eval

# Share HF cache via gpfs2 (avoid re-downloading 7GB weights per rjob)
export HF_HOME="/mnt/shared-storage-gpfs2/liyanchao2/model/huggingface"
mkdir -p "${HF_HOME}"

# GPU/rjob nodes have no outbound internet. Force HF to use the cache directly,
# skip the default ETag HEAD check that would otherwise time out + retry for ~1min.
# Pre-condition: run scripts/prepare_weights.sh on a node WITH internet first.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RUN_ID="batch_v1_$(date +%m%d-%H%M%S)"
OUT="results/${RUN_ID}"
mkdir -p "${OUT}"

CASES=(BetV1 DerF21 DerF7 HER2 IFNAR2 PD1 PDL1 SpCas9
       TCR_1QRN TCR_3QDJ TCR_5BS0 TCR_6BJ8 TCR_9C3E)

# 1) Per-case manifests, then concat (header-only files concat safely thanks to v1.1 fix)
echo "[1] generate per-case manifests"
HEADER_WRITTEN=0
> "${OUT}/manifest.csv"
for CASE in "${CASES[@]}"; do
    CASE_OUT="${OUT}/manifest_${CASE}.csv"
    if ! python -m adapters.bcv1_adapter \
        --batch-dir /mnt/shared-storage-user/liyanchao/af3/BindCraft/outputs/batch_v1 \
        --case "${CASE}" \
        --out "${CASE_OUT}"; then
        echo "WARN: ${CASE} adapter failed; skipping"
        continue
    fi
    if [ ${HEADER_WRITTEN} -eq 0 ]; then
        cat "${CASE_OUT}" >> "${OUT}/manifest.csv"
        HEADER_WRITTEN=1
    else
        tail -n +2 "${CASE_OUT}" >> "${OUT}/manifest.csv"
    fi
done

N_ROWS=$(($(wc -l < "${OUT}/manifest.csv") - 1))
echo "  ${N_ROWS} total rows in combined manifest"
if [ "${N_ROWS}" -eq 0 ]; then
    echo "  combined manifest is empty (no Accepted/*.pdb in any case yet); aborting"
    exit 0
fi

# 2) Run pipeline
echo "[2] run ESMFold pipeline"
python -m eval_pipeline.cli \
    --manifest "${OUT}/manifest.csv" \
    --out-dir "${OUT}" \
    --device cuda:0

# 3) Per-case error summary
echo "[3] error summary"
python <<PY
import pandas as pd
df = pd.read_csv("${OUT}/eval_long.csv", keep_default_na=False, na_values=["nan","NaN"])
print(df.groupby("case").agg(
    n=("design_id", "count"),
    n_err=("error_msg", lambda s: int(s.astype(str).str.len().gt(0).sum())),
    tier1_pass=("tier1_pass", lambda s: int((s.astype(str).str.lower()=="true").sum())),
    tier2_pass=("tier2_pass", lambda s: int((s.astype(str).str.lower()=="true").sum())),
))
PY

echo ""
echo "Run dir: ${OUT}"
echo "Next: python -m eval_pipeline.report --long-csv ${OUT}/eval_long.csv --out-dir ${OUT} --run-id $(basename ${OUT})"
