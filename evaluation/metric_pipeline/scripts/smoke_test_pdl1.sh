#!/usr/bin/env bash
# M4 smoke test: PDL1 case (10 designs) on dev GPU.
set -euo pipefail

ROOT="/mnt/shared-storage-user/liyanchao/af3/BCv2/eval0501"
cd "${ROOT}"

source /mnt/shared-storage-user/liyanchao/conda/miniconda3/etc/profile.d/conda.sh
conda activate esmfold-eval

# Redirect HuggingFace cache to gpfs2 shared storage (~7GB weights, share across nodes/jobs)
export HF_HOME="/mnt/shared-storage-gpfs2/liyanchao2/model/huggingface"
mkdir -p "${HF_HOME}"

# GPU/rjob nodes have no outbound internet. Force HF to use the cache directly,
# skip the default ETag HEAD check that would otherwise time out + retry for ~1min.
# Pre-condition: run scripts/prepare_weights.sh on a node WITH internet first.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RUN_ID="smoke_pdl1_$(date +%m%d-%H%M%S)"
OUT="results/${RUN_ID}"
mkdir -p "${OUT}"

echo "[1/3] generate manifest"
python -m adapters.bcv1_adapter \
    --batch-dir /mnt/shared-storage-user/liyanchao/af3/BindCraft/outputs/batch_v1 \
    --case PDL1 \
    --out "${OUT}/manifest.csv"

echo "[2/3] run ESMFold pipeline (~5-15min for 10 designs on H200)"
python -m eval_pipeline.cli \
    --manifest "${OUT}/manifest.csv" \
    --out-dir "${OUT}" \
    --device cuda:0

echo "[3/3] code-correctness gates"
python <<'PY'
import os, json, sys
import pandas as pd
import numpy as np
from pathlib import Path

run_dir = sorted(Path("results").glob("smoke_pdl1_*"))[-1]
print(f"checking {run_dir}")

df = pd.read_csv(run_dir/"eval_long.csv", keep_default_na=False, na_values=["nan","NaN"])
manifest = pd.read_csv(run_dir/"manifest.csv")

# Gate 1: row count
assert len(df) == len(manifest), f"long CSV {len(df)} rows != manifest {len(manifest)}"
print(f"  ✓ row count = {len(df)}")

# Gate 2: 0 errors
n_err = int(df['error_msg'].astype(str).str.len().gt(0).sum())
assert n_err == 0, f"{n_err} rows with errors:\n" + str(df.loc[df['error_msg'].astype(str).str.len() > 0, ['design_id','error_msg']])
print(f"  ✓ 0 errors")

# Gate 3: critical metrics non-NaN
critical = ["binder_pLDDT","complex_pLDDT","ipAE","binder_scRMSD","inter_chain_clash_count"]
for c in critical:
    nans = pd.to_numeric(df[c], errors="coerce").isna().sum()
    assert nans == 0, f"{c}: {nans} NaN values"
print(f"  ✓ critical metrics all non-NaN")

# Gate 4: artifacts
for _, r in df.iterrows():
    pdb = Path(r['esmfold_pdb_path'])
    plddt = pdb.parent / "plddt.npy"
    pae = pdb.parent / "pae.npy"
    meta = pdb.parent / "meta.json"
    for f in (pdb, plddt, pae, meta):
        assert f.exists(), f"missing {f}"
    m = json.loads(meta.read_text())
    assert m["binder_len"] == r["binder_length"], f"{r['design_id']}: meta binder_len {m['binder_len']} != manifest binder_length {r['binder_length']}"
    chains = {s["pdb_chain_id"] for s in m["clean_segments"]}
    assert chains == {"A","B"}, f"{r['design_id']}: pdb_chain_ids {chains}, expected {{A,B}}"
print(f"  ✓ artifacts complete + chain IDs ∈ {{A,B}} + binder_len consistent")

# Gate 5: scRMSD finite + reasonable
for _, r in df.iterrows():
    scr = float(r["binder_scRMSD"])
    assert np.isfinite(scr) and 0 <= scr < 100, f"{r['design_id']}: scRMSD {scr} suspicious"
print(f"  ✓ binder_scRMSD all finite + < 100Å")

# === Biological observations (report only) ===
print("\nBiological observations (NOT gates):")
print(f"  binder_pLDDT  mean={pd.to_numeric(df['binder_pLDDT'], errors='coerce').mean():.1f}, max={pd.to_numeric(df['binder_pLDDT'], errors='coerce').max():.1f}")
print(f"  complex_pLDDT mean={pd.to_numeric(df['complex_pLDDT'], errors='coerce').mean():.1f}, max={pd.to_numeric(df['complex_pLDDT'], errors='coerce').max():.1f}")
print(f"  ipAE          mean={pd.to_numeric(df['ipAE'], errors='coerce').mean():.2f}, min={pd.to_numeric(df['ipAE'], errors='coerce').min():.2f}")
print(f"  binder_scRMSD mean={pd.to_numeric(df['binder_scRMSD'], errors='coerce').mean():.2f}, min={pd.to_numeric(df['binder_scRMSD'], errors='coerce').min():.2f}")
print(f"  tier1_plddt80_pass: {(df['tier1_plddt80_pass'].astype(str).str.lower()=='true').sum()}/{len(df)}")
print(f"  tier1_plddt90_pass: {(df['tier1_plddt90_pass'].astype(str).str.lower()=='true').sum()}/{len(df)}")
print(f"  tier2_complexa_pass: {(df['tier2_complexa_pass'].astype(str).str.lower()=='true').sum()}/{len(df)}")
print(f"  tier2_strict_physical_pass: {(df['tier2_strict_physical_pass'].astype(str).str.lower()=='true').sum()}/{len(df)}")
print(f"  mean ESMFold runtime: {pd.to_numeric(df['esmfold_runtime_s'], errors='coerce').mean():.1f}s")
print("\nALL CODE-CORRECTNESS GATES PASSED ✓")
PY
