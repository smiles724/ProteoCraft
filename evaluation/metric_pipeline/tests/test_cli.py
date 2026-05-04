"""Monkeypatched ESMFold tests (no GPU required) — v1.1 reviewer P1 #7."""
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_pipeline.cli import (  # noqa: E402
    CSV_COLS,
    _TIER_FLAG_COLS,
    _empty_row,
    evaluate_one,
)
from eval_pipeline.esmfold_runner import ESMFoldResult  # noqa: E402


def _fake_run(esm_seq, tok, mdl, device):
    """Deterministic fake: high pLDDT, low PAE → tier1+tier2 should pass."""
    L = len(esm_seq)
    plddt = np.full(L, 95.0, dtype=np.float32)
    pae = np.full((L, L), 3.0, dtype=np.float32)
    # Build a minimal valid PDB string with L Cα + N + C atoms in chain "A"
    # (rechain will rewrite the chain assignment).
    lines = []
    for i in range(L):
        x = float(i * 3.0)
        # CA only — sufficient for our metrics (Kabsch / pLDDT / scRMSD all use CA)
        lines.append(
            f"ATOM  {i + 1:>5}  CA  ALA A{i + 1:>4}    "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 95.00           C"
        )
    return ESMFoldResult(
        pdb_str="\n".join(lines) + "\nEND\n",
        plddt=plddt,
        pae=pae,
        ptm=0.85,
        runtime_s=0.5,
    )


def _make_row(target_seq="A" * 10, binder_seq="BBB"):
    return pd.Series({
        "method": "BCv1", "case": "PDL1", "design_id": "d1",
        "target_pdb_path": "/tmp/t.pdb", "target_chain": "A",
        "target_seq": target_seq, "binder_seq": binder_seq,
        "designed_complex_pdb_path": "/tmp/c.pdb",
        "target_chain_in_complex": "A", "binder_chain_in_complex": "B",
        "n_generated_trajectories": 100, "n_mpnn_attempted": 50,
        "n_accepted_by_method": 5, "candidate_rank": 1,
        "source_filter_status": "accepted", "verifier_calls": 250,
        "design_gpu_seconds": 60.0,
    })


def test_csv_cols_unique():
    assert len(CSV_COLS) == len(set(CSV_COLS))
    for c in _TIER_FLAG_COLS:
        assert c in CSV_COLS


def test_empty_row_defaults():
    """v1.1 reviewer P1 #5: tier flags default False, error_msg empty (not NaN)."""
    r = _empty_row()
    for c in _TIER_FLAG_COLS:
        assert r[c] is False
    assert r["error_msg"] == ""


def _build_ref_pdb(path: Path, target_len: int, binder_len: int):
    """Two-chain ref: target=A, binder=B with simple linear coords."""
    lines = []
    for i in range(target_len):
        lines.append(
            f"ATOM  {i + 1:>5}  CA  ALA A{i + 1:>4}    "
            f"{i * 3.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 90.00           C"
        )
    base = target_len + 100  # binder 起始原子编号别冲突
    for j in range(binder_len):
        lines.append(
            f"ATOM  {base + j:>5}  CA  ALA B{j + 1:>4}    "
            f"{j * 3.0:8.3f}{10.0:8.3f}{0.0:8.3f}  1.00 90.00           C"
        )
    path.write_text("\n".join(lines) + "\nEND\n")


def test_evaluate_one_happy_path(tmp_path):
    """Mock ESMFold → verify metric pipeline + tier flags work end-to-end on CPU."""
    target_len = 10
    binder_len = 3
    ref_pdb = tmp_path / "ref.pdb"
    _build_ref_pdb(ref_pdb, target_len, binder_len)
    row = _make_row(target_seq="A" * target_len, binder_seq="BBB")
    row["designed_complex_pdb_path"] = str(ref_pdb)
    with patch("eval_pipeline.cli.run_esmfold", side_effect=_fake_run), \
         patch("eval_pipeline.cli.load_esmfold", return_value=(None, None)):
        out = evaluate_one(row, tmp_path, None, None, "cpu")
    assert out["error_msg"] == "", f"unexpected error: {out['error_msg']}"
    assert out["complex_pLDDT"] == pytest.approx(95.0)
    assert out["binder_pLDDT"] == pytest.approx(95.0)
    assert out["ipAE"] == pytest.approx(3.0)
    assert out["tier1_pass"] is True


def test_evaluate_one_failure_keeps_row(tmp_path):
    """If ESMFold raises, row still has all CSV_COLS, error_msg non-empty, tier flags False."""
    def _broken(*a, **kw):
        raise RuntimeError("CUDA OOM (mock)")

    row = _make_row()
    with patch("eval_pipeline.cli.run_esmfold", side_effect=_broken):
        out = evaluate_one(row, tmp_path, None, None, "cpu")
    assert "OOM" in out["error_msg"]
    for c in _TIER_FLAG_COLS:
        assert out[c] is False
    assert set(CSV_COLS).issubset(set(out.keys()))


def test_evaluator_set_before_exception(tmp_path):
    """Bookkeeping fields must survive evaluator exceptions."""
    def _broken(*a, **kw):
        raise RuntimeError("forced evaluator failure")

    row = _make_row()
    with patch("eval_pipeline.cli.run_esmfold", side_effect=_broken):
        out = evaluate_one(row, tmp_path, None, None, "cpu", evaluator="esmfold")

    assert out["evaluator"] == "esmfold"
    assert out["eval_n_calls"] == 1
    assert out["eval_n_successes"] == 0
    assert "forced evaluator failure" in out["error_msg"]
