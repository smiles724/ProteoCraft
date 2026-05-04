"""Tests for binder_scRMSD multi-chain target alignment."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from eval_pipeline import metrics as M


def _write_pdb(path: Path, atoms: list[tuple]):
    lines = []
    serial = 1
    for chain, resi, atom_name, x, y, z, b in atoms:
        lines.append(
            f"ATOM  {serial:>5d}  {atom_name:<3s} ALA {chain}{resi:>4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}           "
            f"{atom_name[0]}\n"
        )
        serial += 1
    lines.append("TER\nEND\n")
    path.write_text("".join(lines))


def test_binder_scrmsd_multichain_target_concat(tmp_path):
    pred_atoms = [
        ("D", 1, "CA", 0, 0, 0, 80),
        ("D", 2, "CA", 1, 0, 0, 80),
        ("B", 1, "CA", 5, 0, 0, 90),
        ("B", 2, "CA", 6, 0, 0, 90),
        ("C", 1, "CA", 5, 5, 0, 90),
        ("C", 2, "CA", 6, 5, 0, 90),
    ]
    ref_atoms = [
        ("B", 1, "CA", 0.1, 0.1, 0, 80),
        ("B", 2, "CA", 1.1, 0.1, 0, 80),
        ("A", 1, "CA", 5, 0, 0, 90),
        ("A", 2, "CA", 6, 0, 0, 90),
        ("A", 3, "CA", 5, 5, 0, 90),
        ("A", 4, "CA", 6, 5, 0, 90),
    ]
    pred_pdb = tmp_path / "pred.pdb"
    ref_pdb = tmp_path / "ref.pdb"
    _write_pdb(pred_pdb, pred_atoms)
    _write_pdb(ref_pdb, ref_atoms)

    sc, fold = M.binder_scRMSD(
        pred_pdb_path=str(pred_pdb),
        ref_pdb_path=str(ref_pdb),
        target_chains_pred=["B", "C"],
        target_chains_ref=["A"],
        binder_chain_pred="D",
        binder_chain_ref="B",
    )

    assert math.isfinite(sc)
    assert math.isfinite(fold)


def test_binder_scrmsd_ca_count_mismatch_raises(tmp_path):
    pred_atoms = [
        ("D", 1, "CA", 0, 0, 0, 80),
        ("D", 2, "CA", 1, 0, 0, 80),
        ("B", 1, "CA", 5, 0, 0, 90),
    ]
    ref_atoms = [
        ("B", 1, "CA", 0, 0, 0, 80),
        ("B", 2, "CA", 1, 0, 0, 80),
        ("A", 1, "CA", 5, 0, 0, 90),
        ("A", 2, "CA", 6, 0, 0, 90),
    ]
    pred_pdb = tmp_path / "pred.pdb"
    ref_pdb = tmp_path / "ref.pdb"
    _write_pdb(pred_pdb, pred_atoms)
    _write_pdb(ref_pdb, ref_atoms)

    with pytest.raises(ValueError, match="target Cα count mismatch"):
        M.binder_scRMSD(
            pred_pdb_path=str(pred_pdb),
            ref_pdb_path=str(ref_pdb),
            target_chains_pred=["B"],
            target_chains_ref=["A"],
            binder_chain_pred="D",
            binder_chain_ref="B",
        )
