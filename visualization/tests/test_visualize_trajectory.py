"""
Tests for scripts/visualize_trajectory.py — data loading, parsing, RMSD.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import math
import tempfile

import numpy as np
import pytest

from visualize_trajectory import (
    discover_steps,
    parse_ca_atoms,
    kabsch_rmsd,
    _apply_target_alignment,
    compute_all_rmsds,
)


# ---------------------------------------------------------------------------
# PDB / directory helpers
# ---------------------------------------------------------------------------

def _make_pdb_line(serial, atom_name, res_name, chain, res_seq, x, y, z,
                   occupancy=1.00, temp_factor=0.00):
    """Create a PDB ATOM line (non-standard width, split-based compatible)."""
    return (
        f"ATOM  {serial:5d}  {atom_name:<3s}  {res_name:<3s} {chain}{res_seq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{temp_factor:6.2f}           C\n"
    )


def _make_step_dir(parent_dir, step_idx, binder_coords, target_coords):
    """Create a minimal step directory with a PDB and meta.json.

    Args:
        parent_dir:     parent checkpoint directory path.
        step_idx:       integer step number.
        binder_coords:  list of (x, y, z) tuples for chain-A CA atoms.
        target_coords:  list of (x, y, z) tuples for chain-B CA atoms.

    Returns:
        Path to the created step directory.
    """
    step_name = f"step_{step_idx:04d}"
    step_dir = os.path.join(parent_dir, step_name)
    os.makedirs(step_dir, exist_ok=True)

    pdb_name = f"{step_name}_beam_0_complex.pdb"
    pdb_path = os.path.join(step_dir, pdb_name)

    lines = ["REMARK  1  Test PDB\n"]
    serial = 1

    # Chain A (binder) — one CA per residue
    for res_idx, (x, y, z) in enumerate(binder_coords, start=1):
        lines.append(_make_pdb_line(serial, "CA", "ALA", "A", res_idx, x, y, z))
        serial += 1

    # Chain B (target) — one CA per residue
    for res_idx, (x, y, z) in enumerate(target_coords, start=1):
        lines.append(_make_pdb_line(serial, "CA", "ALA", "B", res_idx, x, y, z))
        serial += 1

    lines.append("END\n")
    with open(pdb_path, "w") as f:
        f.writelines(lines)

    meta = {
        "step": step_idx,
        "beam_id": 0,
        "pdb_file": pdb_name,
        "n_binder_res": len(binder_coords),
        "n_target_res": len(target_coords),
        "total_loss": 1.0,
        "loss_dict": {},
        "summary_confidence": {},
        "accepted": False,
        "is_global_best": False,
    }
    meta_path = os.path.join(step_dir, f"{step_name}_beam_0_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    return step_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def checkpoint_dir(tmp_path):
    """3-step checkpoint where binder shifts by [0, 1, 0] each step; target fixed.

    Step 0: binder at y=0, step 1: binder at y=1, step 2: binder at y=2.
    Target is always a 5-atom square in the XZ plane.
    """
    n_binder = 5
    n_target = 5

    # Fixed target: a regular 5-point star-ish shape in XZ plane
    target_base = [(float(i), 0.0, float(i * 2)) for i in range(n_target)]

    ckpt = str(tmp_path)
    for step_idx in range(3):
        shift = float(step_idx)
        binder = [(float(j), shift, 0.0) for j in range(n_binder)]
        _make_step_dir(ckpt, step_idx, binder, target_base)

    return ckpt


@pytest.fixture
def checkpoint_dir_identical_binder(tmp_path):
    """3-step checkpoint where binder AND target are identical across steps."""
    n_binder = 4
    n_target = 4

    binder_base = [(float(i), float(i), 0.0) for i in range(n_binder)]
    target_base = [(float(i) * 2, 0.0, float(i)) for i in range(n_target)]

    ckpt = str(tmp_path)
    for step_idx in range(3):
        _make_step_dir(ckpt, step_idx, binder_base, target_base)

    return ckpt


# ---------------------------------------------------------------------------
# TestDiscoverSteps
# ---------------------------------------------------------------------------

class TestDiscoverSteps:
    def test_finds_all_steps(self, checkpoint_dir):
        steps = discover_steps(checkpoint_dir)
        assert len(steps) == 3

    def test_each_step_has_required_fields(self, checkpoint_dir):
        steps = discover_steps(checkpoint_dir)
        for s in steps:
            assert "step" in s
            assert "pdb_path" in s
            assert "meta" in s
            assert os.path.isfile(s["pdb_path"])
            assert isinstance(s["step"], int)
            assert isinstance(s["meta"], dict)

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(ValueError):
            discover_steps(str(tmp_path))


# ---------------------------------------------------------------------------
# TestParseCA
# ---------------------------------------------------------------------------

class TestParseCA:
    def test_correct_counts(self, checkpoint_dir):
        steps = discover_steps(checkpoint_dir)
        binder_ca, target_ca = parse_ca_atoms(steps[0]["pdb_path"])
        # We set n_binder=5, n_target=5 in the fixture
        assert binder_ca.shape == (5, 3)
        assert target_ca.shape == (5, 3)


# ---------------------------------------------------------------------------
# TestKabschRMSD
# ---------------------------------------------------------------------------

class TestKabschRMSD:
    def test_identical_is_zero(self):
        P = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]])
        assert kabsch_rmsd(P, P) == pytest.approx(0.0, abs=1e-10)

    def test_pure_translation_is_zero(self):
        """After Kabsch alignment, a pure translation should give RMSD ~ 0."""
        rng = np.random.default_rng(42)
        P = rng.standard_normal((10, 3))
        Q = P + np.array([5.0, -3.0, 2.0])   # shift
        assert kabsch_rmsd(P, Q) == pytest.approx(0.0, abs=1e-10)

    def test_non_rigid_deformation_is_nonzero(self):
        """Perturbing one atom should give nonzero RMSD."""
        rng = np.random.default_rng(7)
        P = rng.standard_normal((10, 3))
        Q = P.copy()
        Q[0] += np.array([1.0, 0.0, 0.0])
        assert kabsch_rmsd(P, Q) > 1e-6


# ---------------------------------------------------------------------------
# TestComputeTargetAlignedBinderRMSD
# ---------------------------------------------------------------------------

class TestComputeTargetAlignedBinderRMSD:
    def test_identical_binder_gives_zero(self, checkpoint_dir_identical_binder):
        steps = discover_steps(checkpoint_dir_identical_binder)
        result = compute_all_rmsds(steps)
        for rmsd in result["binder_vs_0"]:
            assert rmsd == pytest.approx(0.0, abs=1e-9)
        for rmsd in result["target_vs_0"]:
            assert rmsd == pytest.approx(0.0, abs=1e-9)

    def test_binder_movement_gives_increasing_rmsd(self, checkpoint_dir):
        """Binder shifts by 1 Å in Y each step; vs-step-0 RMSD should increase."""
        steps = discover_steps(checkpoint_dir)
        result = compute_all_rmsds(steps)
        vs0 = result["binder_vs_0"]
        # Step 0 RMSD vs itself is 0
        assert vs0[0] == pytest.approx(0.0, abs=1e-9)
        # Each subsequent step moves further from step 0
        assert vs0[1] > vs0[0]
        assert vs0[2] > vs0[1]

    def test_adjacent_frames(self, checkpoint_dir):
        """Step 0 adjacent RMSD should be NaN; subsequent steps should be > 0."""
        steps = discover_steps(checkpoint_dir)
        result = compute_all_rmsds(steps)
        adj = result["binder_adjacent"]
        assert math.isnan(adj[0])
        # Steps 1 and 2 each moved the same amount, so adjacent RMSD should be equal
        assert adj[1] == pytest.approx(adj[2], rel=1e-6)
        assert adj[1] > 0.0
