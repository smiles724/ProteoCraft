import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_pipeline.esmfold_runner import build_esmfold_input, compute_clean_segments  # noqa: E402
from eval_pipeline.metrics import (  # noqa: E402
    _assert_pae_unit,
    binder_pLDDT,
    binder_scRMSD,
    complex_pLDDT,
    i_pTM_approx_from_PAE,
    interface_pLDDT,
    inter_chain_clash_count,
    ipAE,
    mean_pAE,
    min_ipAE,
    n_contacts_5A,
    radius_of_gyration,
    target_pLDDT,
)
from eval_pipeline.pdb_utils import kabsch_align, rmsd  # noqa: E402


def _segs(target_seq="AAA", binder_seq="BB"):
    _, raw = build_esmfold_input(target_seq, binder_seq)
    return compute_clean_segments(raw)


# ─── pLDDT family ───────────────────────────────────────────────────
def test_complex_plddt_mean_all():
    plddt = np.array([90, 80, 70, 60, 50], dtype=np.float32)
    assert complex_pLDDT(plddt, _segs()) == pytest.approx(70.0)


def test_binder_plddt_slices_binder_only():
    plddt = np.array([90, 90, 90, 60, 60], dtype=np.float32)
    assert binder_pLDDT(plddt, _segs()) == pytest.approx(60.0)


def test_target_plddt_slices_all_target_chains():
    plddt = np.array([100, 100, 100, 50, 50, 50, 0, 0], dtype=np.float32)
    segs = _segs("AAA:CCC", "BB")
    assert target_pLDDT(plddt, segs) == pytest.approx(75.0)
    assert binder_pLDDT(plddt, segs) == pytest.approx(0.0)


def test_interface_plddt_uses_target_5A_proximity():
    plddt = np.array([90, 90, 90, 60, 60], dtype=np.float32)
    target_ca = np.array([[0, 0, 0], [0, 0, 3], [0, 0, 6]], dtype=np.float64)
    binder_ca = np.array([[0, 0, 1], [0, 0, 100]], dtype=np.float64)
    val = interface_pLDDT(plddt, _segs(), target_ca, binder_ca, cutoff=5.0)
    assert val == pytest.approx(60.0)


# ─── PAE family ─────────────────────────────────────────────────────
def test_mean_pae_average_all():
    pae = np.full((5, 5), 4.0, dtype=np.float32)
    assert mean_pAE(pae, _segs()) == pytest.approx(4.0)


def test_ipae_symmetric():
    pae = np.array(
        [
            [0, 0, 0, 5, 5],
            [0, 0, 0, 5, 5],
            [0, 0, 0, 5, 5],
            [9, 9, 9, 0, 0],
            [9, 9, 9, 0, 0],
        ],
        dtype=np.float32,
    )
    # binder→target = mean[9,9,9, 9,9,9] = 9; target→binder = mean[5,5,5,5,5,5] = 5
    assert ipAE(pae, _segs()) == pytest.approx(7.0)


def test_min_ipae_takes_min_cross_block():
    pae = np.full((5, 5), 4.0, dtype=np.float32)
    pae[3, 0] = 1.5  # binder→target row=binder_idx=3, col=target_idx=0
    assert min_ipAE(pae, _segs()) == pytest.approx(1.5)


def test_iptm_known_bounds():
    segs = _segs("A" * 50, "B" * 50)
    pae_zero = np.zeros((100, 100), dtype=np.float32)
    pae_large = np.full((100, 100), 30.0, dtype=np.float32)
    assert i_pTM_approx_from_PAE(pae_zero, segs) > 0.99
    assert i_pTM_approx_from_PAE(pae_large, segs) < 0.05


def test_pae_unit_assert_passes_raw_angstrom():
    # ESMFold raw Å values up to 31 and AF3 raw Å values up to ~99.9 must pass.
    _assert_pae_unit(np.array([0.0, 5.5, 31.0, 100.0], dtype=np.float32))


def test_pae_unit_assert_rejects_double_multiplied():
    # values > 100.5 must raise (e.g. someone × 31 the already-raw values → up to 961)
    bad = np.array([0.0, 5.5, 100.6], dtype=np.float32)
    with pytest.raises(AssertionError, match="exceeds tolerated range"):
        _assert_pae_unit(bad)


def test_pae_unit_assert_rejects_negative():
    with pytest.raises(AssertionError, match="must be non-negative"):
        _assert_pae_unit(np.array([0.0, -0.1], dtype=np.float32))


def test_ipae_propagates_pae_unit_assert():
    # ipAE should also raise if upstream double-multiplied
    bad = np.full((5, 5), 101.0, dtype=np.float32)
    with pytest.raises(AssertionError, match="exceeds tolerated range"):
        ipAE(bad, _segs())


# ─── Structural family ──────────────────────────────────────────────
def test_kabsch_translation_invariant():
    P = np.random.RandomState(0).randn(10, 3)
    Q = P + np.array([5.0, 3.0, -2.0])
    R, t = kabsch_align(P, Q)
    aligned = P @ R.T + t
    assert rmsd(aligned, Q) < 1e-6


def test_kabsch_rotation_invariant():
    rng = np.random.RandomState(1)
    P = rng.randn(15, 3)
    # Random rotation
    A = rng.randn(3, 3)
    Rng, _ = np.linalg.qr(A)
    if np.linalg.det(Rng) < 0:
        Rng[:, 0] = -Rng[:, 0]
    Q = P @ Rng.T + np.array([1.0, 2.0, 3.0])
    R, t = kabsch_align(P, Q)
    aligned = P @ R.T + t
    assert rmsd(aligned, Q) < 1e-6


def test_clash_count_one_pair():
    binder_atoms = np.array([[0, 0, 1.5]], dtype=np.float64)
    target_atoms = np.array([[0, 0, 0]], dtype=np.float64)
    assert inter_chain_clash_count(binder_atoms, target_atoms, cutoff=2.0) == 1


def test_clash_count_zero_far():
    binder_atoms = np.array([[0, 0, 100]], dtype=np.float64)
    target_atoms = np.array([[0, 0, 0]], dtype=np.float64)
    assert inter_chain_clash_count(binder_atoms, target_atoms, cutoff=2.0) == 0


def test_contacts_5A():
    binder_ca = np.array([[0, 0, 0], [0, 0, 3], [0, 0, 100]], dtype=np.float64)
    target_ca = np.array([[0, 0, 1], [0, 0, 4]], dtype=np.float64)
    assert n_contacts_5A(binder_ca, target_ca) == 2


def test_radius_of_gyration_unit_sphere():
    pts = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float64,
    )
    assert radius_of_gyration(pts) == pytest.approx(1.0, abs=0.01)


def test_binder_scrmsd_identical_complex(tmp_path):
    """Real BCv1 PDB: pred and ref are the same file → both RMSDs ≈ 0."""
    src = Path("/mnt/shared-storage-user/liyanchao/af3/BindCraft/outputs/batch_v1/PDL1/Accepted/PDL1_l85_s131547_mpnn1_model2.pdb")
    if not src.exists():
        pytest.skip("BCv1 PDL1 fixture not available")
    p = tmp_path / "fixture.pdb"
    shutil.copy(src, p)
    sc, fold = binder_scRMSD(str(p), str(p), ["A"], ["A"], "B", "B")
    assert sc < 1e-6
    assert fold < 1e-6
