"""Metrics derivation (spec §6). All functions are pure."""
from __future__ import annotations

from typing import Optional

import numpy as np


# ─── pLDDT family ──────────────────────────────────────────────────
def complex_pLDDT(plddt_clean: np.ndarray, clean_segments: list[dict]) -> float:
    return float(plddt_clean.mean())


def _binder_seg(clean_segments):
    return next(s for s in clean_segments if s["kind"] == "binder")


def _target_clean_idx(clean_segments) -> np.ndarray:
    parts = [
        np.arange(s["clean_start"], s["clean_end"])
        for s in clean_segments
        if s["kind"].startswith("target_chain_")
    ]
    return np.concatenate(parts) if parts else np.array([], dtype=int)


def binder_pLDDT(plddt_clean: np.ndarray, clean_segments: list[dict]) -> float:
    s = _binder_seg(clean_segments)
    return float(plddt_clean[s["clean_start"]:s["clean_end"]].mean())


def target_pLDDT(plddt_clean: np.ndarray, clean_segments: list[dict]) -> float:
    idx = _target_clean_idx(clean_segments)
    if idx.size == 0:
        return float("nan")
    return float(plddt_clean[idx].mean())


def interface_pLDDT(
    plddt_clean: np.ndarray,
    clean_segments: list[dict],
    target_ca: np.ndarray,
    binder_ca: np.ndarray,
    cutoff: float = 5.0,
) -> float:
    """Mean pLDDT over binder residues whose Cα is within `cutoff` Å of any target Cα."""
    if target_ca.size == 0 or binder_ca.size == 0:
        return float("nan")
    dists = np.linalg.norm(binder_ca[:, None, :] - target_ca[None, :, :], axis=-1)
    is_interface = dists.min(axis=1) < cutoff
    if not is_interface.any():
        return float("nan")
    bs = _binder_seg(clean_segments)
    binder_plddt = plddt_clean[bs["clean_start"]:bs["clean_end"]]
    if binder_plddt.shape[0] != is_interface.shape[0]:
        # binder_ca may have fewer entries than binder_plddt if some residues lack CA
        # → conservative: only use the first N entries
        n = min(binder_plddt.shape[0], is_interface.shape[0])
        binder_plddt = binder_plddt[:n]
        is_interface = is_interface[:n]
    return float(binder_plddt[is_interface].mean())


# ─── PAE family ────────────────────────────────────────────────────
def _assert_pae_unit(pae: np.ndarray) -> None:
    """Sanity check that PAE is in raw Å, not double-multiplied by 31.

    HF ESMFold returns raw Å directly (max ~31 Å, pinned by max_bin=31).
    AF3 returns raw Å clipped to ~99.9.
    Proteina-Complexa stores normalized [0,1] values requiring × 31 for raw Å.
    A common bug is double-multiplying, yielding values up to ~961 Å.
    """
    if pae.size == 0:
        return
    pae_max = float(np.nanmax(pae))
    if pae_max > 100.5:
        raise AssertionError(
            f"PAE max {pae_max:.2f} Å exceeds tolerated range (max 100.5 Å). "
            f"Likely double-multiplied by 31 or unit error."
        )
    pae_min = float(np.nanmin(pae))
    if pae_min < 0:
        raise AssertionError(f"PAE min {pae_min:.2f} < 0; PAE must be non-negative.")


def mean_pAE(pae_clean: np.ndarray, clean_segments: list[dict]) -> float:
    _assert_pae_unit(pae_clean)
    return float(pae_clean.mean())


def _cross_block(pae_clean, clean_segments):
    _assert_pae_unit(pae_clean)
    bs = _binder_seg(clean_segments)
    binder_idx = np.arange(bs["clean_start"], bs["clean_end"])
    target_idx = _target_clean_idx(clean_segments)
    bt = pae_clean[np.ix_(binder_idx, target_idx)]
    tb = pae_clean[np.ix_(target_idx, binder_idx)]
    return bt, tb


def ipAE(pae_clean: np.ndarray, clean_segments: list[dict]) -> float:
    bt, tb = _cross_block(pae_clean, clean_segments)
    if bt.size == 0:
        return float("nan")
    return float(0.5 * (bt.mean() + tb.mean()))


def min_ipAE(pae_clean: np.ndarray, clean_segments: list[dict]) -> float:
    bt, _ = _cross_block(pae_clean, clean_segments)
    if bt.size == 0:
        return float("nan")
    return float(bt.min())


def i_pTM_approx_from_PAE(
    pae_clean: np.ndarray,
    clean_segments: list[dict],
    max_seq_len: Optional[int] = None,
) -> float:
    """PAE-based approximation of i_pTM. NOT native AF2-Multimer i_pTM."""
    bt, _ = _cross_block(pae_clean, clean_segments)
    if bt.size == 0:
        return float("nan")
    n = max_seq_len if max_seq_len else pae_clean.shape[0]
    d0 = 1.24 * (max(n - 15, 1)) ** (1 / 3) - 1.8
    tm_per_pair = 1.0 / (1.0 + (bt / d0) ** 2)
    return float(tm_per_pair.max(axis=1).mean())


# ─── Structural family ─────────────────────────────────────────────
def inter_chain_clash_count(
    binder_atoms: np.ndarray, target_atoms: np.ndarray, cutoff: float = 2.0
) -> int:
    """Heavy-atom pairs <cutoff Å. Caller passes heavy-atom coords (no H)."""
    if binder_atoms.size == 0 or target_atoms.size == 0:
        return 0
    d = np.linalg.norm(binder_atoms[:, None, :] - target_atoms[None, :, :], axis=-1)
    return int((d < cutoff).sum())


def n_contacts_5A(
    binder_ca: np.ndarray, target_ca: np.ndarray, cutoff: float = 5.0
) -> int:
    if binder_ca.size == 0 or target_ca.size == 0:
        return 0
    d = np.linalg.norm(binder_ca[:, None, :] - target_ca[None, :, :], axis=-1)
    return int((d.min(axis=1) < cutoff).sum())


def radius_of_gyration(ca_coords: np.ndarray) -> float:
    if ca_coords.size == 0:
        return float("nan")
    com = ca_coords.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((ca_coords - com) ** 2, axis=1))))


def binder_scRMSD(
    pred_pdb_path: str,
    ref_pdb_path: str,
    target_chains_pred: list[str],
    target_chains_ref: list[str],
    binder_chain_pred: str,
    binder_chain_ref: str,
) -> tuple[float, float]:
    """Returns (target_aligned_binder_rmsd, binder_fold_rmsd) — see spec §6.6."""
    from eval_pipeline.pdb_utils import kabsch_align
    from eval_pipeline.pdb_utils import rmsd as _rmsd
    from vendor.proteina_complexa.pdb_chain_utils import get_chain_ca_coords

    pred_t = np.concatenate(
        [get_chain_ca_coords(pred_pdb_path, c) for c in target_chains_pred]
    )
    ref_t = np.concatenate(
        [get_chain_ca_coords(ref_pdb_path, c) for c in target_chains_ref]
    )
    pred_b = get_chain_ca_coords(pred_pdb_path, binder_chain_pred)
    ref_b = get_chain_ca_coords(ref_pdb_path, binder_chain_ref)
    if pred_t.shape != ref_t.shape:
        raise ValueError(
            f"target Cα count mismatch: pred {pred_t.shape} vs ref {ref_t.shape}"
        )
    if pred_b.shape != ref_b.shape:
        raise ValueError(
            f"binder Cα count mismatch: pred {pred_b.shape} vs ref {ref_b.shape}"
        )
    R, t = kabsch_align(pred_t, ref_t)
    pred_b_aligned = pred_b @ R.T + t
    sc_target_aligned = _rmsd(pred_b_aligned, ref_b)
    R_b, t_b = kabsch_align(pred_b, ref_b)
    pred_b_only = pred_b @ R_b.T + t_b
    fold_rmsd = _rmsd(pred_b_only, ref_b)
    return sc_target_aligned, fold_rmsd
