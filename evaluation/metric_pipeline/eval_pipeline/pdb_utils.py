"""Light PDB / geometry helpers."""
from __future__ import annotations

import numpy as np


def kabsch_align(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find rotation R and translation t so that P @ R.T + t aligns onto Q.

    P, Q both shape (N, 3). Returns (R, t).
    """
    assert P.shape == Q.shape, f"shape mismatch {P.shape} vs {Q.shape}"
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = Q.mean(axis=0) - R @ P.mean(axis=0)
    return R, t


def rmsd(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1))))
