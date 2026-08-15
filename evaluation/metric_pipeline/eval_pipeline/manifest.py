"""Manifest CSV schema (spec §4)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

REQUIRED_COLS = [
    "method",
    "case",
    "design_id",
    "target_pdb_path",
    "target_chain",
    "target_seq",
    "binder_seq",
    "designed_complex_pdb_path",
    "target_chain_in_complex",
    "binder_chain_in_complex",
]
DENOMINATOR_COLS = [
    "n_generated_trajectories",
    "n_mpnn_attempted",
    "n_accepted_by_method",
    "candidate_rank",
    "source_filter_status",
    "verifier_calls",
    "design_gpu_seconds",
]
OPTIONAL_COLS = [
    "binder_length",
    "target_chain_roles",
    "target_hotspot_concat_indices",
    "notes",
]


@dataclass
class ManifestRow:
    method: str
    case: str
    design_id: str
    target_pdb_path: str
    target_chain: str
    target_seq: str
    binder_seq: str
    designed_complex_pdb_path: str
    target_chain_in_complex: str
    binder_chain_in_complex: str
    n_generated_trajectories: int = -1
    n_mpnn_attempted: int = -1
    n_accepted_by_method: int = -1
    candidate_rank: int = -1
    source_filter_status: str = "unknown"
    verifier_calls: int = -1
    design_gpu_seconds: float = float("nan")
    target_chain_roles: str = ""
    target_hotspot_concat_indices: str = ""
    notes: str = ""

    @property
    def binder_length(self) -> int:
        return len(self.binder_seq)


def validate_manifest(df: pd.DataFrame) -> None:
    missing_required = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_required:
        raise ValueError(f"manifest missing required columns: {missing_required}")
    missing_denom = [c for c in DENOMINATOR_COLS if c not in df.columns]
    if missing_denom:
        raise ValueError(
            f"manifest missing denominator columns (use -1/NaN sentinels if unknown): {missing_denom}"
        )


def load_manifest(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={
            "target_chain": str,
            "target_chain_in_complex": str,
            "binder_chain_in_complex": str,
        },
    )
    validate_manifest(df)
    return df


def write_manifest(rows: list, path: str) -> None:
    """Write manifest CSV. Empty rows → header-only CSV (v1.1 reviewer P0 #3)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pd.DataFrame([asdict(r) for r in rows])
    else:
        all_cols = REQUIRED_COLS + DENOMINATOR_COLS + OPTIONAL_COLS
        df = pd.DataFrame(columns=all_cols)
    df.to_csv(path, index=False)
