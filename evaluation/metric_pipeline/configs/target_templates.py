"""Target PDB template paths — method-agnostic, shared by all method adapters.

Each method's adapter (bcv1_adapter, future rfdiff_adapter / pxdesign_adapter / bcv2_adapter)
imports this module to find the canonical target structure for a given case. Per-method
binder-chain conventions (which chain in *method output* is the binder) stay in the adapter
itself, since they vary by method.

The constants here are about *source data* (yanjie's pdb_template + tcr_case PDBs) and are
identical across all methods.
"""
from __future__ import annotations

from pathlib import Path

DATA0429_ROOT = Path(
    "/mnt/shared-storage-user/liyanchao/af3/BCv2/docs/BCv2-meeting-0428/data0429"
)

# v1.1 reviewer P0 #2: explicit per-case template path.
# yanjie templates have inconsistent naming (PD-L1.pdb hyphen; TCR in tcr_case/<id>.pdb).
TARGET_TEMPLATE_MAP: dict[str, Path] = {
    "BetV1":    DATA0429_ROOT / "bc_v1_benchmark_data/pdb_template/BetV1.pdb",
    "DerF21":   DATA0429_ROOT / "bc_v1_benchmark_data/pdb_template/DerF21.pdb",
    "DerF7":    DATA0429_ROOT / "bc_v1_benchmark_data/pdb_template/DerF7.pdb",
    "HER2":     DATA0429_ROOT / "bc_v1_benchmark_data/pdb_template/HER2.pdb",
    "IFNAR2":   DATA0429_ROOT / "bc_v1_benchmark_data/pdb_template/IFNAR2.pdb",
    "PD1":      DATA0429_ROOT / "bc_v1_benchmark_data/pdb_template/PD1.pdb",
    "PDL1":     DATA0429_ROOT / "bc_v1_benchmark_data/pdb_template/PD-L1.pdb",  # ⚠ hyphen
    "SpCas9":   DATA0429_ROOT / "bc_v1_benchmark_data/pdb_template/SpCas9.pdb",
    "TCR_1QRN": DATA0429_ROOT / "tcr_case/1QRN.pdb",
    "TCR_3QDJ": DATA0429_ROOT / "tcr_case/3QDJ.pdb",
    "TCR_5BS0": DATA0429_ROOT / "tcr_case/5BS0.pdb",
    "TCR_6BJ8": DATA0429_ROOT / "tcr_case/6BJ8.pdb",
    "TCR_9C3E": DATA0429_ROOT / "tcr_case/9C3E.pdb",
}

# Target chains in TARGET_TEMPLATE_MAP[case].pdb (which chain ID(s) hold the target).
# Method-agnostic: this is a property of the source PDB.
# Multi-chain TCR cases use both αβ chains; pMHC / β2m / EGFP-fusion chains are excluded.
TARGET_CHAINS_IN_TEMPLATE: dict[str, list[str]] = {
    "BetV1":    ["A"],
    "DerF21":   ["A"],
    "DerF7":    ["A"],
    "HER2":     ["A"],
    "IFNAR2":   ["A"],
    "PD1":      ["A"],
    "PDL1":     ["A"],
    "SpCas9":   ["A"],
    "TCR_1QRN": ["D", "E"],
    "TCR_3QDJ": ["D", "E"],
    "TCR_5BS0": ["D", "E"],
    "TCR_6BJ8": ["D", "H"],
    "TCR_9C3E": ["A", "B"],
}

EXPECTED_CASES: frozenset[str] = frozenset(TARGET_TEMPLATE_MAP.keys())
assert set(TARGET_CHAINS_IN_TEMPLATE.keys()) == EXPECTED_CASES, (
    "TARGET_TEMPLATE_MAP and TARGET_CHAINS_IN_TEMPLATE keys must match"
)
