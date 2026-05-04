"""BCv1 batch_v1 outputs → manifest CSV (spec §4.4)."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Allow `python -m adapters.bcv1_adapter` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_pipeline.manifest import ManifestRow, write_manifest  # noqa: E402
from vendor.proteina_complexa.pdb_chain_utils import (  # noqa: E402
    extract_chains_to_pdb,
    get_chain_sequence,
    list_chain_ids,
)

# ─── Case-specific config ───────────────────────────────────────────
# Spec §4.4: target chain map (yanjie pdb_template) + binder chain in BCv1 Accepted PDB.
# For TCR cases binder chain "auto" means "first chain not in target".
BCV1_CHAIN_MAP: dict[str, dict[str, list[str] | str]] = {
    "BetV1":    {"target": ["A"], "binder": "B"},
    "DerF21":   {"target": ["A"], "binder": "B"},
    "DerF7":    {"target": ["A"], "binder": "B"},
    "HER2":     {"target": ["A"], "binder": "B"},
    "IFNAR2":   {"target": ["A"], "binder": "B"},
    "PD1":      {"target": ["A"], "binder": "B"},
    "PDL1":     {"target": ["A"], "binder": "B"},
    "SpCas9":   {"target": ["A"], "binder": "B"},
    "TCR_1QRN": {"target": ["D", "E"], "binder": "auto"},
    "TCR_3QDJ": {"target": ["D", "E"], "binder": "auto"},
    "TCR_5BS0": {"target": ["D", "E"], "binder": "auto"},
    "TCR_6BJ8": {"target": ["D", "H"], "binder": "auto"},
    "TCR_9C3E": {"target": ["A", "B"], "binder": "auto"},
}

# Spec §3.4 sub-run → paper case mapping. Sub-run JSONs (`batch_v1_<sub>.json`) were
# split because paper SI Table 1 lists OR-semantics hotspots that BindCraft requires
# as separate runs. Every sub-run shares the parent paper case's TARGET_TEMPLATE_MAP /
# BCV1_CHAIN_MAP entry; only the design output dir name (and hotspot config inside
# the batch JSON) differ.
#
# The 9 paper cases without splits map to themselves implicitly via
# `resolve_paper_case()` below.
BCV1_SUB_RUN_TO_PAPER_CASE: dict[str, str] = {
    "BetV1_h23":     "BetV1",
    "BetV1_h24":     "BetV1",
    "DerF7_h132":    "DerF7",
    "DerF7_h178":    "DerF7",
    "DerF21_h17":    "DerF21",
    "DerF21_h35":    "DerF21",
    "DerF21_h70":    "DerF21",
    "DerF21_h88":    "DerF21",
    "DerF21_h99":    "DerF21",
    "HER2_h148":     "HER2",
    "HER2_h143_148": "HER2",
}


def resolve_paper_case(case: str) -> str:
    """Map sub-run name → paper case. Identity for non-split paper cases.

    Examples:
      resolve_paper_case("BetV1_h23")  → "BetV1"
      resolve_paper_case("PD1")        → "PD1"
      resolve_paper_case("TCR_1QRN")   → "TCR_1QRN"
    """
    return BCV1_SUB_RUN_TO_PAPER_CASE.get(case, case)


def is_sub_run(case: str) -> bool:
    """True iff `case` is a split sub-run (not a paper case)."""
    return case in BCV1_SUB_RUN_TO_PAPER_CASE

# Method-agnostic config (target PDB paths + target chains in source) lives in
# configs/target_templates.py so future adapters (rfdiff/pxdesign/bcv2) can share
# the same canonical target structures. Re-exported here for backward compat with
# existing tests that import via `from adapters.bcv1_adapter import TARGET_TEMPLATE_MAP`.
from configs.target_templates import (  # noqa: E402, F401
    DATA0429_ROOT,
    TARGET_CHAINS_IN_TEMPLATE,
    TARGET_TEMPLATE_MAP,
)

# BCv1 Accepted PDB always consolidates target to chain "A" regardless of multi-chain template.
BCV1_TARGET_CHAIN_IN_COMPLEX = "A"

# ─── Helpers ────────────────────────────────────────────────────────
_TIME_RE = re.compile(r"(\d+)\s*hours?,\s*(\d+)\s*minutes?,\s*(\d+)\s*seconds?")
_MODEL_SUFFIX_RE = re.compile(r"_model\d+$")


def parse_design_time_to_seconds(s) -> float:
    if s is None or pd.isna(s) or str(s).strip() == "":
        return float("nan")
    m = _TIME_RE.search(str(s))
    if not m:
        return float("nan")
    h, mn, sec = (int(x) for x in m.groups())
    return float(h * 3600 + mn * 60 + sec)


def _design_id_to_design_key(design_id: str) -> str:
    """Strip _model\\d+$ suffix to match 'Design' col in final_design_stats.csv.

    PDB stem 'PDL1_l155_s752946_mpnn5_model1' → 'PDL1_l155_s752946_mpnn5'.
    """
    return _MODEL_SUFFIX_RE.sub("", design_id)


def _resolve_binder_chain(complex_pdb: Path, target_chains: list[str]) -> str:
    chains = list_chain_ids(str(complex_pdb))
    for c in chains:
        if c not in target_chains:
            return c
    raise ValueError(
        f"No binder chain found in {complex_pdb}; chains={chains}, target={target_chains}"
    )


# ─── Main ────────────────────────────────────────────────────────────
def build_bcv1_manifest_for_case(
    batch_dir: Path,
    case: str,
    target_template: Optional[Path] = None,
    target_only_dir: Path = Path("./target_only_pdbs"),
    out_csv: Path = Path("./manifest.csv"),
) -> int:
    """Returns count of rows written. Empty Accepted/ → header-only CSV (returns 0).

    `case` may be either a paper case (e.g. "PDL1", "TCR_1QRN") or a split sub-run
    (e.g. "BetV1_h23", "DerF7_h132"). For sub-runs the function reads from
    `<batch_dir>/<sub_run>/...` but reuses the parent paper case's chain config
    and target template. The manifest's `case` column keeps the sub-run name so
    downstream aggregation (spec §3.4) can apply per-paper-case rollup rules
    (OR-merge for DerF7/DerF21/HER2; pick yield-higher sub-run for BetV1).
    """
    case_dir = Path(batch_dir) / case
    accepted_dir = case_dir / "Accepted"
    paper_case = resolve_paper_case(case)
    if paper_case not in BCV1_CHAIN_MAP:
        raise KeyError(
            f"Unknown case '{case}' (resolved paper_case='{paper_case}' not in BCV1_CHAIN_MAP). "
            f"Allowed paper cases: {sorted(BCV1_CHAIN_MAP)}. "
            f"Allowed sub-runs: {sorted(BCV1_SUB_RUN_TO_PAPER_CASE)}."
        )
    if target_template is None:
        target_template = TARGET_TEMPLATE_MAP[paper_case]
    target_template = Path(target_template)
    if not target_template.exists():
        raise FileNotFoundError(f"Target template not found: {target_template}")

    if not accepted_dir.exists():
        write_manifest([], str(out_csv))
        print(f"[{case}] no Accepted/ dir; wrote header-only manifest to {out_csv}")
        return 0

    spec = BCV1_CHAIN_MAP[paper_case]
    target_chains: list[str] = spec["target"]  # type: ignore
    binder_spec: str = spec["binder"]  # type: ignore

    # Extract target-only PDB once. Filename keyed by paper_case so multiple
    # sub-runs of the same paper case (BetV1_h23 + BetV1_h24) reuse the same
    # extracted target PDB — saves disk + keeps target_pdb_path stable per case.
    Path(target_only_dir).mkdir(parents=True, exist_ok=True)
    target_only_pdb = Path(target_only_dir) / f"{paper_case}.pdb"
    extract_chains_to_pdb(str(target_template), target_chains, str(target_only_pdb))
    target_seq = ":".join(get_chain_sequence(str(target_only_pdb), c) for c in target_chains)

    # Denominator columns
    traj_csv = case_dir / "trajectory_stats.csv"
    mpnn_csv = case_dir / "mpnn_design_stats.csv"
    final_csv = case_dir / "final_design_stats.csv"
    n_traj = len(pd.read_csv(traj_csv)) if traj_csv.exists() else -1
    n_mpnn = len(pd.read_csv(mpnn_csv)) if mpnn_csv.exists() else -1

    design_time_map: dict[str, float] = {}
    if final_csv.exists():
        fdf = pd.read_csv(final_csv)
        if "Design" in fdf.columns and "DesignTime" in fdf.columns:
            for _, r in fdf.iterrows():
                design_time_map[str(r["Design"])] = parse_design_time_to_seconds(r["DesignTime"])

    # Ranked/<n>_<design_id>.pdb gives candidate_rank
    rank_map: dict[str, int] = {}
    ranked_dir = accepted_dir / "Ranked"
    if ranked_dir.exists():
        for p in ranked_dir.glob("*.pdb"):
            m = re.match(r"^(\d+)_(.+)$", p.stem)
            if m:
                rank_map[m.group(2)] = int(m.group(1))

    # Top-level Accepted/*.pdb only (do NOT recurse into Ranked/)
    accepted_pdbs = sorted(p for p in accepted_dir.glob("*.pdb") if p.is_file())
    n_accepted = len(accepted_pdbs)

    rows: list[ManifestRow] = []
    for pdb_path in accepted_pdbs:
        design_id = pdb_path.stem
        design_key = _design_id_to_design_key(design_id)

        complex_chains = list_chain_ids(str(pdb_path))
        if binder_spec == "auto":
            binder_chain = _resolve_binder_chain(pdb_path, [BCV1_TARGET_CHAIN_IN_COMPLEX])
        else:
            binder_chain = binder_spec
        binder_seq = get_chain_sequence(str(pdb_path), binder_chain)

        if BCV1_TARGET_CHAIN_IN_COMPLEX not in complex_chains:
            raise ValueError(
                f"BCv1 convention violation: target chain "
                f"'{BCV1_TARGET_CHAIN_IN_COMPLEX}' not in {pdb_path} "
                f"(chains={complex_chains})"
            )

        # Stash paper_case in `notes` so report.py / future aggregator can roll up
        # sub-runs without touching ManifestRow schema. Paper cases (no split) get
        # empty notes — keep the column noise-free.
        notes_str = f"paper_case={paper_case}" if is_sub_run(case) else ""
        rows.append(
            ManifestRow(
                method="BCv1",
                case=case,
                design_id=design_id,
                target_pdb_path=str(target_only_pdb),
                target_chain="+".join(target_chains),
                target_seq=target_seq,
                binder_seq=binder_seq,
                designed_complex_pdb_path=str(pdb_path),
                target_chain_in_complex=BCV1_TARGET_CHAIN_IN_COMPLEX,
                binder_chain_in_complex=binder_chain,
                n_generated_trajectories=n_traj,
                n_mpnn_attempted=n_mpnn,
                n_accepted_by_method=n_accepted,
                candidate_rank=rank_map.get(design_id, -1),
                source_filter_status="accepted_after_default_filters",
                verifier_calls=(n_mpnn * 5) if n_mpnn > 0 else -1,
                design_gpu_seconds=design_time_map.get(design_key, float("nan")),
                notes=notes_str,
            )
        )
    write_manifest(rows, str(out_csv))
    print(f"[{case}] wrote {len(rows)} rows to {out_csv}")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--target-template", default=None,
                    help="Override TARGET_TEMPLATE_MAP[case] with explicit path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-only-dir", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    target_only_dir = (
        Path(args.target_only_dir) if args.target_only_dir else (out.parent / "target_only_pdbs")
    )
    target_template = Path(args.target_template) if args.target_template else None
    build_bcv1_manifest_for_case(
        batch_dir=Path(args.batch_dir),
        case=args.case,
        target_template=target_template,
        target_only_dir=target_only_dir,
        out_csv=out,
    )


if __name__ == "__main__":
    main()
