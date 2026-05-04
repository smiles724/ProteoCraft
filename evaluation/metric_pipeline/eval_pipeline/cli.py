"""End-to-end orchestration: manifest → per-design ESMFold/AF3 → metrics → long CSV."""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `python -m eval_pipeline.cli` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_pipeline import af3_runner  # noqa: E402
from eval_pipeline import metrics as M  # noqa: E402
from eval_pipeline.esmfold_runner import (  # noqa: E402
    build_esmfold_input,
    dump_design_artifacts,
    load_esmfold,
    run_esmfold,
)
from eval_pipeline.manifest import load_manifest  # noqa: E402
from eval_pipeline.tier_gating import compute_tier_flags  # noqa: E402
from vendor.proteina_complexa.pdb_chain_utils import get_chain_ca_coords  # noqa: E402

CSV_COLS_BASE = [
    # metadata
    "method", "case", "design_id", "target_chain", "binder_length",
    "designed_complex_pdb_path", "esmfold_pdb_path",
    # denominator
    "n_generated_trajectories", "n_mpnn_attempted", "n_accepted_by_method",
    "candidate_rank", "source_filter_status", "verifier_calls", "design_gpu_seconds",
    # Tier 0 raw
    "complex_pLDDT", "binder_pLDDT", "target_pLDDT", "interface_pLDDT",
    "mean_pAE", "ipAE", "min_ipAE", "i_pTM_approx_from_PAE", "ptm",
    "binder_scRMSD", "binder_fold_RMSD",
    "inter_chain_clash_count", "n_contacts_5A", "radius_of_gyration_binder",
    # Tier flags
    "tier1_plddt80_pass", "tier1_plddt90_pass", "tier1_pass",
    "tier2_complexa_pass", "tier2_strict_physical_pass", "tier2_pass",
    # Tier 3 (stub, NaN until later)
    "n_diverse_clusters", "tm_score_to_pdb_nearest", "ipsae_score",
    # Compute
    "esmfold_runtime_s",
    # Errors
    "error_msg",
]

CSV_COLS_NEW = [
    "evaluator",
    "paper_case",
    "eval_n_calls",
    "eval_n_successes",
    "eval_runtime_s",
    "af3_ranking_score",
    "af3_iptm",
    "af3_ptm_native",
    "af3_has_clash",
    "af3_fraction_disordered",
]

CSV_COLS = CSV_COLS_BASE + CSV_COLS_NEW

_TIER_FLAG_COLS = [
    "tier1_plddt80_pass", "tier1_plddt90_pass", "tier1_pass",
    "tier2_complexa_pass", "tier2_strict_physical_pass", "tier2_pass",
]


def _empty_row() -> dict:
    """v1.1 reviewer P1 #5: tier flags default False, error_msg empty (not NaN)."""
    out = {c: float("nan") for c in CSV_COLS}
    for c in _TIER_FLAG_COLS:
        out[c] = False
    out["error_msg"] = ""
    return out


def _align_eval_csv_rows(rows: list[dict]) -> pd.DataFrame:
    """Return rows aligned to the current eval CSV schema."""
    if not rows:
        return pd.DataFrame(columns=CSV_COLS)
    df = pd.DataFrame(rows)
    for col in CSV_COLS:
        if col not in df.columns:
            df[col] = float("nan")
    df["error_msg"] = df["error_msg"].where(~df["error_msg"].isna(), "")
    return df[CSV_COLS]


def _read_existing_eval_rows(output_csv: Path) -> list[dict]:
    return pd.read_csv(
        output_csv,
        keep_default_na=False,
        na_values=[],
    ).to_dict("records")


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--evaluator", choices=["esmfold", "af3"], default="esmfold")
    ap.add_argument("--paper-case", default=None)
    ap.add_argument("--sub-run", default=None)
    ap.add_argument("--af3-python", default=os.environ.get("AF3_PYTHON"))
    ap.add_argument("--af3-run-script", default=os.environ.get("AF3_RUN_SCRIPT"))
    ap.add_argument("--af3-model-dir", default=os.environ.get("AF3_MODEL_DIR"))
    ap.add_argument("--af3-db-dir", default=os.environ.get("AF3_DB_DIR"))
    ap.add_argument("--af3-msa-root", default=None)
    ap.add_argument("--af3-staging-root", default=None)
    ap.add_argument("--af3-output-root", default=None)
    ap.add_argument("--af3-timeout-per-sub-run", type=int, default=43200)
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return ap


def _design_dir_for_row(out_root: Path, row) -> Path:
    return Path(out_root) / "per_design" / row["method"] / row["case"] / row["design_id"]


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _row_get(row, key: str, default):
    value = row.get(key, default) if hasattr(row, "get") else default
    return default if _is_missing(value) else value


def _resolve_output_csv(args, sub_run: str | None) -> Path:
    out_dir = Path(args.out_dir) / "eval"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        print(f"WARNING: could not create eval output directory {out_dir}: {exc}")
    paper_case = args.paper_case or "unknown"
    sub_run_name = sub_run or paper_case
    return out_dir / f"{paper_case}__{sub_run_name}__{args.evaluator}.csv"


def _heavy_atoms_by_chain(pdb_path: str, chains: list[str]) -> np.ndarray:
    """Concat heavy-atom (non-H) coords for all residues across given chains."""
    from Bio.PDB import PDBParser
    structure = PDBParser(QUIET=True).get_structure("e", pdb_path)
    coords = []
    for ch in structure[0]:
        if ch.id not in chains:
            continue
        for res in ch:
            if res.id[0] != " ":
                continue
            for atom in res:
                if atom.element == "H":
                    continue
                coords.append(atom.coord.copy())
    return np.array(coords, dtype=np.float64) if coords else np.zeros((0, 3))


def _compute_eval_n_successes(out: dict) -> int:
    """One evaluator call succeeded if core metrics exist and no error was recorded."""
    error_msg = out.get("error_msg", "")
    if pd.notna(error_msg) and str(error_msg) != "":
        return 0
    key_cols = ["binder_pLDDT", "ipAE", "binder_scRMSD"]
    return int(all(pd.notna(out.get(c)) for c in key_cols))


def evaluate_one(
    row: pd.Series,
    out_root: Path,
    tokenizer=None,
    model=None,
    device: str = "cuda:0",
    *,
    af3_result: af3_runner.AF3Result | None = None,
    evaluator: str = "esmfold",
) -> dict:
    out: dict = _empty_row()
    out["evaluator"] = evaluator
    out["eval_n_calls"] = 1
    out.update({
        "method": row["method"],
        "case": row["case"],
        "paper_case": _row_get(row, "paper_case", row["case"]),
        "design_id": row["design_id"],
        "target_chain": row["target_chain"],
        "binder_length": len(row["binder_seq"]),
        "designed_complex_pdb_path": row["designed_complex_pdb_path"],
        "n_generated_trajectories": row.get("n_generated_trajectories", -1),
        "n_mpnn_attempted": row.get("n_mpnn_attempted", -1),
        "n_accepted_by_method": row.get("n_accepted_by_method", -1),
        "candidate_rank": row.get("candidate_rank", -1),
        "source_filter_status": row.get("source_filter_status", "unknown"),
        "verifier_calls": row.get("verifier_calls", -1),
        "design_gpu_seconds": row.get("design_gpu_seconds", float("nan")),
    })
    design_dir = _design_dir_for_row(out_root, row)

    try:
        if af3_result is None:
            # 1. Build input
            esm_seq, segs = build_esmfold_input(row["target_seq"], row["binder_seq"])
            # 2. ESMFold inference
            result = run_esmfold(esm_seq, tokenizer, model, device)
            # 3. Dump artifacts (rechain happens inside)
            meta = dump_design_artifacts(design_dir, esm_seq, segs, result)
            out["esmfold_pdb_path"] = str(design_dir / "esmfold.pdb")
            out["ptm"] = result.ptm
            out["esmfold_runtime_s"] = result.runtime_s
            out["eval_runtime_s"] = result.runtime_s

            plddt_clean = np.load(design_dir / "plddt.npy")
            pae_clean = np.load(design_dir / "pae.npy")
            clean_segs = meta["clean_segments"]
        else:
            out["esmfold_pdb_path"] = af3_result.pdb_path
            out["eval_runtime_s"] = af3_result.runtime_s
            plddt_clean = af3_result.plddt
            pae_clean = af3_result.pae
            clean_segs = af3_result.clean_segments
            out.update({
                "af3_ranking_score": af3_result.native.get("ranking_score", float("nan")),
                "af3_iptm": af3_result.native.get("iptm", float("nan")),
                "af3_ptm_native": af3_result.native.get("ptm", float("nan")),
                "af3_has_clash": af3_result.native.get("has_clash", float("nan")),
                "af3_fraction_disordered": af3_result.native.get(
                    "fraction_disordered", float("nan")
                ),
            })

        # 4. Tier 0 metrics
        out["complex_pLDDT"] = M.complex_pLDDT(plddt_clean, clean_segs)
        out["binder_pLDDT"] = M.binder_pLDDT(plddt_clean, clean_segs)
        out["target_pLDDT"] = M.target_pLDDT(plddt_clean, clean_segs)
        out["mean_pAE"] = M.mean_pAE(pae_clean, clean_segs)
        out["ipAE"] = M.ipAE(pae_clean, clean_segs)
        out["min_ipAE"] = M.min_ipAE(pae_clean, clean_segs)
        out["i_pTM_approx_from_PAE"] = M.i_pTM_approx_from_PAE(pae_clean, clean_segs)

        # v1.1 reviewer P0 #1: pred chain IDs come from meta clean_segments[i]["pdb_chain_id"]
        pred_t_chains = [
            s["pdb_chain_id"] for s in clean_segs if s["kind"].startswith("target_chain_")
        ]
        pred_b_chain = next(s["pdb_chain_id"] for s in clean_segs if s["kind"] == "binder")

        # Cα-based metrics from rechained ESMFold PDB
        target_ca_pred = np.concatenate(
            [get_chain_ca_coords(out["esmfold_pdb_path"], c) for c in pred_t_chains]
        ) if pred_t_chains else np.zeros((0, 3))
        binder_ca_pred = get_chain_ca_coords(out["esmfold_pdb_path"], pred_b_chain)

        out["interface_pLDDT"] = M.interface_pLDDT(
            plddt_clean, clean_segs, target_ca_pred, binder_ca_pred
        )
        out["n_contacts_5A"] = M.n_contacts_5A(binder_ca_pred, target_ca_pred)
        out["radius_of_gyration_binder"] = M.radius_of_gyration(binder_ca_pred)

        # Heavy-atom clash count
        binder_atoms = _heavy_atoms_by_chain(out["esmfold_pdb_path"], [pred_b_chain])
        target_atoms = _heavy_atoms_by_chain(out["esmfold_pdb_path"], pred_t_chains)
        out["inter_chain_clash_count"] = M.inter_chain_clash_count(binder_atoms, target_atoms)

        # scRMSD vs designed_complex_pdb_path
        # v1.1 reviewer P1 #6: ref target chains from manifest's target_chain_in_complex
        ref_target_chains = str(row["target_chain_in_complex"]).split("+")
        sc, fold = M.binder_scRMSD(
            pred_pdb_path=out["esmfold_pdb_path"],
            ref_pdb_path=row["designed_complex_pdb_path"],
            target_chains_pred=pred_t_chains,
            target_chains_ref=ref_target_chains,
            binder_chain_pred=pred_b_chain,
            binder_chain_ref=row["binder_chain_in_complex"],
        )
        out["binder_scRMSD"] = sc
        out["binder_fold_RMSD"] = fold

        # 5. Tier flags (overwrites the False defaults)
        flags = compute_tier_flags(out, evaluator=evaluator)
        out.update(flags)

    except Exception as e:
        out["error_msg"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    out["eval_n_successes"] = _compute_eval_n_successes(out)
    return out


def _collect_af3_env(args) -> dict:
    env = {
        "AF3_PYTHON": args.af3_python,
        "AF3_RUN_SCRIPT": args.af3_run_script,
        "AF3_MODEL_DIR": args.af3_model_dir,
        "AF3_DB_DIR": args.af3_db_dir,
    }
    missing = [k for k, v in env.items() if not v]
    if missing:
        raise SystemExit(
            f"--evaluator af3 requires {', '.join(missing)} (or env vars)"
        )
    if os.environ.get("AF3_MOCK") == "1":
        return env
    for key, must_be in [
        ("AF3_PYTHON", "file"),
        ("AF3_RUN_SCRIPT", "file"),
        ("AF3_MODEL_DIR", "dir"),
        ("AF3_DB_DIR", "dir"),
    ]:
        path = Path(env[key])
        ok = path.is_file() if must_be == "file" else path.is_dir()
        if not ok:
            flag = key.lower().replace("_", "-")
            raise SystemExit(f"--{flag}={path} does not exist as {must_be}")
    return env


def _count_ca_in_chains(pdb_path: str, chain_ids: list[str]) -> int:
    from Bio.PDB import PDBParser

    structure = PDBParser(QUIET=True).get_structure("ref", pdb_path)
    model = structure[0]
    available = [chain.id for chain in model]
    total = 0
    for chain_id in chain_ids:
        if chain_id not in available:
            raise SystemExit(
                f"chain {chain_id!r} missing from {pdb_path} (have {available})"
            )
        chain = model[chain_id]
        total += sum(
            1
            for res in chain
            if res.id[0] == " "
            for atom in res
            if atom.get_name() == "CA"
        )
    return total


def _af3_preflight(args, manifest_df: pd.DataFrame) -> None:
    """Fail fast on AF3 runtime inputs that would otherwise waste a sub-run."""
    if len(manifest_df) == 0:
        return

    msa_root = Path(args.af3_msa_root or Path(args.out_dir) / "per_paper_case_msa")
    paper_case_col = "paper_case" if "paper_case" in manifest_df.columns else "case"

    for paper_case in sorted(set(manifest_df[paper_case_col].astype(str))):
        sample_row = manifest_df[manifest_df[paper_case_col].astype(str) == paper_case].iloc[0]
        target_seqs = str(sample_row["target_seq"]).split(":")
        expected_ids = [chr(ord("B") + i) for i in range(len(target_seqs))]
        case_dir = msa_root / paper_case
        for chain_id in expected_ids:
            a3m = case_dir / f"target_chain_{chain_id}.a3m"
            if not a3m.exists():
                raise SystemExit(
                    f"missing MSA: {a3m} "
                    f"(paper_case={paper_case}, expected chains={expected_ids})"
                )
            if a3m.stat().st_size == 0:
                raise SystemExit(
                    f"empty MSA: {a3m} (re-run prepare_af3_target_msa for {paper_case})"
                )

    for _, row in manifest_df.iterrows():
        target_seqs = str(row["target_seq"]).split(":")
        target_chains_ref = str(row.get("target_chain_in_complex", "A")).split("+")
        ref_n_ca = _count_ca_in_chains(row["designed_complex_pdb_path"], target_chains_ref)
        target_total_residues = sum(len(seq) for seq in target_seqs)
        if ref_n_ca != target_total_residues:
            raise SystemExit(
                f"CA count mismatch for {row['case']} {row['design_id']}: "
                f"ref chain(s) {target_chains_ref!r} have {ref_n_ca} CA, "
                f"target_seq sum has {target_total_residues} residues"
            )


def _build_failed_row(staged: dict, paper_case: str, sub_run: str, error: str) -> dict:
    out = _empty_row()
    row = staged.get("row", {}) if isinstance(staged, dict) else {}
    try:
        binder_length = len(row["binder_seq"]) if "binder_seq" in row else -1
    except Exception:
        binder_length = -1
    out.update({
        "evaluator": "af3",
        "method": _row_get(row, "method", "unknown"),
        "case": _row_get(row, "case", sub_run),
        "paper_case": _row_get(row, "paper_case", paper_case),
        "design_id": _row_get(row, "design_id", "_unknown"),
        "target_chain": _row_get(row, "target_chain", ""),
        "binder_length": binder_length,
        "designed_complex_pdb_path": _row_get(row, "designed_complex_pdb_path", ""),
        "n_generated_trajectories": _row_get(row, "n_generated_trajectories", -1),
        "n_mpnn_attempted": _row_get(row, "n_mpnn_attempted", -1),
        "n_accepted_by_method": _row_get(row, "n_accepted_by_method", -1),
        "candidate_rank": _row_get(row, "candidate_rank", -1),
        "source_filter_status": _row_get(row, "source_filter_status", "unknown"),
        "verifier_calls": _row_get(row, "verifier_calls", -1),
        "design_gpu_seconds": _row_get(row, "design_gpu_seconds", float("nan")),
        "eval_n_calls": 1,
        "eval_n_successes": 0,
        "eval_runtime_s": 0.0,
        "error_msg": error,
    })
    return out


def _run_esmfold_loop(df: pd.DataFrame, args, out_root: Path) -> None:
    output_csv = (
        _resolve_output_csv(args, sub_run=args.sub_run)
        if args.paper_case
        else Path(args.out_dir) / "eval_long.csv"
    )
    if args.skip_existing:
        before = len(df)
        df = df[
            df.apply(
                lambda r: not (_design_dir_for_row(out_root, r) / "esmfold.pdb").exists(),
                axis=1,
            )
        ]
        if len(df) < before:
            print(f"skip_existing: {before - len(df)} designs already have esmfold.pdb")
        if len(df) == 0:
            if output_csv.exists():
                print(f"all designs already evaluated; preserving existing {output_csv}")
                return
            raise SystemExit(
                f"all designs already have esmfold.pdb but no eval csv at {output_csv}; "
                "use --no-skip-existing to rebuild"
            )

    print(f"Loading ESMFold on {args.device}...")
    tok, mdl = load_esmfold(args.device)
    rows_out: list[dict] = (
        _read_existing_eval_rows(output_csv)
        if args.skip_existing and output_csv.exists()
        else []
    )
    for i, row in df.iterrows():
        row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        print(
            f"[{i + 1}/{len(df)}] "
            f"{row_dict['method']}/{row_dict['case']}/{row_dict['design_id']}",
            flush=True,
        )
        rows_out.append(
            evaluate_one(row_dict, out_root, tok, mdl, args.device, evaluator="esmfold")
        )
        _align_eval_csv_rows(rows_out).to_csv(output_csv, index=False)
    print(f"wrote {output_csv}")


def _run_af3_for_one_sub_run(df: pd.DataFrame, args, out_root: Path) -> None:
    if not args.paper_case:
        raise SystemExit("AF3 evaluator: --paper-case required")
    if not args.sub_run:
        raise SystemExit(
            "AF3 evaluator: --sub-run required (one cli.py invocation per sub-run)"
        )

    paper_case = args.paper_case
    sub_run = args.sub_run
    af3_env = _collect_af3_env(args)
    if os.environ.get("AF3_MOCK") != "1":
        _af3_preflight(args, df)

    staging_dir = Path(args.af3_staging_root or out_root / "staging") / paper_case / sub_run
    msa_dir = Path(args.af3_msa_root or out_root / "per_paper_case_msa") / paper_case
    af3_output_dir = Path(args.af3_output_root or out_root / "af3_outputs") / paper_case / sub_run

    if args.skip_existing:
        before = len(df)
        df = df[
            df.apply(
                lambda r: not (
                    _design_dir_for_row(out_root, r) / "af3" / "af3_pred.pdb"
                ).exists(),
                axis=1,
            )
        ]
        if len(df) < before:
            print(f"skip_existing: {before - len(df)} designs already have af3_pred.pdb")
        if len(df) == 0:
            output_csv = _resolve_output_csv(args, sub_run=sub_run)
            if output_csv.exists():
                print(f"all designs already evaluated; preserving existing {output_csv}")
                return
            raise SystemExit(
                f"all designs already have af3_pred.pdb but no eval csv at {output_csv}; "
                "use --no-skip-existing to rebuild"
            )

    af3_runner.cleanup_for_rerun(staging_dir, af3_output_dir, df)

    rows = df.to_dict("records")
    for row in rows:
        if _is_missing(row.get("paper_case")):
            row["paper_case"] = paper_case
    output_csv = _resolve_output_csv(args, sub_run=sub_run)
    try:
        staged = af3_runner.stage_inputs_for_sub_run(
            rows=rows,
            staging_dir=staging_dir,
            target_msa_dir=msa_dir,
        )
    except Exception as exc:
        print(f"WARNING: staging failed: {type(exc).__name__}: {exc}")
        rows_out = [
            _build_failed_row(
                {"row": row},
                paper_case,
                sub_run,
                f"staging_failed: {type(exc).__name__}: {exc}",
            )
            for row in rows
        ]
        _align_eval_csv_rows(rows_out).to_csv(output_csv, index=False)
        print(f"wrote {output_csv} ({len(rows_out)} stage-failed rows)")
        return
    for staged_design in staged:
        staged_design["out_dir"] = _design_dir_for_row(out_root, staged_design["row"]) / "af3"

    if len(staged) == 0:
        _align_eval_csv_rows([]).to_csv(output_csv, index=False)
        print(f"no AF3 designs staged; wrote header-only {output_csv}")
        return

    print(f"Running AF3 batched: {len(staged)} designs in {sub_run}")
    subp_result = af3_runner.run_af3_batched(
        staging_dir=staging_dir,
        output_dir=af3_output_dir,
        af3_env=af3_env,
        timeout_s=args.af3_timeout_per_sub_run,
    )
    if subp_result["timed_out"]:
        print(f"WARNING: AF3 timed out at {subp_result['wall_clock_s']:.0f}s")
    if subp_result["exit_code"] != 0:
        print(
            f"WARNING: AF3 exit code {subp_result['exit_code']}; "
            f"stderr tail: {subp_result['stderr_tail'][-500:]}"
        )

    amortized_s = subp_result["wall_clock_s"] / max(len(staged), 1)

    rows_out: list[dict] = []
    for staged_design in staged:
        try:
            row = staged_design["row"]
            af3_result = af3_runner.parse_one_af3_design(
                af3_sub_run_output_dir=af3_output_dir,
                staged_design=staged_design,
                target_seqs=str(row["target_seq"]).split(":"),
                binder_seq=str(row["binder_seq"]),
                amortized_runtime_s=amortized_s,
                out_dir=staged_design["out_dir"],
            )
            row_out = evaluate_one(
                row,
                out_root,
                af3_result=af3_result,
                evaluator="af3",
            )
        except FileNotFoundError as exc:
            row_out = _build_failed_row(
                staged_design,
                paper_case,
                sub_run,
                f"af3_output_missing: {exc}",
            )
        except Exception as exc:
            row_out = _build_failed_row(
                staged_design,
                paper_case,
                sub_run,
                f"parse/metrics: {type(exc).__name__}: {exc}",
            )
        rows_out.append(row_out)
        _align_eval_csv_rows(rows_out).to_csv(output_csv, index=False)
    print(f"wrote {output_csv}")


def main():
    args = _build_argparser().parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = load_manifest(args.manifest)
    if args.paper_case and "paper_case" in df.columns:
        paper_values = df["paper_case"]
        fallback_case = args.sub_run or args.paper_case
        mask = paper_values.eq(args.paper_case) | (
            paper_values.isna() & df["case"].eq(fallback_case)
        )
        df = df[mask]
    elif args.paper_case:
        df = df[df["case"] == args.paper_case]
    if args.sub_run:
        df = df[df["case"] == args.sub_run]
    if args.limit > 0:
        df = df.head(args.limit)

    if len(df) == 0:
        sub_run = args.sub_run or args.paper_case or "_empty"
        output_csv = _resolve_output_csv(args, sub_run=sub_run)
        pd.DataFrame(columns=CSV_COLS).to_csv(output_csv, index=False)
        print(f"manifest empty after filter; wrote header-only {output_csv}")
        return

    if args.evaluator == "esmfold":
        _run_esmfold_loop(df, args, out_root)
    else:
        _run_af3_for_one_sub_run(df, args, out_root)


if __name__ == "__main__":
    main()
