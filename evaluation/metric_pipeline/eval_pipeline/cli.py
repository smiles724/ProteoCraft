"""End-to-end orchestration: manifest → evaluator → metrics → long CSV."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `python -m eval_pipeline.cli` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_pipeline import af3_runner, atlasfold_runner, esmfold2_runner  # noqa: E402
from eval_pipeline import metrics as M  # noqa: E402
from eval_pipeline.esmfold_runner import (  # noqa: E402
    build_esmfold_input,
    dump_design_artifacts,
    load_esmfold,
    run_esmfold,
)
from eval_pipeline.manifest import load_manifest  # noqa: E402
from eval_pipeline.hotspot_coverage import calculate_hotspot_coverage  # noqa: E402
from eval_pipeline.tier_gating import compute_tier_flags  # noqa: E402
from vendor.proteina_complexa.pdb_chain_utils import get_chain_ca_coords  # noqa: E402


DEFAULT_EVALUATOR = "esmfold2"
LEGACY_ESMFOLD_EVALUATOR = "esmfold_202605"
LEGACY_ESMFOLD_ALIAS = "esmfold"
ATLASFOLD_EVALUATOR = atlasfold_runner.EVALUATOR_NAME

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
    # AtlasFold-M v1.0.2 native sequence-only cross-evaluation metrics.
    "atlasfold_ranking_score",
    "atlasfold_pair_chain_iptm",
    "atlasfold_scaffold_pair_chain_iptm",
    "atlasfold_iptm",
    "atlasfold_ptm",
    "atlasfold_complex_plddt",
    "atlasfold_target_plddt",
    "atlasfold_binder_plddt",
    "atlasfold_binder_ptm",
    "atlasfold_avg_pde",
    "atlasfold_fraction_disordered",
    "atlasfold_interface_iptm_json",
    "atlasfold_mmcif_path",
    "atlasfold_pdb_path",
    "atlasfold_confidence_json_path",
    "atlasfold_raw_output_dir",
    "atlasfold_reference_metric_status",
    "atlasfold_reference_metric_error",
    "atlasfold_model",
    "atlasfold_version",
    "atlasfold_source_commit",
    "atlasfold_num_recycles",
    "atlasfold_mlm_prob",
    "atlasfold_num_samples",
    "atlasfold_seeds",
    "atlasfold_num_steps",
    "atlasfold_max_tokens_per_batch",
    "atlasfold_kernel",
    # Current ProteoCraft ESMFold2 full-verifier metrics.
    "pair_chain_iptm",
    "iptm",
    "esmfold2_total_loss",
    "esmfold2_plddt",
    "esmfold2_binder_plddt",
    "esmfold2_binder_rg_excess",
    "esmfold2_mmcif_path",
    "esmfold2_model",
    "esmfold2_num_loops",
    "esmfold2_num_sampling_steps",
    "esmfold2_seed",
    "esmfold2_loss_mode",
    "target_chain_roles",
    "hotspot_mean_min_heavy_atom_distance",
    "hotspot_heavy_atom_coverage",
    "hotspot_heavy_atom_coverage_5a",
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
    ap.add_argument(
        "--evaluator",
        choices=[
            DEFAULT_EVALUATOR,
            LEGACY_ESMFOLD_EVALUATOR,
            LEGACY_ESMFOLD_ALIAS,
            "af3",
            ATLASFOLD_EVALUATOR,
        ],
        default=DEFAULT_EVALUATOR,
        help=(
            "evaluation backend (default: esmfold2). 'esmfold' remains a "
            "compatibility alias for esmfold_202605"
        ),
    )
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
    ap.add_argument(
        "--atlasfold-python",
        default=os.environ.get("ATLASFOLD_PYTHON"),
        help="Python executable in the isolated AtlasFold environment",
    )
    ap.add_argument(
        "--atlasfold-source-root",
        default=os.environ.get("ATLASFOLD_SOURCE_ROOT"),
        help="Pinned full AtlasFold v1.0.2 source checkout",
    )
    ap.add_argument(
        "--atlasfold-cache-dir",
        default=os.environ.get("ATLASFOLD_CACHE_DIR"),
    )
    ap.add_argument(
        "--atlasfold-model-path", default=os.environ.get("ATLASFOLD_MODEL_PATH")
    )
    ap.add_argument("--atlasfold-lm-path", default=os.environ.get("ATLASFOLD_LM_PATH"))
    ap.add_argument(
        "--atlasfold-weights-receipt",
        default=os.environ.get("ATLASFOLD_WEIGHTS_RECEIPT"),
    )
    ap.add_argument("--atlasfold-staging-root", default=None)
    ap.add_argument("--atlasfold-output-root", default=None)
    ap.add_argument("--atlasfold-timeout-per-sub-run", type=int, default=86400)
    ap.add_argument(
        "--atlasfold-num-recycles",
        type=int,
        default=atlasfold_runner.DEFAULT_NUM_RECYCLES,
    )
    ap.add_argument(
        "--atlasfold-mlm-prob",
        type=float,
        default=atlasfold_runner.DEFAULT_MLM_PROB,
    )
    ap.add_argument(
        "--atlasfold-num-samples",
        type=int,
        default=atlasfold_runner.DEFAULT_NUM_SAMPLES,
    )
    ap.add_argument(
        "--atlasfold-seed",
        type=int,
        nargs="+",
        default=list(atlasfold_runner.DEFAULT_SEEDS),
    )
    ap.add_argument(
        "--atlasfold-num-steps",
        type=int,
        default=atlasfold_runner.DEFAULT_NUM_STEPS,
    )
    ap.add_argument(
        "--atlasfold-max-tokens-per-batch",
        type=int,
        default=atlasfold_runner.DEFAULT_MAX_TOKENS_PER_BATCH,
    )
    ap.add_argument(
        "--atlasfold-kernel",
        choices=["auto", "torch", "cuequiv"],
        default=atlasfold_runner.DEFAULT_KERNEL,
    )
    ap.add_argument(
        "--esmfold2-model",
        default=esmfold2_runner.DEFAULT_MODEL_NAME,
    )
    ap.add_argument(
        "--esmfold2-num-loops",
        type=int,
        default=esmfold2_runner.DEFAULT_NUM_LOOPS,
    )
    ap.add_argument(
        "--esmfold2-num-sampling-steps",
        type=int,
        default=esmfold2_runner.DEFAULT_NUM_SAMPLING_STEPS,
    )
    ap.add_argument(
        "--esmfold2-seed",
        type=int,
        default=esmfold2_runner.DEFAULT_SEED,
    )
    ap.add_argument(
        "--esmfold2-loss-mode",
        choices=["pair_chain_iptm", "aligned_common"],
        default=esmfold2_runner.DEFAULT_LOSS_MODE,
    )
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
    if out.get("evaluator") == DEFAULT_EVALUATOR:
        key_cols = ["pair_chain_iptm", "esmfold2_total_loss"]
    elif out.get("evaluator") == ATLASFOLD_EVALUATOR:
        key_cols = ["pair_chain_iptm", "atlasfold_ranking_score"]
    else:
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
    atlasfold_result: atlasfold_runner.AtlasFoldResult | None = None,
    evaluator: str = LEGACY_ESMFOLD_EVALUATOR,
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
        if af3_result is not None and atlasfold_result is not None:
            raise ValueError("only one external evaluator result may be supplied")
        if af3_result is None and atlasfold_result is None:
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
        elif af3_result is not None:
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
        else:
            assert atlasfold_result is not None
            native = atlasfold_result.native
            out["esmfold_pdb_path"] = atlasfold_result.pdb_path
            out["eval_runtime_s"] = atlasfold_result.runtime_s
            plddt_clean = atlasfold_result.plddt
            pae_clean = atlasfold_result.pae
            clean_segs = atlasfold_result.clean_segments
            out.update({
                "pair_chain_iptm": native.get("pair_chain_iptm", float("nan")),
                "iptm": native.get("iptm", float("nan")),
                "ptm": native.get("ptm", float("nan")),
                "atlasfold_ranking_score": native.get(
                    "ranking_score", float("nan")
                ),
                "atlasfold_pair_chain_iptm": native.get(
                    "pair_chain_iptm", float("nan")
                ),
                "atlasfold_scaffold_pair_chain_iptm": native.get(
                    "scaffold_pair_chain_iptm", float("nan")
                ),
                "atlasfold_iptm": native.get("iptm", float("nan")),
                "atlasfold_ptm": native.get("ptm", float("nan")),
                "atlasfold_complex_plddt": native.get(
                    "complex_plddt", float("nan")
                ),
                "atlasfold_target_plddt": native.get(
                    "target_plddt", float("nan")
                ),
                "atlasfold_binder_plddt": native.get(
                    "binder_plddt", float("nan")
                ),
                "atlasfold_binder_ptm": native.get(
                    "binder_ptm", float("nan")
                ),
                "atlasfold_avg_pde": native.get("avg_pde", float("nan")),
                "atlasfold_fraction_disordered": native.get(
                    "fraction_disordered", float("nan")
                ),
                "atlasfold_interface_iptm_json": json.dumps(
                    native.get("interface_iptm", {}), sort_keys=True
                ),
                "atlasfold_mmcif_path": atlasfold_result.mmcif_path,
                "atlasfold_pdb_path": atlasfold_result.pdb_path,
                "atlasfold_confidence_json_path": (
                    atlasfold_result.confidence_json_path
                ),
                "atlasfold_raw_output_dir": atlasfold_result.raw_output_dir,
                "atlasfold_model": atlasfold_runner.MODEL_NAME,
                "atlasfold_version": atlasfold_runner.UPSTREAM_VERSION,
                "atlasfold_source_commit": atlasfold_runner.UPSTREAM_COMMIT,
            })

        # 4. Tier 0 metrics
        out["complex_pLDDT"] = M.complex_pLDDT(plddt_clean, clean_segs)
        out["binder_pLDDT"] = M.binder_pLDDT(plddt_clean, clean_segs)
        out["target_pLDDT"] = M.target_pLDDT(plddt_clean, clean_segs)
        out["mean_pAE"] = M.mean_pAE(pae_clean, clean_segs)
        out["ipAE"] = M.ipAE(pae_clean, clean_segs)
        out["min_ipAE"] = M.min_ipAE(pae_clean, clean_segs)
        out["i_pTM_approx_from_PAE"] = M.i_pTM_approx_from_PAE(pae_clean, clean_segs)
        if atlasfold_result is not None:
            out["complex_pLDDT"] = atlasfold_result.native["complex_plddt"]
            out["target_pLDDT"] = atlasfold_result.native["target_plddt"]
            out["binder_pLDDT"] = atlasfold_result.native["binder_plddt"]

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

        # scRMSD is optional for AtlasFold-M because this evaluator is
        # sequence-only and several production manifests intentionally omit a
        # reference complex. Other evaluators retain their original contract.
        reference_path = _row_get(row, "designed_complex_pdb_path", None)
        if atlasfold_result is not None:
            out["atlasfold_reference_metric_status"] = "missing"
            if reference_path and Path(str(reference_path)).is_file():
                try:
                    reference_pdb = Path(str(reference_path))
                    if reference_pdb.suffix.lower() in {".cif", ".mmcif"}:
                        converted_reference = (
                            design_dir
                            / ATLASFOLD_EVALUATOR
                            / "reference_complex.pdb"
                        )
                        af3_runner.mmcif_to_pdb(reference_pdb, converted_reference)
                        reference_pdb = converted_reference
                    ref_target_chains = str(row["target_chain_in_complex"]).split("+")
                    sc, fold = M.binder_scRMSD(
                        pred_pdb_path=out["esmfold_pdb_path"],
                        ref_pdb_path=str(reference_pdb),
                        target_chains_pred=pred_t_chains,
                        target_chains_ref=ref_target_chains,
                        binder_chain_pred=pred_b_chain,
                        binder_chain_ref=row["binder_chain_in_complex"],
                    )
                    out["binder_scRMSD"] = sc
                    out["binder_fold_RMSD"] = fold
                    out["atlasfold_reference_metric_status"] = "computed"
                except Exception as exc:
                    out["atlasfold_reference_metric_status"] = "failed"
                    out["atlasfold_reference_metric_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
        else:
            # v1.1 reviewer P1 #6: ref target chains from the manifest.
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

        if atlasfold_result is not None:
            roles = tuple(segment.get("role", "target") for segment in clean_segs[:-1])
            out["target_chain_roles"] = "+".join(roles)
            hotspots = esmfold2_runner.parse_hotspot_indices(
                _row_get(row, "target_hotspot_concat_indices", None)
            )
            if hotspots:
                target_lengths = [
                    int(segment["length"])
                    for segment in clean_segs
                    if segment["kind"].startswith("target_chain_")
                ]
                coverage_8a = calculate_hotspot_coverage(
                    atlasfold_result.pdb_path,
                    target_chain_ids=pred_t_chains,
                    binder_chain_id=pred_b_chain,
                    hotspot_concat_indices_0based=hotspots,
                    expected_target_chain_lengths=target_lengths,
                    heavy_atom_cutoff=8.0,
                )
                coverage_5a = calculate_hotspot_coverage(
                    atlasfold_result.pdb_path,
                    target_chain_ids=pred_t_chains,
                    binder_chain_id=pred_b_chain,
                    hotspot_concat_indices_0based=hotspots,
                    expected_target_chain_lengths=target_lengths,
                    heavy_atom_cutoff=5.0,
                )
                out.update({
                    "hotspot_mean_min_heavy_atom_distance": coverage_8a[
                        "mean_min_heavy_atom_distance"
                    ],
                    "hotspot_heavy_atom_coverage": coverage_8a[
                        "heavy_atom_coverage"
                    ],
                    "hotspot_heavy_atom_coverage_5a": coverage_5a[
                        "heavy_atom_coverage"
                    ],
                })
        else:
            # AtlasFold-M has no calibrated ProteoCraft pass thresholds yet.
            flags = compute_tier_flags(out, evaluator=evaluator)
            out.update(flags)

    except Exception as e:
        out["error_msg"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    out["eval_n_successes"] = _compute_eval_n_successes(out)
    return out


def _score_as_percent(value) -> float:
    """Normalize confidence scores to the legacy report's 0-100 pLDDT scale."""

    score = float(value)
    return score * 100.0 if score <= 1.0 else score


def evaluate_one_esmfold2(
    row: pd.Series,
    out_root: Path,
    backend,
    *,
    seed: int,
) -> dict:
    """Evaluate one manifest row with the current ProteoCraft full verifier."""

    out: dict = _empty_row()
    out["evaluator"] = DEFAULT_EVALUATOR
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
        "esmfold2_model": backend.config.verifier_model_name,
        "esmfold2_num_loops": backend.config.verifier_num_loops,
        "esmfold2_num_sampling_steps": backend.config.verifier_num_sampling_steps,
        "esmfold2_seed": int(seed),
        "esmfold2_loss_mode": backend.config.verifier_loss_mode,
    })
    design_dir = _design_dir_for_row(out_root, row)
    try:
        target_sequences = esmfold2_runner.parse_target_sequences(row["target_seq"])
        target_chain_roles = esmfold2_runner.parse_target_chain_roles(
            _row_get(row, "target_chain_roles", None),
            len(target_sequences),
        )
        evaluation = esmfold2_runner.evaluate(
            backend,
            binder_seq=row["binder_seq"],
            target_seq=target_sequences,
            design_dir=design_dir,
            seed=int(seed),
            target_chain_roles=target_chain_roles,
            hotspot_indices=_row_get(row, "target_hotspot_concat_indices", None),
        )
        result = evaluation.verifier_result
        summary = result.summary_confidence
        out.update({
            "esmfold2_total_loss": float(result.total_loss),
            "pair_chain_iptm": summary.get("pair_chain_iptm", float("nan")),
            "iptm": summary.get("iptm", float("nan")),
            "ptm": summary.get("ptm", float("nan")),
            "esmfold2_plddt": summary.get("plddt", float("nan")),
            "esmfold2_binder_plddt": summary.get("binder_plddt", float("nan")),
            "esmfold2_binder_rg_excess": summary.get(
                "binder_rg_excess", float("nan")
            ),
            "esmfold2_mmcif_path": evaluation.mmcif_path,
            "target_chain_roles": "+".join(target_chain_roles),
            "hotspot_mean_min_heavy_atom_distance": summary.get(
                "hotspot_mean_min_heavy_atom_distance", float("nan")
            ),
            "hotspot_heavy_atom_coverage": summary.get(
                "hotspot_heavy_atom_coverage", float("nan")
            ),
            "hotspot_heavy_atom_coverage_5a": summary.get(
                "hotspot_heavy_atom_coverage_5a", float("nan")
            ),
            "eval_runtime_s": evaluation.runtime_s,
            "esmfold_runtime_s": evaluation.runtime_s,
        })
        if pd.notna(out["esmfold2_plddt"]):
            out["complex_pLDDT"] = _score_as_percent(out["esmfold2_plddt"])
        if pd.notna(out["esmfold2_binder_plddt"]):
            out["binder_pLDDT"] = _score_as_percent(
                out["esmfold2_binder_plddt"]
            )
        radius = summary.get("binder_radius_gyration")
        if radius is not None:
            out["radius_of_gyration_binder"] = float(radius)
        # Tier-1 remains a confidence diagnostic. Tier-2 stays false because the
        # legacy ipAE/scRMSD contract is not emitted by the ESMFold2 verifier.
        flags = compute_tier_flags(out, evaluator=DEFAULT_EVALUATOR)
        out.update(flags)
    except Exception as exc:
        out["error_msg"] = f"{type(exc).__name__}: {exc}"
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


def _build_failed_row(
    staged: dict,
    paper_case: str,
    sub_run: str,
    error: str,
    *,
    evaluator: str = "af3",
) -> dict:
    out = _empty_row()
    row = staged.get("row", {}) if isinstance(staged, dict) else {}
    try:
        binder_length = len(row["binder_seq"]) if "binder_seq" in row else -1
    except Exception:
        binder_length = -1
    out.update({
        "evaluator": evaluator,
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
    """Run the May 2026 HuggingFace ESMFold evaluator."""
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
            evaluate_one(
                row_dict,
                out_root,
                tok,
                mdl,
                args.device,
                evaluator=LEGACY_ESMFOLD_EVALUATOR,
            )
        )
        _align_eval_csv_rows(rows_out).to_csv(output_csv, index=False)
    print(f"wrote {output_csv}")


def _run_esmfold2_loop(df: pd.DataFrame, args, out_root: Path) -> None:
    output_csv = (
        _resolve_output_csv(args, sub_run=args.sub_run)
        if args.paper_case
        else Path(args.out_dir) / "eval_long.csv"
    )
    if args.skip_existing:
        before = len(df)
        df = df[
            df.apply(
                lambda row: not (
                    _design_dir_for_row(out_root, row) / "esmfold2.cif"
                ).exists(),
                axis=1,
            )
        ]
        if len(df) < before:
            print(f"skip_existing: {before - len(df)} designs already have esmfold2.cif")
        if len(df) == 0:
            if output_csv.exists():
                print(f"all designs already evaluated; preserving existing {output_csv}")
                return
            raise SystemExit(
                "all designs already have esmfold2.cif but no eval csv at "
                f"{output_csv}; use --no-skip-existing to rebuild"
            )

    print(
        "Loading ESMFold2 verifier on "
        f"{args.device} ({args.esmfold2_num_loops}x"
        f"{args.esmfold2_num_sampling_steps}, seed={args.esmfold2_seed})..."
    )
    backend = esmfold2_runner.build_backend(
        device=args.device,
        model_name=args.esmfold2_model,
        num_loops=args.esmfold2_num_loops,
        num_sampling_steps=args.esmfold2_num_sampling_steps,
        loss_mode=args.esmfold2_loss_mode,
    )
    rows_out: list[dict] = (
        _read_existing_eval_rows(output_csv)
        if args.skip_existing and output_csv.exists()
        else []
    )
    for index, row in df.iterrows():
        row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        print(
            f"[{index + 1}/{len(df)}] "
            f"{row_dict['method']}/{row_dict['case']}/{row_dict['design_id']}",
            flush=True,
        )
        rows_out.append(
            evaluate_one_esmfold2(
                row_dict,
                out_root,
                backend,
                seed=args.esmfold2_seed,
            )
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


def _collect_atlasfold_env(args) -> dict:
    if os.environ.get("ATLASFOLD_MOCK") == "1":
        return {
            "atlasfold_python": args.atlasfold_python or "mock-python",
            "source_root": args.atlasfold_source_root or "mock-source",
            "source_metadata": {"mock": True},
            "environment_metadata": {"mock": True},
            "weights_metadata": {"mock": True},
        }

    required = {
        "atlasfold_python": args.atlasfold_python,
        "source_root": args.atlasfold_source_root,
        "model_path": args.atlasfold_model_path,
        "lm_path": args.atlasfold_lm_path,
        "weights_receipt": args.atlasfold_weights_receipt,
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        flags = ", ".join(f"--{key.replace('_', '-')}" for key in missing)
        raise SystemExit(
            f"--evaluator {ATLASFOLD_EVALUATOR} requires {flags} (or env vars)"
        )
    # Preserve a venv entrypoint symlink. Resolving it selects the base interpreter
    # and silently drops packages installed only in the isolated AtlasFold venv.
    python_path = Path(args.atlasfold_python).expanduser().absolute()
    source_root = Path(args.atlasfold_source_root)
    if not python_path.is_file():
        raise SystemExit(f"--atlasfold-python={python_path} is not a file")
    if not source_root.is_dir():
        raise SystemExit(f"--atlasfold-source-root={source_root} is not a directory")
    for flag_name, value in (
        ("atlasfold-model-path", args.atlasfold_model_path),
        ("atlasfold-lm-path", args.atlasfold_lm_path),
        ("atlasfold-weights-receipt", args.atlasfold_weights_receipt),
    ):
        if value and not Path(value).is_file():
            raise SystemExit(f"--{flag_name}={value} is not a file")
    try:
        source_metadata = atlasfold_runner.verify_source_checkout(source_root)
        environment_metadata = atlasfold_runner.probe_environment(
            str(python_path), source_root
        )
        weights_metadata = atlasfold_runner.validate_weights_receipt(
            Path(args.atlasfold_weights_receipt),
            model_path=Path(args.atlasfold_model_path),
            lm_path=Path(args.atlasfold_lm_path),
        )
    except Exception as exc:
        raise SystemExit(f"AtlasFold-M provenance preflight failed: {exc}") from exc
    return {
        "atlasfold_python": str(python_path),
        "source_root": str(source_root.resolve()),
        "source_metadata": source_metadata,
        "environment_metadata": environment_metadata,
        "weights_metadata": weights_metadata,
    }


def _atlasfold_preflight(manifest_df: pd.DataFrame) -> None:
    """Validate sequence/role inputs before spending AtlasFold GPU time."""

    for _, row in manifest_df.iterrows():
        atlasfold_runner.build_chain_layout(
            row["target_seq"],
            row["binder_seq"],
            _row_get(row, "target_chain_roles", None),
        )


def _row_identity(row) -> tuple[str, str, str]:
    return tuple(str(_row_get(row, key, "")) for key in ("method", "case", "design_id"))


def _drop_existing_rows_for_rerun(
    existing_rows: list[dict], rows_to_run: list[dict]
) -> list[dict]:
    rerun_keys = {_row_identity(row) for row in rows_to_run}
    return [row for row in existing_rows if _row_identity(row) not in rerun_keys]


def _write_or_validate_atlasfold_run_config(
    output_dir: Path, run_config: dict, *, allow_replace: bool
) -> None:
    output_dir = Path(output_dir)
    config_path = output_dir / "adapter_provenance.json"
    done_markers = list(output_dir.glob("*/done.txt")) if output_dir.is_dir() else []
    if config_path.is_file() and not allow_replace:
        previous = json.loads(config_path.read_text())
        if previous != run_config:
            raise SystemExit(
                "AtlasFold-M raw outputs use a different run configuration; "
                "use --no-skip-existing to replace them"
            )
        return
    if done_markers and not config_path.is_file() and not allow_replace:
        raise SystemExit(
            "AtlasFold-M raw outputs have completion markers but no adapter provenance; "
            "use --no-skip-existing to replace them"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n")


def _run_atlasfold_for_one_sub_run(df: pd.DataFrame, args, out_root: Path) -> None:
    if not args.paper_case:
        raise SystemExit(f"{ATLASFOLD_EVALUATOR} evaluator: --paper-case required")
    if not args.sub_run:
        raise SystemExit(
            f"{ATLASFOLD_EVALUATOR} evaluator: --sub-run required "
            "(one cli.py invocation per sub-run)"
        )

    paper_case = args.paper_case
    sub_run = args.sub_run
    output_csv = _resolve_output_csv(args, sub_run=sub_run)
    existing_rows: list[dict] = []
    existing_keys: set[tuple[str, str, str]] = set()
    if args.skip_existing and output_csv.exists():
        existing_rows = _read_existing_eval_rows(output_csv)
        existing_keys = {_row_identity(row) for row in existing_rows}

    if args.skip_existing:
        before = len(df)
        df = df[
            df.apply(
                lambda row: not (
                    (
                        _design_dir_for_row(out_root, row)
                        / ATLASFOLD_EVALUATOR
                        / "atlasfold_m.cif"
                    ).exists()
                    and _row_identity(row) in existing_keys
                ),
                axis=1,
            )
        ]
        if len(df) < before:
            print(
                f"skip_existing: {before - len(df)} designs already have "
                "AtlasFold-M artifacts and CSV rows"
            )
        if len(df) == 0:
            print(f"all designs already evaluated; preserving existing {output_csv}")
            return

    atlas_env = _collect_atlasfold_env(args)
    if os.environ.get("ATLASFOLD_MOCK") != "1":
        _atlasfold_preflight(df)

    staging_dir = Path(
        args.atlasfold_staging_root or out_root / "atlasfold_staging"
    ) / paper_case / sub_run
    atlas_output_dir = Path(
        args.atlasfold_output_root or out_root / "atlasfold_outputs"
    ) / paper_case / sub_run
    rows = df.to_dict("records")
    for row in rows:
        if _is_missing(row.get("paper_case")):
            row["paper_case"] = paper_case
    existing_rows = _drop_existing_rows_for_rerun(existing_rows, rows)

    atlasfold_runner.cleanup_for_rerun(
        staging_dir,
        atlas_output_dir,
        reset_outputs=not args.skip_existing,
    )
    try:
        staged = atlasfold_runner.stage_inputs_for_sub_run(
            rows=rows,
            staging_dir=staging_dir,
        )
    except Exception as exc:
        print(f"WARNING: AtlasFold-M staging failed: {type(exc).__name__}: {exc}")
        rows_out = existing_rows + [
            _build_failed_row(
                {"row": row},
                paper_case,
                sub_run,
                f"staging_failed: {type(exc).__name__}: {exc}",
                evaluator=ATLASFOLD_EVALUATOR,
            )
            for row in rows
        ]
        _align_eval_csv_rows(rows_out).to_csv(output_csv, index=False)
        print(f"wrote {output_csv} ({len(rows)} stage-failed rows)")
        return

    for staged_design in staged:
        staged_design["out_dir"] = (
            _design_dir_for_row(out_root, staged_design["row"])
            / ATLASFOLD_EVALUATOR
        )
    if not staged:
        _align_eval_csv_rows(existing_rows).to_csv(output_csv, index=False)
        print(f"no AtlasFold-M designs staged; wrote {output_csv}")
        return

    run_config = {
        "model": atlasfold_runner.MODEL_NAME,
        "upstream_version": atlasfold_runner.UPSTREAM_VERSION,
        "upstream_commit": atlasfold_runner.UPSTREAM_COMMIT,
        "source_metadata": atlas_env["source_metadata"],
        "environment_metadata": atlas_env["environment_metadata"],
        "weights_metadata": atlas_env["weights_metadata"],
        "weights_receipt": args.atlasfold_weights_receipt,
        "device": args.device,
        "num_recycles": int(args.atlasfold_num_recycles),
        "mlm_prob": float(args.atlasfold_mlm_prob),
        "num_samples": int(args.atlasfold_num_samples),
        "seeds": [int(seed) for seed in args.atlasfold_seed],
        "num_steps": int(args.atlasfold_num_steps),
        "max_tokens_per_batch": int(args.atlasfold_max_tokens_per_batch),
        "kernel": args.atlasfold_kernel,
        "cache_dir": args.atlasfold_cache_dir,
        "model_path": args.atlasfold_model_path,
        "lm_path": args.atlasfold_lm_path,
        "sequence_only": True,
    }
    _write_or_validate_atlasfold_run_config(
        atlas_output_dir,
        run_config,
        allow_replace=not args.skip_existing,
    )
    print(f"Running AtlasFold-M batched: {len(staged)} designs in {sub_run}")
    subp_result = atlasfold_runner.run_atlasfold_batched(
        staging_dir=staging_dir,
        output_dir=atlas_output_dir,
        atlasfold_python=atlas_env["atlasfold_python"],
        source_root=Path(atlas_env["source_root"]),
        device=args.device,
        timeout_s=args.atlasfold_timeout_per_sub_run,
        num_recycles=args.atlasfold_num_recycles,
        mlm_prob=args.atlasfold_mlm_prob,
        num_samples=args.atlasfold_num_samples,
        seeds=args.atlasfold_seed,
        num_steps=args.atlasfold_num_steps,
        max_tokens_per_batch=args.atlasfold_max_tokens_per_batch,
        kernel=args.atlasfold_kernel,
        cache_dir=(
            Path(args.atlasfold_cache_dir) if args.atlasfold_cache_dir else None
        ),
        model_path=(
            Path(args.atlasfold_model_path) if args.atlasfold_model_path else None
        ),
        lm_path=Path(args.atlasfold_lm_path) if args.atlasfold_lm_path else None,
        overwrite=not args.skip_existing,
    )
    if subp_result["timed_out"]:
        print(f"WARNING: AtlasFold-M timed out at {subp_result['wall_clock_s']:.0f}s")
    if subp_result["exit_code"] != 0:
        print(
            f"WARNING: AtlasFold-M exit code {subp_result['exit_code']}; "
            f"stderr tail: {subp_result['stderr_tail'][-500:]}"
        )

    amortized_s = subp_result["wall_clock_s"] / max(len(staged), 1)
    rows_out = list(existing_rows)
    for staged_design in staged:
        try:
            atlas_result = atlasfold_runner.parse_one_atlasfold_design(
                atlasfold_output_dir=atlas_output_dir,
                staged_design=staged_design,
                amortized_runtime_s=amortized_s,
                out_dir=staged_design["out_dir"],
                run_config=run_config,
            )
            row_out = evaluate_one(
                staged_design["row"],
                out_root,
                atlasfold_result=atlas_result,
                evaluator=ATLASFOLD_EVALUATOR,
            )
            row_out.update({
                "atlasfold_num_recycles": args.atlasfold_num_recycles,
                "atlasfold_mlm_prob": args.atlasfold_mlm_prob,
                "atlasfold_num_samples": args.atlasfold_num_samples,
                "atlasfold_seeds": ",".join(map(str, args.atlasfold_seed)),
                "atlasfold_num_steps": args.atlasfold_num_steps,
                "atlasfold_max_tokens_per_batch": args.atlasfold_max_tokens_per_batch,
                "atlasfold_kernel": args.atlasfold_kernel,
            })
        except FileNotFoundError as exc:
            row_out = _build_failed_row(
                staged_design,
                paper_case,
                sub_run,
                f"atlasfold_output_missing: {exc}",
                evaluator=ATLASFOLD_EVALUATOR,
            )
        except Exception as exc:
            row_out = _build_failed_row(
                staged_design,
                paper_case,
                sub_run,
                f"parse/metrics: {type(exc).__name__}: {exc}",
                evaluator=ATLASFOLD_EVALUATOR,
            )
        rows_out.append(row_out)
        _align_eval_csv_rows(rows_out).to_csv(output_csv, index=False)
    print(f"wrote {output_csv}")


def main():
    args = _build_argparser().parse_args()
    if args.evaluator == LEGACY_ESMFOLD_ALIAS:
        print(
            "WARNING: --evaluator esmfold is deprecated; "
            f"use --evaluator {LEGACY_ESMFOLD_EVALUATOR}"
        )
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

    if args.evaluator == DEFAULT_EVALUATOR:
        _run_esmfold2_loop(df, args, out_root)
    elif args.evaluator in {LEGACY_ESMFOLD_EVALUATOR, LEGACY_ESMFOLD_ALIAS}:
        _run_esmfold_loop(df, args, out_root)
    elif args.evaluator == "af3":
        _run_af3_for_one_sub_run(df, args, out_root)
    elif args.evaluator == ATLASFOLD_EVALUATOR:
        _run_atlasfold_for_one_sub_run(df, args, out_root)
    else:
        raise SystemExit(f"unsupported evaluator: {args.evaluator}")


if __name__ == "__main__":
    main()
