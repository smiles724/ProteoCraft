#!/usr/bin/env python
"""Matched post-hoc evaluation of the formal 6-case run.

Every arm is re-scored identically -- same verifier, same seed, contact metrics
computed for all arms including the ones that did not optimise them -- so the
comparison never depends on an arm-specific total_loss. That confound is what
broke the old Protenix analysis (see OLD_PROTENIX_PRIOR_AUDIT.md).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path("/root/mingxuan/ProteoCraft")
sys.path.insert(0, str(REPO))
SPEC_DIR = Path(
    "/root/Bingchen/experiments/20260819_liubo_cdr6_native_20x200/inputs/case_specs"
)
TCRLM_REPO = "/root/mingxuan/tcr_prior_backends/tcrLM"
TCRLM_CKPT = f"{TCRLM_REPO}/pretrained_model/pretrained_model_tcr.bin"
ARMS = ["A", "B", "C", "D", "E", "F"]


def load_rh():
    spec = importlib.util.spec_from_file_location(
        "rh", str(REPO / "scripts" / "run_hallucination.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def best_sequence(path: Path) -> str | None:
    f = path / "hallucination_result.txt"
    if not f.exists():
        return None
    for line in f.read_text().splitlines():
        if line.startswith("Best Sequence:"):
            return line.split(":", 1)[1].strip()
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/root/mingxuan/formal_tcr6_20260820")
    parser.add_argument("--out", default="/root/mingxuan/formal_eval.json")
    parser.add_argument("--seed", type=int, default=4242)  # fixed eval seed
    args_cli = parser.parse_args()
    root = Path(args_cli.root)

    rh = load_rh()
    results: dict = {}

    for case_path in sorted(SPEC_DIR.glob("*.json")):
        case = json.loads(case_path.read_text())
        case_id = case["case_id"]
        binder0 = case["binder_sequence"]
        start = case["cdr_start_0based"]
        end = case["cdr_end_0based_exclusive"]
        targets = case["partner_sequences"]
        mutable = [bool(v) for v in case["mutable_mask"]]

        present = {
            arm: seq
            for arm in ARMS
            if (seq := best_sequence(root / case_id / f"arm_{arm}")) is not None
        }
        if not present:
            continue
        print(f"\n=== {case_id}: evaluating {sorted(present)} ===", flush=True)

        # One backend per case (chain lengths / masks differ), contact ENABLED so
        # every arm gets contact metrics regardless of what it optimised.
        argv = [
            "--profile", "esmfold2_d",
            "--model_backend", "esmfold2",
            "--esmfold2_task_type", "cdr_redesign",
            "--target", "|".join(targets),
            "--esmfold2_target_chain_roles", ",".join(case["target_chain_roles"]),
            "--binder_length", str(len(binder0)),
            "--init_binder_seq", binder0,
            "--esmfold2_mutable_mask", "".join("1" if v else "0" for v in mutable),
            "--esmfold2_verifier_num_loops", "4",
            "--esmfold2_verifier_num_sampling_steps", "50",
            "--use_tcr_contact_loss", "true",
            "--tcr_contact_peptide_chain_index", "0",
            "--tcr_contact_mhc_chain_indices", "1",
            "--tcr_contact_partner_tcr_chain_index", "2",
            "--use_tcr_prior_gradient", "false",
            "--use_tcr_prior_verifier", "false",
            "--output_dir", "/tmp/eval_dummy",
        ]
        # Go through the same profile-aware entry point the real run uses, so the
        # evaluator inherits exactly the esmfold2_d defaults.
        parsed, _parser = rh.parse_args_with_profile(argv)
        # Same resolver order main() uses; resolve_model_backend fills in the
        # derived esmfold2_* defaults that the profile does not set directly.
        backend_name = rh.resolve_model_backend(parsed)
        rh.resolve_esmfold2_hotspot_defaults(
            parsed, model_backend=backend_name, hotspot_params=None
        )
        backend = rh.load_esmfold2_backend(
            parsed, device="cuda",
            mutable_mask=mutable,
            target_chain_roles=tuple(case["target_chain_roles"]),
        )

        for arm, seq in sorted(present.items()):
            result = backend.verify(
                binder_seq=seq, target_seq=targets, seed=args_cli.seed
            )
            ld = dict(result.loss_dict or {})
            summary = getattr(result, "summary_confidence", {}) or {}
            row = {
                "cdr3": seq[start:end],
                "framework_unchanged": (
                    seq[:start] == binder0[:start] and seq[end:] == binder0[end:]
                ),
                "n_mut_outside_cdr3": sum(
                    1 for i, (a, b) in enumerate(zip(seq, binder0))
                    if a != b and not (start <= i < end)
                ),
                # arm-comparable structure metrics
                "esm_verify_loss": ld.get("esm_verify_loss"),
                "pair_chain_iptm_loss": ld.get("pair_chain_iptm_loss"),
                "raw_common_iptm": ld.get("raw_common_iptm"),
                "raw_common_i_pae": ld.get("raw_common_i_pae"),
                "raw_common_i_con": ld.get("raw_common_i_con"),
                "raw_common_plddt": ld.get("raw_common_plddt"),
                "raw_common_clash": ld.get("raw_common_clash"),
                # contact, computed identically for every arm
                "cdr3_peptide_contact_nll": ld.get("cdr3_peptide_contact_nll"),
                "framework_pmhc_contact_nll": ld.get("framework_pmhc_contact_nll"),
                "framework_penalty": ld.get("framework_penalty"),
                "cdr3_peptide_contact_mass": ld.get("cdr3_peptide_contact_mass"),
                "tcr_pmhc_contact_mass": ld.get("tcr_pmhc_contact_mass"),
                "r_pep": ld.get("r_pep"),
                "best_sequence": seq,
            }
            results.setdefault(case_id, {})[arm] = row
            print(f"  arm {arm}: cdr3={row['cdr3']} "
                  f"verify={row['esm_verify_loss']:.4f} "
                  f"r_pep={row['r_pep']:.4f}", flush=True)

        del backend
        torch.cuda.empty_cache()

    # --- naturalness, cross-evaluated with BOTH scorers ---
    from bindcraft.hallucination.tcr_prior import build_tcr_prior_scorer

    for kind, cfg in (
        ("tcrlm", {"tcr_prior_model_name": f"tcrlm:{TCRLM_REPO}",
                   "tcr_prior_checkpoint": TCRLM_CKPT}),
        ("esmc", {"tcr_prior_model_name": "esmc:biohub/ESMC-6B"}),
    ):
        print(f"\n=== naturalness: {kind} ===", flush=True)
        full = {"use_tcr_prior_verifier": True, "tcr_prior_verifier_weight": 1.0,
                "require_real_tcr_prior": True, **cfg}
        backend = build_tcr_prior_scorer(full, device=torch.device("cuda")).backend
        for case_path in sorted(SPEC_DIR.glob("*.json")):
            case = json.loads(case_path.read_text())
            case_id = case["case_id"]
            if case_id not in results:
                continue
            start = case["cdr_start_0based"]
            end = case["cdr_end_0based_exclusive"]
            # the natural A6 CDR3 is a reference point, never an optimisation target
            results[case_id].setdefault("_natural_A6", {})[f"{kind}_nll"] = round(
                backend.region_nll(case["binder_sequence"], list(range(start, end))), 4)
            for arm, row in results[case_id].items():
                if arm.startswith("_"):
                    continue
                row[f"{kind}_nll"] = round(
                    backend.region_nll(row["best_sequence"], list(range(start, end))), 4)
        del backend
        torch.cuda.empty_cache()

    Path(args_cli.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args_cli.out}")


if __name__ == "__main__":
    main()
