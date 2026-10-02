#!/usr/bin/env python
"""Score our matched-run results and Bingchen's baselines with ONE verifier.

Neither side's own reported loss is comparable: ours carries arm-specific TCR
terms and his is a rescore-selected candidate. Everything here is re-folded at
the same seed with contact metrics computed for every entry.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, sys
from pathlib import Path
import torch

REPO = Path("/root/mingxuan/ProteoCraft"); sys.path.insert(0, str(REPO))
SPEC = Path("/root/Bingchen/experiments/20260819_liubo_cdr6_native_20x200/inputs/case_specs")
BING = Path("/root/Bingchen/experiments/20260814_liubo_cdr6_fast_search/outputs/search")
OURS = Path("/root/mingxuan/matched_tcr6_20260826")


def load_rh():
    spec = importlib.util.spec_from_file_location("rh", str(REPO / "scripts" / "run_hallucination.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def our_seq(case, arm):
    f = OURS / case / f"arm_{arm}" / "hallucination_result.txt"
    if not f.exists(): return None
    for line in f.read_text().splitlines():
        if line.startswith("Best Sequence:"): return line.split(":", 1)[1].strip()
    return None


def bing_seqs(case):
    """His raw search best AND his 20x200-selected best -- they differ."""
    out = {}
    f = BING / case / "cdr_result.json"
    if f.exists(): out["BC_search"] = json.loads(f.read_text())["best_sequence"]
    h = BING / case / "high_precision_20x200" / "rescore_summary.json"
    if h.exists():
        b = json.loads(h.read_text()).get("best")
        if b: out["BC_rescored"] = b["sequence"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="ALGRRIPPV_alpha,ALGRRIPPV_beta")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--loops", type=int, default=4)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", default="/root/mingxuan/compare_bingchen.json")
    a = ap.parse_args()
    rh = load_rh(); results = {}

    for case_id in [c.strip() for c in a.cases.split(",") if c.strip()]:
        case = json.loads((SPEC / f"{case_id}.json").read_text())
        binder0 = case["binder_sequence"]; targets = case["partner_sequences"]
        mutable = [bool(v) for v in case["mutable_mask"]]
        s, e = case["cdr_start_0based"], case["cdr_end_0based_exclusive"]

        entries = {f"ours_{arm}": our_seq(case_id, arm) for arm in "ABCD"}
        entries.update(bing_seqs(case_id))
        entries["natural_A6"] = binder0
        entries = {k: v for k, v in entries.items() if v}
        if not entries: continue
        print(f"\n=== {case_id}: {sorted(entries)} ===", flush=True)

        argv = ["--profile", "esmfold2_d", "--model_backend", "esmfold2",
                "--esmfold2_task_type", "cdr_redesign",
                "--target", "|".join(targets),
                "--esmfold2_target_chain_roles", ",".join(case["target_chain_roles"]),
                "--binder_length", str(len(binder0)), "--init_binder_seq", binder0,
                "--esmfold2_mutable_mask", "".join("1" if v else "0" for v in mutable),
                "--esmfold2_verifier_num_loops", str(a.loops),
                "--esmfold2_verifier_num_sampling_steps", str(a.steps),
                "--use_tcr_contact_loss", "true",
                "--tcr_contact_peptide_chain_index", "0",
                "--tcr_contact_mhc_chain_indices", "1",
                "--tcr_contact_partner_tcr_chain_index", "2",
                "--use_tcr_prior_gradient", "false", "--use_tcr_prior_verifier", "false",
                "--output_dir", "/tmp/cmp_dummy"]
        parsed, _ = rh.parse_args_with_profile(argv)
        rh.resolve_model_backend(parsed)
        rh.resolve_esmfold2_hotspot_defaults(parsed, model_backend="esmfold2", hotspot_params=None)
        backend = rh.load_esmfold2_backend(parsed, device="cuda", mutable_mask=mutable,
                                           target_chain_roles=tuple(case["target_chain_roles"]))
        for label, seq in sorted(entries.items()):
            r = backend.verify(binder_seq=seq, target_seq=targets, seed=a.seed)
            ld = dict(r.loss_dict or {})
            results.setdefault(case_id, {})[label] = {
                "cdr3": seq[s:e],
                "esm_verify_loss": ld.get("esm_verify_loss"),
                "pair_chain_iptm_loss": ld.get("pair_chain_iptm_loss"),
                "raw_common_iptm": ld.get("raw_common_iptm"),
                "raw_common_plddt": ld.get("raw_common_plddt"),
                "cdr3_peptide_contact_nll": ld.get("cdr3_peptide_contact_nll"),
                "framework_pmhc_contact_nll": ld.get("framework_pmhc_contact_nll"),
                "r_pep": ld.get("r_pep"),
                "framework_unchanged": seq[:s] == binder0[:s] and seq[e:] == binder0[e:],
            }
            print(f"  {label:14s} {seq[s:e]:18s} verify={ld.get('esm_verify_loss'):.4f} "
                  f"r_pep={ld.get('r_pep'):.4f}", flush=True)
        del backend; torch.cuda.empty_cache()

    Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
