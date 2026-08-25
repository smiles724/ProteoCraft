#!/usr/bin/env python
"""Independently recompute the old naturalness table from raw best sequences.

The script that produced naturalness_cross_score_results.json is gone, so this
recomputes TCRLM / ESM-C NLL from the recovered best_seq values and checks the
stored numbers rather than trusting them.

Two scoring conventions are tried, because the original convention was not
recorded:
  cdr_only  -- score the CDR string on its own
  full_ctx  -- score the whole TCR chain, average NLL over the CDR positions
The second is the one that mattered for the old TCRLM window bug.
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, "/root/mingxuan/ProteoCraft")

from bindcraft.hallucination.tcr_prior import build_tcr_prior_scorer  # noqa: E402

ROOT = "/root/mingxuan"
ARM_DIRS = {
    "baseline": "cdr_prior_full_out_off",
    "tcrlm_fixed": "cdr_prior_full_out",
    "esmc6b": "cdr_prior_full_out_esmc6b",
    "ligandmpnn": "cdr_prior_full_out_ligandmpnn",
}
CASES = ["9NMW_alpha", "9NMW_beta", "1QRN_alpha", "1QRN_beta"]

TCRLM_REPO = "/root/mingxuan/tcr_prior_backends/tcrLM"
TCRLM_CKPT = f"{TCRLM_REPO}/pretrained_model/pretrained_model_tcr.bin"


def load_backend(kind: str):
    if kind == "tcrlm":
        cfg = {
            "use_tcr_prior_verifier": True,
            "tcr_prior_verifier_weight": 1.0,
            "tcr_prior_model_name": f"tcrlm:{TCRLM_REPO}",
            "tcr_prior_checkpoint": TCRLM_CKPT,
            "require_real_tcr_prior": True,
        }
    else:
        cfg = {
            "use_tcr_prior_verifier": True,
            "tcr_prior_verifier_weight": 1.0,
            "tcr_prior_model_name": "esmc:biohub/ESMC-6B",
            "require_real_tcr_prior": True,
        }
    scorer = build_tcr_prior_scorer(cfg, device=torch.device("cuda"))
    return scorer.backend


def main() -> None:
    recovered = {
        (r["arm"], r["case"]): r
        for r in json.load(open(os.path.join(ROOT, "old_protenix_recovered.json")))
        if r.get("status") == "ok"
    }
    stored = json.load(open(os.path.join(ROOT, "naturalness_cross_score_results.json")))

    # 1. confirm which output dir each label in the stored table came from
    print("=== arm-label -> output-dir verification ===")
    import pickle

    baseline_pkl = pickle.load(open(os.path.join(ROOT, "baseline_best_seqs.pkl"), "rb"))
    our_pkl = pickle.load(open(os.path.join(ROOT, "our_best_seqs.pkl"), "rb"))
    seqs: dict = {}
    for case in CASES:
        for label, dirname in ARM_DIRS.items():
            row = recovered.get((dirname, case))
            if row is None:
                continue
            start, end = row["cdr_start"] - 1, row["cdr_end"]
            cdr = row["best_seq"][start:end]
            if label == "baseline":
                pkl_cdr = baseline_pkl[case]["cdr"]
            else:
                pkl_cdr = our_pkl[case].get(label, {}).get("cdr")
            match = (pkl_cdr == cdr)
            print(f"  {case:12s} {label:12s} dir={dirname:34s} "
                  f"cdr={cdr:18s} pkl_match={match}")
            seqs[(case, label)] = {
                "cdr": cdr, "full": row["best_seq"],
                "start": start, "end": end,
            }

    # 2. recompute
    results: dict = {}
    for kind in ("tcrlm", "esmc"):
        print(f"\n=== loading {kind} ===", flush=True)
        backend = load_backend(kind)
        for (case, label), payload in seqs.items():
            cdr, full = payload["cdr"], payload["full"]
            start, end = payload["start"], payload["end"]
            nll_cdr_only = backend.region_nll(cdr, list(range(len(cdr))))
            nll_full_ctx = backend.region_nll(full, list(range(start, end)))
            results.setdefault(case, {}).setdefault(label, {})[
                f"{kind}_cdr_only"] = round(nll_cdr_only, 4)
            results[case][label][f"{kind}_full_ctx"] = round(nll_full_ctx, 4)
        del backend
        torch.cuda.empty_cache()

    # 3. compare against the stored table
    print("\n" + "=" * 104)
    print(f"{'case':12s} {'arm':12s} {'stored_tcrlm':>13s} {'cdr_only':>9s} "
          f"{'full_ctx':>9s} | {'stored_esmc':>12s} {'cdr_only':>9s} {'full_ctx':>9s}")
    print("=" * 104)
    for case in CASES:
        for label in ("baseline", "tcrlm_fixed", "esmc6b", "ligandmpnn"):
            s = stored[case][label]
            r = results[case][label]
            print(f"{case:12s} {label:12s} {s['tcrlm_nll']:>13.4f} "
                  f"{r['tcrlm_cdr_only']:>9.4f} {r['tcrlm_full_ctx']:>9.4f} | "
                  f"{s['esmc_nll']:>12.4f} {r['esmc_cdr_only']:>9.4f} "
                  f"{r['esmc_full_ctx']:>9.4f}")

    json.dump(results, open(os.path.join(ROOT, "old_naturalness_recomputed.json"), "w"),
              indent=2)
    print(f"\nwrote {ROOT}/old_naturalness_recomputed.json")


if __name__ == "__main__":
    main()
