#!/usr/bin/env python
"""Recover the old Protenix + TCR sequence-prior experiment from raw outputs."""
from __future__ import annotations

import glob
import json
import os

CASES = ["9NMW_alpha", "9NMW_beta", "1QRN_alpha", "1QRN_beta"]
ROOT = "/root/mingxuan"

rows = []
for d in sorted(glob.glob(os.path.join(ROOT, "cdr_prior_full_out*"))):
    arm = os.path.basename(d)
    for case in CASES:
        f = os.path.join(d, case, "result.json")
        if not os.path.exists(f):
            rows.append({"arm": arm, "case": case, "status": "MISSING"})
            continue
        r = json.load(open(f))
        p = r.get("params", {})
        tp = p.get("tcr_prior_config", {}) or {}
        rows.append({
            "arm": arm,
            "case": case,
            "status": r.get("status"),
            "best_loss": r.get("best_loss"),
            "best_step": r.get("best_step"),
            "final_loss": r.get("final_loss"),
            "elapsed_h": round((r.get("elapsed_sec") or 0) / 3600, 2),
            "fw_ok": r.get("best_framework_unchanged"),
            "n_steps": p.get("n_steps"),
            "k_top": p.get("k_top"),
            "b_cand": p.get("b_candidates"),
            "model": p.get("model_name"),
            "grad": tp.get("use_tcr_prior_gradient"),
            "ver": tp.get("use_tcr_prior_verifier"),
            "gw": tp.get("tcr_prior_gradient_weight"),
            "vw": tp.get("tcr_prior_verifier_weight"),
            "backend": (tp.get("tcr_prior_model_name") or "none"),
            "region": tp.get("tcr_prior_region"),
            "strict": tp.get("require_real_tcr_prior"),
            "seed_in_params": "seed" in p,
            "best_seq": r.get("best_seq"),
            "cdr_start": r.get("cdr_start_1based"),
            "cdr_end": r.get("cdr_end_1based"),
            "binder_sequence": r.get("binder_sequence"),
        })

print("=" * 118)
print(f"{'arm':38s} {'case':12s} {'st':4s} {'best_loss':>10s} {'stp':>4s} "
      f"{'h':>5s} {'fw':>3s} {'grad':>5s} {'ver':>5s} {'backend':22s}")
print("=" * 118)
for row in rows:
    if row["status"] == "MISSING":
        print(f"{row['arm']:38s} {row['case']:12s} MISSING")
        continue
    backend = str(row["backend"]).split(":")[0]
    print(f"{row['arm']:38s} {row['case']:12s} {str(row['status'])[:3]:4s} "
          f"{row['best_loss']:>10.4f} {str(row['best_step']):>4s} "
          f"{row['elapsed_h']:>5.1f} {str(row['fw_ok'])[:3]:>3s} "
          f"{str(row['grad'])[:5]:>5s} {str(row['ver'])[:5]:>5s} {backend:22s}")

ok = [r for r in rows if r.get("status") == "ok"]
print()
print("=== matched-ness across arms ===")
for key in ("n_steps", "k_top", "b_cand", "model", "region", "gw", "vw", "strict"):
    values = sorted({str(r.get(key)) for r in ok})
    print(f"  {key:10s}: {values}")
print(f"  seed recorded in params: {sorted({r['seed_in_params'] for r in ok})}")
print(f"  framework unchanged    : {sorted({str(r['fw_ok']) for r in ok})}")
print(f"  completed runs         : {len(ok)} / {len(rows)}")

json.dump(rows, open(os.path.join(ROOT, "old_protenix_recovered.json"), "w"), indent=2)
print(f"\nwrote {ROOT}/old_protenix_recovered.json")
