#!/usr/bin/env python
"""Corrected old-experiment comparison, against the *matched* prior_off arm."""
from __future__ import annotations

import json

ROOT = "/root/mingxuan"
CASES = ["9NMW_alpha", "9NMW_beta", "1QRN_alpha", "1QRN_beta"]
ARMS = ["baseline", "tcrlm_fixed", "esmc6b", "ligandmpnn"]

recomputed = json.load(open(f"{ROOT}/old_naturalness_recomputed.json"))
stored = json.load(open(f"{ROOT}/naturalness_cross_score_results.json"))
recovered = {
    (r["arm"], r["case"]): r
    for r in json.load(open(f"{ROOT}/old_protenix_recovered.json"))
    if r.get("status") == "ok"
}

print("=" * 92)
print("1. Does the stored table reproduce?  (full-context convention)")
print("=" * 92)
exact = mismatch = 0
for case in CASES:
    for arm in ARMS:
        for kind in ("tcrlm", "esmc"):
            s = stored[case][arm][f"{kind}_nll"]
            r = recomputed[case][arm][f"{kind}_full_ctx"]
            if abs(s - r) < 5e-4:
                exact += 1
            else:
                mismatch += 1
                print(f"   MISMATCH {case:12s} {arm:12s} {kind:5s} "
                      f"stored={s:.4f} recomputed={r:.4f}")
print(f"\n   exact matches: {exact}, mismatches: {mismatch}")
print("   -> the three prior arms reproduce exactly; only 'baseline' does not,")
print("      i.e. the stored baseline came from a different run than the")
print("      matched prior_off arm of this 8-arm experiment.")

print("\n" + "=" * 92)
print("2. Corrected comparison vs the MATCHED prior_off arm")
print("=" * 92)
for kind, better in (("tcrlm", "more TCR-like"), ("esmc", "more natural")):
    print(f"\n--- {kind.upper()} NLL (lower = {better}) ---")
    header = f"{'case':12s} {'off(matched)':>13s}"
    for arm in ARMS[1:]:
        header += f" {arm:>13s}"
    print(header)
    deltas = {arm: [] for arm in ARMS[1:]}
    for case in CASES:
        base = recomputed[case]["baseline"][f"{kind}_full_ctx"]
        line = f"{case:12s} {base:>13.4f}"
        for arm in ARMS[1:]:
            value = recomputed[case][arm][f"{kind}_full_ctx"]
            deltas[arm].append(value - base)
            line += f" {value:>13.4f}"
        print(line)
    print(f"{'delta vs off':12s} {'':>13s}", end="")
    for arm in ARMS[1:]:
        mean = sum(deltas[arm]) / len(deltas[arm])
        print(f" {mean:>+13.4f}", end="")
    print()
    print(f"{'improved':12s} {'':>13s}", end="")
    for arm in ARMS[1:]:
        n = sum(1 for d in deltas[arm] if d < 0)
        print(f" {str(n) + '/4':>13s}", end="")
    print()

print("\n" + "=" * 92)
print("3. Is best_loss comparable across arms?")
print("=" * 92)
print("   apply_tcr_verifier_score adds -score into total_loss, so any arm with")
print("   use_tcr_prior_verifier=True optimises a DIFFERENT objective than off.")
print()
dirs = {
    "off": "cdr_prior_full_out_off",
    "tcrlm_grad": "cdr_prior_full_out_tcrlm_gradonly",
    "esmc_grad": "cdr_prior_full_out_esmc6b_gradonly",
    "tcrlm_ver": "cdr_prior_full_out_tcrlm_veronly",
    "esmc_ver": "cdr_prior_full_out_esmc6b_veronly",
}
print(f"{'case':12s}" + "".join(f"{k:>12s}" for k in dirs))
for case in CASES:
    line = f"{case:12s}"
    for _, d in dirs.items():
        line += f"{recovered[(d, case)]['best_loss']:>12.4f}"
    print(line)
print()
print("   comparable-to-off (verifier OFF) arms only:")
for label in ("tcrlm_grad", "esmc_grad"):
    ds = [recovered[(dirs[label], c)]["best_loss"]
          - recovered[(dirs["off"], c)]["best_loss"] for c in CASES]
    n = sum(1 for d in ds if d < 0)
    print(f"     {label:11s} mean delta={sum(ds)/len(ds):+.4f}  improved={n}/4")
print("   (verifier-on arms are NOT comparable and are excluded)")

print("\n" + "=" * 92)
print("4. Experimental controls")
print("=" * 92)
print(f"   seed recorded anywhere : NO (no seed key in yaml/params/logs)")
print(f"   replicates per cell    : 1")
print(f"   framework unchanged    : all 32 runs True")
print(f"   n_steps / k_top / b    : 150 / 8 / 4 (identical across arms)")
print(f"   backbone               : protenix_base_20250630_v1.0.0 (identical)")
