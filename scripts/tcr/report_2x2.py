#!/usr/bin/env python
"""The 2x2 under the new r_pep_target objective.

A and B come from the formal run (contact was off there, so the objective
change cannot have touched them); C030 from round 2; D030 from round 3.
Every cell was scored by the same verifier at the same fixed eval seed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

NAT = {"alpha": 0.2953, "beta": 0.3468}
COLS = [
    ("A off/off", "formal", "A"),
    ("B prior", "formal", "B"),
    ("C contact", "r2", "C030"),
    ("D both", "r3", "D030"),
]


def nat_for(case: str) -> float:
    return NAT["alpha" if case.endswith("alpha") else "beta"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--formal", default="/root/mingxuan/formal_eval_full.json")
    ap.add_argument("--r2", default="/root/mingxuan/sweep_r2_eval.json")
    ap.add_argument("--r3", default="/root/mingxuan/sweep_r3_eval.json")
    a = ap.parse_args()
    src = {k: json.load(open(getattr(a, k))) for k in ("formal", "r2", "r3")}
    cases = sorted(src["formal"])
    cols = [(lab, src[s], arm) for lab, s, arm in COLS
            if all(arm in src[s].get(c, {}) for c in cases)]
    if len(cols) < len(COLS):
        have = {c[0] for c in cols}
        print("missing arms: " + ", ".join(l for l, _, _ in COLS if l not in have))

    def block(title, note, get, lower_better=True):
        print(f"\n--- {title} ---\n    ({note})")
        print(f"{'case':18s}" + "".join(f"{l:>13s}" for l, _, _ in cols))
        for c in cases:
            print(f"{c:18s}" + "".join(f"{get(s, arm, c):13.4f}" for _, s, arm in cols))
        print(f"{'mean':18s}" + "".join(
            f"{mean([get(s, arm, c) for c in cases]):13.4f}" for _, s, arm in cols))
        base = cols[0]
        print(f"{'better than A':18s}", end="")
        for _, s, arm in cols:
            n = sum(1 for c in cases
                    if (get(s, arm, c) < get(base[1], base[2], c)) == lower_better)
            print(f"{f'{n}/{len(cases)}':>13s}", end="")
        print()

    print("2x2 under the new r_pep_target objective (lambda_contact = 0.3)")
    print(f"cases: {len(cases)}   seed 42   CDR3-only")

    block("|r_pep - natural|", "lower better; natural = 0.295 alpha / 0.347 beta",
          lambda s, arm, c: abs(s[c][arm]["r_pep"] - nat_for(c)))
    block("structure loss (esm_verify_loss)", "lower better; excludes all TCR terms",
          lambda s, arm, c: s[c][arm]["esm_verify_loss"])
    block("TCRLM NLL", "lower = closer to natural TCR repertoire",
          lambda s, arm, c: s[c][arm]["tcrlm_nll"])
    block("ESM-C NLL", "lower better",
          lambda s, arm, c: s[c][arm]["esmc_nll"])
    block("pair_chain_iptm_loss", "lower better",
          lambda s, arm, c: s[c][arm]["pair_chain_iptm_loss"])

    # does D inherit B's naturalness and C's structure?
    print("\n--- superposition check ---")
    print("    (does D pick up B's naturalness gain and C's structure gain?)")
    d = {lab: (s, arm) for lab, s, arm in cols}
    if "D both" in d and "B prior" in d and "C contact" in d:
        for metric, label, key in (
            ("tcrlm_nll", "TCRLM NLL", lambda s, arm, c: s[c][arm]["tcrlm_nll"]),
            ("esm_verify_loss", "structure", lambda s, arm, c: s[c][arm]["esm_verify_loss"]),
            ("r_pep", "|r_pep-nat|",
             lambda s, arm, c: abs(s[c][arm]["r_pep"] - nat_for(c))),
        ):
            vals = {lab: mean([key(s, arm, c) for c in cases])
                    for lab, s, arm in cols}
            print(f"  {label:14s} A={vals['A off/off']:.4f}  B={vals['B prior']:.4f}  "
                  f"C={vals['C contact']:.4f}  D={vals['D both']:.4f}")
            gain_b = vals["A off/off"] - vals["B prior"]
            gain_c = vals["A off/off"] - vals["C contact"]
            gain_d = vals["A off/off"] - vals["D both"]
            print(f"                 gain vs A:  B {gain_b:+.4f}  C {gain_c:+.4f}  "
                  f"D {gain_d:+.4f}   (additive would be {gain_b + gain_c:+.4f})")

    print("\n--- controls ---")
    bad = [(c, arm) for _, s, arm in cols for c in cases
           if not s[c][arm]["framework_unchanged"]]
    seqs = [s[c][arm]["cdr3"] for _, s, arm in cols for c in cases]
    print(f"  framework violations: {len(bad)}")
    print(f"  distinct CDR3: {len(set(seqs))}/{len(seqs)}")


if __name__ == "__main__":
    main()
