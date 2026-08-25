#!/usr/bin/env python
"""Report the formal run per spec 12: each family separately, never one blended loss."""
from __future__ import annotations

import argparse
import json
from statistics import mean

ARMS = ["A", "B", "C", "D"]
LABEL = {"A": "A off/off", "B": "B TCRLM", "C": "C contact", "D": "D both",
         "E": "E ESM-C", "F": "F ESMC+ct"}


def table(data, metric, title, note, lower_better=True):
    cases = [c for c in sorted(data) if any(a in data[c] for a in ARMS)]
    if not cases:
        return
    print(f"\n--- {title} ---")
    print(f"    ({note})")
    head = f"{'case':18s}" + "".join(f"{LABEL.get(a, a):>13s}" for a in ARMS)
    print(head)
    for case in cases:
        line = f"{case:18s}"
        for arm in ARMS:
            v = data[case].get(arm, {}).get(metric)
            line += f"{v:>13.4f}" if isinstance(v, (int, float)) else f"{'-':>13s}"
        print(line)
    # deltas vs arm A
    print(f"{'mean delta vs A':18s}", end="")
    for arm in ARMS:
        ds = [data[c][arm][metric] - data[c]["A"][metric]
              for c in cases
              if arm in data[c] and "A" in data[c]
              and isinstance(data[c][arm].get(metric), (int, float))
              and isinstance(data[c]["A"].get(metric), (int, float))]
        print(f"{mean(ds):>+13.4f}" if ds else f"{'-':>13s}", end="")
    print()
    print(f"{'better than A':18s}", end="")
    for arm in ARMS:
        ds = [data[c][arm][metric] - data[c]["A"][metric]
              for c in cases
              if arm in data[c] and "A" in data[c]
              and isinstance(data[c][arm].get(metric), (int, float))
              and isinstance(data[c]["A"].get(metric), (int, float))]
        if not ds:
            print(f"{'-':>13s}", end="")
            continue
        n = sum(1 for d in ds if (d < 0) == lower_better)
        print(f"{f'{n}/{len(ds)}':>13s}", end="")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="/root/mingxuan/formal_eval_partial.json")
    args = ap.parse_args()
    data = json.load(open(args.eval))

    done = sum(1 for c in data for a in data[c] if not a.startswith("_"))
    print(f"evaluated cells: {done}   cases: {len(data)}")

    print("\n" + "=" * 74)
    print("1. STRUCTURE / INTERFACE  (arm-comparable: excludes all TCR terms)")
    print("=" * 74)
    table(data, "esm_verify_loss", "esm_verify_loss", "lower better")
    table(data, "pair_chain_iptm_loss", "pair_chain_iptm_loss", "lower better")
    table(data, "raw_common_i_pae", "interface PAE", "lower better")
    table(data, "raw_common_clash", "clash", "lower better")

    print("\n" + "=" * 74)
    print("2. TCR NATURALNESS  (cross-evaluated with BOTH scorers)")
    print("=" * 74)
    table(data, "tcrlm_nll", "TCRLM NLL", "lower = more TCR-like")
    table(data, "esmc_nll", "ESM-C NLL", "lower = more natural")
    print("\n  natural A6 CDR3 reference (evaluation-only, never optimised):")
    for case in sorted(data):
        ref = data[case].get("_natural_A6", {})
        if ref:
            print(f"    {case:18s} tcrlm={ref.get('tcrlm_nll')}  "
                  f"esmc={ref.get('esmc_nll')}")

    print("\n" + "=" * 74)
    print("3. CONTACT  (recomputed identically for every arm)")
    print("=" * 74)
    table(data, "cdr3_peptide_contact_nll", "CDR3-peptide contact nll",
          "lower = more contact")
    table(data, "framework_pmhc_contact_nll", "framework-pMHC contact nll",
          "HIGHER is better here", lower_better=False)
    table(data, "framework_penalty", "framework hinge penalty",
          "0 = guardrail never fired")
    table(data, "r_pep", "r_pep = CDR3-pep / TCR-pMHC contact mass",
          "diagnostic only, higher = more peptide-centric", lower_better=False)

    print("\n" + "=" * 74)
    print("4. SEQUENCES / CONTROLS")
    print("=" * 74)
    for case in sorted(data):
        print(f"  {case}")
        for arm in ARMS:
            row = data[case].get(arm)
            if not row:
                continue
            print(f"    {LABEL[arm]:12s} {row['cdr3']:18s} "
                  f"fw_unchanged={row['framework_unchanged']} "
                  f"n_mut_outside={row['n_mut_outside_cdr3']}")
    all_cdr = [data[c][a]["cdr3"] for c in data for a in ARMS if a in data[c]]
    print(f"\n  distinct CDR3 across all evaluated cells: "
          f"{len(set(all_cdr))}/{len(all_cdr)}")
    bad = [(c, a) for c in data for a in ARMS
           if a in data[c] and not data[c][a]["framework_unchanged"]]
    print(f"  framework violations: {len(bad)}")


if __name__ == "__main__":
    main()
