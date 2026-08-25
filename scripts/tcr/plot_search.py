#!/usr/bin/env python
"""Plot the ESMFold2 search trajectory: every structure-loss term, the TCR
sequence-prior term, the contact terms, and the proposal-gradient diagnostics.

Follows the METRICS convention of benchmark/results/analysis/plot_loss_histories.py
so the panels line up with what the lab already reads.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ["A", "B", "C", "D"]
COLOR = {"A": "#6B7480", "B": "#0E6B6B", "C": "#B4482F", "D": "#9A6B12"}
LABEL = {"A": "A off/off", "B": "B prior", "C": "C contact", "D": "D both"}

# structure terms, same naming as plot_loss_histories.py METRICS
STRUCT = ["raw_common_total", "raw_common_plddt", "raw_common_pae", "raw_common_i_pae",
          "raw_common_con", "raw_common_i_con", "raw_common_iptm", "raw_common_rg",
          "raw_common_helix", "raw_common_clash", "raw_common_i_clash"]
COMPOSITE = ["esm_verify_loss", "pair_chain_iptm_loss", "binder_confidence_loss",
             "global_confidence_loss", "binder_rg_excess_loss"]
TCR = ["tcr_prior_hard", "tcr_contact_loss", "cdr3_peptide_contact_nll",
       "framework_pmhc_contact_nll", "framework_penalty", "r_pep"]

# empirical targets from 216 human class-I TCR-pMHC structures
NAT_RPEP = {"alpha": 0.2953, "beta": 0.3468}

GRAD = re.compile(
    r"\|contact\|=([-\d.e+]+) \|tcr\|=([-\d.e+]+) \|structure\|=([-\d.e+]+) \| "
    r"cos\(contact,struct\)=([-+\d.]+) cos\(contact,esmc\)=([-+\d.]+) "
    r"cos\(contact,tcr\)=([-+\d.]+)")
COS_TCR = re.compile(r"cos\(tcr,struct\)=([-+\d.]+)")


def load_candidates(case_dir: Path, arm: str):
    f = case_dir / f"arm_{arm}" / "artifacts" / "candidates.jsonl"
    if not f.exists():
        return []
    out = []
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("step") is None:
            continue
        out.append(d)
    return out


def series(cands, key):
    """Best (min) value of `key` seen at each step -- the search frontier."""
    per = defaultdict(list)
    for d in cands:
        v = (d.get("loss_dict") or {}).get(key)
        if isinstance(v, (int, float)):
            per[d["step"]].append(v)
    steps = sorted(per)
    return steps, [min(per[s]) for s in steps]


def series_mean(cands, key):
    per = defaultdict(list)
    for d in cands:
        v = (d.get("loss_dict") or {}).get(key)
        if isinstance(v, (int, float)):
            per[d["step"]].append(v)
    steps = sorted(per)
    return steps, [sum(per[s]) / len(per[s]) for s in steps]


def plot_case(case_dir: Path, out_dir: Path):
    case_id = case_dir.name
    chain = "alpha" if case_id.endswith("alpha") else "beta"
    data = {a: load_candidates(case_dir, a) for a in ARMS}
    data = {a: v for a, v in data.items() if v}
    if not data:
        return None

    panels = [k for k in STRUCT + COMPOSITE + TCR
              if any((d.get("loss_dict") or {}).get(k) is not None
                     for a in data for d in data[a])]
    ncol = 4
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 2.9 * nrow))
    axes = axes.flatten()

    for ax, key in zip(axes, panels):
        for a, cands in data.items():
            # r_pep is a diagnostic ratio -- mean is the honest summary, not min
            xs, ys = (series_mean(cands, key) if key == "r_pep"
                      else series(cands, key))
            if xs:
                ax.plot(xs, ys, color=COLOR[a], label=LABEL[a], lw=1.6, marker="o", ms=2.5)
        if key == "r_pep":
            ax.axhline(NAT_RPEP[chain], color="#14181D", ls="--", lw=1.4)
            ax.text(0.02, NAT_RPEP[chain], f" natural {NAT_RPEP[chain]:.2f}",
                    transform=ax.get_yaxis_transform(), va="bottom",
                    fontsize=7.5, color="#14181D")
        ax.set_title(key, fontsize=9.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=.18, lw=.6)
    for ax in axes[len(panels):]:
        ax.axis("off")
    axes[0].legend(fontsize=7.5, loc="best")
    fig.suptitle(f"{case_id} — search trajectory, per-term", fontsize=13, y=.997)
    fig.tight_layout()
    p = out_dir / f"{case_id}_terms.png"
    fig.savefig(p, dpi=135)
    plt.close(fig)
    return p


def plot_gradients(root: Path, out_dir: Path):
    """Proposal-path gradient diagnostics, pooled over cases."""
    stat = {a: defaultdict(list) for a in ARMS}
    for case_dir in sorted(root.iterdir()):
        if not case_dir.is_dir():
            continue
        for a in ARMS:
            log = case_dir / f"arm_{a}" / "run.log"
            if not log.exists():
                continue
            for line in log.open(errors="ignore"):
                m = GRAD.search(line)
                if m:
                    c, t, s, cs, ce, ct = (float(x) for x in m.groups())
                    stat[a]["|contact|/|struct|"].append(c / s if s > 0 else 0.0)
                    stat[a]["cos(contact,struct)"].append(cs)
                m2 = COS_TCR.search(line)
                if m2:
                    stat[a]["cos(tcr,struct)"].append(float(m2.group(1)))
                    mt = re.search(r"\|tcr\|=([-\d.e+]+)", line)
                    if mt:
                        stat[a]["|tcr|"].append(float(mt.group(1)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    # left: cos vs structure, prior vs contact
    for a, key, col, lab in [("B", "cos(tcr,struct)", "#0E6B6B", "sequence prior (arm B)"),
                             ("C", "cos(contact,struct)", "#B4482F", "contact loss (arm C)")]:
        v = stat[a].get(key, [])
        if v:
            axes[0].hist(v, bins=40, range=(-1, 1), alpha=.62, color=col,
                         label=f"{lab}  mean={sum(v)/len(v):+.3f}")
    axes[0].axvline(0, color="#14181D", ls="--", lw=1.2)
    axes[0].set_title("cos(signal grad, structure grad)", fontsize=11)
    axes[0].set_xlabel("cosine"); axes[0].legend(fontsize=9)
    axes[0].grid(alpha=.18, lw=.6)
    # right: magnitude ratio
    v = stat["C"].get("|contact|/|struct|", [])
    if v:
        axes[1].hist([min(x, 12) for x in v], bins=40, color="#B4482F", alpha=.7)
        axes[1].axvline(1.0, color="#14181D", ls="--", lw=1.2)
        axes[1].set_title(f"|contact_grad| / |structure_grad|  median={sorted(v)[len(v)//2]:.2f}",
                          fontsize=11)
        axes[1].set_xlabel("ratio (clipped at 12)")
        axes[1].grid(alpha=.18, lw=.6)
    fig.suptitle("Proposal-path gradient diagnostics (pooled over 6 cases)", fontsize=13)
    fig.tight_layout()
    p = out_dir / "gradient_diagnostics.png"
    fig.savefig(p, dpi=135)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/mingxuan/formal_tcr6_20260820")
    ap.add_argument("--out", default="/root/mingxuan/search_plots")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for case_dir in sorted(root.iterdir()):
        if case_dir.is_dir() and (case_dir / "arm_A").exists():
            p = plot_case(case_dir, out)
            if p:
                print("wrote", p)
    print("wrote", plot_gradients(root, out))


if __name__ == "__main__":
    main()
