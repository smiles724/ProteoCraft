#!/usr/bin/env python3
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean, median

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/data/huangyanjie/binder_design/BindCraft-v2")
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
RESULTS_DIR = BENCHMARK_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "analysis" / "template_effects"

CASE_ORDER = ["BetV1", "DerF21", "DerF7", "HER2", "IFNAR2", "PD-L1", "PD1", "SpCas9"]

EXPERIMENTS = [
    {
        "loss_family": "default_loss",
        "label": "Default loss",
        "no_template": RESULTS_DIR / "bc_v1_random_avglen_hybrid" / "bc_v1_random_avglen_hybrid",
        "template": RESULTS_DIR / "bc_v1_random_avglen_hybrid_template" / "bc_v1_random_avglen_hybrid_template",
    },
    {
        "loss_family": "v1_loss",
        "label": "V1 loss",
        "no_template": RESULTS_DIR / "bc_v1_random_avglen_hybrid_v1loss" / "bc_v1_random_avglen_hybrid_v1loss",
        "template": RESULTS_DIR / "bc_v1_random_avglen_hybrid_v1loss_template" / "bc_v1_random_avglen_hybrid_v1loss_template",
    },
    {
        "loss_family": "custom_loss",
        "label": "Custom loss",
        "no_template": RESULTS_DIR / "custom_loss_no_hotspot_random_avglen" / "custom_loss_no_hotspot_random_avglen",
        "template": RESULTS_DIR
        / "custom_loss_with_template_no_hotspot_random_avglen"
        / "custom_loss_with_template_no_hotspot_random_avglen",
    },
]

LOSS_METRICS = [
    "total",
    "plddt",
    "pae",
    "i_pae",
    "con",
    "i_con",
    "iptm",
    "rg",
    "helix",
    "clash",
    "i_clash",
    "pde",
    "i_pde",
]

CONF_METRICS = [
    "plddt",
    "gpde",
    "ptm",
    "iptm",
    "chain_ptm",
    "chain_iptm",
    "ranking_score",
]


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def find_best_meta(case_dir: Path, result: dict) -> dict:
    best_step = int(result["best_step"])
    meta_paths = sorted((case_dir / "checkpoints" / f"step_{best_step:04d}").glob("*_meta.json"))
    if not meta_paths:
        return {}
    best_seq = result.get("best_seq")
    exact = []
    for path in meta_paths:
        meta = load_json(path)
        if meta.get("binder_seq") == best_seq:
            exact.append(meta)
    if exact:
        return min(exact, key=lambda meta: float(meta.get("loss_dict", {}).get("total", math.inf)))
    return min((load_json(path) for path in meta_paths), key=lambda meta: float(meta.get("loss_dict", {}).get("total", math.inf)))


def load_case(case_dir: Path) -> dict:
    result_path = case_dir / "result.json"
    result = load_json(result_path)
    meta = find_best_meta(case_dir, result)
    return {
        "entry_id": result["entry_id"],
        "binder_length": result.get("binder_length"),
        "best_loss": float(result["best_loss"]),
        "final_loss": float(result["final_loss"]),
        "best_step": int(result["best_step"]),
        "elapsed_sec": float(result["elapsed_sec"]),
        "target_template_path": result.get("params", {}).get("target_template_path"),
        "loss_dict": {k: float(v) for k, v in meta.get("loss_dict", {}).items()},
        "confidence": {k: float(v) for k, v in meta.get("summary_confidence", {}).items()},
    }


def pct_delta(template_value: float, base_value: float) -> float:
    if base_value == 0:
        return math.nan
    return (template_value - base_value) / abs(base_value) * 100.0


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.{digits}f}"


def collect_rows() -> list[dict]:
    rows = []
    for exp in EXPERIMENTS:
        for case in CASE_ORDER:
            no_case_dir = exp["no_template"] / case
            template_case_dir = exp["template"] / case
            if not no_case_dir.exists() or not template_case_dir.exists():
                raise FileNotFoundError(f"Missing paired case for {exp['loss_family']} {case}")

            base = load_case(no_case_dir)
            templ = load_case(template_case_dir)
            row = {
                "loss_family": exp["loss_family"],
                "loss_label": exp["label"],
                "case": case,
                "binder_length": base["binder_length"],
                "base_best_loss": base["best_loss"],
                "template_best_loss": templ["best_loss"],
                "delta_best_loss": templ["best_loss"] - base["best_loss"],
                "pct_delta_best_loss": pct_delta(templ["best_loss"], base["best_loss"]),
                "template_lower_best_loss": templ["best_loss"] < base["best_loss"],
                "base_final_loss": base["final_loss"],
                "template_final_loss": templ["final_loss"],
                "delta_final_loss": templ["final_loss"] - base["final_loss"],
                "base_best_step": base["best_step"],
                "template_best_step": templ["best_step"],
                "base_elapsed_sec": base["elapsed_sec"],
                "template_elapsed_sec": templ["elapsed_sec"],
                "delta_elapsed_sec": templ["elapsed_sec"] - base["elapsed_sec"],
                "template_path": templ["target_template_path"],
            }
            for metric in LOSS_METRICS:
                base_value = base["loss_dict"].get(metric)
                templ_value = templ["loss_dict"].get(metric)
                if base_value is not None and templ_value is not None:
                    row[f"base_loss_{metric}"] = base_value
                    row[f"template_loss_{metric}"] = templ_value
                    row[f"delta_loss_{metric}"] = templ_value - base_value
            for metric in CONF_METRICS:
                base_value = base["confidence"].get(metric)
                templ_value = templ["confidence"].get(metric)
                if base_value is not None and templ_value is not None:
                    row[f"base_conf_{metric}"] = base_value
                    row[f"template_conf_{metric}"] = templ_value
                    row[f"delta_conf_{metric}"] = templ_value - base_value
            rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "loss_family",
        "loss_label",
        "case",
        "binder_length",
        "base_best_loss",
        "template_best_loss",
        "delta_best_loss",
        "pct_delta_best_loss",
        "template_lower_best_loss",
        "base_conf_ranking_score",
        "template_conf_ranking_score",
        "delta_conf_ranking_score",
        "base_conf_plddt",
        "template_conf_plddt",
        "delta_conf_plddt",
        "base_conf_iptm",
        "template_conf_iptm",
        "delta_conf_iptm",
        "base_elapsed_sec",
        "template_elapsed_sec",
        "delta_elapsed_sec",
        "template_path",
    ]
    ordered = [name for name in preferred if name in fieldnames]
    ordered.extend(name for name in fieldnames if name not in set(ordered))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for exp in EXPERIMENTS:
        exp_rows = [row for row in rows if row["loss_family"] == exp["loss_family"]]
        deltas = [row["delta_best_loss"] for row in exp_rows]
        pct_deltas = [row["pct_delta_best_loss"] for row in exp_rows]
        ranking_deltas = [row.get("delta_conf_ranking_score") for row in exp_rows if "delta_conf_ranking_score" in row]
        iptm_deltas = [row.get("delta_conf_iptm") for row in exp_rows if "delta_conf_iptm" in row]
        plddt_deltas = [row.get("delta_conf_plddt") for row in exp_rows if "delta_conf_plddt" in row]
        summaries.append(
            {
                "loss_family": exp["loss_family"],
                "loss_label": exp["label"],
                "n_cases": len(exp_rows),
                "template_lower_best_loss_count": sum(1 for row in exp_rows if row["template_lower_best_loss"]),
                "mean_delta_best_loss": mean(deltas),
                "median_delta_best_loss": median(deltas),
                "mean_pct_delta_best_loss": mean(pct_deltas),
                "median_pct_delta_best_loss": median(pct_deltas),
                "mean_delta_ranking_score": mean(ranking_deltas) if ranking_deltas else math.nan,
                "mean_delta_iptm": mean(iptm_deltas) if iptm_deltas else math.nan,
                "mean_delta_plddt": mean(plddt_deltas) if plddt_deltas else math.nan,
                "mean_delta_elapsed_sec": mean(row["delta_elapsed_sec"] for row in exp_rows),
            }
        )
    return summaries


def write_markdown(rows: list[dict], summaries: list[dict], path: Path) -> None:
    lines = [
        "# Template Effect Analysis",
        "",
        "Paired comparison of no-template vs target-template runs on the eight bc_v1 targets.",
        "For loss values, lower is better. For confidence values such as ranking_score, pLDDT and ipTM, higher is better.",
        "",
        "## Overall Summary",
        "",
        "| Loss family | Cases | Template lower best_loss | Mean delta best_loss | Median delta best_loss | Mean % delta best_loss | Mean delta ranking_score | Mean delta ipTM | Mean delta pLDDT | Mean delta runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {label} | {n} | {wins}/{n} | {d} | {md} | {pd}% | {rs} | {iptm} | {plddt} | {rt} |".format(
                label=item["loss_label"],
                n=item["n_cases"],
                wins=item["template_lower_best_loss_count"],
                d=fmt(item["mean_delta_best_loss"]),
                md=fmt(item["median_delta_best_loss"]),
                pd=fmt(item["mean_pct_delta_best_loss"], 1),
                rs=fmt(item["mean_delta_ranking_score"]),
                iptm=fmt(item["mean_delta_iptm"]),
                plddt=fmt(item["mean_delta_plddt"]),
                rt=fmt(item["mean_delta_elapsed_sec"], 1),
            )
        )
    lines.extend(
        [
            "",
            "## Per-Target Best Loss",
            "",
            "| Loss family | Target | No template | Template | Delta | Delta % | Template better | Ranking delta | ipTM delta | pLDDT delta |",
            "|---|---|---:|---:|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {family} | {case} | {base} | {templ} | {delta} | {pct}% | {better} | {rank} | {iptm} | {plddt} |".format(
                family=row["loss_label"],
                case=row["case"],
                base=fmt(row["base_best_loss"]),
                templ=fmt(row["template_best_loss"]),
                delta=fmt(row["delta_best_loss"]),
                pct=fmt(row["pct_delta_best_loss"], 1),
                better="yes" if row["template_lower_best_loss"] else "no",
                rank=fmt(row.get("delta_conf_ranking_score")),
                iptm=fmt(row.get("delta_conf_iptm")),
                plddt=fmt(row.get("delta_conf_plddt")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_delta_bars(rows: list[dict]) -> None:
    for exp in EXPERIMENTS:
        exp_rows = [row for row in rows if row["loss_family"] == exp["loss_family"]]
        labels = [row["case"] for row in exp_rows]
        values = [row["delta_best_loss"] for row in exp_rows]
        colors = ["#2f9e44" if value < 0 else "#c92a2a" for value in values]
        plt.figure(figsize=(10, 4.8))
        plt.bar(labels, values, color=colors)
        plt.axhline(0, color="#333333", linewidth=1)
        plt.title(f"{exp['label']}: template - no template best_loss")
        plt.ylabel("Delta best_loss")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{exp['loss_family']}_delta_best_loss.png", dpi=200)
        plt.close()


def plot_scatter(rows: list[dict]) -> None:
    """Normalized scatter: each loss family is min-max scaled independently.

    The three loss families use different weighted objectives, so global
    normalization would incorrectly imply their raw best_loss values are
    directly comparable. Per-family normalization keeps template vs
    no-template comparisons within each loss definition.

    Points below the diagonal (y < x) mean template improved (lower loss).
    """
    plt.figure(figsize=(7, 7))

    markers = {"default_loss": "o", "v1_loss": "s", "custom_loss": "^"}
    colors = {"default_loss": "#3498db", "v1_loss": "#e74c3c", "custom_loss": "#2ecc71"}

    for exp in EXPERIMENTS:
        exp_rows = [row for row in rows if row["loss_family"] == exp["loss_family"]]
        exp_vals = []
        for row in exp_rows:
            exp_vals.extend([row["base_best_loss"], row["template_best_loss"]])
        vmin, vmax = min(exp_vals), max(exp_vals)
        rng = vmax - vmin

        def norm(v):
            return (v - vmin) / rng if rng != 0 else 0.0

        x = [norm(row["base_best_loss"]) for row in exp_rows]
        y = [norm(row["template_best_loss"]) for row in exp_rows]
        plt.scatter(x, y, label=exp["label"], marker=markers[exp["loss_family"]], s=70,
                    color=colors[exp["loss_family"]], alpha=0.85)
        for row in exp_rows:
            plt.annotate(row["case"], (norm(row["base_best_loss"]), norm(row["template_best_loss"])), fontsize=7)

    plt.plot([0, 1], [0, 1], color="#555555", linewidth=1.2, linestyle="--", label="y = x (no change)")
    plt.xlabel("No-template best_loss (per-loss min-max normalized)")
    plt.ylabel("Template best_loss (per-loss min-max normalized)")
    plt.title("Template vs no-template best_loss (per-loss normalized)")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "best_loss_scatter.png", dpi=200)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    summaries = summarize(rows)
    write_csv(rows, OUTPUT_DIR / "template_vs_no_template_by_case.csv")
    write_csv(summaries, OUTPUT_DIR / "template_vs_no_template_summary.csv")
    write_markdown(rows, summaries, OUTPUT_DIR / "template_effects_report.md")
    plot_delta_bars(rows)
    plot_scatter(rows)
    print(f"Wrote analysis to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
