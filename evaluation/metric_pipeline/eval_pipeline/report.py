"""Per-case aggregate + markdown report (spec §7)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

NUMERIC_COLS = [
    "complex_pLDDT", "binder_pLDDT", "target_pLDDT", "interface_pLDDT",
    "mean_pAE", "ipAE", "min_ipAE", "i_pTM_approx_from_PAE", "ptm",
    "binder_scRMSD", "binder_fold_RMSD",
    "inter_chain_clash_count", "n_contacts_5A", "radius_of_gyration_binder",
    "esmfold_runtime_s",
    "n_generated_trajectories", "n_mpnn_attempted", "n_accepted_by_method",
    "candidate_rank", "verifier_calls", "design_gpu_seconds",
    "eval_n_calls", "eval_n_successes", "eval_runtime_s",
    "af3_ranking_score", "af3_iptm", "af3_ptm_native",
    "af3_has_clash", "af3_fraction_disordered",
]
BOOL_COLS = [
    "tier1_pass", "tier2_pass",
    "tier1_plddt80_pass", "tier1_plddt90_pass",
    "tier2_complexa_pass", "tier2_strict_physical_pass",
]


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric + bool columns after read with keep_default_na=False."""
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().isin(["true", "1", "yes"])
    return df


def load_long_csv(long_csv: str) -> pd.DataFrame:
    """Load + coerce an eval CSV."""
    # v1.1 reviewer P1 #5: keep_default_na=False so error_msg "" stays "" (not NaN)
    df = pd.read_csv(long_csv, keep_default_na=False, na_values=["nan", "NaN"])
    return _coerce(df)


def aggregate_per_case(df: pd.DataFrame) -> pd.DataFrame:
    def _agg(g):
        n = len(g)
        n_t1 = int(g["tier1_pass"].sum())
        n_t1_loose = int(g["tier1_plddt80_pass"].sum()) if "tier1_plddt80_pass" in g.columns else 0
        n_t2 = int(g["tier2_pass"].sum())
        n_t2_strict = int(g["tier2_strict_physical_pass"].sum()) if "tier2_strict_physical_pass" in g.columns else 0
        n_gen_series = g["n_generated_trajectories"]
        n_gen = int(n_gen_series.iloc[0]) if (n_gen_series >= 0).all() and not n_gen_series.empty else -1
        n_acc_series = g["n_accepted_by_method"]
        n_acc = int(n_acc_series.iloc[0]) if (n_acc_series >= 0).all() and not n_acc_series.empty else -1
        n_err = int(g["error_msg"].astype(str).str.len().gt(0).sum())
        return pd.Series({
            "n_designs_evaluated": n,
            "n_generated_trajectories": n_gen,
            "n_accepted_by_method": n_acc,
            "n_tier1_pass": n_t1,
            "n_tier1_plddt80_pass": n_t1_loose,
            "n_tier2_pass": n_t2,
            "n_tier2_strict_pass": n_t2_strict,
            "tier1_pass_rate_over_evaluated": (n_t1 / n) if n else 0.0,
            "tier1_loose_rate_over_evaluated": (n_t1_loose / n) if n else 0.0,
            "tier1_pass_rate_over_generated": (n_t1 / n_gen) if n_gen > 0 else float("nan"),
            "tier2_pass_rate_over_evaluated": (n_t2 / n) if n else 0.0,
            "tier2_pass_rate_over_generated": (n_t2 / n_gen) if n_gen > 0 else float("nan"),
            "best_binder_pLDDT": float(g["binder_pLDDT"].max()) if g["binder_pLDDT"].notna().any() else float("nan"),
            "best_complex_pLDDT": float(g["complex_pLDDT"].max()) if g["complex_pLDDT"].notna().any() else float("nan"),
            "best_ipAE_min": float(g["ipAE"].min()) if g["ipAE"].notna().any() else float("nan"),
            "best_binder_scRMSD": float(g["binder_scRMSD"].min()) if g["binder_scRMSD"].notna().any() else float("nan"),
            "mean_esmfold_runtime_s": float(g["esmfold_runtime_s"].mean()) if g["esmfold_runtime_s"].notna().any() else float("nan"),
            "n_errors": n_err,
        })

    return df.groupby(["method", "case"]).apply(_agg, include_groups=False).reset_index()


def compute_bio_summary(df: pd.DataFrame) -> dict:
    """Whole-batch decomposition that disentangles 'binder folds correctly'
    from 'ESMFold docks binder at target interface'. See report bio section."""
    n = len(df)
    def _safe_count(series, op):
        s = pd.to_numeric(series, errors="coerce")
        return int(op(s).sum())
    return {
        "n_total": n,
        "n_bplddt_gt_90": _safe_count(df["binder_pLDDT"], lambda s: s > 90),
        "n_bplddt_gt_80": _safe_count(df["binder_pLDDT"], lambda s: s > 80),
        "n_cplddt_gt_90": _safe_count(df["complex_pLDDT"], lambda s: s > 90),
        "n_ipae_lt_7":    _safe_count(df["ipAE"], lambda s: s < 7),
        "n_scrmsd_lt_1p5": _safe_count(df["binder_scRMSD"], lambda s: s < 1.5),
        "n_clash_eq_0":   _safe_count(df["inter_chain_clash_count"], lambda s: s == 0),
        "n_fold_lt_1p5":  _safe_count(df["binder_fold_RMSD"], lambda s: s < 1.5),
        "n_contacts_gt_0": _safe_count(df["n_contacts_5A"], lambda s: s > 0),
        "scrmsd_median":  float(pd.to_numeric(df["binder_scRMSD"], errors="coerce").median()),
    }


def get_top_designs(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Top-K designs ranked by binder_pLDDT (the most informative single metric)."""
    cols = ["case", "design_id", "binder_pLDDT", "complex_pLDDT",
            "ipAE", "binder_scRMSD", "n_contacts_5A", "inter_chain_clash_count"]
    cols = [c for c in cols if c in df.columns]
    sub = df.dropna(subset=["binder_pLDDT"]).nlargest(k, "binder_pLDDT")[cols]
    return sub.reset_index(drop=True)


def normalize_eval_schema(df: pd.DataFrame, source_path: Path | None = None) -> pd.DataFrame:
    """Normalize legacy and per-sub-run eval CSV schema for paper_case aggregation."""
    df = df.copy()
    if "evaluator" not in df.columns:
        df["evaluator"] = "esmfold"
    if "case" not in df.columns:
        df["case"] = "unknown"
    if "paper_case" not in df.columns:
        df["paper_case"] = df["case"]
    if "eval_n_calls" not in df.columns:
        df["eval_n_calls"] = 1 if len(df) > 0 else 0
    if "eval_runtime_s" not in df.columns:
        df["eval_runtime_s"] = (
            df["esmfold_runtime_s"] if "esmfold_runtime_s" in df.columns else 0.0
        )
    if "eval_n_successes" not in df.columns:
        if len(df) > 0:
            err = (
                df.get("error_msg", pd.Series([""] * len(df), index=df.index))
                .fillna("")
                .astype(str)
            )
            ok_metrics = pd.Series([True] * len(df), index=df.index)
            for col in ("binder_pLDDT", "ipAE", "binder_scRMSD"):
                if col in df.columns:
                    ok_metrics = ok_metrics & df[col].notna()
            df["eval_n_successes"] = (err.eq("") & ok_metrics).astype(int)
        else:
            df["eval_n_successes"] = 0
    for col in ("tier1_pass", "tier2_pass"):
        if col not in df.columns:
            df[col] = False
    if "design_id" not in df.columns:
        df["design_id"] = "unknown_legacy"
    return df


def fill_missing_paper_cases(
    df: pd.DataFrame,
    roster: list[str],
    evaluators: list[str],
) -> pd.DataFrame:
    """Append _NO_DESIGNS stub rows for missing evaluator x paper_case combos."""
    df = normalize_eval_schema(df)
    stubs = []
    for evaluator in evaluators:
        sub = df[df["evaluator"] == evaluator] if len(df) > 0 else df
        present = set(sub["paper_case"].unique()) if len(sub) > 0 else set()
        for paper_case in sorted(set(roster) - present):
            stub = {col: float("nan") for col in df.columns}
            stub.update({
                "evaluator": evaluator,
                "method": "BCv1",
                "case": paper_case,
                "paper_case": paper_case,
                "design_id": "_NO_DESIGNS",
                "n_accepted_by_method": 0,
                "eval_n_calls": 0,
                "eval_n_successes": 0,
                "eval_runtime_s": 0.0,
                "tier1_pass": False,
                "tier2_pass": False,
                "error_msg": "case_has_zero_accepted_designs",
            })
            stubs.append(stub)
    if stubs:
        return pd.concat([df, pd.DataFrame(stubs)], ignore_index=True)
    return df


def aggregate_per_paper_case(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate evaluator x method x paper_case, with _NO_DESIGNS as zero denominator."""
    df = normalize_eval_schema(df)
    is_stub = df["design_id"] == "_NO_DESIGNS"
    real = df[~is_stub]
    stub_keys = (
        df[is_stub][["evaluator", "method", "paper_case"]].drop_duplicates()
        if is_stub.any() else pd.DataFrame()
    )

    if len(real) > 0:
        real_agg = real.groupby(["evaluator", "method", "paper_case"]).agg(
            n_designs_evaluated=("design_id", "count"),
            n_calls=("eval_n_calls", "sum"),
            n_successes=("eval_n_successes", "sum"),
            success_rate=("eval_n_successes", lambda s: s.sum() / max(len(s), 1)),
            eval_runtime_s_total=("eval_runtime_s", "sum"),
            tier1_rate=("tier1_pass", "mean"),
            tier2_rate=("tier2_pass", "mean"),
        ).reset_index()
    else:
        real_agg = pd.DataFrame(columns=[
            "evaluator", "method", "paper_case", "n_designs_evaluated", "n_calls",
            "n_successes", "success_rate", "eval_runtime_s_total",
            "tier1_rate", "tier2_rate",
        ])

    if not stub_keys.empty:
        stub_agg = stub_keys.copy()
        stub_agg["n_designs_evaluated"] = 0
        stub_agg["n_calls"] = 0
        stub_agg["n_successes"] = 0
        stub_agg["success_rate"] = 0.0
        stub_agg["eval_runtime_s_total"] = 0.0
        stub_agg["tier1_rate"] = float("nan")
        stub_agg["tier2_rate"] = float("nan")
        if real_agg.empty:
            return stub_agg
        return pd.concat([real_agg, stub_agg], ignore_index=True)
    return real_agg


BIO_SUMMARY_COLS = {
    "binder_pLDDT", "complex_pLDDT", "ipAE", "binder_scRMSD",
    "inter_chain_clash_count", "binder_fold_RMSD", "n_contacts_5A",
}

AF3_NATIVE_COLS = [
    "af3_ranking_score", "af3_iptm", "af3_ptm_native",
    "af3_has_clash", "af3_fraction_disordered",
]


def _af3_native_metric_lines(df: pd.DataFrame) -> list[str]:
    present = [col for col in AF3_NATIVE_COLS if col in df.columns]
    if len(df) == 0 or not present:
        return []

    native = df[present].apply(pd.to_numeric, errors="coerce")
    if not native.notna().any().any():
        return []

    lines = ["## AF3 native metrics", ""]
    for col in ("af3_ranking_score", "af3_iptm", "af3_ptm_native"):
        if col in native.columns and native[col].notna().any():
            lines.append(f"- `{col} mean`: {native[col].mean():.3f}")
    if "af3_has_clash" in native.columns and native["af3_has_clash"].notna().any():
        clash = native["af3_has_clash"].dropna()
        lines.append(f"- `af3_has_clash != 0`: {int(clash.ne(0).sum())}/{len(clash)}")
    if (
        "af3_fraction_disordered" in native.columns
        and native["af3_fraction_disordered"].notna().any()
    ):
        lines.append(
            f"- `af3_fraction_disordered mean`: "
            f"{native['af3_fraction_disordered'].mean():.3f}"
        )
    lines.append("")
    return lines


def render_markdown_for_evaluator(
    sub_agg: pd.DataFrame,
    sub_df: pd.DataFrame,
    run_id: str,
    evaluator: str,
    out_md: str,
) -> None:
    """Render spec §9.5 per-evaluator markdown with a chart link."""
    method_values = (
        sorted(str(method) for method in sub_df["method"].dropna().unique())
        if "method" in sub_df.columns and len(sub_df) > 0 else []
    )
    methods = ", ".join(method_values) if method_values else "-"
    n_designs = int(sub_agg["n_designs_evaluated"].sum()) if "n_designs_evaluated" in sub_agg else 0
    tier1_mean = (
        float(sub_agg["tier1_rate"].mean())
        if "tier1_rate" in sub_agg and sub_agg["tier1_rate"].notna().any()
        else float("nan")
    )
    tier2_mean = (
        float(sub_agg["tier2_rate"].mean())
        if "tier2_rate" in sub_agg and sub_agg["tier2_rate"].notna().any()
        else float("nan")
    )
    aggregate_table = (
        sub_agg.to_markdown(index=False, floatfmt=".3f")
        if len(sub_agg) > 0 else "_(empty)_"
    )

    lines = [
        f"# {run_id} - evaluator: {evaluator}",
        "",
        "## TL;DR",
        "",
        f"- Methods: {methods}",
        f"- paper_cases evaluated: {len(sub_agg)}",
        f"- Total designs evaluated: {n_designs}",
        f"- Mean Tier-1 rate: {tier1_mean:.3f}" if not np.isnan(tier1_mean) else "- Mean Tier-1 rate: NaN",
        f"- Mean Tier-2 rate: {tier2_mean:.3f}" if not np.isnan(tier2_mean) else "- Mean Tier-2 rate: NaN",
        "",
        "## Per-paper_case aggregate",
        "",
        aggregate_table,
        "",
    ]

    if len(sub_df) > 0 and "binder_pLDDT" in sub_df.columns:
        top = get_top_designs(sub_df, k=5)
        if len(top) > 0:
            lines.extend([
                "## Top designs",
                "",
                top.to_markdown(index=False, floatfmt=".2f"),
                "",
            ])

    lines.extend([
        "## Biological interpretation",
        "",
    ])

    af3_native_lines = (
        _af3_native_metric_lines(sub_df)
        if evaluator == "af3" else []
    )
    if len(sub_df) > 0 and BIO_SUMMARY_COLS <= set(sub_df.columns):
        bio = compute_bio_summary(sub_df)
        lines.extend([
            f"- `binder_pLDDT > 90`: {bio['n_bplddt_gt_90']}/{bio['n_total']}",
            f"- `binder_pLDDT > 80`: {bio['n_bplddt_gt_80']}/{bio['n_total']}",
            f"- `complex_pLDDT > 90`: {bio['n_cplddt_gt_90']}/{bio['n_total']}",
            f"- `ipAE < 7`: {bio['n_ipae_lt_7']}/{bio['n_total']}",
            f"- `binder_scRMSD < 1.5`: {bio['n_scrmsd_lt_1p5']}/{bio['n_total']}",
            f"- `inter_chain_clash == 0`: {bio['n_clash_eq_0']}/{bio['n_total']}",
            f"- `n_contacts_5A > 0`: {bio['n_contacts_gt_0']}/{bio['n_total']}",
            f"- `binder_scRMSD median`: {bio['scrmsd_median']:.2f}",
        ])
    elif af3_native_lines:
        lines.append("- AF3 native metrics are listed below.")
    else:
        lines.append("- Component metrics unavailable for this evaluator/schema.")

    if af3_native_lines:
        lines.extend(["", *af3_native_lines])

    lines.extend([
        "",
        "## Bar chart",
        "",
        f"![pass rates](bar_chart_{evaluator}.png)",
        "",
    ])
    Path(out_md).write_text("\n".join(lines) + "\n")


def render_bar_chart_paper_case(sub_agg: pd.DataFrame, evaluator: str, out_png: str) -> None:
    """Render spec §9.5 per-evaluator paper_case pass-rate chart.

    Spec §7.4 keeps NaN rates distinct from 0% pass rates: NaN means there
    were no accepted designs for that paper_case, not that evaluated designs
    all failed.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    if "paper_case" not in sub_agg.columns or len(sub_agg) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_png, dpi=140)
        plt.close(fig)
        return

    cases = sub_agg["paper_case"].astype(str).tolist()
    x = np.arange(len(cases))
    w = 0.4
    tier1_raw = (
        sub_agg["tier1_rate"]
        if "tier1_rate" in sub_agg else pd.Series([float("nan")] * len(sub_agg), index=sub_agg.index)
    )
    tier2_raw = (
        sub_agg["tier2_rate"]
        if "tier2_rate" in sub_agg else pd.Series([float("nan")] * len(sub_agg), index=sub_agg.index)
    )
    tier1_mask = tier1_raw.notna().to_numpy()
    tier2_mask = tier2_raw.notna().to_numpy()
    tier1 = (tier1_raw.fillna(0.0).astype(float) * 100).to_numpy()
    tier2 = (tier2_raw.fillna(0.0).astype(float) * 100).to_numpy()

    if tier1_mask.any():
        ax.bar((x - w / 2)[tier1_mask], tier1[tier1_mask], w, label="Tier-1 rate")
    if tier2_mask.any():
        ax.bar((x + w / 2)[tier2_mask], tier2[tier2_mask], w, label="Tier-2 rate")

    no_data = ~(tier1_mask | tier2_mask)
    if no_data.any():
        ax.bar(x[no_data], np.full(no_data.sum(), 1.0), w * 2, color="lightgray",
               alpha=0.4, label="no designs")
        for xi in x[no_data]:
            ax.text(xi, 2, "no designs", ha="center", va="bottom",
                    fontsize=8, color="gray", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("pass rate (%)")
    ax.set_title(f"{evaluator} pass rates by paper_case")
    if tier1_mask.any() or tier2_mask.any() or no_data.any():
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def render_table2(agg: pd.DataFrame, out_csv: str, out_md: str) -> None:
    """Render spec §9.5 combined Table 2 as paper_case rows x method/evaluator columns."""
    if len(agg) == 0:
        Path(out_csv).write_text("paper_case\n")
        Path(out_md).write_text("# Table 2\n\n_(empty)_\n")
        return

    pivot_metrics = [
        "tier1_rate", "tier2_rate", "success_rate", "n_designs_evaluated",
    ]
    pivot_keys = [key for key in ("method", "evaluator") if key in agg.columns]
    cols_to_keep = [
        col for col in ["paper_case", *pivot_keys, *pivot_metrics]
        if col in agg.columns
    ]
    sub = agg[cols_to_keep].copy()
    if not pivot_keys:
        wide = sub.copy()
        wide.to_csv(out_csv, index=False)
        md_lines = [
            "# Table 2: BCv2 evaluator pass rates by paper_case",
            "",
            wide.to_markdown(index=False, floatfmt=".3f"),
        ]
        Path(out_md).write_text("\n".join(md_lines) + "\n")
        return

    duplicate_keys = ["paper_case", *pivot_keys]
    dup_mask = sub.duplicated(subset=duplicate_keys, keep=False)
    if dup_mask.any():
        first_dup = sub.loc[dup_mask, duplicate_keys].iloc[0].to_dict()
        raise ValueError(
            f"render_table2: duplicate ({', '.join(duplicate_keys)}) rows in agg; "
            f"first dup: {first_dup}"
        )

    wide = sub.pivot(index="paper_case", columns=pivot_keys, values=pivot_metrics)
    wide.columns = [
        "_".join([*(str(value) for value in key_values), str(metric)])
        for metric, *key_values in wide.columns.to_flat_index()
    ]
    wide = wide.reset_index()
    wide.to_csv(out_csv, index=False)

    md_lines = [
        "# Table 2: BCv2 evaluator pass rates by paper_case",
        "",
        wide.to_markdown(index=False, floatfmt=".3f"),
    ]
    Path(out_md).write_text("\n".join(md_lines) + "\n")


def main():
    import json as _json

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--long-csv",
        nargs="+",
        default=None,
        help="one or more eval CSV paths",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--auto-discover",
        action="store_true",
        help="glob <out-dir>/eval/*.csv",
    )
    ap.add_argument(
        "--case-roster",
        default=None,
        help="path to bcv2_case_roster.json for missing-case stubs",
    )
    ap.add_argument(
        "--evaluators",
        nargs="+",
        default=["esmfold", "af3"],
        help=(
            "evaluators expected during roster fill (default assumes BCv2 "
            "esmfold+af3 protocol; pass explicit list for other evaluators)"
        ),
    )
    ap.add_argument("--run-id", default="report")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.long_csv:
        csv_paths = [Path(p) for p in args.long_csv]
        missing = [path for path in csv_paths if not path.exists()]
        if missing:
            raise SystemExit(
                f"--long-csv: {len(missing)} path(s) do not exist: "
                + ", ".join(str(path) for path in missing)
            )
    elif args.auto_discover:
        csv_paths = sorted(out.glob("eval/*.csv"))
        if not csv_paths:
            raise SystemExit(f"--auto-discover: no csv found in {out / 'eval'}")
    else:
        raise SystemExit("provide --long-csv <paths> or --auto-discover")

    dfs = [
        normalize_eval_schema(load_long_csv(str(path)), source_path=path)
        for path in csv_paths
    ]
    total_rows = sum(len(df) for df in dfs)
    print(f"loaded {len(csv_paths)} csv(s), {total_rows} total rows")
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    df = normalize_eval_schema(df)

    if args.case_roster:
        roster_obj = _json.loads(Path(args.case_roster).read_text())
        df = fill_missing_paper_cases(
            df,
            roster=list(roster_obj["paper_cases"]),
            evaluators=args.evaluators,
        )

    agg = aggregate_per_paper_case(df)
    agg_path = out / "eval_per_paper_case_aggregate.csv"
    agg.to_csv(agg_path, index=False)

    if len(agg) > 0:
        for evaluator in sorted(agg["evaluator"].dropna().unique()):
            sub_agg = agg[agg["evaluator"] == evaluator]
            sub_df = df[df["evaluator"] == evaluator]
            render_markdown_for_evaluator(
                sub_agg,
                sub_df,
                args.run_id,
                evaluator,
                str(out / f"report_{evaluator}.md"),
            )
            render_bar_chart_paper_case(
                sub_agg,
                evaluator,
                str(out / f"bar_chart_{evaluator}.png"),
            )
    render_table2(agg, str(out / "table2.csv"), str(out / "table2.md"))

    print(f"wrote {agg_path} (n_rows={len(agg)})")


if __name__ == "__main__":
    main()
