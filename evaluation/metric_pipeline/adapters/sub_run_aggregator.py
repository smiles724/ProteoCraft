"""Sub-run to paper_case aggregation before evaluator execution."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.bcv1_adapter import (  # noqa: E402
    BCV1_SUB_RUN_TO_PAPER_CASE as SUB_RUN_TO_PAPER_CASE,
)
from adapters.bcv1_adapter import _design_id_to_design_key  # noqa: E402

BETV1_RULE_CASES = {"BetV1"}
OR_MERGE_CASES = {"DerF7", "DerF21", "HER2"}
TOP_K = 100
RANKING_KEY = "design_avg_iptm"
PASSTHROUGH_PAPER_CASES = {
    "PD1", "PDL1", "IFNAR2", "SpCas9",
    "TCR_1QRN", "TCR_3QDJ", "TCR_5BS0", "TCR_6BJ8", "TCR_9C3E",
}
KNOWN_PAPER_CASES = set(SUB_RUN_TO_PAPER_CASE.values()) | PASSTHROUGH_PAPER_CASES


def _sub_run_from_manifest_path(path: str | Path) -> str:
    name = Path(path).name
    for suffix in (".raw.manifest.csv", ".selected.manifest.csv", ".manifest.csv", ".csv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def aggregate_sub_runs(
    sub_run_manifests: dict[str, pd.DataFrame],
    bindcraft_stats: dict[str, pd.DataFrame],
    out_dir: Path,
    return_counts: bool = False,
) -> dict[str, Path] | tuple[dict[str, Path], dict[str, int]]:
    """Aggregate sub-run manifests into selected paper_case manifests."""
    out_dir = Path(out_dir)
    paper_cases: dict[str, list[pd.DataFrame]] = {}

    for sub_run, manifest_df in sub_run_manifests.items():
        if sub_run not in bindcraft_stats:
            raise KeyError(f"bindcraft_stats missing sub_run={sub_run}")
        stats = bindcraft_stats[sub_run]
        stats_lookup = stats.set_index("Design")["Average_i_pTM"].to_dict()

        df = manifest_df.copy()
        df["_design_key"] = df["design_id"].map(_design_id_to_design_key)
        df[RANKING_KEY] = df["_design_key"].map(stats_lookup)

        nan_rate = df[RANKING_KEY].isna().mean() if len(df) else 0.0
        if nan_rate > 0.5:
            raise RuntimeError(
                f"sub_run={sub_run}: {RANKING_KEY} NaN rate {nan_rate:.0%}, "
                f"likely design_id to stats.Design key mismatch"
            )

        df = df.drop(columns=["_design_key"])
        paper_case = SUB_RUN_TO_PAPER_CASE.get(sub_run, sub_run)
        if paper_case not in KNOWN_PAPER_CASES:
            raise KeyError(
                f"Unknown sub_run/paper_case {sub_run!r} (resolved to {paper_case!r}); "
                f"allowed paper_cases: {sorted(KNOWN_PAPER_CASES)}"
            )
        paper_cases.setdefault(paper_case, []).append(df)

    out_paths: dict[str, Path] = {}
    out_counts: dict[str, int] = {}
    for paper_case, frames in paper_cases.items():
        merged = pd.concat(frames, ignore_index=True)
        if len(merged) == 0:
            kept = merged.copy()
        elif paper_case in BETV1_RULE_CASES:
            sub_yields = merged.groupby("case")["n_accepted_by_method"].max()
            best_yield = sub_yields.max()
            candidates = sorted(sub_yields[sub_yields == best_yield].index.tolist())
            if len(candidates) > 1:
                print(
                    f"WARNING: BetV1 yield tie at {best_yield} between {candidates}; "
                    f"choosing alphabetically first: {candidates[0]}"
                )
            best_sub_run = candidates[0]
            kept = merged[merged["case"] == best_sub_run].copy()
        elif paper_case in OR_MERGE_CASES:
            kept = merged.sort_values(
                [RANKING_KEY, "design_id"],
                ascending=[False, True],
                kind="mergesort",
                na_position="last",
            ).head(TOP_K).copy()
        else:
            kept = merged.copy()

        kept["paper_case"] = paper_case
        out_path = out_dir / "manifests" / f"{paper_case}.selected.manifest.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        kept.to_csv(out_path, index=False)
        out_paths[paper_case] = out_path
        out_counts[paper_case] = len(kept)
    if return_counts:
        return out_paths, out_counts
    return out_paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub-run-manifests", nargs="+", required=True)
    ap.add_argument(
        "--bindcraft-outputs",
        required=True,
        help="root containing <sub_run>/final_design_stats.csv per sub-run",
    )
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    sub_run_manifests = {}
    bindcraft_stats = {}
    for path in args.sub_run_manifests:
        df = pd.read_csv(path)
        if "case" not in df.columns:
            print(f"skip headerless: {path}")
            continue
        sub_run = df["case"].iloc[0] if len(df) > 0 else _sub_run_from_manifest_path(path)
        sub_run_manifests[sub_run] = df
        stats_path = Path(args.bindcraft_outputs) / sub_run / "final_design_stats.csv"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"missing BindCraft stats for sub_run={sub_run!r}: {stats_path}\n"
                "expected layout: <bindcraft-outputs>/<sub_run>/final_design_stats.csv\n"
                f"(--bindcraft-outputs={args.bindcraft_outputs})"
            )
        bindcraft_stats[sub_run] = pd.read_csv(stats_path)

    out_paths, out_counts = aggregate_sub_runs(
        sub_run_manifests,
        bindcraft_stats,
        Path(args.out_dir),
        return_counts=True,
    )
    for paper_case, path in sorted(out_paths.items()):
        n_rows = out_counts[paper_case]
        print(f"  {paper_case}: {n_rows} rows -> {path}")


if __name__ == "__main__":
    main()
