#!/usr/bin/env python3
"""Summarize on-target versus decoy scores for frozen Affibody candidates."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bindcraft.affibody.data import read_csv_rows, sha256_file, write_csv, write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="计算 frozen Affibody 的 target-decoy specificity margin")
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-key", default="pair_chain_iptm")
    parser.add_argument("--lower-is-better", action="store_true")
    return parser


def _score(record: dict, key: str) -> float:
    if key in record:
        return float(record[key])
    for container in ("summary_confidence", "loss_dict"):
        if key in record.get(container, {}):
            return float(record[container][key])
    raise KeyError(f"score key {key!r} missing for {record.get('complex_id')}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    panel = {row["complex_id"]: row for row in read_csv_rows(args.panel)}
    score_values: dict[str, list[float]] = defaultdict(list)
    hotspot_values: dict[str, list[float]] = defaultdict(list)
    with args.scores.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            complex_id = str(record["complex_id"])
            if complex_id not in panel:
                raise ValueError(f"score has unknown decoy complex ID: {complex_id}")
            value = _score(record, args.score_key)
            score_values[complex_id].append(-value if args.lower_is_better else value)
            summary = record.get("summary_confidence", {})
            for key in (
                "hotspot_heavy_atom_coverage_5a",
                "hotspot_heavy_atom_coverage_near",
            ):
                if key in summary:
                    hotspot_values[complex_id].append(float(summary[key]))
                    break
    missing = sorted(set(panel) - set(score_values))
    if missing:
        raise ValueError(f"panel complexes lack scores: {missing[:10]}")
    detailed: list[dict[str, object]] = []
    for complex_id, row in panel.items():
        detailed.append(
            {
                **row,
                "model_score": statistics.fmean(score_values[complex_id]),
                "score_seed_count": len(score_values[complex_id]),
                "mean_hotspot_coverage": (
                    statistics.fmean(hotspot_values[complex_id])
                    if hotspot_values[complex_id]
                    else ""
                ),
            }
        )
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in detailed:
        grouped[str(row["candidate_id"])].append(row)
    summary_rows: list[dict[str, object]] = []
    for candidate_id, rows in sorted(grouped.items()):
        on_target = [row for row in rows if str(row["is_on_target"]).lower() == "true"]
        off_target = [row for row in rows if str(row["is_on_target"]).lower() != "true"]
        if len(on_target) != 1 or not off_target:
            raise ValueError(f"candidate {candidate_id} needs one target and at least one decoy")
        target_score = float(on_target[0]["model_score"])
        best_off = max(off_target, key=lambda row: float(row["model_score"]))
        ordered = sorted(rows, key=lambda row: float(row["model_score"]), reverse=True)
        summary_rows.append(
            {
                "candidate_id": candidate_id,
                "on_target_id": on_target[0]["target_id"],
                "target_score": target_score,
                "best_off_target_id": best_off["target_id"],
                "best_off_target_score": best_off["model_score"],
                "specificity_margin": target_score - float(best_off["model_score"]),
                "target_rank_among_panel": next(
                    index + 1
                    for index, row in enumerate(ordered)
                    if str(row["is_on_target"]).lower() == "true"
                ),
                "panel_size": len(rows),
            }
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "specificity_scores_detailed.csv", detailed)
    write_csv(output_dir / "specificity_summary.csv", summary_rows)
    write_json(
        output_dir / "specificity_manifest.json",
        {
            "schema_version": 1,
            "panel_sha256": sha256_file(args.panel),
            "scores_sha256": sha256_file(args.scores),
            "score_key": args.score_key,
            "candidate_count": len(summary_rows),
        },
    )
    print(f"Specificity summary: {output_dir / 'specificity_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
