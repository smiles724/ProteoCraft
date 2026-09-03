#!/usr/bin/env python3
"""Collect best sequences from completed Affibody ESMFold2 run summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bindcraft.affibody.data import write_csv  # noqa: E402
from bindcraft.affibody.profiles import get_affibody_profile  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="收集已完成设计的最优 Affibody 序列")
    parser.add_argument("--runs", type=Path, required=True, help="design_runs.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--top-per-run",
        type=int,
        default=10,
        help="从每个 candidates.jsonl 按 verifier loss 保留多少个去重候选",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records: list[dict[str, object]] = []
    with args.runs.open("r", encoding="utf-8") as handle:
        runs = [json.loads(line) for line in handle if line.strip()]
    missing: list[str] = []
    if args.top_per_run < 1:
        raise ValueError("--top-per-run must be positive")
    seen_global: set[tuple[str, str, str]] = set()
    for run in runs:
        if run.get("model_backend") != "esmfold2":
            raise ValueError(
                f"{run.get('run_id', '<unknown>')} is not an ESMFold2 Affibody run"
            )
        if run.get("task_type") != "affibody_redesign":
            raise ValueError(
                f"{run.get('run_id', '<unknown>')} has unexpected task_type: "
                f"{run.get('task_type')!r}"
            )
        summary_path = Path(run["artifact_summary"])
        if not summary_path.is_file():
            missing.append(str(summary_path))
            continue
        profile = get_affibody_profile(run["library"])
        candidate_path = summary_path.with_name("candidates.jsonl")
        candidates: dict[str, dict[str, object]] = {}
        if candidate_path.is_file():
            with candidate_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    candidate = json.loads(line)
                    sequence = str(candidate.get("sequence", "")).upper()
                    if not sequence:
                        continue
                    loss = float(candidate.get("total_loss", float("inf")))
                    previous = candidates.get(sequence)
                    if previous is None or loss < float(previous["total_loss"]):
                        candidates[sequence] = candidate
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        best_sequence = str(
            summary.get("best_sequence")
            or summary.get("selected_sequence")
            or summary.get("binder_seq")
            or ""
        ).upper()
        if best_sequence and best_sequence not in candidates:
            candidates[best_sequence] = {
                "sequence": best_sequence,
                "total_loss": summary.get("best_loss", float("inf")),
                "stage": "run_summary",
            }
        ordered = sorted(
            candidates.values(), key=lambda candidate: float(candidate["total_loss"])
        )[: args.top_per_run]
        for local_rank, candidate in enumerate(ordered, start=1):
            sequence = profile.validate_sequence(str(candidate["sequence"]).upper())
            key = (str(run["target_id"]), profile.name, sequence)
            if key in seen_global:
                continue
            seen_global.add(key)
            records.append(
                {
                    "candidate_id": f"C{len(records) + 1:04d}",
                    "target_id": run["target_id"],
                    "library": profile.name,
                    "affibody_sequence": sequence,
                    "affibody_tuple": profile.extract_tuple(sequence),
                    "design_score": candidate.get("total_loss", ""),
                    "design_rank": local_rank,
                    "source_run": run["run_id"],
                    "source_summary": str(summary_path),
                    "source_stage": candidate.get("stage", ""),
                }
            )
    write_csv(args.output, records)
    print(f"Collected {len(records)} candidate(s): {args.output.resolve()}")
    if missing:
        print(f"Skipped {len(missing)} unfinished run(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
