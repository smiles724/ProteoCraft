#!/usr/bin/env python3
"""Freeze candidates and join them to existing cleavage/sequencing evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bindcraft.affibody.candidates import (  # noqa: E402
    annotate_candidates,
    frozen_file_record,
    normalize_candidates,
)
from bindcraft.affibody.data import (  # noqa: E402
    read_csv_rows,
    write_csv,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="冻结候选清单，并查询已有 cleavage/测序证据；不会运行模型或实验"
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=PROJECT_ROOT / "data_prepared" / "affibody",
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "datafolder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-output", type=Path, default=None)
    parser.add_argument(
        "--with-sequencing-landscape",
        action="store_true",
        help="额外扫描 R009/R010 全部 pair，计算 target/off-target 测序特异性",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepared_dir = args.prepared_dir.resolve()
    targets_path = prepared_dir / "targets.csv"
    cleavage_path = prepared_dir / "cleavage_labels.csv"
    targets = read_csv_rows(targets_path)
    cleavage = read_csv_rows(cleavage_path)
    candidates = normalize_candidates(
        read_csv_rows(args.candidates), targets=targets
    )
    frozen_path = (
        args.frozen_output.resolve()
        if args.frozen_output is not None
        else args.output.resolve().with_name("candidates_frozen.csv")
    )
    write_csv(frozen_path, candidates)
    annotated = annotate_candidates(
        candidates,
        cleavage_records=cleavage,
        data_dir=args.data_dir.resolve(),
        with_sequencing_landscape=args.with_sequencing_landscape,
    )
    output_path = args.output.resolve()
    write_csv(output_path, annotated)
    write_json(
        output_path.with_suffix(output_path.suffix + ".manifest.json"),
        {
            "schema_version": 1,
            "input_candidates": frozen_file_record(args.candidates),
            "frozen_candidates": frozen_file_record(frozen_path),
            "annotated_candidates": frozen_file_record(output_path),
            "targets": frozen_file_record(targets_path),
            "cleavage_labels": frozen_file_record(cleavage_path),
            "candidate_count": len(candidates),
        },
    )
    print(f"Frozen candidates: {frozen_path}")
    print(f"Annotated candidates: {output_path}")
    print("No model, GPU design, or wet-lab experiment was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
