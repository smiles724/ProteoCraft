#!/usr/bin/env python3
"""Build a fixed target/off-target pMHC panel for frozen Affibody candidates."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bindcraft.affibody.data import read_csv_rows, sha256_file, write_csv, write_json  # noqa: E402
from bindcraft.affibody.profiles import get_affibody_profile  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="为冻结候选建立同 HLA、不同 peptide 的 decoy panel")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--targets",
        type=Path,
        default=PROJECT_ROOT / "data_prepared" / "affibody" / "targets.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--decoy-target-id",
        action="append",
        default=[],
        help="显式 decoy target ID，可重复或逗号分隔；不传则按固定 seed 抽取",
    )
    parser.add_argument("--max-decoys", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_decoys < 1:
        raise ValueError("--max-decoys must be positive")
    candidates = read_csv_rows(args.candidates)
    targets = read_csv_rows(args.targets)
    target_by_id = {row["target_id"]: row for row in targets}
    requested = {
        token.strip()
        for value in args.decoy_target_id
        for token in value.split(",")
        if token.strip()
    }
    unknown = requested - set(target_by_id)
    if unknown:
        raise ValueError(f"unknown decoy target ID(s): {sorted(unknown)}")
    rng = random.Random(args.seed)
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        on_target_id = candidate["target_id"]
        if on_target_id not in target_by_id:
            raise ValueError(f"candidate {candidate_id} has unknown target {on_target_id}")
        profile = get_affibody_profile(candidate["library"])
        sequence = profile.validate_sequence(candidate["affibody_sequence"])
        if requested:
            decoy_ids = sorted(requested - {on_target_id})[: args.max_decoys]
        else:
            pool = [target_id for target_id in target_by_id if target_id != on_target_id]
            rng.shuffle(pool)
            controls = [target_id for target_id in ("T001", "T003") if target_id in pool]
            decoy_ids = list(dict.fromkeys(controls + pool))[: args.max_decoys]
        if not decoy_ids:
            raise ValueError(
                f"candidate {candidate_id} has no off-target after decoy selection"
            )
        for target_id in [on_target_id, *decoy_ids]:
            target = target_by_id[target_id]
            rows.append(
                {
                    "complex_id": f"{candidate_id}__{target_id}",
                    "candidate_id": candidate_id,
                    "target_id": target_id,
                    "on_target_id": on_target_id,
                    "is_on_target": target_id == on_target_id,
                    "library": profile.name,
                    "peptide_pair": target["up_facing_pair"],
                    "peptide": target["peptide"],
                    "register": target["register"],
                    "affibody_tuple": profile.extract_tuple(sequence),
                    "affibody_sequence": sequence,
                }
            )
    write_csv(args.output, rows)
    write_json(
        args.output.with_suffix(args.output.suffix + ".manifest.json"),
        {
            "schema_version": 1,
            "candidate_file_sha256": sha256_file(args.candidates),
            "target_file_sha256": sha256_file(args.targets),
            "candidate_count": len(candidates),
            "complex_count": len(rows),
            "max_decoys": args.max_decoys,
            "selection_seed": args.seed,
            "explicit_decoy_ids": sorted(requested),
        },
    )
    print(f"Decoy panel: {len(rows)} complexes -> {args.output.resolve()}")
    print("No model or experiment was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
