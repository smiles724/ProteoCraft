#!/usr/bin/env python3
"""Prepare compact, auditable Affibody validation manifests (CPU only)."""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bindcraft.affibody.data import (  # noqa: E402
    extract_cleavage_records,
    extract_target_records,
    sha256_file,
    write_csv,
    write_json,
)
from bindcraft.affibody.profiles import AFFIBODY_PROFILES  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从现有 Excel/PPTX 生成 Affibody 验证用规范表；不会运行模型或实验"
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "datafolder")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data_prepared" / "affibody"
    )
    parser.add_argument("--binder-threshold", type=float, default=75.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    target_source = data_dir / "CoEvo_target_panel_v3.xlsx"
    cleavage_source = data_dir / "Affibody-MHC data summary.pptx"
    missing = [str(path) for path in (target_source, cleavage_source) if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required source file(s): " + ", ".join(missing))

    cleavage = extract_cleavage_records(
        cleavage_source, binder_threshold=args.binder_threshold
    )
    targets = extract_target_records(target_source, cleavage_records=cleavage)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets_path = output_dir / "targets.csv"
    cleavage_path = output_dir / "cleavage_labels.csv"
    blind_complexes_path = output_dir / "known_complexes_blind.csv"
    profiles_path = output_dir / "affibody_profiles.json"
    write_csv(targets_path, targets)
    write_csv(cleavage_path, cleavage)
    write_csv(
        blind_complexes_path,
        [
            {
                "complex_id": row["experiment_id"],
                "library": row["library"],
                "peptide_pair": row["peptide_pair"],
                "peptide": row["experimental_peptide"],
                "register": "P4/P5",
                "affibody_tuple": row["affibody_tuple"],
                "affibody_sequence": row["affibody_sequence"],
            }
            for row in cleavage
        ],
    )
    write_json(
        profiles_path,
        {key: profile.as_dict() for key, profile in AFFIBODY_PROFILES.items()},
    )

    manifest = {
        "schema_version": 1,
        "purpose": "Affibody computational-design validation data preparation",
        "sources": {
            "target_workbook": {
                "path": str(target_source),
                "sha256": sha256_file(target_source),
            },
            "cleavage_powerpoint": {
                "path": str(cleavage_source),
                "sha256": sha256_file(cleavage_source),
            },
        },
        "outputs": {
            path.name: {"path": str(path), "sha256": sha256_file(path)}
            for path in (
                targets_path,
                cleavage_path,
                blind_complexes_path,
                profiles_path,
            )
        },
        "counts": {
            "targets": len(targets),
            "targets_by_tier": dict(Counter(str(row["tier"]) for row in targets)),
            "targets_by_support": dict(
                Counter(str(row["support_level"]) for row in targets)
            ),
            "cleavage_labels": len(cleavage),
            "cleavage_binders": sum(bool(row["cleavage_binder"]) for row in cleavage),
        },
        "environment": {"cwd": os.getcwd()},
    }
    write_json(output_dir / "preparation_manifest.json", manifest)
    print(f"Prepared {len(targets)} targets and {len(cleavage)} cleavage labels")
    print(f"Output: {output_dir}")
    print("No model, GPU design, or wet-lab experiment was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
