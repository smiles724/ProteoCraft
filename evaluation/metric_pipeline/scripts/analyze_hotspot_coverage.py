#!/usr/bin/env python3
"""Calculate residue-level hotspot coverage for a manifest of structures."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from eval_pipeline.hotspot_coverage import calculate_hotspot_coverage


OUTPUT_FIELDS = [
    "candidate_id",
    "structure_path",
    "target_chain_ids",
    "binder_chain_id",
    "hotspot_indices_0based",
    "hotspot_count",
    "heavy_atom_cutoff",
    "heavy_atom_covered_count",
    "heavy_atom_coverage",
    "mean_min_heavy_atom_distance",
    "ca_cutoff",
    "ca_covered_count",
    "ca_coverage",
    "mean_min_ca_distance",
    "status",
    "error",
    "per_hotspot_json",
]


def _json_list(row: dict[str, str], key: str) -> list:
    value = row.get(key, "")
    if not value:
        raise ValueError(f"manifest field {key!r} is required")
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError(f"manifest field {key!r} must be a JSON list")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--heavy-atom-cutoff", type=float, default=5.0)
    parser.add_argument("--ca-cutoff", type=float, default=8.0)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8"), delimiter="\t"))
    failures = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            candidate_id = row.get("candidate_id", "")
            try:
                result = calculate_hotspot_coverage(
                    row["structure_path"],
                    target_chain_ids=[str(value) for value in _json_list(row, "target_chain_ids_json")],
                    binder_chain_id=row["binder_chain_id"],
                    hotspot_concat_indices_0based=_json_list(
                        row, "hotspot_concat_indices_0based_json"
                    ),
                    expected_target_chain_lengths=_json_list(
                        row, "target_chain_lengths_json"
                    ),
                    heavy_atom_cutoff=args.heavy_atom_cutoff,
                    ca_cutoff=args.ca_cutoff,
                )
                output_row = {
                    "candidate_id": candidate_id,
                    **result,
                    "target_chain_ids": json.dumps(result["target_chain_ids"]),
                    "hotspot_indices_0based": json.dumps(result["hotspot_indices_0based"]),
                    "status": "ok",
                    "error": "",
                }
            except Exception as exc:
                failures += 1
                output_row = {
                    "candidate_id": candidate_id,
                    "structure_path": row.get("structure_path", ""),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            writer.writerow({field: output_row.get(field, "") for field in OUTPUT_FIELDS})

    if failures:
        print(f"hotspot coverage: {len(rows) - failures}/{len(rows)} succeeded", file=sys.stderr)
    else:
        print(f"hotspot coverage: {len(rows)}/{len(rows)} succeeded")
    return 1 if failures and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
