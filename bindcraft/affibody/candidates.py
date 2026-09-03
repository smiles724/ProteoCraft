"""Candidate validation, freezing, and experimental-evidence annotation."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from bindcraft.affibody.data import (
    lookup_sequencing_evidence,
    lookup_sequencing_landscapes,
    sha256_file,
)
from bindcraft.affibody.profiles import get_affibody_profile


def normalize_candidates(
    rows: Sequence[Mapping[str, object]],
    *,
    targets: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Validate full Affibody sequences and attach their target peptide."""

    target_by_id = {str(row["target_id"]): dict(row) for row in targets}
    normalized: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(rows, start=1):
        row = dict(raw)
        candidate_id = str(row.get("candidate_id") or f"C{index:04d}").strip()
        if candidate_id in seen_ids:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        seen_ids.add(candidate_id)
        target_id = str(row.get("target_id", "")).strip()
        if target_id not in target_by_id:
            raise ValueError(f"candidate {candidate_id} has unknown target_id {target_id!r}")
        library = str(row.get("library", "")).strip()
        profile = get_affibody_profile(library)
        sequence = str(
            row.get("affibody_sequence") or row.get("sequence") or ""
        ).strip().upper()
        sequence = profile.validate_sequence(sequence)
        variable_tuple = profile.extract_tuple(sequence)
        target = target_by_id[target_id]
        normalized.append(
            {
                "candidate_id": candidate_id,
                "target_id": target_id,
                "library": profile.name,
                "affibody_sequence": sequence,
                "affibody_tuple": variable_tuple,
                "peptide": str(target["peptide"]).upper(),
                "register": str(target["register"]).upper(),
                "peptide_pair": str(target["up_facing_pair"]).upper(),
                "target_support_level": target.get("support_level", ""),
                "design_score": row.get("design_score", ""),
                "design_rank": row.get("design_rank", ""),
                "source_run": row.get("source_run", ""),
            }
        )
    return normalized


def annotate_candidates(
    candidates: Sequence[Mapping[str, object]],
    *,
    cleavage_records: Sequence[Mapping[str, object]],
    data_dir: str | Path,
    with_sequencing_landscape: bool = False,
) -> list[dict[str, object]]:
    """Join cleavage labels and four sequencing rounds to frozen candidates."""

    cleavage_index = {
        (
            str(record["library"]),
            str(record["peptide_pair"]).upper(),
            str(record["affibody_tuple"]).upper(),
        ): dict(record)
        for record in cleavage_records
    }
    keys = [
        (
            str(candidate["library"]),
            str(candidate["peptide_pair"]),
            str(candidate["affibody_tuple"]),
        )
        for candidate in candidates
    ]
    sequencing = lookup_sequencing_evidence(data_dir, keys)
    landscapes = (
        lookup_sequencing_landscapes(data_dir, keys)
        if with_sequencing_landscape
        else {}
    )
    output: list[dict[str, object]] = []
    for candidate in candidates:
        record = dict(candidate)
        key = (
            str(record["library"]),
            str(record["peptide_pair"]),
            str(record["affibody_tuple"]),
        )
        cleavage = cleavage_index.get(key)
        sequence_evidence = sequencing.get(key, {})
        if cleavage is not None:
            record.update(
                {
                    "cleavage_label_available": True,
                    "cleavage_retention_percent": cleavage["retention_percent"],
                    "cleavage_binder": cleavage["cleavage_binder"],
                    "cleavage_assay_time_min": cleavage["assay_time_min"],
                    "cleavage_source_slide": cleavage["source_slide"],
                    "cleavage_is_exact_full_peptide": (
                        str(record["peptide"])
                        == str(cleavage["experimental_peptide"])
                    ),
                }
            )
        else:
            record.update(
                {
                    "cleavage_label_available": False,
                    "cleavage_retention_percent": "",
                    "cleavage_binder": "",
                    "cleavage_assay_time_min": "",
                    "cleavage_source_slide": "",
                    "cleavage_is_exact_full_peptide": False,
                }
            )
        record.update(sequence_evidence)
        record.update(landscapes.get(key, {}))
        if record["cleavage_is_exact_full_peptide"]:
            evidence_level = "exact_existing_assay"
        elif record["cleavage_label_available"]:
            evidence_level = "pair_transfer"
        elif bool(record.get("sequencing_observed")):
            evidence_level = "sequencing_only"
        else:
            evidence_level = "computational_only"
        record["validation_evidence_level"] = evidence_level
        output.append(record)
    return output


def frozen_file_record(path: str | Path) -> dict[str, object]:
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
