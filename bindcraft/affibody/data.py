"""Prepare experimental target, cleavage, and sequencing evidence tables."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from bindcraft.affibody.office import read_pptx_tables, read_xlsx_tables
from bindcraft.affibody.profiles import AFFIBODY_PROFILES, STANDARD_AAS


PEPTIDE_LIBRARY_AAS = frozenset("ADEFHIKLNPQSTVY")
CLEAVAGE_SLIDES = {"libb": 3, "liba": 6}
DEFAULT_SEQUENCING_ROUNDS = ("R000", "R001", "R009", "R010")


def _float_or_none(value: object) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_get(row: Sequence[str], index: int) -> str:
    return str(row[index]).strip() if index < len(row) else ""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def extract_cleavage_records(
    pptx_path: str | Path,
    *,
    binder_threshold: float = 75.0,
) -> list[dict[str, object]]:
    """Flatten the LibA/LibB retention matrices from the source PowerPoint."""

    records: list[dict[str, object]] = []
    for library, slide_number in CLEAVAGE_SLIDES.items():
        tables = read_pptx_tables(pptx_path, slide_number=slide_number)
        if not tables:
            raise ValueError(f"slide {slide_number} contains no table")
        matrix = tables[0]
        if len(matrix) < 2 or len(matrix[0]) < 2:
            raise ValueError(f"slide {slide_number} cleavage matrix is malformed")
        affibody_tuples = [value.strip().upper() for value in matrix[0][1:]]
        expected_tuple_length = len(
            AFFIBODY_PROFILES[library].mutable_positions_1based
        )
        for affibody_tuple in affibody_tuples:
            if len(affibody_tuple) != expected_tuple_length:
                raise ValueError(
                    f"slide {slide_number} has invalid {library} tuple "
                    f"{affibody_tuple!r}"
                )
        for row in matrix[1:]:
            peptide_pair = _safe_get(row, 0).upper()
            if len(peptide_pair) != 2 or any(
                aa not in STANDARD_AAS for aa in peptide_pair
            ):
                continue
            for column, affibody_tuple in enumerate(affibody_tuples, start=1):
                retention = _float_or_none(_safe_get(row, column))
                if retention is None:
                    continue
                records.append(
                    {
                        "experiment_id": f"E{len(records) + 1:04d}",
                        "library": AFFIBODY_PROFILES[library].name,
                        "peptide_pair": peptide_pair,
                        "experimental_peptide": f"SLL{peptide_pair}ITQV",
                        "affibody_tuple": affibody_tuple,
                        "affibody_sequence": AFFIBODY_PROFILES[library].render(
                            affibody_tuple
                        ),
                        "retention_percent": retention,
                        "binder_threshold_percent": float(binder_threshold),
                        "cleavage_binder": retention >= float(binder_threshold),
                        "assay_time_min": AFFIBODY_PROFILES[
                            library
                        ].assay_time_min,
                        "source_file": Path(pptx_path).name,
                        "source_slide": slide_number,
                    }
                )
    return records


def _computed_pair(peptide: str, register: str) -> str:
    register = register.strip().upper()
    if register == "P4/P5":
        return peptide[3:5]
    if register == "P5/P6":
        return peptide[4:6]
    raise ValueError(f"unsupported register: {register!r}")


def _support_annotation(
    *,
    peptide: str,
    register: str,
    measured_pairs: Mapping[str, set[str]],
) -> tuple[str, str, str]:
    pair = _computed_pair(peptide, register)
    applicable = [
        AFFIBODY_PROFILES[library].name
        for library in ("liba", "libb")
        if pair in measured_pairs[library]
    ]
    exact_libraries = [
        AFFIBODY_PROFILES[library].name
        for library in ("liba", "libb")
        if peptide == f"SLL{pair}ITQV" and pair in measured_pairs[library]
    ]
    if exact_libraries:
        return "exact_full_peptide", ";".join(exact_libraries), (
            "完整 9-mer 与 cleavage 实验肽完全相同，可做直接回溯验证"
        )
    if register.upper() != "P4/P5":
        return "register_extrapolation", ";".join(applicable), (
            "P5/P6 展示 register 没有直接训练支持，必须作为外推单独评估"
        )
    if applicable:
        return "cleavage_pair_transfer", ";".join(applicable), (
            "P4/P5 上举二肽已有 cleavage 标签，但完整 pMHC 不同，只能做迁移验证"
        )
    if all(aa in PEPTIDE_LIBRARY_AAS for aa in pair):
        return "sequencing_pair_supported", "LibA;LibB", (
            "二肽属于已测序的 15×15 肽库，但目前没有对应 cleavage 标签"
        )
    return "computational_only", "", (
        "二肽含训练肽库未覆盖的残基，当前只有计算外推证据"
    )


def extract_target_records(
    xlsx_path: str | Path,
    *,
    cleavage_records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Read the 69 design targets and attach an explicit evidence level."""

    workbook = read_xlsx_tables(xlsx_path)
    try:
        rows = workbook["Targets_v3"]
    except KeyError as exc:
        raise ValueError("workbook is missing worksheet 'Targets_v3'") from exc
    if not rows:
        raise ValueError("Targets_v3 is empty")
    header = [str(value).strip() for value in rows[0]]
    required = {"Tier", "Sequence", "Register", "Up-facing pair"}
    if not required.issubset(header):
        raise ValueError(f"Targets_v3 is missing columns: {sorted(required - set(header))}")
    measured_pairs: dict[str, set[str]] = {"liba": set(), "libb": set()}
    for record in cleavage_records:
        library = str(record["library"]).lower()
        measured_pairs[library].add(str(record["peptide_pair"]).upper())

    records: list[dict[str, object]] = []
    for row in rows[1:]:
        values = {
            name: _safe_get(row, index)
            for index, name in enumerate(header)
        }
        tier = values["Tier"].strip()
        if tier not in {"1", "2", "3"}:
            continue
        peptide = values["Sequence"].strip().upper()
        register = values["Register"].strip().upper()
        if len(peptide) != 9 or any(aa not in STANDARD_AAS for aa in peptide):
            raise ValueError(f"invalid 9-mer target peptide: {peptide!r}")
        computed_pair = _computed_pair(peptide, register)
        reported_pair = values["Up-facing pair"].strip().upper()
        if computed_pair != reported_pair:
            raise ValueError(
                f"target {peptide} reported pair {reported_pair} != {computed_pair}"
            )
        support_level, libraries, rationale = _support_annotation(
            peptide=peptide,
            register=register,
            measured_pairs=measured_pairs,
        )
        target_id = f"T{len(records) + 1:03d}"
        records.append(
            {
                "target_id": target_id,
                "tier": int(tier),
                "peptide": peptide,
                "register": register,
                "up_facing_pair": computed_pair,
                "support_level": support_level,
                "applicable_libraries": libraries,
                "support_rationale": rationale,
                "oncogenic": values.get("Oncogenic", ""),
                "confidence": values.get("Confidence", ""),
                "epitope": values.get("Epitope", ""),
                "antigen": values.get("Antigen", ""),
                "source": values.get("Source", ""),
                "pdb": values.get("PDB", ""),
                "tier_meaning": values.get("Tier meaning", ""),
                "transfer_dependent_residue": values.get(
                    "Transfer-dependent residue", ""
                ),
                "source_file": Path(xlsx_path).name,
                "source_sheet": "Targets_v3",
                "source_row": len(records) + 2,
            }
        )
    return records


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sequencing_file_identity(path: Path) -> tuple[str, str] | None:
    name = path.name.upper()
    library = "LibA" if "LIBA" in name else "LibB" if "LIBB" in name else None
    if library is None:
        return None
    patterns = (
        r"_R1_R(\d{3})(?:_|\.)",
        r"_R1_(\d{3})(?:_|\.)",
        r"_R(\d{3})(?:_|\.)",
    )
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return library, f"R{match.group(1)}"
    return None


def discover_sequencing_files(
    data_dir: str | Path,
    *,
    rounds: Sequence[str] = DEFAULT_SEQUENCING_ROUNDS,
) -> dict[tuple[str, str], Path]:
    """Choose one raw TSV per library/round, preferring the Raw data folders."""

    wanted = {str(round_name).upper() for round_name in rounds}
    candidates: dict[tuple[str, str], list[Path]] = {}
    for path in Path(data_dir).rglob("*.tsv"):
        identity = _sequencing_file_identity(path)
        if identity is not None and identity[1] in wanted:
            candidates.setdefault(identity, []).append(path)
    selected: dict[tuple[str, str], Path] = {}
    for identity, paths in candidates.items():
        selected[identity] = sorted(
            paths,
            key=lambda path: (
                "raw data" not in str(path.parent).lower(),
                "top2pct" in str(path.parent).lower(),
                str(path),
            ),
        )[0]
    return selected


def lookup_sequencing_evidence(
    data_dir: str | Path,
    candidate_keys: Iterable[tuple[str, str, str]],
    *,
    rounds: Sequence[str] = DEFAULT_SEQUENCING_ROUNDS,
    pseudocount: float = 1e-12,
) -> dict[tuple[str, str, str], dict[str, object]]:
    """Lookup only requested (library, peptide-pair, affibody-tuple) rows."""

    def canonical_library(value: object) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"liba", "libb"}:
            raise ValueError(f"unknown sequencing library: {value!r}")
        return AFFIBODY_PROFILES[normalized].name

    normalized_keys = {
        (canonical_library(library), str(peptide).upper(), str(affibody).upper())
        for library, peptide, affibody in candidate_keys
    }
    evidence: dict[tuple[str, str, str], dict[str, object]] = {
        key: {} for key in normalized_keys
    }
    files = discover_sequencing_files(data_dir, rounds=rounds)
    for (library, round_name), path in sorted(files.items()):
        library_keys = {key for key in normalized_keys if key[0] == library}
        if not library_keys:
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"pep", "aff", "count", "frequency", "pvalue"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"unexpected sequencing TSV header in {path}")
            for row in reader:
                key = (library, row["pep"].upper(), row["aff"].upper())
                if key not in library_keys:
                    continue
                evidence[key][f"{round_name}_count"] = int(float(row["count"]))
                evidence[key][f"{round_name}_frequency"] = float(row["frequency"])
                evidence[key][f"{round_name}_pvalue"] = float(row["pvalue"])
                evidence[key][f"{round_name}_source"] = str(path)
    for key, record in evidence.items():
        r009_count = int(record.get("R009_count", 0))
        r010_count = int(record.get("R010_count", 0))
        r009_frequency = float(record.get("R009_frequency", 0.0))
        r010_frequency = float(record.get("R010_frequency", 0.0))
        record["R009_R010_count"] = r009_count + r010_count
        record["R009_R010_frequency"] = r009_frequency + r010_frequency
        selected_frequency = r009_frequency + r010_frequency
        for baseline in ("R000", "R001"):
            baseline_frequency = float(record.get(f"{baseline}_frequency", 0.0))
            record[f"log2_enrichment_vs_{baseline}"] = math.log2(
                (selected_frequency + float(pseudocount))
                / (baseline_frequency + float(pseudocount))
            )
        record["sequencing_observed"] = any(
            int(record.get(f"{round_name}_count", 0)) > 0
            for round_name in rounds
        )
    return evidence


def lookup_sequencing_landscapes(
    data_dir: str | Path,
    candidate_keys: Iterable[tuple[str, str, str]],
) -> dict[tuple[str, str, str], dict[str, object]]:
    """Measure target-pair specificity across all observed R009/R010 pairs.

    Keys are ``(library, target_peptide_pair, affibody_tuple)``. Frequencies
    are pooled across R009 and R010 before target/off-target ranking.
    """

    def canonical_library(value: object) -> str:
        key = str(value).strip().lower()
        if key not in AFFIBODY_PROFILES:
            raise ValueError(f"unknown sequencing library: {value!r}")
        return AFFIBODY_PROFILES[key].name

    normalized = {
        (canonical_library(library), str(pair).upper(), str(affibody).upper())
        for library, pair, affibody in candidate_keys
    }
    requested_tuples: dict[str, set[str]] = {"LibA": set(), "LibB": set()}
    for library, _pair, affibody in normalized:
        requested_tuples[library].add(affibody)
    landscapes: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    files = discover_sequencing_files(data_dir, rounds=("R009", "R010"))
    for (library, _round_name), path in sorted(files.items()):
        wanted = requested_tuples[library]
        if not wanted:
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                affibody = row["aff"].upper()
                if affibody not in wanted:
                    continue
                peptide_pair = row["pep"].upper()
                key = (library, affibody)
                landscapes[key][peptide_pair] = landscapes[key].get(
                    peptide_pair, 0.0
                ) + float(row["frequency"])
    output: dict[tuple[str, str, str], dict[str, object]] = {}
    for key in normalized:
        library, target_pair, affibody = key
        values = landscapes.get((library, affibody), {})
        target_frequency = float(values.get(target_pair, 0.0))
        off_target = [
            frequency for pair, frequency in values.items() if pair != target_pair
        ]
        best_off_target = max(off_target, default=0.0)
        percentile = (
            sum(frequency <= target_frequency for frequency in values.values())
            / len(values)
            if values
            else 0.0
        )
        output[key] = {
            "landscape_observed_pair_count": len(values),
            "landscape_target_frequency": target_frequency,
            "landscape_target_percentile": percentile,
            "landscape_best_off_target_frequency": best_off_target,
            "landscape_target_minus_best_off_target": (
                target_frequency - best_off_target
            ),
            "landscape_pairs_above_target": sum(
                frequency > target_frequency for frequency in values.values()
            ),
        }
    return output
