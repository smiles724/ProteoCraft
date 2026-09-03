#!/usr/bin/env python3
"""Generate a self-contained HTML report for completed Affibody design runs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bindcraft.affibody.candidates import annotate_candidates  # noqa: E402
from bindcraft.affibody.data import (  # noqa: E402
    discover_sequencing_files,
    read_csv_rows,
)
from bindcraft.affibody.profiles import get_affibody_profile  # noqa: E402


SEQUENCING_ROUNDS = ("R000", "R001", "R009", "R010")
RUN_COLORS = ("#2563eb", "#ea580c", "#16845b", "#9333ea", "#db2777")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "读取每个 Affibody design seed 的 Top-N 与最终候选，匹配历史 cleavage/"
            "测序记录，并生成自包含中文 HTML 报告"
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="run_affibody_design.py 的输出根目录",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=None,
        help="design_runs.jsonl；默认 <output-root>/design_runs.jsonl",
    )
    parser.add_argument("--top-per-run", type=int, default=10)
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=PROJECT_ROOT / "data_prepared" / "affibody",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "datafolder",
        help="包含 LibA/LibB Raw data TSV 的历史数据目录",
    )
    parser.add_argument(
        "--blind-scores",
        type=Path,
        default=PROJECT_ROOT / "validation" / "known_pair_scores_blind.jsonl",
        help="已知候选独立盲评分；文件不存在时只省略该项",
    )
    parser.add_argument(
        "--reference-tuple",
        default=None,
        help="对照 tuple；默认使用每个 run 的 init_tuple（本例为 NNYYF）",
    )
    parser.add_argument(
        "--skip-landscape",
        action="store_true",
        help="不扫描 R009/R010 的跨 peptide-pair landscape（运行更快）",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _metric(record: Mapping[str, object], key: str) -> float | None:
    for container_name in ("summary_confidence", "loss_dict"):
        container = record.get(container_name, {})
        if isinstance(container, Mapping) and key in container:
            return _number(container[key])
    return _number(record.get(key))


def _candidate_record(
    raw: Mapping[str, object],
    *,
    run: Mapping[str, object],
    rank: int | None,
    is_final: bool,
    candidate_id: str,
) -> dict[str, object]:
    profile = get_affibody_profile(str(run["library"]))
    sequence = profile.validate_sequence(str(raw.get("sequence", "")))
    return {
        "candidate_id": candidate_id,
        "run_id": str(run["run_id"]),
        "seed": int(run["seed"]),
        "replicate_index": int(run.get("replicate_index", 0)),
        "target_id": str(run["target_id"]),
        "library": profile.name,
        "peptide": str(run["peptide"]),
        "register": str(run["register"]),
        "peptide_pair": str(run["peptide_pair"]),
        "target_support_level": str(run.get("target_support_level", "")),
        "sequence": sequence,
        "affibody_tuple": profile.extract_tuple(sequence),
        "rank": rank,
        "is_final": bool(is_final),
        "total_loss": _number(raw.get("total_loss")),
        "pair_chain_iptm": _metric(raw, "pair_chain_iptm"),
        "binder_plddt": _metric(raw, "binder_plddt"),
        "ptm": _metric(raw, "ptm"),
        "iptm": _metric(raw, "iptm"),
        "quality_floor": _metric(raw, "stageb_pair_chain_quality_floor"),
        "quality_ok": raw.get("summary_confidence", {}).get(
            "stageb_pair_chain_quality_ok"
        )
        if isinstance(raw.get("summary_confidence"), Mapping)
        else None,
        "hotspot_coverage_5a": _metric(raw, "hotspot_heavy_atom_coverage_5a"),
        "hotspot_mean_distance": _metric(
            raw, "hotspot_mean_min_heavy_atom_distance"
        ),
        "hotspot_too_close_loss": _metric(raw, "hotspot_too_close_loss"),
        "stage": str(raw.get("stage", "")),
        "step": raw.get("step"),
        "mutation_count": raw.get("mutation_count"),
    }


def _summary_candidate(summary: Mapping[str, object]) -> dict[str, object]:
    surfaces = summary.get("stageb_selection_surfaces", {})
    best = {}
    if isinstance(surfaces, Mapping):
        value = surfaces.get("best_by_total_loss", {})
        if isinstance(value, Mapping):
            best = dict(value)
    confidence = {
        key: best[key]
        for key in (
            "pair_chain_iptm",
            "quality_floor",
            "hotspot_coverage_5a",
            "hotspot_mean_distance",
        )
        if key in best
    }
    if "quality_floor" in confidence:
        confidence["stageb_pair_chain_quality_floor"] = confidence.pop(
            "quality_floor"
        )
    if "hotspot_coverage_5a" in confidence:
        confidence["hotspot_heavy_atom_coverage_5a"] = confidence.pop(
            "hotspot_coverage_5a"
        )
    if "hotspot_mean_distance" in confidence:
        confidence["hotspot_mean_min_heavy_atom_distance"] = confidence.pop(
            "hotspot_mean_distance"
        )
    return {
        "sequence": summary.get("final_sequence") or summary.get("best_sequence"),
        "total_loss": summary.get("best_loss") or best.get("total_loss"),
        "summary_confidence": confidence,
        "stage": "run_summary",
        "step": summary.get("n_steps"),
    }


def _collect_runs(
    runs_path: Path, *, top_per_run: int
) -> tuple[list[dict], list[dict]]:
    if top_per_run < 1:
        raise ValueError("--top-per-run must be positive")
    runs = _load_jsonl(runs_path)
    if not runs:
        raise ValueError(f"run manifest is empty: {runs_path}")
    output_runs: list[dict] = []
    candidates: list[dict] = []
    for run in runs:
        run_id = str(run.get("run_id", ""))
        if not run_id:
            raise ValueError("design run is missing run_id")
        if run.get("model_backend") != "esmfold2":
            raise ValueError(f"{run_id} is not an ESMFold2 run")
        profile = get_affibody_profile(str(run["library"]))
        artifact_summary = Path(str(run["artifact_summary"])).resolve()
        candidate_path = artifact_summary.with_name("candidates.jsonl")
        result_path = Path(str(run["output_dir"])).resolve() / "affibody_result.json"
        if not artifact_summary.is_file():
            raise FileNotFoundError(f"unfinished run; missing {artifact_summary}")
        if not candidate_path.is_file():
            raise FileNotFoundError(f"unfinished run; missing {candidate_path}")
        summary = _load_json(artifact_summary)
        result = _load_json(result_path) if result_path.is_file() else summary
        final_sequence = profile.validate_sequence(
            str(result.get("final_sequence") or summary.get("final_sequence") or "")
        )

        best_by_sequence: dict[str, dict] = {}
        for raw in _load_jsonl(candidate_path):
            sequence = profile.validate_sequence(str(raw.get("sequence", "")))
            loss = _number(raw.get("total_loss"))
            if loss is None:
                continue
            previous = best_by_sequence.get(sequence)
            if previous is None or loss < float(previous["total_loss"]):
                best_by_sequence[sequence] = raw
        if not best_by_sequence:
            raise ValueError(f"{candidate_path} contains no scored candidates")
        ordered = sorted(
            best_by_sequence.values(), key=lambda row: float(row["total_loss"])
        )
        rank_by_sequence = {
            str(row["sequence"]): rank for rank, row in enumerate(ordered, start=1)
        }
        selected = list(ordered[:top_per_run])
        if final_sequence not in {str(row["sequence"]) for row in selected}:
            selected.append(
                best_by_sequence.get(final_sequence) or _summary_candidate(summary)
            )

        run_candidates: list[dict] = []
        for raw in selected:
            sequence = profile.validate_sequence(str(raw.get("sequence", "")))
            rank = rank_by_sequence.get(sequence)
            is_final = sequence == final_sequence
            suffix = f"rank{rank}" if rank is not None else "final"
            row = _candidate_record(
                raw,
                run=run,
                rank=rank,
                is_final=is_final,
                candidate_id=f"{run_id}::{suffix}",
            )
            run_candidates.append(row)
            candidates.append(row)

        history_path = Path(str(run["output_dir"])).resolve() / "hallucination_result.txt"
        output_runs.append(
            {
                **run,
                "profile": profile,
                "artifact_summary_path": artifact_summary,
                "candidate_path": candidate_path,
                "result_path": result_path,
                "history_path": history_path,
                "summary": summary,
                "result": result,
                "final_sequence": final_sequence,
                "unique_candidate_count": len(ordered),
                "displayed_candidates": run_candidates,
            }
        )
    return output_runs, candidates


def _reference_key(run: Mapping[str, object]) -> tuple[str, str, str, str, str]:
    return (
        str(run["target_id"]),
        str(run["library"]),
        str(run["peptide"]),
        str(run["register"]),
        str(run["peptide_pair"]),
    )


def _evidence_inputs(
    runs: Sequence[dict],
    candidates: Sequence[dict],
    *,
    reference_tuple: str | None,
) -> tuple[list[dict], dict[tuple[str, str, str, str, str], dict]]:
    rows: list[dict] = []
    for candidate in candidates:
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "target_id": candidate["target_id"],
                "library": candidate["library"],
                "affibody_sequence": candidate["sequence"],
                "affibody_tuple": candidate["affibody_tuple"],
                "peptide": candidate["peptide"],
                "register": candidate["register"],
                "peptide_pair": candidate["peptide_pair"],
                "target_support_level": candidate["target_support_level"],
            }
        )
    references: dict[tuple[str, str, str, str, str], dict] = {}
    for run in runs:
        key = _reference_key(run)
        if key in references:
            continue
        profile = get_affibody_profile(str(run["library"]))
        tuple_value = str(
            reference_tuple or run.get("init_tuple") or profile.default_tuple
        ).upper()
        sequence = profile.render(tuple_value)
        candidate_id = f"REFERENCE::{key[0]}::{key[1]}::{tuple_value}"
        row = {
            "candidate_id": candidate_id,
            "target_id": key[0],
            "library": key[1],
            "affibody_sequence": sequence,
            "affibody_tuple": tuple_value,
            "peptide": key[2],
            "register": key[3],
            "peptide_pair": key[4],
            "target_support_level": str(run.get("target_support_level", "")),
        }
        rows.append(row)
        references[key] = row
    return rows, references


def _history_status(evidence: Mapping[str, object]) -> tuple[str, str]:
    if bool(evidence.get("cleavage_is_exact_full_peptide")):
        return "exact_assay", "完整 peptide cleavage 实验"
    if bool(evidence.get("cleavage_label_available")):
        return "pair_transfer", "同 pair cleavage 记录"
    if bool(evidence.get("sequencing_observed")):
        return "sequencing", "目标 pair 测序出现"
    if int(evidence.get("landscape_observed_pair_count", 0) or 0) > 0:
        return "other_pair", "仅在其他 peptide pair 出现"
    return "none", "历史记录未找到"


def _attach_evidence(
    candidates: Sequence[dict],
    references: Mapping[tuple[str, str, str, str, str], dict],
    annotated: Sequence[dict],
) -> dict[tuple[str, str, str, str, str], dict]:
    evidence_by_id = {str(row["candidate_id"]): row for row in annotated}
    for candidate in candidates:
        evidence = evidence_by_id[str(candidate["candidate_id"])]
        status_key, status_label = _history_status(evidence)
        candidate["evidence"] = evidence
        candidate["history_status"] = status_key
        candidate["history_label"] = status_label
    reference_output: dict[tuple[str, str, str, str, str], dict] = {}
    for key, reference in references.items():
        evidence = evidence_by_id[str(reference["candidate_id"])]
        status_key, status_label = _history_status(evidence)
        reference_output[key] = {
            **reference,
            "evidence": evidence,
            "history_status": status_key,
            "history_label": status_label,
        }
    return reference_output


def _attach_other_pair_details(
    candidates: Sequence[dict], *, data_dir: Path
) -> dict[tuple[str, str], Path]:
    """Attach per-round R009/R010 rows for tuples seen away from the target pair."""

    eligible = [
        candidate
        for candidate in candidates
        if candidate.get("history_status") == "other_pair"
    ]
    for candidate in candidates:
        candidate["other_pair_details"] = []
    if not eligible:
        return {}

    wanted: dict[str, set[str]] = {}
    for candidate in eligible:
        wanted.setdefault(str(candidate["library"]), set()).add(
            str(candidate["affibody_tuple"]).upper()
        )
    files = discover_sequencing_files(data_dir, rounds=("R009", "R010"))
    details: dict[tuple[str, str, str], dict[str, object]] = {}
    for (library, round_name), path in sorted(files.items()):
        library_tuples = wanted.get(library, set())
        if not library_tuples:
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"pep", "aff", "count", "frequency", "pvalue"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"unexpected sequencing TSV header in {path}")
            for row in reader:
                affibody_tuple = str(row["aff"]).upper()
                if affibody_tuple not in library_tuples:
                    continue
                peptide_pair = str(row["pep"]).upper()
                key = (library, affibody_tuple, peptide_pair)
                record = details.setdefault(
                    key,
                    {
                        "library": library,
                        "affibody_tuple": affibody_tuple,
                        "peptide_pair": peptide_pair,
                    },
                )
                count_key = f"{round_name}_count"
                frequency_key = f"{round_name}_frequency"
                record[count_key] = int(record.get(count_key, 0)) + int(
                    float(row["count"])
                )
                record[frequency_key] = float(
                    record.get(frequency_key, 0.0)
                ) + float(row["frequency"])
                record[f"{round_name}_pvalue"] = float(row["pvalue"])
                record[f"{round_name}_source"] = str(path)

    for candidate in eligible:
        target_pair = str(candidate["peptide_pair"]).upper()
        rows = [
            dict(record)
            for (library, affibody_tuple, pair), record in details.items()
            if library == candidate["library"]
            and affibody_tuple == candidate["affibody_tuple"]
            and pair != target_pair
        ]
        for row in rows:
            row["total_count"] = sum(
                int(row.get(f"{round_name}_count", 0))
                for round_name in ("R009", "R010")
            )
            row["total_frequency"] = sum(
                float(row.get(f"{round_name}_frequency", 0.0))
                for round_name in ("R009", "R010")
            )
            row["observed_rounds"] = [
                round_name
                for round_name in ("R009", "R010")
                if int(row.get(f"{round_name}_count", 0)) > 0
            ]
        candidate["other_pair_details"] = sorted(
            rows,
            key=lambda row: (
                -int(row["total_count"]),
                -float(row["total_frequency"]),
                str(row["peptide_pair"]),
            ),
        )
    return files


def _blind_reference_scores(
    path: Path,
    references: Mapping[tuple[str, str, str, str, str], dict],
) -> dict[tuple[str, str, str, str, str], dict]:
    output: dict[tuple[str, str, str, str, str], dict] = {}
    if not path.is_file():
        return output
    records = _load_jsonl(path)
    for key, reference in references.items():
        matches = [
            row
            for row in records
            if str(row.get("library")) == key[1]
            and str(row.get("peptide", "")).upper() == key[2].upper()
            and str(row.get("peptide_pair", "")).upper() == key[4].upper()
            and str(row.get("affibody_tuple", "")).upper()
            == str(reference["affibody_tuple"]).upper()
        ]
        losses = [
            value
            for row in matches
            if (value := _number(row.get("total_loss"))) is not None
        ]
        pair_iptm = [
            value
            for row in matches
            if (value := _metric(row, "pair_chain_iptm")) is not None
        ]
        if losses:
            output[key] = {
                "seed_count": len(losses),
                "total_loss_mean": statistics.fmean(losses),
                "total_loss_sd": statistics.stdev(losses) if len(losses) > 1 else 0.0,
                "pair_chain_iptm_mean": (
                    statistics.fmean(pair_iptm) if pair_iptm else None
                ),
                "pair_chain_iptm_sd": (
                    statistics.stdev(pair_iptm) if len(pair_iptm) > 1 else 0.0
                ),
                "seeds": sorted({int(row["seed"]) for row in matches if "seed" in row}),
            }
    return output


def _loss_history(path: Path) -> list[tuple[int, float]]:
    if not path.is_file():
        return []
    values: list[tuple[int, float]] = []
    pattern = re.compile(r"^Step\s+(\d+):\s+([-+0-9.eE]+)$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            value = _number(match.group(2))
            if value is not None:
                values.append((int(match.group(1)), value))
    return values


def _esc(value: object) -> str:
    return html.escape(str(value))


def _fmt(value: object, digits: int = 3) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:.{digits}f}"


def _fmt_frequency(value: object) -> str:
    number = _number(value)
    if number is None:
        return "—"
    return f"{number:.3e}" if abs(number) < 0.001 else f"{number:.6f}"


def _badge(status: str, label: str) -> str:
    return f"<span class='badge badge-{_esc(status)}'>{_esc(label)}</span>"


def _sequence_details(sequence: object) -> str:
    return (
        "<details class='sequence'><summary>查看完整序列</summary>"
        f"<code>{_esc(sequence)}</code></details>"
    )


def _history_detail(evidence: Mapping[str, object]) -> str:
    rows = []
    for round_name in SEQUENCING_ROUNDS:
        rows.append(
            "<tr>"
            f"<td>{round_name}</td>"
            f"<td>{int(evidence.get(f'{round_name}_count', 0) or 0)}</td>"
            f"<td>{_fmt_frequency(evidence.get(f'{round_name}_frequency', 0.0))}</td>"
            f"<td>{_fmt(evidence.get(f'{round_name}_pvalue'), 4)}</td>"
            "</tr>"
        )
    return (
        "<details><summary>四轮测序明细</summary><table class='nested'><thead><tr>"
        "<th>轮次</th><th>count</th><th>frequency</th><th>p-value</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></details>"
    )


def _other_pair_detail(candidate: Mapping[str, object]) -> str:
    details = candidate.get("other_pair_details", [])
    if not isinstance(details, Sequence) or not details:
        return ""
    body: list[str] = []
    source_paths: set[str] = set()
    for row in details:
        if not isinstance(row, Mapping):
            continue
        rounds = row.get("observed_rounds", [])
        for round_name in ("R009", "R010"):
            source = str(row.get(f"{round_name}_source", ""))
            if source:
                source_paths.add(source)
        body.append(
            "<tr>"
            f"<td><strong>{_esc(row.get('peptide_pair', '—'))}</strong></td>"
            f"<td>{_esc(', '.join(str(value) for value in rounds) or '—')}</td>"
            f"<td>{int(row.get('R009_count', 0) or 0)}</td>"
            f"<td>{_fmt_frequency(row.get('R009_frequency', 0.0))}</td>"
            f"<td>{_fmt(row.get('R009_pvalue'), 4)}</td>"
            f"<td>{int(row.get('R010_count', 0) or 0)}</td>"
            f"<td>{_fmt_frequency(row.get('R010_frequency', 0.0))}</td>"
            f"<td>{_fmt(row.get('R010_pvalue'), 4)}</td>"
            f"<td><strong>{int(row.get('total_count', 0) or 0)}</strong></td>"
            f"<td>{_fmt_frequency(row.get('total_frequency', 0.0))}</td>"
            "</tr>"
        )
    sources = " · ".join(
        f"<code>{_esc(Path(path).name)}</code>" for path in sorted(source_paths)
    )
    return (
        "<details class='other-pair-detail'><summary>其他 pair 明细（"
        f"{len(body)} 个 pair）</summary><div class='table-wrap'>"
        "<table class='nested other-pair-table'><thead><tr>"
        "<th>Pair</th><th>出现轮次</th>"
        "<th>R009 count</th><th>R009 freq</th><th>R009 p</th>"
        "<th>R010 count</th><th>R010 freq</th><th>R010 p</th>"
        "<th>合计 reads</th><th>合计 freq</th>"
        f"</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
        f"<p class='muted'>原始文件：{sources or '—'}</p></details>"
    )


def _candidate_table(rows: Sequence[dict], reference: Mapping[str, object]) -> str:
    reference_blind = reference.get("blind_score", {})
    reference_loss = _number(reference_blind.get("total_loss_mean"))
    body: list[str] = []
    for row in sorted(
        rows,
        key=lambda item: (
            item["rank"] is None,
            int(item["rank"] or 10**9),
            float(item["total_loss"] or math.inf),
        ),
    ):
        evidence = row["evidence"]
        final_badge = " <span class='badge badge-final'>最终候选</span>" if row["is_final"] else ""
        delta_loss = (
            float(row["total_loss"]) - reference_loss
            if row["total_loss"] is not None and reference_loss is not None
            else None
        )
        cleavage = "—"
        if evidence.get("cleavage_label_available"):
            binder = _truth(evidence.get("cleavage_binder"))
            cleavage = (
                f"{_fmt(evidence.get('cleavage_retention_percent'), 2)}% "
                f"({'binder' if binder else 'non-binder'})"
            )
        observed_pairs = int(evidence.get("landscape_observed_pair_count", 0) or 0)
        percentile = (
            f"{100 * float(evidence['landscape_target_percentile']):.1f}%"
            if observed_pairs and evidence.get("landscape_target_percentile") is not None
            else "—"
        )
        body.append(
            "<tr class='candidate-row'>"
            f"<td>{row['rank'] or 'final'}{final_badge}</td>"
            f"<td><strong>{_esc(row['affibody_tuple'])}</strong>"
            f"{_sequence_details(row['sequence'])}</td>"
            f"<td>{_fmt(row['total_loss'], 4)}</td>"
            f"<td class='comparison'>{_fmt(delta_loss, 4)}</td>"
            f"<td>{_fmt(row['pair_chain_iptm'], 3)}</td>"
            f"<td>{_fmt(row['binder_plddt'], 3)}</td>"
            f"<td>{_fmt(row['quality_floor'], 3)}</td>"
            f"<td>{_fmt(row['hotspot_coverage_5a'], 2)}</td>"
            f"<td>{_fmt(row['hotspot_mean_distance'], 2)}</td>"
            f"<td>{_esc(row['stage'])}<br><span class='muted'>step "
            f"{_esc(row['step'] if row['step'] is not None else '—')}</span></td>"
            f"<td>{_badge(row['history_status'], row['history_label'])}"
            f"{_other_pair_detail(row)}{_history_detail(evidence)}</td>"
            f"<td>{cleavage}</td>"
            f"<td>{int(evidence.get('R009_R010_count', 0) or 0)}<br>"
            f"<span class='muted'>"
            f"{_fmt_frequency(evidence.get('R009_R010_frequency', 0.0))}"
            f"</span></td>"
            f"<td>"
            f"{_fmt(evidence.get('log2_enrichment_vs_R000'), 2) if evidence.get('sequencing_observed') else '—'}"
            f"</td>"
            f"<td>{percentile}<br><span class='muted'>"
            f"{int(evidence.get('landscape_pairs_above_target', 0) or 0)} "
            f"pair 更高</span></td>"
            f"<td>"
            f"{_fmt_frequency(evidence.get('landscape_target_minus_best_off_target')) if observed_pairs else '—'}"
            f"</td>"
            "</tr>"
        )
    return f"""
<div class="table-wrap"><table class="candidate-table">
<thead><tr>
<th>Run rank</th><th>Tuple / sequence</th><th>设计 loss ↓</th>
<th>Δloss vs 对照*</th><th>Pair iPTM ↑</th><th>Binder pLDDT</th>
<th>Quality floor</th><th>Hotspot 5Å coverage</th><th>Hotspot 距离 Å</th>
<th>来源</th><th>历史记录</th><th>Cleavage</th><th>R009+R010<br>count / freq</th>
<th>log2 enrich<br>vs R000</th><th>Target landscape</th><th>Target − best off-target</th>
</tr></thead><tbody>{''.join(body)}</tbody></table></div>
"""


def _final_table(runs: Sequence[dict], references: Mapping[tuple, dict]) -> str:
    rows: list[str] = []
    for run in runs:
        final = next(row for row in run["displayed_candidates"] if row["is_final"])
        reference = references[_reference_key(run)]
        blind = reference.get("blind_score", {})
        reference_loss = _number(blind.get("total_loss_mean"))
        delta = (
            float(final["total_loss"]) - reference_loss
            if final["total_loss"] is not None and reference_loss is not None
            else None
        )
        rows.append(
            "<tr>"
            f"<td>{_esc(run['seed'])}</td>"
            f"<td><strong>{_esc(final['affibody_tuple'])}</strong></td>"
            f"<td>{_fmt(final['total_loss'], 4)}</td>"
            f"<td>{_fmt(delta, 4)}</td>"
            f"<td>{_fmt(final['pair_chain_iptm'], 3)}</td>"
            f"<td>{_fmt(final['binder_plddt'], 3)}</td>"
            f"<td>{_fmt(final['quality_floor'], 3)}</td>"
            f"<td>{_fmt(final['hotspot_coverage_5a'], 2)}</td>"
            f"<td>{_fmt(final['hotspot_mean_distance'], 2)}</td>"
            f"<td>{_badge(final['history_status'], final['history_label'])}</td>"
            "</tr>"
        )
    return f"""
<div class="table-wrap"><table><thead><tr><th>Seed</th><th>最终 tuple</th>
<th>设计 loss ↓</th><th>Δloss vs 对照*</th><th>Pair iPTM ↑</th>
<th>Binder pLDDT</th><th>Quality floor</th><th>Hotspot coverage</th>
<th>Hotspot 距离 Å</th><th>历史记录</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
"""


def _convergence_svg(runs: Sequence[dict]) -> str:
    histories = [(run, _loss_history(run["history_path"])) for run in runs]
    histories = [(run, values) for run, values in histories if values]
    if not histories:
        return "<p class='muted'>未找到 Loss History。</p>"
    width, height = 820, 300
    left, right, top, bottom = 58, 25, 22, 48
    max_step = max(step for _, values in histories for step, _ in values)
    losses = [loss for _, values in histories for _, loss in values]
    y_min, y_max = min(losses), max(losses)
    padding = max((y_max - y_min) * 0.08, 0.01)
    y_min -= padding
    y_max += padding

    def x(step: int) -> float:
        return left + step / max(max_step, 1) * (width - left - right)

    def y(loss: float) -> float:
        return top + (y_max - loss) / (y_max - y_min) * (height - top - bottom)

    parts = [f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='各 seed loss 收敛曲线'>"]
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        position = y(value)
        parts.append(
            f"<line class='grid' x1='{left}' y1='{position:.1f}' x2='{width-right}' y2='{position:.1f}'/>"
            f"<text x='{left-8}' y='{position+4:.1f}' text-anchor='end'>{value:.3f}</text>"
        )
    for tick in range(6):
        step = round(max_step * tick / 5)
        position = x(step)
        parts.append(
            f"<text x='{position:.1f}' y='{height-bottom+21}' text-anchor='middle'>{step}</text>"
        )
    legend: list[str] = []
    for index, (run, values) in enumerate(histories):
        color = RUN_COLORS[index % len(RUN_COLORS)]
        points = " ".join(f"{x(step):.1f},{y(loss):.1f}" for step, loss in values)
        parts.append(
            f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='2.5'/>"
        )
        legend.append(
            f"<span><i style='background:{color}'></i>seed {_esc(run['seed'])}: "
            f"{values[0][1]:.3f} → {values[-1][1]:.3f}</span>"
        )
    parts.append(
        f"<text x='{width/2:.1f}' y='{height-5}' text-anchor='middle'>Optimization step</text>"
        f"<text transform='translate(15 {height/2:.1f}) rotate(-90)' text-anchor='middle'>Best total_loss</text></svg>"
    )
    return "".join(parts) + f"<div class='legend'>{''.join(legend)}</div>"


def _reference_cards(references: Mapping[tuple, dict]) -> str:
    cards: list[str] = []
    for key, reference in references.items():
        evidence = reference["evidence"]
        blind = reference.get("blind_score", {})
        retention = (
            f"{_fmt(evidence.get('cleavage_retention_percent'), 2)}%"
            if evidence.get("cleavage_label_available")
            else "—"
        )
        blind_text = "—"
        if blind:
            blind_text = (
                f"{_fmt(blind.get('total_loss_mean'), 3)} ± "
                f"{_fmt(blind.get('total_loss_sd'), 3)}"
            )
        cards.append(
            "<article class='reference-card'>"
            f"<h3>{_esc(reference['affibody_tuple'])} 阳性/起始基准</h3>"
            f"<p class='muted'>{_esc(key[0])} · {key[1]} · peptide {_esc(key[2])} · pair {_esc(key[4])}</p>"
            "<div class='metric-grid'>"
            f"<div><strong>{retention}</strong><span>Cleavage retention</span></div>"
            f"<div><strong>{int(evidence.get('R009_R010_count', 0) or 0)}</strong><span>R009+R010 reads</span></div>"
            f"<div><strong>{_fmt_frequency(evidence.get('R009_R010_frequency', 0.0))}</strong><span>R009+R010 frequency</span></div>"
            f"<div><strong>{blind_text}</strong><span>独立盲评 total_loss</span></div>"
            f"<div><strong>{_fmt(blind.get('pair_chain_iptm_mean'), 3)}</strong><span>独立盲评 Pair iPTM</span></div>"
            f"<div><strong>{_esc(blind.get('seed_count', '—'))}</strong><span>盲评 seeds</span></div>"
            "</div>"
            f"{_sequence_details(reference['affibody_sequence'])}"
            f"{_history_detail(evidence)}"
            "</article>"
        )
    return "".join(cards)


def _render_html(
    *,
    runs: Sequence[dict],
    candidates: Sequence[dict],
    references: Mapping[tuple, dict],
    sources: Mapping[str, Path],
    top_per_run: int,
    landscape_enabled: bool,
) -> str:
    unique_candidates: dict[tuple[str, str, str], dict] = {}
    for row in candidates:
        unique_candidates[(str(row["target_id"]), str(row["library"]), str(row["sequence"]))] = row
    history_counts = Counter(row["history_status"] for row in unique_candidates.values())
    final_candidates = [row for row in candidates if row["is_final"]]
    run_sections: list[str] = []
    for run in runs:
        reference = references[_reference_key(run)]
        run_sections.append(
            "<details class='run-section' open>"
            f"<summary>Seed {_esc(run['seed'])} · {len(run['displayed_candidates'])} 条展示记录 "
            f"· candidates.jsonl 中 {run['unique_candidate_count']} 条去重序列</summary>"
            f"{_candidate_table(run['displayed_candidates'], reference)}"
            "</details>"
        )
    source_rows = "".join(
        f"<tr><td>{_esc(name)}</td><td><code>{_esc(path)}</code></td></tr>"
        for name, path in sources.items()
    )
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    landscape_note = "已扫描" if landscape_enabled else "已跳过"
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Affibody 设计候选报告</title>
<style>
:root{{--ink:#18212f;--muted:#64748b;--line:#dbe2ea;--blue:#2563eb;--green:#16845b;--orange:#c56a14;--red:#b42318;--purple:#7c3aed;--paper:#fff;--wash:#f4f7fa}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--wash);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1540px;margin:0 auto;padding:28px}}section,.hero,.reference-card{{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:22px;margin-bottom:18px}}
h1{{font-size:29px;margin:0 0 7px}}h2{{font-size:21px;margin:0 0 10px}}h3{{margin:5px 0}}.muted{{color:var(--muted)}}
.cards,.metric-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:11px;margin-top:16px}}.cards>div,.metric-grid>div{{background:#f8fafc;border:1px solid var(--line);border-radius:10px;padding:12px}}.cards strong,.metric-grid strong{{display:block;font-size:22px}}.cards span,.metric-grid span{{color:var(--muted);font-size:12px}}
.warning{{border-left:4px solid #ca8a04;background:#fffbeb;padding:11px 13px;margin:13px 0}}.info{{border-left-color:var(--blue);background:#eff6ff}}
.table-wrap{{overflow:auto;margin-top:12px}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border-bottom:1px solid var(--line);padding:8px;text-align:right;white-space:nowrap;vertical-align:top}}th{{background:#f8fafc;position:sticky;top:0;z-index:1}}th:nth-child(2),td:nth-child(2),th:nth-child(10),td:nth-child(10),th:nth-child(11),td:nth-child(11){{text-align:left}}
.candidate-table{{min-width:1900px}}.candidate-row:hover{{background:#f8fbff}}.comparison{{background:#fffdf3}}details{{border-top:1px solid var(--line);padding:10px 0}}summary{{cursor:pointer;font-weight:650}}.run-section{{border:1px solid var(--line);border-radius:11px;padding:12px;margin:12px 0}}.run-section>summary{{font-size:16px}}
.sequence{{border:0;padding:2px 0;max-width:240px}}.sequence summary{{font-size:11px;color:var(--blue)}}code{{white-space:normal;overflow-wrap:anywhere;background:#eef2f7;border-radius:4px;padding:2px 4px}}.nested{{margin:5px 0;font-size:11px}}.nested th,.nested td{{padding:4px}}
.badge{{display:inline-block;border-radius:99px;padding:2px 7px;font-size:11px;font-weight:650}}.badge-final{{background:#dbeafe;color:#174c9b}}.badge-exact_assay{{background:#dcfce7;color:#17663e}}.badge-pair_transfer{{background:#d1fae5;color:#17663e}}.badge-sequencing{{background:#ede9fe;color:#6d28d9}}.badge-other_pair{{background:#ffedd5;color:#9a3412}}.badge-none{{background:#f1f5f9;color:#64748b}}
svg{{display:block;width:100%;max-width:900px;height:auto}}svg text{{fill:var(--muted);font-size:11px}}.grid{{stroke:#e5eaf0;stroke-width:1}}.legend{{display:flex;flex-wrap:wrap;gap:14px;margin:8px 0;color:var(--muted)}}.legend span{{display:inline-flex;align-items:center;gap:6px}}.legend i{{width:11px;height:11px;border-radius:50%}}
.search{{width:100%;max-width:500px;padding:9px 11px;border:1px solid var(--line);border-radius:8px;font:inherit}}
@media(max-width:800px){{main{{padding:10px}}section,.hero,.reference-card{{padding:15px}}}}
</style></head><body><main>
<header class="hero"><h1>Affibody 设计候选报告</h1>
<p class="muted">每个 seed 的去重 Top-{top_per_run}、最终候选、历史实验/测序证据与起始阳性基准比较。生成时间：{_esc(generated)}</p>
<div class="cards">
<div><strong>{len(runs)}</strong><span>完成的 design seeds</span></div>
<div><strong>{len(candidates)}</strong><span>seed 内展示记录</span></div>
<div><strong>{len(unique_candidates)}</strong><span>跨 seed 去重候选</span></div>
<div><strong>{len(final_candidates)}</strong><span>最终候选</span></div>
<div><strong>{history_counts.get('exact_assay', 0) + history_counts.get('sequencing', 0)}</strong><span>目标组合有历史记录</span></div>
<div><strong>{history_counts.get('none', 0)}</strong><span>历史记录未找到</span></div>
</div>
<div class="warning"><b>解释边界：</b>候选的 design loss 是搜索过程中选出的优化值；对照的 loss 来自独立盲评 seed。Δloss 仅用于初步参考，不能当作同条件的严格优劣检验，也不能直接预测实验 retention。公平比较需要统一 fresh-seed 固定序列重评分。</div>
</header>

<section><h2>1. 最终候选概览</h2>{_final_table(runs, references)}
<p class="muted">* Δloss = 候选的优化内 loss − 对照的独立盲评平均 loss；负值在数值上更低，但两者并非完全同条件。</p></section>

<section><h2>2. 搜索收敛</h2>{_convergence_svg(runs)}</section>

<section><h2>3. 起始/阳性基准</h2>
<div class="warning info">基准用于证明现有实验体系存在阳性、提供固定 Affibody framework，并校验历史数据匹配。它不是负对照，也不保证是实验中最强 binder。</div>
{_reference_cards(references)}</section>

<section><h2>4. 每个 seed 的候选明细</h2>
<p class="muted">历史匹配键为 library + peptide pair + Affibody tuple。跨 pair landscape：{landscape_note}。表格可以横向滚动。</p>
<input id="candidate-search" class="search" placeholder="按 seed、tuple、序列或历史状态筛选……">
{''.join(run_sections)}</section>

<section><h2>5. 标签说明</h2>
<ul>
<li>{_badge('exact_assay', '完整 peptide cleavage 实验')}：同 library、完整 peptide、pair 和 tuple 均有实验记录。</li>
<li>{_badge('sequencing', '目标 pair 测序出现')}：没有 cleavage 标签，但四轮测序中精确组合至少出现一次；少量 reads 不能自动视为 binder。</li>
<li>{_badge('other_pair', '仅在其他 peptide pair 出现')}：目标 pair 未出现，但同一 tuple 在其他 pair 的 R009/R010 中出现。</li>
<li>{_badge('none', '历史记录未找到')}：现有 cleavage 与所扫描测序记录均未找到；不等于 non-binder。</li>
<li><b>富集值注意：</b>当 R000 基线为 0 时，即使 R009/R010 只有 1 read，带 pseudocount 的 log2 enrichment 也可能很大；必须与 count、frequency 和跨 pair landscape 一起解释。</li>
</ul></section>

<section><h2>6. 数据来源</h2><div class="table-wrap"><table><thead><tr><th>来源</th><th>路径</th></tr></thead><tbody>{source_rows}</tbody></table></div></section>
</main>
<script>
const search = document.getElementById('candidate-search');
search.addEventListener('input', () => {{
  const query = search.value.trim().toLowerCase();
  document.querySelectorAll('.candidate-row').forEach(row => {{
    row.style.display = !query || row.textContent.toLowerCase().includes(query) ? '' : 'none';
  }});
}});
</script></body></html>"""


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = args.output_root.resolve()
    runs_path = (
        args.runs.resolve()
        if args.runs is not None
        else output_root / "design_runs.jsonl"
    )
    prepared_dir = args.prepared_dir.resolve()
    data_dir = args.data_dir.resolve()
    cleavage_path = prepared_dir / "cleavage_labels.csv"
    targets_path = prepared_dir / "targets.csv"
    for required in (runs_path, cleavage_path, targets_path, data_dir):
        if not required.exists():
            raise FileNotFoundError(f"required input not found: {required}")

    runs, candidates = _collect_runs(runs_path, top_per_run=args.top_per_run)
    evidence_rows, raw_references = _evidence_inputs(
        runs, candidates, reference_tuple=args.reference_tuple
    )
    print(
        f"Matching {len(evidence_rows)} seed-level candidate/reference rows against "
        f"{data_dir}",
        flush=True,
    )
    annotated = annotate_candidates(
        evidence_rows,
        cleavage_records=read_csv_rows(cleavage_path),
        data_dir=data_dir,
        with_sequencing_landscape=not args.skip_landscape,
    )
    references = _attach_evidence(candidates, raw_references, annotated)
    if not args.skip_landscape:
        _attach_other_pair_details(candidates, data_dir=data_dir)
    blind_scores = _blind_reference_scores(args.blind_scores.resolve(), references)
    for key, score in blind_scores.items():
        references[key]["blind_score"] = score

    sources = {
        "Design runs": runs_path,
        "Cleavage labels": cleavage_path,
        "Target manifest": targets_path,
        "Raw sequencing directory": data_dir,
        "Known-pair blind scores": args.blind_scores.resolve(),
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _render_html(
            runs=runs,
            candidates=candidates,
            references=references,
            sources=sources,
            top_per_run=args.top_per_run,
            landscape_enabled=not args.skip_landscape,
        ),
        encoding="utf-8",
    )
    unique_count = len(
        {
            (row["target_id"], row["library"], row["sequence"])
            for row in candidates
        }
    )
    print(
        f"HTML report: {output_path} "
        f"({len(runs)} runs, {unique_count} unique candidates)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
