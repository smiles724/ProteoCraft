#!/usr/bin/env python3
"""Explain blind affibody scores with decision-oriented schemes 1, 4, and 6.

The analyses are within-group ranking, raw/rank scatter plots, and score-bin
experimental outcomes. Internal model scores are normalized so that larger is
better, while human-readable output retains the unmodified raw score.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bindcraft.affibody.data import read_csv_rows, sha256_file, write_csv, write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "将冻结盲评分连接到 cleavage 标签，并仅生成方案 1 的组内排序、"
            "方案 4 的原始分数散点图和方案 6 的分箱实验结果"
        )
    )
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "data_prepared" / "affibody" / "cleavage_labels.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-key", default="pair_chain_iptm")
    parser.add_argument(
        "--lower-is-better",
        action="store_true",
        help="原始 score 越低越好；分析时会取负，使 model_score 始终越高越好",
    )
    return parser


def _rank(values: Sequence[float]) -> list[float]:
    """Return average ranks in ascending order, preserving ties."""
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for index in ordered[cursor:end]:
            ranks[index] = average
        cursor = end
    return ranks


def _descending_ranks(values: Sequence[float]) -> list[float]:
    return _rank([-value for value in values])


def _score_value(record: dict, key: str) -> float:
    if key in record:
        return float(record[key])
    for container in ("summary_confidence", "loss_dict"):
        values = record.get(container, {})
        if isinstance(values, dict) and key in values:
            return float(values[key])
    raise KeyError(f"score key {key!r} is absent from record {record.get('complex_id')}")


def _parse_binder(value: object, *, experiment_id: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(
        f"invalid cleavage_binder value {value!r} for experiment {experiment_id}"
    )


def _read_score_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"score file contains no records: {path}")
    return records


def _build_candidate_rows(
    score_records: Sequence[dict],
    label_rows: Sequence[dict[str, str]],
    *,
    score_key: str,
    lower_is_better: bool,
) -> list[dict]:
    labels: dict[str, dict[str, str]] = {}
    for row in label_rows:
        experiment_id = str(row.get("experiment_id", "")).strip()
        if not experiment_id:
            raise ValueError("labels contain an empty experiment_id")
        if experiment_id in labels:
            raise ValueError(f"duplicate experiment_id in labels: {experiment_id}")
        labels[experiment_id] = row

    by_complex: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for record_index, record in enumerate(score_records, start=1):
        complex_id = str(record.get("complex_id", "")).strip()
        if not complex_id:
            raise ValueError(f"score record {record_index} has no complex_id")
        if "seed" not in record:
            raise ValueError(f"score record for {complex_id} has no seed")
        seed = str(record["seed"])
        if seed in by_complex[complex_id]:
            raise ValueError(f"duplicate score for complex_id={complex_id}, seed={seed}")
        raw_score = _score_value(record, score_key)
        if not math.isfinite(raw_score):
            raise ValueError(
                f"non-finite score for complex_id={complex_id}, seed={seed}: {raw_score}"
            )
        model_score = -raw_score if lower_is_better else raw_score
        by_complex[complex_id][seed] = (raw_score, model_score)

    missing_labels = sorted(set(by_complex) - set(labels))
    if missing_labels:
        raise ValueError(f"scores contain unknown experiment ID(s): {missing_labels[:10]}")

    rows: list[dict] = []
    for complex_id, seed_values in sorted(by_complex.items()):
        label = labels[complex_id]
        retention = float(label["retention_percent"])
        if not math.isfinite(retention):
            raise ValueError(f"non-finite retention for experiment {complex_id}")
        raw_scores = [pair[0] for pair in seed_values.values()]
        model_scores = [pair[1] for pair in seed_values.values()]
        rows.append(
            {
                "complex_id": complex_id,
                "library": label["library"],
                "peptide_pair": label["peptide_pair"],
                "affibody_tuple": label["affibody_tuple"],
                "raw_score_mean": statistics.fmean(raw_scores),
                "model_score": statistics.fmean(model_scores),
                "score_seed_sd": (
                    statistics.stdev(model_scores) if len(model_scores) > 1 else 0.0
                ),
                "score_seed_count": len(model_scores),
                "retention_percent": retention,
                "cleavage_binder": _parse_binder(
                    label["cleavage_binder"], experiment_id=complex_id
                ),
                "_seed_scores": {
                    seed: values[1] for seed, values in seed_values.items()
                },
            }
        )
    return rows


def _group_rows(rows: Sequence[dict]) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["library"]), str(row["peptide_pair"]))].append(row)
    return dict(sorted(groups.items()))


def _group_id(key: tuple[str, str]) -> str:
    return f"{key[0]}::{key[1]}"


def _ordered_by_model(
    rows: Sequence[dict], score_key: str = "model_score"
) -> list[dict]:
    return sorted(rows, key=lambda row: (-float(row[score_key]), str(row["complex_id"])))


def _scheme_1_rankings(
    groups: dict[tuple[str, str], list[dict]],
) -> tuple[list[dict], dict]:
    output: list[dict] = []
    top1_best_count = 0
    for key, values in groups.items():
        ordered = _ordered_by_model(values)
        retention_ranks = _descending_ranks(
            [float(row["retention_percent"]) for row in ordered]
        )
        best_retention = max(float(row["retention_percent"]) for row in ordered)
        top1_best_count += int(
            float(ordered[0]["retention_percent"]) == best_retention
        )
        for model_rank, (row, experimental_rank) in enumerate(
            zip(ordered, retention_ranks), start=1
        ):
            selection_percentile = 100.0 * (model_rank - 0.5) / len(ordered)
            output.append(
                {
                    "group_id": _group_id(key),
                    "library": key[0],
                    "peptide_pair": key[1],
                    "group_size": len(ordered),
                    "model_rank": model_rank,
                    "within_group_selection_percentile": selection_percentile,
                    "experimental_retention_rank": experimental_rank,
                    "complex_id": row["complex_id"],
                    "affibody_tuple": row["affibody_tuple"],
                    "raw_score_mean": row["raw_score_mean"],
                    "model_score": row["model_score"],
                    "score_seed_sd": row["score_seed_sd"],
                    "score_seed_count": row["score_seed_count"],
                    "retention_percent": row["retention_percent"],
                    "cleavage_binder": row["cleavage_binder"],
                    "model_top1": model_rank == 1,
                    "experimental_best": (
                        float(row["retention_percent"]) == best_retention
                    ),
                }
            )
    group_count = len(groups)
    return output, {
        "group_count": group_count,
        "candidate_count": len(output),
        "model_top1_is_experimental_best_count": top1_best_count,
        "model_top1_is_experimental_best_rate": (
            top1_best_count / group_count if group_count else None
        ),
        "percentile_definition": (
            "100 * (model_rank - 0.5) / group_size; lower means the model "
            "ranked the candidate closer to the top."
        ),
    }


def _kendall_tau_b(rows: Sequence[dict]) -> dict[str, float | int | None]:
    concordant = 0
    discordant = 0
    model_ties_only = 0
    retention_ties_only = 0
    ties_in_both = 0
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            model_delta = float(left["model_score"]) - float(
                right["model_score"]
            )
            retention_delta = float(left["retention_percent"]) - float(
                right["retention_percent"]
            )
            if model_delta == 0.0 and retention_delta == 0.0:
                ties_in_both += 1
            elif model_delta == 0.0:
                model_ties_only += 1
            elif retention_delta == 0.0:
                retention_ties_only += 1
            elif model_delta * retention_delta > 0.0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + model_ties_only)
        * (concordant + discordant + retention_ties_only)
    )
    return {
        "kendall_tau_b": (
            (concordant - discordant) / denominator if denominator else None
        ),
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "model_tied_pairs": model_ties_only,
        "retention_tied_pairs": retention_ties_only,
        "pairs_tied_in_both": ties_in_both,
    }


def _scheme_1_ranking_agreement(
    groups: dict[tuple[str, str], list[dict]], *, top_k: int = 3
) -> tuple[list[dict], list[dict]]:
    per_group: list[dict] = []
    for key, values in groups.items():
        k = min(int(top_k), len(values))
        model_order = _ordered_by_model(values)
        retention_order = sorted(
            values,
            key=lambda row: (
                -float(row["retention_percent"]),
                str(row["complex_id"]),
            ),
        )
        model_top = {str(row["complex_id"]) for row in model_order[:k]}
        retention_top = {
            str(row["complex_id"]) for row in retention_order[:k]
        }
        overlap_count = len(model_top & retention_top)
        overlap = overlap_count / k if k else None
        random_expected = k / len(values) if values else None
        adjusted = (
            (overlap - random_expected) / (1.0 - random_expected)
            if overlap is not None
            and random_expected is not None
            and random_expected < 1.0
            else None
        )
        tau = _kendall_tau_b(values)
        per_group.append(
            {
                "group_id": _group_id(key),
                "library": key[0],
                "peptide_pair": key[1],
                "group_size": len(values),
                "top_k": k,
                "overlap_count": overlap_count,
                "overlap_at_3": overlap,
                "random_expected_overlap_at_3": random_expected,
                "adjusted_overlap_at_3": adjusted,
                **tau,
                "model_top3_complex_ids": ";".join(
                    str(row["complex_id"]) for row in model_order[:k]
                ),
                "retention_top3_complex_ids": ";".join(
                    str(row["complex_id"]) for row in retention_order[:k]
                ),
                "top3_tie_policy": (
                    "retention ties at the cutoff are resolved by complex_id; "
                    "Kendall tau-b retains ties"
                ),
            }
        )

    def mean_or_none(rows: Sequence[dict], key: str) -> float | None:
        values = [
            float(row[key]) for row in rows if row.get(key) is not None
        ]
        return statistics.fmean(values) if values else None

    summaries: list[dict] = []
    scopes = [("Overall", per_group)]
    scopes.extend(
        (
            library,
            [row for row in per_group if row["library"] == library],
        )
        for library in sorted({str(row["library"]) for row in per_group})
    )
    for scope, rows in scopes:
        summaries.append(
            {
                "scope": scope,
                "group_count": len(rows),
                "mean_overlap_at_3": mean_or_none(rows, "overlap_at_3"),
                "mean_random_expected_overlap_at_3": mean_or_none(
                    rows, "random_expected_overlap_at_3"
                ),
                "mean_adjusted_overlap_at_3": mean_or_none(
                    rows, "adjusted_overlap_at_3"
                ),
                "mean_kendall_tau_b": mean_or_none(rows, "kendall_tau_b"),
            }
        )
    return per_group, summaries


def _linear_trend(rows: Sequence[dict], x_key: str) -> dict[str, float | int | None]:
    """Return a descriptive least-squares line, not a prediction model."""
    if len(rows) < 2:
        return {"point_count": len(rows), "slope": None, "intercept": None}
    x_values = [float(row[x_key]) for row in rows]
    y_values = [float(row["retention_percent"]) for row in rows]
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator == 0.0:
        return {"point_count": len(rows), "slope": None, "intercept": y_mean}
    slope = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values)
    ) / denominator
    return {
        "point_count": len(rows),
        "slope": slope,
        "intercept": y_mean - slope * x_mean,
    }


def _pearson_correlation(
    x_values: Sequence[float], y_values: Sequence[float]
) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values)
    )
    x_sum_squares = sum((value - x_mean) ** 2 for value in x_values)
    y_sum_squares = sum((value - y_mean) ** 2 for value in y_values)
    denominator = math.sqrt(x_sum_squares * y_sum_squares)
    return numerator / denominator if denominator else None


def _library_correlations(rows: Sequence[dict], *, score_key: str) -> list[dict]:
    output: list[dict] = []
    for library in sorted({str(row["library"]) for row in rows}):
        library_rows = [row for row in rows if str(row["library"]) == library]
        negative_scores = [
            -float(row["raw_score_mean"]) for row in library_rows
        ]
        retentions = [
            float(row["retention_percent"]) for row in library_rows
        ]
        output.append(
            {
                "library": library,
                "candidate_count": len(library_rows),
                "correlation_method": "Pearson",
                "x_variable": f"-{score_key}",
                "y_variable": "retention_percent",
                "pearson_r": _pearson_correlation(
                    negative_scores, retentions
                ),
                "direction_interpretation": (
                    "positive r means lower raw score tends to accompany "
                    "higher retention"
                ),
            }
        )
    return output


def _scheme_4_scatter(
    rankings: Sequence[dict], *, lower_is_better: bool
) -> tuple[list[dict], dict]:
    output = [
        {
            "library": row["library"],
            "peptide_pair": row["peptide_pair"],
            "complex_id": row["complex_id"],
            "affibody_tuple": row["affibody_tuple"],
            "raw_score_mean": row["raw_score_mean"],
            "model_rank": row["model_rank"],
            "group_size": row["group_size"],
            "retention_percent": row["retention_percent"],
            "cleavage_binder": row["cleavage_binder"],
        }
        for row in rankings
    ]
    facets: dict[str, dict] = {}
    for library in sorted({str(row["library"]) for row in output}):
        values = [row for row in output if row["library"] == library]
        facets[library] = {
            "point_count": len(values),
            "peptide_pair_count": len(
                {str(row["peptide_pair"]) for row in values}
            ),
            "raw_score_descriptive_trend": _linear_trend(
                values, "raw_score_mean"
            ),
        }
    return output, {
        "point_count": len(output),
        "facets": facets,
        "raw_score_axis": (
            "raw score is shown without sign conversion; lower values are "
            "placed on the left and labeled better"
            if lower_is_better
            else "raw score is shown without sign conversion; higher values are better"
        ),
        "marker_encoding": {
            "color": "peptide_pair",
            "filled": "cleavage_binder",
            "hollow": "non_binder",
            "hover_text": "affibody_tuple and candidate details",
        },
        "trend_warning": (
            "Trend lines are descriptive visual guides only and must not be "
            "interpreted as retention prediction models."
        ),
    }


SCORE_BIN_LABELS = (
    "Top 20%",
    "20–40%",
    "40–60%",
    "60–80%",
    "Bottom 20%",
)


def _quartiles(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), float(values[0])
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return quartiles[0], quartiles[2]


def _scheme_6_score_bins(
    rankings: Sequence[dict],
) -> tuple[list[dict], dict]:
    libraries = sorted({str(row["library"]) for row in rankings})
    scopes: list[tuple[str, list[dict]]] = [("Overall", list(rankings))]
    scopes.extend(
        (
            library,
            [row for row in rankings if str(row["library"]) == library],
        )
        for library in libraries
    )
    output: list[dict] = []
    for scope, values in scopes:
        binned: dict[int, list[dict]] = defaultdict(list)
        for row in values:
            percentile = float(row["within_group_selection_percentile"])
            bin_index = min(int(percentile // 20.0), 4)
            binned[bin_index].append(row)
        for bin_index, label in enumerate(SCORE_BIN_LABELS):
            bin_rows = binned[bin_index]
            retentions = [
                float(row["retention_percent"]) for row in bin_rows
            ]
            binder_count = sum(
                bool(row["cleavage_binder"]) for row in bin_rows
            )
            q1, q3 = _quartiles(retentions)
            output.append(
                {
                    "scope": scope,
                    "bin_order": bin_index + 1,
                    "score_bin": label,
                    "percentile_start": 20 * bin_index,
                    "percentile_end": 20 * (bin_index + 1),
                    "candidate_count": len(bin_rows),
                    "binder_count": binder_count,
                    "binder_rate": (
                        binder_count / len(bin_rows) if bin_rows else None
                    ),
                    "mean_retention": (
                        statistics.fmean(retentions) if retentions else None
                    ),
                    "median_retention": (
                        statistics.median(retentions) if retentions else None
                    ),
                    "retention_q1": q1,
                    "retention_q3": q3,
                    "retention_iqr": (
                        q3 - q1 if q1 is not None and q3 is not None else None
                    ),
                }
            )

    overall = [row for row in output if row["scope"] == "Overall"]
    top = overall[0]
    bottom = overall[-1]
    adjacent_binder_steps = sum(
        left["binder_rate"] is not None
        and right["binder_rate"] is not None
        and float(left["binder_rate"]) >= float(right["binder_rate"])
        for left, right in zip(overall, overall[1:])
    )
    adjacent_retention_steps = sum(
        left["mean_retention"] is not None
        and right["mean_retention"] is not None
        and float(left["mean_retention"]) >= float(right["mean_retention"])
        for left, right in zip(overall, overall[1:])
    )
    return output, {
        "bin_count": len(SCORE_BIN_LABELS),
        "scope_count": len(scopes),
        "scopes": [scope for scope, _ in scopes],
        "candidate_count": len(rankings),
        "top_minus_bottom_binder_rate": (
            float(top["binder_rate"]) - float(bottom["binder_rate"])
            if top["binder_rate"] is not None
            and bottom["binder_rate"] is not None
            else None
        ),
        "top_minus_bottom_mean_retention": (
            float(top["mean_retention"]) - float(bottom["mean_retention"])
            if top["mean_retention"] is not None
            and bottom["mean_retention"] is not None
            else None
        ),
        "non_increasing_binder_rate_steps_top_to_bottom": adjacent_binder_steps,
        "non_increasing_mean_retention_steps_top_to_bottom": (
            adjacent_retention_steps
        ),
        "maximum_adjacent_steps": len(SCORE_BIN_LABELS) - 1,
        "binning_definition": (
            "Within each library x peptide-pair group, percentile = "
            "100 * (model_rank - 0.5) / group_size. Boundaries are "
            "[0,20), [20,40), [40,60), [60,80), [80,100]."
        ),
    }


def _fmt(value: object, *, percent: bool = False) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value * 100:.1f}%" if percent else f"{value:.3f}"
    return html.escape(str(value))


def _html_table(
    rows: Sequence[dict], columns: Sequence[tuple[str, str, bool]]
) -> str:
    header = "".join(
        f"<th>{html.escape(label)}</th>" for _, label, _ in columns
    )
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{_fmt(row.get(key), percent=percent)}</td>"
            for key, _, percent in columns
        )
        + "</tr>"
        for row in rows
    )
    return (
        "<div class='table-wrap'><table><thead><tr>"
        f"{header}</tr></thead><tbody>{body}</tbody></table></div>"
    )


SCATTER_COLORS = (
    "#2563eb",
    "#ea580c",
    "#16845b",
    "#9333ea",
    "#db2777",
    "#0891b2",
    "#ca8a04",
    "#4f46e5",
    "#dc2626",
    "#059669",
    "#7c3aed",
    "#475569",
    "#0f766e",
    "#be123c",
    "#65a30d",
    "#c026d3",
    "#0369a1",
    "#b45309",
    "#6d28d9",
    "#15803d",
    "#e11d48",
    "#1d4ed8",
    "#a16207",
    "#0e7490",
)


def _scatter_facet_svg(
    rows: Sequence[dict],
    *,
    x_key: str,
    x_label: str,
    pair_colors: dict[str, str],
) -> str:
    width, height = 600, 360
    left, right, top, bottom = 58, 24, 24, 58
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values = [float(row[x_key]) for row in rows]
    x_min, x_max = min(x_values), max(x_values)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    padding = (x_max - x_min) * 0.04
    x_min -= padding
    x_max += padding

    def x_position(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_position(value: float) -> float:
        clipped = max(0.0, min(100.0, value))
        return top + (100.0 - clipped) / 100.0 * plot_height

    parts = [
        f"<svg viewBox='0 0 {width} {height}' role='img' "
        f"aria-label='{html.escape(x_label)} 与 retention 散点图'>"
    ]
    for tick in range(0, 101, 25):
        y = y_position(float(tick))
        parts.append(
            f"<line x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}' "
            f"class='grid'/><text x='{left-8}' y='{y+4:.1f}' "
            f"text-anchor='end'>{tick}</text>"
        )
    for tick_index in range(5):
        value = x_min + (x_max - x_min) * tick_index / 4
        x = x_position(value)
        parts.append(
            f"<line x1='{x:.1f}' y1='{top}' x2='{x:.1f}' "
            f"y2='{height-bottom}' class='grid'/><text x='{x:.1f}' "
            f"y='{height-bottom+18}' text-anchor='middle'>{value:.3g}</text>"
        )

    trend = _linear_trend(rows, x_key)
    if trend["slope"] is not None and trend["intercept"] is not None:
        start_y = float(trend["intercept"]) + float(trend["slope"]) * x_min
        end_y = float(trend["intercept"]) + float(trend["slope"]) * x_max
        parts.append(
            f"<line x1='{x_position(x_min):.1f}' y1='{y_position(start_y):.1f}' "
            f"x2='{x_position(x_max):.1f}' y2='{y_position(end_y):.1f}' "
            "class='trend-line'/>"
        )

    for row in rows:
        color = pair_colors[str(row["peptide_pair"])]
        binder = bool(row["cleavage_binder"])
        fill = color if binder else "#ffffff"
        tooltip = html.escape(
            f"{row['affibody_tuple']} | pair={row['peptide_pair']} | "
            f"raw score={float(row['raw_score_mean']):.5g} | "
            f"group rank={row['model_rank']}/{row['group_size']} | "
            f"retention={float(row['retention_percent']):.1f}% | "
            f"{'cleavage binder' if binder else 'non-binder'}"
        )
        parts.append(
            f"<circle cx='{x_position(float(row[x_key])):.1f}' "
            f"cy='{y_position(float(row['retention_percent'])):.1f}' r='5' "
            f"fill='{fill}' stroke='{color}' stroke-width='2' opacity='0.9'>"
            f"<title>{tooltip}</title></circle>"
        )
    parts.append(
        f"<text x='{width/2:.1f}' y='{height-6}' text-anchor='middle'>"
        f"{html.escape(x_label)}</text>"
        f"<text transform='translate(15 {height/2:.1f}) rotate(-90)' "
        "text-anchor='middle'>retention (%)</text></svg>"
    )
    return "".join(parts)


def _scatter_facets_html(
    rows: Sequence[dict], *, x_key: str, x_label: str
) -> str:
    pairs = sorted({str(row["peptide_pair"]) for row in rows})
    pair_colors = {
        pair: SCATTER_COLORS[index % len(SCATTER_COLORS)]
        for index, pair in enumerate(pairs)
    }
    facets: list[str] = []
    for library in sorted({str(row["library"]) for row in rows}):
        library_rows = [row for row in rows if row["library"] == library]
        library_pairs = sorted(
            {str(row["peptide_pair"]) for row in library_rows}
        )
        legend = "".join(
            f"<span><i style='background:{pair_colors[pair]}'></i>"
            f"{html.escape(pair)}</span>"
            for pair in library_pairs
        )
        facets.append(
            f"<div class='scatter-facet'><h3>{html.escape(library)}</h3>"
            f"{_scatter_facet_svg(library_rows, x_key=x_key, x_label=x_label, pair_colors=pair_colors)}"
            "<div class='marker-key'><span><b class='solid-dot'></b>binder</span>"
            "<span><b class='hollow-dot'></b>non-binder</span>"
            "<span><b class='dash-line'></b>描述性趋势线</span></div>"
            f"<div class='pair-legend'>{legend}</div></div>"
        )
    return f"<div class='scatter-grid'>{''.join(facets)}</div>"


def _ranking_html(rankings: Sequence[dict]) -> str:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rankings:
        grouped[str(row["group_id"])].append(row)
    parts: list[str] = []
    for group_id, rows in grouped.items():
        body: list[str] = []
        for row in rows:
            retention = max(0.0, min(100.0, float(row["retention_percent"])))
            label = html.escape(str(row["affibody_tuple"]))
            if row["model_top1"]:
                label += " <span class='badge'>模型 Top-1</span>"
            if row["experimental_best"]:
                label += " <span class='badge best'>实验最佳</span>"
            is_binder = bool(row["cleavage_binder"])
            row_class = "cleavage-binder-row" if is_binder else "non-binder-row"
            fill_class = "binder-fill" if is_binder else "non-binder-fill"
            status_class = "binder-status" if is_binder else "non-binder-status"
            status_text = "cleavage binder" if is_binder else "non-binder"
            body.append(
                f"<tr class='{row_class}'>"
                f"<td>{label}</td><td>{row['model_rank']}</td>"
                f"<td>{_fmt(row['experimental_retention_rank'])}</td>"
                f"<td>{_fmt(row['raw_score_mean'])}</td>"
                f"<td>{float(row['retention_percent']):.1f}"
                f"<div class='bar'><i class='{fill_class}' "
                f"style='width:{retention:.1f}%'></i></div></td>"
                f"<td><span class='status {status_class}'>{status_text}</span></td></tr>"
            )
        parts.append(
            f"<details><summary>{html.escape(group_id)}（{len(rows)} 个候选）"
            "</summary><div class='table-wrap'><table><thead><tr>"
            "<th>候选</th><th>模型名次</th><th>实验名次</th>"
            "<th>raw score</th><th>retention</th><th>binder</th>"
            f"</tr></thead><tbody>{''.join(body)}</tbody></table></div></details>"
        )
    return "".join(parts)


def _ranking_agreement_html(
    per_group: Sequence[dict], summaries: Sequence[dict]
) -> str:
    summary_table = _html_table(
        summaries,
        [
            ("scope", "范围", False),
            ("group_count", "组数", False),
            ("mean_overlap_at_3", "Overlap@3", True),
            (
                "mean_random_expected_overlap_at_3",
                "随机期望 Overlap@3",
                True,
            ),
            (
                "mean_adjusted_overlap_at_3",
                "随机校正 Overlap@3",
                True,
            ),
            ("mean_kendall_tau_b", "Kendall tau-b", False),
        ],
    )
    detail_table = _html_table(
        per_group,
        [
            ("group_id", "组", False),
            ("group_size", "候选数", False),
            ("overlap_count", "Top-3 共同数", False),
            ("overlap_at_3", "Overlap@3", True),
            (
                "random_expected_overlap_at_3",
                "随机期望",
                True,
            ),
            ("adjusted_overlap_at_3", "随机校正", True),
            ("kendall_tau_b", "Kendall tau-b", False),
        ],
    )
    return (
        "<div class='agreement-block'><h3>模型排序与 retention 排序的重合程度</h3>"
        f"{summary_table}"
        "<details class='metric-explanation'><summary>这些指标分别代表什么？"
        "（点击展开）</summary><div class='explanation-body'>"
        "<p><b>Overlap@3</b>：模型 Top-3 与 retention Top-3 的共同候选数除以 3。"
        "100% 表示三个候选完全相同，0% 表示没有共同候选。</p>"
        "<p><b>随机校正 Overlap@3</b>：先扣除同样大小列表在该组内随机重合的期望 "
        "<code>3 / 候选数</code>，再缩放。1 表示完全重合，0 表示与随机相当，"
        "负数表示低于随机期望。</p>"
        "<p><b>Kendall tau-b</b>：检查组内所有候选对的先后次序是否一致，并保留"
        "分数或 retention 并列。1 表示完整顺序一致，0 表示没有稳定的两两排序关系，"
        "-1 表示顺序完全相反。</p>"
        "<p>所有值先在每个 <code>library × peptide_pair</code> 组内计算，再对组做"
        "宏平均。Top-3 边界遇到相同 retention 时使用 complex_id 固定顺序；"
        "Kendall tau-b 本身按并列处理。</p></div></details>"
        f"<details><summary>查看全部 {len(per_group)} 个组的明细</summary>"
        f"{detail_table}</details></div>"
    )


def _score_bin_cards(rows: Sequence[dict]) -> str:
    cards: list[str] = []
    for row in rows:
        binder_rate = row["binder_rate"]
        mean_retention = row["mean_retention"]
        binder_width = (
            max(0.0, min(100.0, 100.0 * float(binder_rate)))
            if binder_rate is not None
            else 0.0
        )
        retention_width = (
            max(0.0, min(100.0, float(mean_retention)))
            if mean_retention is not None
            else 0.0
        )
        cards.append(
            "<div class='bin-card'>"
            f"<h4>{html.escape(str(row['score_bin']))}</h4>"
            f"<div class='bin-n'>n={row['candidate_count']}</div>"
            f"<label>binder rate <strong>{_fmt(binder_rate, percent=True)}</strong></label>"
            f"<div class='metric-bar'><i class='binder-metric' style='width:{binder_width:.1f}%'></i></div>"
            f"<label>mean retention <strong>{_fmt(mean_retention)}</strong></label>"
            f"<div class='metric-bar'><i class='retention-metric' style='width:{retention_width:.1f}%'></i></div>"
            f"<div class='bin-detail'>median {_fmt(row['median_retention'])}<br>"
            f"IQR {_fmt(row['retention_q1'])}–{_fmt(row['retention_q3'])}</div>"
            "</div>"
        )
    return f"<div class='bin-flow'>{''.join(cards)}</div>"


def _score_bins_html(rows: Sequence[dict]) -> str:
    sections: list[str] = []
    scopes = list(dict.fromkeys(str(row["scope"]) for row in rows))
    columns = [
        ("score_bin", "模型档位", False),
        ("candidate_count", "n", False),
        ("binder_count", "binder 数", False),
        ("binder_rate", "binder rate", True),
        ("mean_retention", "平均 retention", False),
        ("median_retention", "中位 retention", False),
        ("retention_q1", "Q1", False),
        ("retention_q3", "Q3", False),
        ("retention_iqr", "IQR 宽度", False),
    ]
    for scope in scopes:
        scope_rows = [row for row in rows if str(row["scope"]) == scope]
        sections.append(
            f"<h3>{html.escape(scope)}</h3>{_score_bin_cards(scope_rows)}"
            f"{_html_table(scope_rows, columns)}"
        )
    return "".join(sections)


def _render_html(
    *,
    report: dict,
    rankings: Sequence[dict],
    ranking_agreement: Sequence[dict],
    ranking_agreement_summaries: Sequence[dict],
    scatter_points: Sequence[dict],
    score_bins: Sequence[dict],
    library_correlations: Sequence[dict],
) -> str:
    scheme1 = report["analyses"]["1_group_rankings"]
    scheme6 = report["analyses"]["6_score_bins"]
    lower_is_better = report["score_direction"].startswith("lower_raw_is_better")
    raw_axis_note = (
        "原始值；越左 / 越低越好"
        if lower_is_better
        else "原始值；越右 / 越高越好"
    )
    raw_scatter = _scatter_facets_html(
        scatter_points,
        x_key="raw_score_mean",
        x_label=f"{report['score_key']}（{raw_axis_note}）",
    )
    correlation_cards = "".join(
        f"<div class='card'><strong>{_fmt(row['pearson_r'])}</strong>"
        f"<span>{html.escape(str(row['library']))}: Pearson r(retention, "
        f"-{html.escape(str(report['score_key']))})</span></div>"
        for row in library_correlations
    )
    correlation_table = _html_table(
        library_correlations,
        [
            ("library", "Library", False),
            ("candidate_count", "n", False),
            ("x_variable", "x", False),
            ("y_variable", "y", False),
            ("correlation_method", "方法", False),
            ("pearson_r", "Pearson r", False),
        ],
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Affibody blind score 直观分析：方案 1 / 4 / 6</title>
<style>
:root{{--ink:#18212f;--muted:#637083;--line:#d9e0e8;--blue:#2563eb;--green:#16845b;--paper:#fff;--wash:#f4f7fa}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--wash);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1300px;margin:0 auto;padding:28px}}.hero,.section{{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:22px;margin-bottom:18px}}
h1{{font-size:28px;margin:0 0 6px}}h2{{font-size:21px;margin:0 0 8px}}h3{{margin:18px 0 8px}}h4{{margin:8px 0}}.muted{{color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-top:16px}}.card{{background:#f8fafc;border:1px solid var(--line);border-radius:10px;padding:13px}}.card strong{{display:block;font-size:23px}}.card span{{color:var(--muted)}}
.table-wrap{{overflow:auto;margin-top:10px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid var(--line);padding:8px 9px;text-align:right;white-space:nowrap}}th{{background:#f8fafc;position:sticky;top:0}}th:first-child,td:first-child{{text-align:left}}
details{{border-top:1px solid var(--line);padding:9px 0}}summary{{cursor:pointer;font-weight:650}}.bar{{height:8px;background:#e7ebf0;border-radius:9px;min-width:100px}}.bar i{{display:block;height:100%;border-radius:9px}}.binder-fill{{background:#16845b}}.non-binder-fill{{background:#9ca3af}}.badge,.status{{font-size:11px;border-radius:99px;padding:2px 6px}}.badge{{background:#dbeafe;color:#174c9b}}.best{{background:#fef3c7;color:#92400e}}.binder-status{{background:#dcfce7;color:#17663e}}.non-binder-status{{background:#fee2e2;color:#991b1b}}.cleavage-binder-row td:first-child{{border-left:3px solid #16845b}}.non-binder-row td:first-child{{border-left:3px solid #9ca3af;color:#6b7280}}
.scatter-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:16px}}.scatter-facet{{border:1px solid var(--line);border-radius:10px;padding:12px}}svg{{width:100%;height:auto}}svg text{{font-size:11px;fill:var(--muted)}}.grid{{stroke:#e6eaf0;stroke-width:1}}.trend-line{{stroke:#111827;stroke-width:2;stroke-dasharray:6 5;opacity:.7}}
.marker-key,.pair-legend{{display:flex;flex-wrap:wrap;gap:8px 13px;margin:5px 8px;color:var(--muted);font-size:12px}}.marker-key span,.pair-legend span{{display:inline-flex;align-items:center;gap:5px}}.pair-legend i{{width:9px;height:9px;border-radius:50%}}.solid-dot,.hollow-dot{{width:10px;height:10px;border:2px solid #475569;border-radius:50%}}.solid-dot{{background:#475569}}.hollow-dot{{background:#fff}}.dash-line{{width:22px;border-top:2px dashed #111827}}
.bin-flow{{display:grid;grid-template-columns:repeat(5,minmax(145px,1fr));gap:9px;margin:10px 0 16px}}.bin-card{{border:1px solid var(--line);border-radius:10px;padding:12px;background:#fbfcfd}}.bin-card h4{{font-size:15px}}.bin-n,.bin-detail{{color:var(--muted);font-size:12px}}.bin-card label{{display:flex;justify-content:space-between;margin-top:10px;font-size:12px}}.metric-bar{{height:8px;background:#e7ebf0;border-radius:9px;margin-top:4px}}.metric-bar i{{display:block;height:100%;border-radius:9px}}.binder-metric{{background:#16845b}}.retention-metric{{background:#2563eb}}
.warning{{border-left:4px solid #ca8a04;background:#fffbeb;padding:10px 12px;margin:10px 0}}.agreement-block{{margin:18px 0 24px;padding:14px;border:1px solid var(--line);border-radius:10px;background:#fbfcfd}}.metric-explanation{{margin-top:12px;border:1px solid #dbe5f0;border-radius:8px;padding:10px 12px;background:#f8fbff}}.explanation-body{{padding:5px 8px;color:#374151}}code{{background:#eef2f7;border-radius:4px;padding:1px 4px}}
@media(max-width:800px){{main{{padding:10px}}.hero,.section{{padding:15px}}.scatter-grid{{grid-template-columns:1fr}}.bin-flow{{grid-template-columns:1fr}}}}
</style></head><body><main>
<section class="hero"><h1>Affibody blind score 直观分析</h1>
<p class="muted">报告仅包含方案 1、4、6。人类可读图使用原始 <b>{html.escape(str(report["score_key"]))}</b>，不显示取负后的内部排序分数。</p>
<div class="cards">
<div class="card"><strong>{report["counts"]["complexes"]}</strong><span>候选复合物</span></div>
<div class="card"><strong>{report["counts"]["groups"]}</strong><span>library × peptide pair 组</span></div>
<div class="card"><strong>{_fmt(scheme1["model_top1_is_experimental_best_rate"], percent=True)}</strong><span>模型 Top-1 恰为实验最佳</span></div>
<div class="card"><strong>{_fmt(scheme6["top_minus_bottom_binder_rate"], percent=True)}</strong><span>Top–Bottom binder rate 差</span></div>
<div class="card"><strong>{_fmt(scheme6["top_minus_bottom_mean_retention"])}</strong><span>Top–Bottom mean retention 差 (pp)</span></div>
{correlation_cards}
</div></section>
<section class="section"><h2>方案 1：逐组对照模型排序与实验 retention</h2>
<p class="muted">cleavage binder 使用绿色；non-binder 使用灰色 retention 条和红色标签。蓝色标记模型 Top-1，黄色只表示实验 retention 最佳。</p>
{_ranking_agreement_html(ranking_agreement, ranking_agreement_summaries)}
{_ranking_html(rankings)}</section>
<section class="section"><h2>方案 4：模型分数与实验 retention 散点图</h2>
<h3>4A. 原始模型指标</h3>
<p class="muted">横轴直接使用原始 {html.escape(str(report["score_key"]))}，没有取负。{html.escape(raw_axis_note)}；纵轴为 retention。颜色表示 peptide pair，实心为 binder，空心为 non-binder；悬停点可查看 Affibody tuple。</p>
{raw_scatter}
<div class="warning"><b>趋势线说明：</b>虚线仅用于帮助观察整体方向，不是 retention 预测模型，也不用于生成候选预测值。</div>
<h3>4C. 分 Library 的相关性</h3>
<p class="muted">分别计算 Pearson r(retention, -{html.escape(str(report["score_key"]))})。这里显式取负，因此正相关表示原始 total_loss 越低时 retention 越高。该数值只描述线性相关，不代表因果关系或预测准确率。</p>
{correlation_table}
</section>
<section class="section"><h2>方案 6：分数分箱后的实验结果</h2>
<p class="muted">每个 library × peptide pair 组先转换为筛选 percentile，再合并为五档。顺序从模型 Top 20% 到 Bottom 20%；每档同时展示 binder rate、平均/中位 retention 和 IQR。</p>
{_score_bins_html(score_bins)}</section>
<section class="section"><h2>输出说明</h2><p class="muted">group_rankings.csv 和 ranking_agreement.csv 对应方案 1；score_retention_scatter.csv 和 library_correlations.csv 对应方案 4；score_bins.csv 对应方案 6；blind_validation_metrics.json 保存汇总和输入 SHA256。</p></section>
</main></body></html>"""


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    score_records = _read_score_records(args.scores)
    rows = _build_candidate_rows(
        score_records,
        read_csv_rows(args.labels),
        score_key=args.score_key,
        lower_is_better=args.lower_is_better,
    )
    groups = _group_rows(rows)
    if not groups:
        raise ValueError("no labeled score groups are available for analysis")

    rankings, scheme1_summary = _scheme_1_rankings(groups)
    ranking_agreement, ranking_agreement_summaries = (
        _scheme_1_ranking_agreement(groups, top_k=3)
    )
    scheme1_summary["ranking_agreement_summary"] = (
        ranking_agreement_summaries
    )
    scatter_points, scheme4_summary = _scheme_4_scatter(
        rankings, lower_is_better=args.lower_is_better
    )
    library_correlations = _library_correlations(
        scatter_points, score_key=args.score_key
    )
    scheme4_summary[
        "retention_vs_negative_score_correlation_by_library"
    ] = library_correlations
    score_bins, scheme6_summary = _scheme_6_score_bins(rankings)

    output_filenames = {
        "1_group_rankings": "group_rankings.csv",
        "1_ranking_agreement": "ranking_agreement.csv",
        "4_score_retention_scatter": "score_retention_scatter.csv",
        "4_library_correlations": "library_correlations.csv",
        "6_score_bins": "score_bins.csv",
        "summary": "blind_validation_metrics.json",
        "html_report": "blind_analysis_report.html",
    }
    binder_count = sum(bool(row["cleavage_binder"]) for row in rows)
    report = {
        "schema_version": 3,
        "scope": "schemes_1_4_6_only",
        "blind_scores": {
            "path": str(args.scores.resolve()),
            "sha256": sha256_file(args.scores),
            "record_count": len(score_records),
        },
        "labels": {
            "path": str(args.labels.resolve()),
            "sha256": sha256_file(args.labels),
        },
        "score_key": args.score_key,
        "score_direction": (
            "lower_raw_is_better; model_score=-raw_score only for ranking; "
            "human-readable plots use unmodified raw_score"
            if args.lower_is_better
            else "higher_raw_is_better; model_score=raw_score"
        ),
        "counts": {
            "complexes": len(rows),
            "groups": len(groups),
            "binders": binder_count,
            "binder_prevalence": binder_count / len(rows),
        },
        "analyses": {
            "1_group_rankings": scheme1_summary,
            "4_score_retention_scatter": scheme4_summary,
            "6_score_bins": scheme6_summary,
        },
        "outputs": output_filenames,
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / output_filenames["1_group_rankings"], rankings)
    write_csv(
        output_dir / output_filenames["1_ranking_agreement"],
        ranking_agreement,
    )
    write_csv(
        output_dir / output_filenames["4_score_retention_scatter"],
        scatter_points,
    )
    write_csv(
        output_dir / output_filenames["4_library_correlations"],
        library_correlations,
    )
    write_csv(
        output_dir / output_filenames["6_score_bins"],
        score_bins,
    )
    write_json(output_dir / output_filenames["summary"], report)
    (output_dir / output_filenames["html_report"]).write_text(
        _render_html(
            report=report,
            rankings=rankings,
            ranking_agreement=ranking_agreement,
            ranking_agreement_summaries=ranking_agreement_summaries,
            scatter_points=scatter_points,
            score_bins=score_bins,
            library_correlations=library_correlations,
        ),
        encoding="utf-8",
    )

    print(f"Analyzed {len(rows)} blinded complexes in {len(groups)} groups")
    print(f"HTML report: {output_dir / output_filenames['html_report']}")
    print(f"JSON summary: {output_dir / output_filenames['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
