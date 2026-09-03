import csv
import importlib.util
import json
import math
import statistics
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analyze_affibody_blind_scores.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_affibody_blind_scores", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def _write_fixture(tmp_path):
    labels_path = tmp_path / "labels.csv"
    fieldnames = [
        "experiment_id",
        "library",
        "peptide_pair",
        "affibody_tuple",
        "retention_percent",
        "cleavage_binder",
    ]
    labels = [
        ("A1", "LibA", "AA", "AAA", 90, True),
        ("A2", "LibA", "AA", "AAB", 50, False),
        ("A3", "LibA", "AA", "AAC", 10, False),
        ("B1", "LibB", "BB", "BBA", 90, True),
        ("B2", "LibB", "BB", "BBB", 70, False),
        ("B3", "LibB", "BB", "BBC", 10, False),
    ]
    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for experiment_id, library, pair, sequence, retention, binder in labels:
            writer.writerow(
                {
                    "experiment_id": experiment_id,
                    "library": library,
                    "peptide_pair": pair,
                    "affibody_tuple": sequence,
                    "retention_percent": retention,
                    "cleavage_binder": binder,
                }
            )

    model_scores = {
        "A1": (0.9, 0.8),
        "A2": (0.8, 0.7),
        "A3": (0.2, 0.1),
        "B1": (0.7, 0.95),
        "B2": (1.0, 0.8),
        "B3": (0.1, 0.2),
    }
    scores_path = tmp_path / "scores.jsonl"
    records = [
        {
            "complex_id": complex_id,
            "seed": seed,
            "total_loss": 1.0 - value,
        }
        for complex_id, values in model_scores.items()
        for seed, value in zip((101, 102), values)
    ]
    scores_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return scores_path, labels_path


def test_only_schemes_1_4_6_are_reported(tmp_path):
    scores_path, labels_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "analysis"

    assert (
        analysis.main(
            [
                "--scores",
                str(scores_path),
                "--labels",
                str(labels_path),
                "--output-dir",
                str(output_dir),
                "--score-key",
                "total_loss",
                "--lower-is-better",
            ]
        )
        == 0
    )

    expected_files = {
        "blind_analysis_report.html",
        "blind_validation_metrics.json",
        "group_rankings.csv",
        "ranking_agreement.csv",
        "library_correlations.csv",
        "score_bins.csv",
        "score_retention_scatter.csv",
    }
    assert {path.name for path in output_dir.iterdir()} == expected_files

    report_text = (output_dir / "blind_validation_metrics.json").read_text()
    report = json.loads(report_text)
    assert report["scope"] == "schemes_1_4_6_only"
    assert list(report["analyses"]) == [
        "1_group_rankings",
        "4_score_retention_scatter",
        "6_score_bins",
    ]
    assert "human-readable plots use unmodified raw_score" in report[
        "score_direction"
    ]

    with (output_dir / "score_retention_scatter.csv").open(newline="") as handle:
        scatter = list(csv.DictReader(handle))
    a1 = next(row for row in scatter if row["complex_id"] == "A1")
    assert math.isclose(float(a1["raw_score_mean"]), 0.15)

    with (output_dir / "group_rankings.csv").open(newline="") as handle:
        rankings = list(csv.DictReader(handle))
    a1_ranking = next(row for row in rankings if row["complex_id"] == "A1")
    assert math.isclose(
        float(a1_ranking["within_group_selection_percentile"]), 100 / 6
    )

    with (output_dir / "library_correlations.csv").open(newline="") as handle:
        correlations = list(csv.DictReader(handle))
    assert [row["library"] for row in correlations] == ["LibA", "LibB"]
    assert all(row["x_variable"] == "-total_loss" for row in correlations)
    expected_liba = statistics.correlation([0.85, 0.75, 0.15], [90, 50, 10])
    assert math.isclose(float(correlations[0]["pearson_r"]), expected_liba)

    with (output_dir / "score_bins.csv").open(newline="") as handle:
        bins = list(csv.DictReader(handle))
    overall = [row for row in bins if row["scope"] == "Overall"]
    assert len(overall) == 5
    assert sum(int(row["candidate_count"]) for row in overall) == 6
    assert overall[0]["score_bin"] == "Top 20%"
    assert math.isclose(float(overall[0]["binder_rate"]), 0.5)
    assert math.isclose(float(overall[0]["mean_retention"]), 80)
    assert overall[-1]["score_bin"] == "Bottom 20%"
    assert float(overall[-1]["binder_rate"]) == 0
    assert float(overall[-1]["mean_retention"]) == 10

    html_report = (output_dir / "blind_analysis_report.html").read_text()
    for scheme in ("方案 1", "方案 4", "方案 6"):
        assert scheme in html_report
    for removed in ("方案 2", "方案 3", "方案 5", "方案 7"):
        assert removed not in html_report
    assert html_report.count(
        "class='status binder-status'>cleavage binder"
    ) == 2
    assert html_report.count(
        "class='status non-binder-status'>non-binder"
    ) == 4
    assert html_report.count("fill='#ffffff'") == 4
    assert "raw score" in html_report
    assert "model score" not in html_report
    assert "Pearson r(retention, -total_loss)" in html_report
    assert "<h3>4B." not in html_report
    assert "这些指标分别代表什么？" in html_report
    assert "<details class='metric-explanation'><summary>" in html_report


def test_lower_is_better_is_normalized_only_for_ranking(tmp_path):
    scores_path, labels_path = _write_fixture(tmp_path)
    records = [
        json.loads(line)
        for line in scores_path.read_text(encoding="utf-8").splitlines()
    ]
    rows = analysis._build_candidate_rows(
        records,
        analysis.read_csv_rows(labels_path),
        score_key="total_loss",
        lower_is_better=True,
    )
    group_a = [row for row in rows if row["peptide_pair"] == "AA"]
    top = analysis._ordered_by_model(group_a)[0]
    assert top["complex_id"] == "A1"
    assert math.isclose(top["raw_score_mean"], 0.15)
    assert math.isclose(top["model_score"], -0.15)


def test_quartiles_use_inclusive_definition():
    q1, q3 = analysis._quartiles([10, 20, 30, 40])
    assert q1 == 17.5
    assert q3 == 32.5


def test_ranking_agreement_metrics():
    model_order = ["A", "B", "C", "D", "E"]
    retention_by_id = {"A": 50, "B": 20, "C": 40, "D": 30, "E": 10}
    rows = [
        {
            "complex_id": complex_id,
            "library": "LibA",
            "peptide_pair": "AA",
            "model_score": 5 - index,
            "retention_percent": retention_by_id[complex_id],
        }
        for index, complex_id in enumerate(model_order)
    ]
    per_group, summaries = analysis._scheme_1_ranking_agreement(
        {("LibA", "AA"): rows}
    )

    assert per_group[0]["overlap_count"] == 2
    assert math.isclose(per_group[0]["overlap_at_3"], 2 / 3)
    assert math.isclose(per_group[0]["random_expected_overlap_at_3"], 3 / 5)
    assert math.isclose(per_group[0]["adjusted_overlap_at_3"], 1 / 6)
    assert math.isclose(per_group[0]["kendall_tau_b"], 0.6)
    assert summaries[0]["scope"] == "Overall"
