"""Tests for report.py CLI auto-discover path."""

import sys
import warnings

import pandas as pd
import pytest


def test_report_main_auto_discover_with_roster_fill(tmp_path, monkeypatch):
    from eval_pipeline import report

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    pd.DataFrame([
        {
            "evaluator": "af3",
            "method": "BCv1",
            "case": "PD1",
            "paper_case": "PD1",
            "design_id": "PD1_d_001",
            "eval_n_calls": 1,
            "eval_n_successes": 1,
            "eval_runtime_s": 2.0,
            "tier1_pass": True,
            "tier2_pass": False,
            "error_msg": "",
        }
    ]).to_csv(eval_dir / "PD1__PD1__af3.csv", index=False)
    roster_path = tmp_path / "roster.json"
    roster_path.write_text('{"paper_cases": ["PD1", "PDL1"]}')

    monkeypatch.setattr(sys, "argv", [
        "report.py",
        "--out-dir", str(tmp_path),
        "--auto-discover",
        "--case-roster", str(roster_path),
        "--evaluators", "af3",
        "--run-id", "unit",
    ])
    report.main()

    agg = pd.read_csv(tmp_path / "eval_per_paper_case_aggregate.csv")
    assert set(agg["paper_case"]) == {"PD1", "PDL1"}
    assert (tmp_path / "report_af3.md").exists()


def test_render_table2_pivot_shape(tmp_path):
    from eval_pipeline.report import render_table2

    agg = pd.DataFrame([
        {
            "evaluator": "esmfold",
            "method": "BCv1",
            "paper_case": "PD1",
            "n_designs_evaluated": 10,
            "tier1_rate": 0.5,
            "tier2_rate": 0.3,
            "success_rate": 0.9,
        },
        {
            "evaluator": "af3",
            "method": "BCv1",
            "paper_case": "PD1",
            "n_designs_evaluated": 10,
            "tier1_rate": 0.6,
            "tier2_rate": 0.4,
            "success_rate": 1.0,
        },
    ])
    out_csv = tmp_path / "table2.csv"
    out_md = tmp_path / "table2.md"
    render_table2(agg, str(out_csv), str(out_md))

    wide = pd.read_csv(out_csv)
    assert len(wide) == 1
    assert "paper_case" in wide.columns
    assert wide.loc[0, "BCv1_esmfold_tier1_rate"] == 0.5
    assert wide.loc[0, "BCv1_af3_tier1_rate"] == 0.6
    assert "Table 2" in out_md.read_text()


def test_render_table2_multi_method_no_crash(tmp_path):
    """CR2-1: Table2 must support method x evaluator columns."""
    from eval_pipeline.report import render_table2

    rows = []
    for paper_case in ("PD1", "PDL1"):
        for method in ("BCv1", "BCv2"):
            for evaluator in ("esmfold", "af3"):
                rows.append({
                    "evaluator": evaluator,
                    "method": method,
                    "paper_case": paper_case,
                    "n_designs_evaluated": 10,
                    "tier1_rate": 0.5,
                    "tier2_rate": 0.25,
                    "success_rate": 0.9,
                })
    out_csv = tmp_path / "table2.csv"
    out_md = tmp_path / "table2.md"

    render_table2(pd.DataFrame(rows), str(out_csv), str(out_md))

    wide = pd.read_csv(out_csv)
    assert len(wide) == 2
    assert "BCv1_esmfold_tier1_rate" in wide.columns
    assert "BCv2_af3_success_rate" in wide.columns


def test_render_table2_duplicate_keys_raises(tmp_path):
    """CR2-1: duplicate paper_case/method/evaluator rows should fail loud."""
    from eval_pipeline.report import render_table2

    agg = pd.DataFrame([
        {
            "evaluator": "esmfold",
            "method": "BCv1",
            "paper_case": "PD1",
            "n_designs_evaluated": 10,
            "tier1_rate": 0.5,
            "tier2_rate": 0.25,
            "success_rate": 0.9,
        },
        {
            "evaluator": "esmfold",
            "method": "BCv1",
            "paper_case": "PD1",
            "n_designs_evaluated": 8,
            "tier1_rate": 0.4,
            "tier2_rate": 0.2,
            "success_rate": 0.8,
        },
    ])

    with pytest.raises(ValueError, match="duplicate"):
        render_table2(agg, str(tmp_path / "table2.csv"), str(tmp_path / "table2.md"))


def test_render_table2_empty_agg_writes_header(tmp_path):
    """SF2-3: empty Table2 CSV must be readable by pandas."""
    from eval_pipeline.report import render_table2

    out_csv = tmp_path / "table2.csv"
    out_md = tmp_path / "table2.md"
    render_table2(pd.DataFrame(), str(out_csv), str(out_md))

    wide = pd.read_csv(out_csv)
    assert wide.columns.tolist() == ["paper_case"]
    assert len(wide) == 0
    assert "empty" in out_md.read_text()


def test_main_writes_table2_and_bar_chart_outputs(tmp_path, monkeypatch):
    from eval_pipeline import report

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    pd.DataFrame([{
        "evaluator": "af3",
        "method": "BCv1",
        "case": "PD1",
        "paper_case": "PD1",
        "design_id": "PD1_d_001",
        "binder_pLDDT": 90.0,
        "ipAE": 5.0,
        "binder_scRMSD": 1.0,
        "tier1_pass": True,
        "tier2_pass": True,
        "eval_n_calls": 1,
        "eval_n_successes": 1,
        "eval_runtime_s": 2.0,
        "error_msg": "",
    }]).to_csv(eval_dir / "PD1__PD1__af3.csv", index=False)

    monkeypatch.setattr(sys, "argv", [
        "report.py",
        "--out-dir", str(tmp_path),
        "--auto-discover",
        "--evaluators", "af3",
        "--run-id", "unit",
    ])
    report.main()

    assert (tmp_path / "table2.csv").exists()
    assert (tmp_path / "table2.md").exists()
    assert (tmp_path / "bar_chart_af3.png").exists()


def test_bar_chart_stub_paper_case_not_drawn_as_zero(tmp_path, monkeypatch):
    """CR2-2: NaN rate means no designs, not a real 0% pass bar."""
    import matplotlib.pyplot as plt
    from eval_pipeline.report import render_bar_chart_paper_case

    monkeypatch.setattr(plt, "close", lambda fig: None)
    sub_agg = pd.DataFrame([
        {"paper_case": "PD1", "tier1_rate": 0.5, "tier2_rate": 0.25},
        {"paper_case": "TCR_1QRN", "tier1_rate": float("nan"), "tier2_rate": float("nan")},
    ])
    out_png = tmp_path / "bar_chart_af3.png"

    render_bar_chart_paper_case(sub_agg, "af3", str(out_png))

    ax = plt.gcf().axes[0]
    assert [round(p.get_height(), 3) for p in ax.patches if p.get_height() > 1.0] == [50.0, 25.0]
    assert not [p for p in ax.patches if abs(p.get_height()) < 1e-9]
    assert any(text.get_text() == "no designs" for text in ax.texts)
    assert out_png.exists() and out_png.stat().st_size > 0
    plt.close("all")


def test_bar_chart_all_real_data_has_no_no_designs_label(tmp_path, monkeypatch):
    """CR2-2: all-real chart should render tier bars without no-designs label."""
    import matplotlib.pyplot as plt
    from eval_pipeline.report import render_bar_chart_paper_case

    monkeypatch.setattr(plt, "close", lambda fig: None)
    sub_agg = pd.DataFrame([
        {"paper_case": "PD1", "tier1_rate": 0.5, "tier2_rate": 0.25},
        {"paper_case": "PDL1", "tier1_rate": 0.0, "tier2_rate": 0.0},
    ])

    render_bar_chart_paper_case(sub_agg, "af3", str(tmp_path / "bar_chart_af3.png"))

    ax = plt.gcf().axes[0]
    assert not any(text.get_text() == "no designs" for text in ax.texts)
    assert len(ax.patches) == 4
    plt.close("all")


def test_bar_chart_all_stub_cases_renders_no_designs(tmp_path, monkeypatch):
    """CR2-2: all-stub chart should render without pretending 0% pass rates."""
    import matplotlib.pyplot as plt
    from eval_pipeline.report import render_bar_chart_paper_case

    monkeypatch.setattr(plt, "close", lambda fig: None)
    sub_agg = pd.DataFrame([
        {"paper_case": "TCR_1QRN", "tier1_rate": float("nan"), "tier2_rate": float("nan")},
    ])
    out_png = tmp_path / "bar_chart_af3.png"

    render_bar_chart_paper_case(sub_agg, "af3", str(out_png))

    ax = plt.gcf().axes[0]
    assert any(text.get_text() == "no designs" for text in ax.texts)
    assert out_png.exists() and out_png.stat().st_size > 0
    plt.close("all")


def test_main_long_csv_with_missing_path_raises(tmp_path, monkeypatch):
    from eval_pipeline import report

    monkeypatch.setattr(sys, "argv", [
        "report.py",
        "--long-csv", str(tmp_path / "does_not_exist.csv"),
        "--out-dir", str(tmp_path),
    ])

    with pytest.raises(SystemExit, match="do not exist"):
        report.main()


def test_main_long_csv_partial_missing_raises(tmp_path, monkeypatch):
    from eval_pipeline import report

    real_csv = tmp_path / "real.csv"
    pd.DataFrame([{
        "evaluator": "af3",
        "method": "BCv1",
        "case": "PD1",
        "paper_case": "PD1",
        "design_id": "PD1_d_001",
        "binder_pLDDT": 90.0,
        "ipAE": 5.0,
        "binder_scRMSD": 1.0,
        "tier1_pass": True,
        "tier2_pass": True,
        "eval_n_calls": 1,
        "eval_n_successes": 1,
        "eval_runtime_s": 5.0,
        "error_msg": "",
    }]).to_csv(real_csv, index=False)

    monkeypatch.setattr(sys, "argv", [
        "report.py",
        "--long-csv", str(real_csv), str(tmp_path / "missing.csv"),
        "--out-dir", str(tmp_path),
    ])

    with pytest.raises(SystemExit, match="do not exist"):
        report.main()


def test_numeric_cols_includes_new_evaluator_cols():
    from eval_pipeline.report import NUMERIC_COLS

    expected_subset = {
        "binder_pLDDT", "complex_pLDDT", "ipAE", "binder_scRMSD",
        "esmfold_runtime_s", "verifier_calls",
        "eval_n_calls", "eval_n_successes", "eval_runtime_s",
        "af3_ranking_score", "af3_iptm", "af3_ptm_native",
        "af3_has_clash", "af3_fraction_disordered",
    }
    assert expected_subset <= set(NUMERIC_COLS)
    assert len(NUMERIC_COLS) == len(set(NUMERIC_COLS))


def test_coerce_converts_new_cols_to_numeric():
    from eval_pipeline.report import _coerce

    df = pd.DataFrame({
        "eval_n_calls": ["1", "2"],
        "eval_n_successes": ["0", "1"],
        "af3_ranking_score": ["0.5", "0.7"],
        "af3_iptm": ["0.6", "nan"],
    })

    out = _coerce(df)
    assert out["eval_n_calls"].dtype.kind in ("i", "f")
    assert out["eval_n_successes"].dtype.kind in ("i", "f")
    assert out["af3_ranking_score"].dtype.kind == "f"
    assert pd.isna(out["af3_iptm"].iloc[1])


def test_normalize_eval_schema_legacy_nan_error_msg():
    """SF-9: legacy CSV with error_msg=NaN should not zero out eval_n_successes."""
    from eval_pipeline.report import normalize_eval_schema

    df = pd.DataFrame([{
        "case": "PD1",
        "design_id": "d1",
        "binder_pLDDT": 90.0,
        "ipAE": 5.0,
        "binder_scRMSD": 1.0,
        "error_msg": float("nan"),
        "tier1_pass": True,
        "tier2_pass": True,
    }])

    out = normalize_eval_schema(df)
    assert out["eval_n_successes"].iloc[0] == 1


def test_normalize_eval_schema_missing_design_id_uses_legacy_sentinel():
    """SF-10: missing design_id should be explicit legacy real-row sentinel."""
    from eval_pipeline.report import normalize_eval_schema

    out = normalize_eval_schema(pd.DataFrame([{"case": "PD1"}]))
    assert out["design_id"].iloc[0] == "unknown_legacy"


def test_aggregate_stub_only_does_not_emit_futurewarning():
    """SF-8: stub-only aggregation should not concat empty real_agg with stub_agg."""
    from eval_pipeline.report import aggregate_per_paper_case

    df = pd.DataFrame([{
        "evaluator": "af3",
        "method": "BCv1",
        "case": "TCR_1QRN",
        "paper_case": "TCR_1QRN",
        "design_id": "_NO_DESIGNS",
        "eval_n_calls": 0,
        "eval_n_successes": 0,
        "eval_runtime_s": 0.0,
        "tier1_pass": False,
        "tier2_pass": False,
        "error_msg": "case_has_zero_accepted_designs",
    }])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agg = aggregate_per_paper_case(df)

    assert len(agg) == 1
    assert not [w for w in caught if issubclass(w.category, FutureWarning)]


def test_render_markdown_for_evaluator_includes_top_designs(tmp_path):
    """SF-12: CR-7 markdown should use get_top_designs instead of leaving it dead."""
    from eval_pipeline.report import render_markdown_for_evaluator

    sub_agg = pd.DataFrame([{
        "evaluator": "af3",
        "method": "BCv1",
        "paper_case": "PD1",
        "n_designs_evaluated": 1,
        "n_successes": 1,
        "success_rate": 1.0,
        "tier1_rate": 1.0,
        "tier2_rate": 0.0,
        "eval_runtime_s_total": 2.0,
    }])
    sub_df = pd.DataFrame([{
        "method": "BCv1",
        "case": "PD1",
        "design_id": "top_d1",
        "binder_pLDDT": 95.0,
        "complex_pLDDT": 90.0,
        "ipAE": 5.0,
        "binder_scRMSD": 1.0,
    }])

    out_md = tmp_path / "report_af3.md"
    render_markdown_for_evaluator(sub_agg, sub_df, "unit", "af3", str(out_md))

    text = out_md.read_text()
    assert "## Top designs" in text
    assert "top_d1" in text


def test_render_markdown_for_af3_native_metrics_without_component_metrics(tmp_path):
    """SF2-5: AF3 reports should summarize native AF3 metrics when present."""
    from eval_pipeline.report import render_markdown_for_evaluator

    sub_agg = pd.DataFrame([{
        "evaluator": "af3",
        "method": "BCv1",
        "paper_case": "PD1",
        "n_designs_evaluated": 2,
        "n_successes": 2,
        "success_rate": 1.0,
        "tier1_rate": 0.5,
        "tier2_rate": 0.0,
        "eval_runtime_s_total": 4.0,
    }])
    sub_df = pd.DataFrame([
        {
            "method": "BCv1",
            "case": "PD1",
            "design_id": "d1",
            "af3_ranking_score": 0.7,
            "af3_iptm": 0.6,
            "af3_ptm_native": 0.8,
            "af3_has_clash": 0,
            "af3_fraction_disordered": 0.1,
        },
        {
            "method": "BCv1",
            "case": "PD1",
            "design_id": "d2",
            "af3_ranking_score": 0.9,
            "af3_iptm": 0.7,
            "af3_ptm_native": 0.85,
            "af3_has_clash": 1,
            "af3_fraction_disordered": 0.2,
        },
    ])

    out_md = tmp_path / "report_af3.md"
    render_markdown_for_evaluator(sub_agg, sub_df, "unit", "af3", str(out_md))

    text = out_md.read_text()
    assert "## AF3 native metrics" in text
    assert "`af3_iptm mean`: 0.650" in text
    assert "`af3_has_clash != 0`: 1/2" in text
    assert "Component metrics unavailable" not in text
