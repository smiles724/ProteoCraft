"""Tests for case roster stub fill and aggregate special-case."""
import math

import pandas as pd


def _make_real_row(evaluator, case, success=True):
    return {
        "evaluator": evaluator,
        "method": "BCv1",
        "case": case,
        "paper_case": case,
        "design_id": f"{case}_d_001",
        "n_accepted_by_method": 80,
        "eval_n_calls": 1,
        "eval_n_successes": 1 if success else 0,
        "eval_runtime_s": 5.0,
        "binder_pLDDT": 88.0 if success else float("nan"),
        "ipAE": 5.5,
        "binder_scRMSD": 1.2,
        "tier1_pass": success,
        "tier2_pass": success,
        "error_msg": "" if success else "fail",
    }


def test_fill_missing_paper_cases_adds_stubs():
    from eval_pipeline.report import fill_missing_paper_cases

    df = pd.DataFrame([
        _make_real_row("esmfold", "PD1"),
        _make_real_row("af3", "PD1"),
        _make_real_row("esmfold", "PDL1"),
    ])
    roster = ["PD1", "PDL1", "TCR_1QRN"]
    out = fill_missing_paper_cases(df, roster=roster, evaluators=["esmfold", "af3"])
    assert len(out) == 6
    stubs = out[out["design_id"] == "_NO_DESIGNS"]
    assert len(stubs) == 3
    actual_pairs = set(zip(stubs["evaluator"], stubs["paper_case"]))
    expected_pairs = {
        ("esmfold", "TCR_1QRN"),
        ("af3", "PDL1"),
        ("af3", "TCR_1QRN"),
    }
    assert actual_pairs == expected_pairs
    assert all(stubs["eval_n_calls"] == 0)
    assert all(stubs["eval_n_successes"] == 0)


def test_aggregate_per_paper_case_stub_special_case():
    from eval_pipeline.report import aggregate_per_paper_case

    df = pd.DataFrame([
        _make_real_row("af3", "PD1"),
        _make_real_row("af3", "PD1"),
        {
            "evaluator": "af3", "method": "BCv1", "case": "TCR_1QRN",
            "paper_case": "TCR_1QRN", "design_id": "_NO_DESIGNS",
            "n_accepted_by_method": 0, "eval_n_calls": 0,
            "eval_n_successes": 0, "eval_runtime_s": 0.0,
            "binder_pLDDT": float("nan"), "ipAE": float("nan"),
            "binder_scRMSD": float("nan"),
            "tier1_pass": False, "tier2_pass": False,
            "error_msg": "case_has_zero_accepted_designs",
        },
    ])
    agg = aggregate_per_paper_case(df)
    pd1_row = agg[agg["paper_case"] == "PD1"].iloc[0]
    tcr_row = agg[agg["paper_case"] == "TCR_1QRN"].iloc[0]
    assert pd1_row["n_designs_evaluated"] == 2
    assert pd1_row["n_successes"] == 2
    assert pd1_row["success_rate"] == 1.0
    assert tcr_row["n_designs_evaluated"] == 0
    assert tcr_row["n_calls"] == 0
    assert tcr_row["n_successes"] == 0
    assert tcr_row["success_rate"] == 0.0
    assert math.isnan(tcr_row["tier1_rate"])
    assert math.isnan(tcr_row["tier2_rate"])
