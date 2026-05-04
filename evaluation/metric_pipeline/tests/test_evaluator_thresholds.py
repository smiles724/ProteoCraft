"""Tests for tier_gating.py THRESHOLDS_BY_EVALUATOR (spec §8)."""
import pytest

from eval_pipeline.tier_gating import (
    THRESHOLDS,
    THRESHOLDS_BY_EVALUATOR,
    compute_tier_flags,
)


def _passing_metrics():
    return {
        "binder_pLDDT": 92.0,
        "complex_pLDDT": 91.0,
        "ipAE": 5.0,
        "binder_scRMSD": 1.2,
        "inter_chain_clash_count": 0,
    }


def test_thresholds_dict_has_both_evaluators():
    assert "esmfold" in THRESHOLDS_BY_EVALUATOR
    assert "af3" in THRESHOLDS_BY_EVALUATOR
    keys = {
        "binder_plddt_strict",
        "binder_plddt_loose",
        "complex_plddt",
        "ipae_max",
        "scrmsd_max",
        "clash_max",
    }
    for ev in ("esmfold", "af3"):
        assert keys <= set(THRESHOLDS_BY_EVALUATOR[ev].keys()), f"{ev} missing keys"


def test_thresholds_backward_compat_alias():
    assert THRESHOLDS == THRESHOLDS_BY_EVALUATOR["esmfold"]


def test_compute_tier_flags_esmfold_default_no_regression():
    flags = compute_tier_flags(_passing_metrics())
    assert flags["tier1_pass"] is True
    assert flags["tier2_pass"] is True


def test_compute_tier_flags_af3_branch():
    flags = compute_tier_flags(_passing_metrics(), evaluator="af3")
    assert flags["tier1_pass"] is True
    assert flags["tier2_pass"] is True


def test_compute_tier_flags_unknown_evaluator_raises():
    with pytest.raises(KeyError):
        compute_tier_flags(_passing_metrics(), evaluator="unknown")
