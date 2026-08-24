"""Tests for the natural (empirical) TCR-pMHC contact prior."""

from __future__ import annotations

import pytest

from bindcraft.hallucination.tcr_natural_prior import (
    load_natural_stats,
    natural_r_pep,
    summary_table,
)


def test_reference_set_size():
    assert load_natural_stats().n_structures == 216


def test_r_pep_targets_are_sane_ratios():
    s = load_natural_stats()
    for v in (s.r_pep_alpha, s.r_pep_beta, s.r_pep_combined):
        assert 0.0 < v < 1.0
    # combined must sit between the two per-chain values
    assert min(s.r_pep_alpha, s.r_pep_beta) <= s.r_pep_combined <= max(
        s.r_pep_alpha, s.r_pep_beta
    )


def test_r_pep_lookup_accepts_chain_aliases():
    assert natural_r_pep("alpha") == natural_r_pep("TCR_alpha")
    assert natural_r_pep("beta") == natural_r_pep("b")
    with pytest.raises(ValueError):
        natural_r_pep("gamma")


def test_cdr3_dominates_peptide_contact():
    """CDR3 must be the largest single contributor -- sanity on the parse."""
    s = load_natural_stats()
    top = max(s.region_share, key=lambda k: s.region_share[k])
    assert top[1] == "CDR3"


def test_cdr1_cdr2_really_do_touch_pmhc():
    """The reason they must not be penalised as framework."""
    s = load_natural_stats()
    for chain in ("alpha", "beta"):
        for reg in ("CDR1", "CDR2"):
            assert s.region_pmhc_prob[(chain, reg)] > 0.25, (chain, reg)


def test_framework_touches_pmhc_far_less_than_cdrs():
    s = load_natural_stats()
    for chain in ("alpha", "beta"):
        fr = max(s.region_pmhc_prob[(chain, r)] for r in ("FR1", "FR2", "FR3", "FR4")
                 if (chain, r) in s.region_pmhc_prob)
        cdr = min(s.region_pmhc_prob[(chain, r)] for r in ("CDR1", "CDR2", "CDR3"))
        assert fr < cdr, chain


def test_framework_peptide_contact_is_near_zero():
    """Framework-peptide contact is what the penalty term targets."""
    s = load_natural_stats()
    for chain in ("alpha", "beta"):
        for reg in ("FR1", "FR3", "FR4"):
            if (chain, reg) in s.region_pep_prob:
                assert s.region_pep_prob[(chain, reg)] < 0.05, (chain, reg)


def test_shares_sum_to_about_one():
    s = load_natural_stats()
    assert 0.97 <= sum(s.region_share.values()) <= 1.03


def test_summary_table_renders():
    text = summary_table()
    assert "r_pep target" in text and "216" in text
