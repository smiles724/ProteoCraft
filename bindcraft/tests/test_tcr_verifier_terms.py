"""Guard against the verifier double-counting the TCR terms.

The pipeline caches verifier results by binder sequence, so the TCR contact
loss and hard prior must be folded in *inside* ``verify`` -- exactly once per
freshly computed result -- and never re-applied on a cache hit.
"""

from __future__ import annotations

import types

import pytest
import torch

from bindcraft.hallucination.esmfold2_runtime import ESMFold2RuntimeBackend
from bindcraft.hallucination.tcr_contact import TCRContactConfig


class _Result:
    def __init__(self, binder_seq: str, total_loss: float):
        self.binder_seq = binder_seq
        self.total_loss = total_loss
        self.loss_dict = {"total": total_loss}


def _backend(**contact_overrides):
    backend = ESMFold2RuntimeBackend.__new__(ESMFold2RuntimeBackend)
    backend.device = torch.device("cpu")
    settings = {"enabled": True, "verifier_weight": 1.0, "use_verifier": True}
    settings.update(contact_overrides)
    backend._tcr_contact_config = TCRContactConfig(**settings)
    backend._tcr_contact_mask_cache = {}
    backend._tcr_prior_raw_config = {}
    backend._tcr_prior_scorer_cached = None
    backend._tcr_prior_scorer_built = True  # no scorer
    backend.config = types.SimpleNamespace(binder_mutable_mask=None)
    return backend


TERMS = {"tcr_contact_loss": 2.5, "cdr3_peptide_contact_nll": 2.5, "r_pep": 0.3}


def test_contact_term_added_exactly_once():
    backend = _backend()
    result = backend._apply_tcr_verifier_terms(_Result("AAA", 1.0), contact_terms=TERMS)
    assert result.total_loss == pytest.approx(3.5)
    assert result.loss_dict["tcr_contact_weighted"] == pytest.approx(2.5)
    assert result.loss_dict["total"] == pytest.approx(3.5)


def test_repeated_verifies_of_same_sequence_agree():
    """Two independent verify() calls must yield the same total, not a sum."""

    backend = _backend()
    first = backend._apply_tcr_verifier_terms(_Result("AAA", 1.0), contact_terms=TERMS)
    second = backend._apply_tcr_verifier_terms(_Result("AAA", 1.0), contact_terms=TERMS)
    assert first.total_loss == pytest.approx(second.total_loss)


def test_verifier_weight_scales_the_contribution():
    backend = _backend(verifier_weight=0.5)
    result = backend._apply_tcr_verifier_terms(_Result("AAA", 1.0), contact_terms=TERMS)
    assert result.total_loss == pytest.approx(2.25)


def test_diagnostics_recorded_without_touching_total_when_verifier_off():
    backend = _backend(use_verifier=False)
    result = backend._apply_tcr_verifier_terms(_Result("AAA", 1.0), contact_terms=TERMS)
    assert result.total_loss == pytest.approx(1.0)
    assert result.loss_dict["r_pep"] == pytest.approx(0.3)
    assert "tcr_contact_weighted" not in result.loss_dict


def test_no_contact_terms_is_a_noop():
    backend = _backend()
    result = backend._apply_tcr_verifier_terms(_Result("AAA", 1.0), contact_terms=None)
    assert result.total_loss == pytest.approx(1.0)


def test_pipeline_verifier_cache_returns_same_object():
    """The cache must hand back the stored result rather than recomputing."""

    from bindcraft.hallucination import esmfold2_pipeline as pipeline

    cache: dict = {}
    calls = {"n": 0}

    class _Backend:
        def verify(self, *, binder_seq, target_seq, seed=None, hotspot_params=None):
            calls["n"] += 1
            return _Result(binder_seq, 1.0 + calls["n"])

    fn = pipeline._verify_with_cache
    backend = _Backend()
    first = fn(backend, cache, binder_seq="AAA", target_seq="TTT", seed=1)
    second = fn(backend, cache, binder_seq="AAA", target_seq="TTT", seed=1)
    assert first is second, "cache hit must return the stored result"
    assert calls["n"] == 1, "verify must not run twice for one sequence"
    assert first.total_loss == pytest.approx(2.0)
