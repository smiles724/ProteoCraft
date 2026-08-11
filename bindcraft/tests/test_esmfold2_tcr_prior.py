"""External TCR sequence prior on the ESMFold2 path.

Covers the two channels the Protenix path already has, checked here without a
GPU or real ESMFold2 weights: the soft prior must reach the proposal gradient
mix, and the hard prior must reach the verifier result that feeds MCTS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pytest
import torch

from bindcraft.hallucination.esmfold2_backend import (
    ESM_AA_ORDER,
    build_gradient_mask,
    combine_normalized_gradients,
)
from bindcraft.hallucination.esmfold2_pipeline import _apply_backend_tcr_prior
from bindcraft.hallucination.esmfold2_runtime import ESMFold2RuntimeBackend


BINDER = "CASRPGLAGGRPEQYF" + "GPGTRLTV"
CDR_MASK = [index < 16 for index in range(len(BINDER))]


@dataclass
class _FakeResult:
    binder_seq: str
    target_seq: str
    total_loss: float
    loss_dict: Dict[str, float] = field(default_factory=dict)
    summary_confidence: Dict[str, Any] = field(default_factory=dict)


class _SequenceDependentScorer:
    """Minimal scorer: prefers alanine, so scores depend on the candidate."""

    gradient_weight = 0.1
    verifier_weight = 0.1
    model_name = "fake"

    def soft_prior_loss(
        self,
        sequence_logits: torch.Tensor,
        region_mask: Optional[Sequence[bool]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        log_probs = torch.log_softmax(sequence_logits, dim=-1)
        target = ESM_AA_ORDER.index("A")
        if region_mask is None:
            rows = list(range(sequence_logits.shape[0]))
        else:
            rows = [index for index, keep in enumerate(region_mask) if keep]
        return -log_probs[rows, target].mean()

    def score_hard_sequence(
        self,
        sequence: str,
        region_mask: Optional[Sequence[bool]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        rows = (
            range(len(sequence))
            if region_mask is None
            else [index for index, keep in enumerate(region_mask) if keep]
        )
        return sum(1.0 for index in rows if sequence[index] == "A")


def _make_backend() -> ESMFold2RuntimeBackend:
    return ESMFold2RuntimeBackend.__new__(ESMFold2RuntimeBackend)


def _attach(backend: ESMFold2RuntimeBackend, **kwargs: Any) -> None:
    ESMFold2RuntimeBackend.attach_tcr_prior(
        backend, _SequenceDependentScorer(), CDR_MASK, **kwargs
    )


def test_backend_starts_without_a_prior_attached() -> None:
    backend = _make_backend()
    assert getattr(backend, "tcr_prior_scorer", None) is None


def test_soft_prior_reaches_the_proposal_gradient() -> None:
    backend = _make_backend()
    _attach(backend, lambda_tcr=0.1)
    logits = torch.zeros(1, len(BINDER), len(ESM_AA_ORDER), requires_grad=True)

    grad = ESMFold2RuntimeBackend._tcr_prior_gradient(
        backend, logits, temperature=1.0, current_seq=BINDER, metadata={}
    )

    assert grad is not None
    assert grad.shape == logits.shape
    # Only CDR positions may carry prior gradient; framework must stay untouched.
    assert torch.any(grad[0, :16] != 0.0)
    assert torch.all(grad[0, 16:] == 0.0)


def test_soft_prior_is_inert_when_lambda_is_zero() -> None:
    backend = _make_backend()
    _attach(backend, lambda_tcr=0.0)
    logits = torch.zeros(1, len(BINDER), len(ESM_AA_ORDER), requires_grad=True)

    assert (
        ESMFold2RuntimeBackend._tcr_prior_gradient(
            backend, logits, temperature=1.0, current_seq=BINDER, metadata={}
        )
        is None
    )


def test_combiner_moves_the_gradient_only_when_lambda_is_nonzero() -> None:
    gradient_mask = build_gradient_mask(BINDER, mutable_mask=CDR_MASK, batch_size=1)
    structure_grad = torch.randn(1, len(BINDER), len(ESM_AA_ORDER))
    tcr_grad = torch.randn(1, len(BINDER), len(ESM_AA_ORDER))

    baseline = combine_normalized_gradients(
        structure_grad, None, gradient_mask=gradient_mask, lambda_esmc=0.0
    )
    off = combine_normalized_gradients(
        structure_grad,
        None,
        gradient_mask=gradient_mask,
        lambda_esmc=0.0,
        tcr_grad=tcr_grad,
        lambda_tcr=0.0,
    )
    on = combine_normalized_gradients(
        structure_grad,
        None,
        gradient_mask=gradient_mask,
        lambda_esmc=0.0,
        tcr_grad=tcr_grad,
        lambda_tcr=0.1,
    )

    assert torch.allclose(baseline, off)
    assert not torch.allclose(baseline, on)


def test_gradient_diagnostics_report_norms_and_cosines() -> None:
    backend = _make_backend()
    _attach(backend, lambda_tcr=0.1)
    gradient_mask = build_gradient_mask(BINDER, mutable_mask=CDR_MASK, batch_size=1)
    structure_grad = torch.randn(1, len(BINDER), len(ESM_AA_ORDER))
    esmc_grad = torch.randn(1, len(BINDER), len(ESM_AA_ORDER))

    diagnostics = ESMFold2RuntimeBackend._log_tcr_prior_gradient_diagnostics(
        backend,
        stage="unit",
        global_step=1,
        soft_step=0,
        structure_grad=structure_grad,
        esmc_grad=esmc_grad,
        tcr_grad=structure_grad.clone(),
        gradient_mask=gradient_mask,
    )

    assert diagnostics["tcr_grad_norm"] > 0.0
    assert diagnostics["cos_tcr_structure"] == pytest.approx(1.0, abs=1e-5)
    assert -1.0 <= diagnostics["cos_tcr_esmc"] <= 1.0


def test_hard_prior_changes_total_loss_and_varies_by_candidate() -> None:
    backend = _make_backend()
    _attach(backend)
    natural = _apply_backend_tcr_prior(
        backend, _FakeResult(binder_seq=BINDER, target_seq="X", total_loss=-1.0)
    )
    alanine = _apply_backend_tcr_prior(
        backend,
        _FakeResult(binder_seq="A" * len(BINDER), target_seq="X", total_loss=-1.0),
    )

    assert "tcr_prior_hard" in natural.loss_dict
    assert natural.total_loss != -1.0
    # A constant score would mean the prior never saw the candidate.
    assert natural.total_loss != alanine.total_loss


def test_hard_prior_is_skipped_when_no_prior_is_attached() -> None:
    backend = _make_backend()
    result = _apply_backend_tcr_prior(
        backend, _FakeResult(binder_seq=BINDER, target_seq="X", total_loss=-1.0)
    )

    assert result.total_loss == -1.0
    assert "tcr_prior_hard" not in result.loss_dict
