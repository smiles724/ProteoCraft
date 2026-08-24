"""Tests for the r_pep_target contact objective."""
from __future__ import annotations

import pytest
import torch

from bindcraft.hallucination.tcr_contact import (
    TCRContactConfig,
    build_tcr_contact_masks,
    resolve_annotation,
    tcr_contact_losses,
)
from bindcraft.hallucination.tcr_natural_prior import natural_r_pep
from bindcraft.tests.test_tcr_contact import (
    A6_BETA_SEQ,
    BETA_TARGET_LENGTHS,
    NUM_BINS,
)


def _setup(**kw):
    cfg = TCRContactConfig(
        enabled=True, peptide_chain_index=0, mhc_chain_indices=(1,), **kw
    )
    masks = build_tcr_contact_masks(
        binder_length=len(A6_BETA_SEQ),
        target_chain_lengths=BETA_TARGET_LENGTHS,
        annotation=resolve_annotation(A6_BETA_SEQ),
        config=cfg,
        device=torch.device("cpu"),
    )
    return cfg, masks


def _dgram(seed: int, requires_grad: bool = False):
    full = sum(BETA_TARGET_LENGTHS) + len(A6_BETA_SEQ)
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(1, full, full, NUM_BINS, generator=g)
    return t.requires_grad_(requires_grad)


def test_target_objective_requires_a_target():
    cfg, masks = _setup(objective="r_pep_target")
    with pytest.raises(ValueError, match="requires r_pep_target"):
        tcr_contact_losses(_dgram(0), masks, config=cfg)


def test_r_pep_carries_gradient_when_it_is_the_objective():
    cfg, masks = _setup(objective="r_pep_target", r_pep_target=natural_r_pep("beta"))
    d = _dgram(1, requires_grad=True)
    out = tcr_contact_losses(d, masks, config=cfg)
    assert out["r_pep"].requires_grad
    grad = torch.autograd.grad(out["tcr_contact_loss"].mean(), d)[0]
    assert torch.isfinite(grad).all() and grad.norm().item() > 0.0


def test_r_pep_stays_detached_under_the_old_objective():
    cfg, masks = _setup(objective="nll_hinge")
    out = tcr_contact_losses(_dgram(2, requires_grad=True), masks, config=cfg)
    assert not out["r_pep"].requires_grad


def test_loss_is_zero_exactly_at_the_target():
    """The defining property: on-target is optimal, not 'more is better'."""
    cfg, masks = _setup(objective="r_pep_target", r_pep_target=0.35)
    out = tcr_contact_losses(_dgram(3), masks, config=cfg)
    observed = float(out["r_pep"].item())
    cfg_on, _ = _setup(objective="r_pep_target", r_pep_target=observed)
    on = tcr_contact_losses(_dgram(3), masks, config=cfg_on)
    assert float(on["tcr_contact_loss"].item()) < 1e-10


def test_overshooting_the_target_is_penalised_like_undershooting():
    """This is what the old 'higher r_pep is better' reading got wrong."""
    cfg, masks = _setup(objective="r_pep_target", r_pep_target=0.30)
    out = tcr_contact_losses(_dgram(4), masks, config=cfg)
    r = float(out["r_pep"].item())
    lo = TCRContactConfig(enabled=True, objective="r_pep_target",
                          r_pep_target=r - 0.1, peptide_chain_index=0,
                          mhc_chain_indices=(1,))
    hi = TCRContactConfig(enabled=True, objective="r_pep_target",
                          r_pep_target=r + 0.1, peptide_chain_index=0,
                          mhc_chain_indices=(1,))
    a = tcr_contact_losses(_dgram(4), masks, config=lo)["tcr_contact_loss"]
    b = tcr_contact_losses(_dgram(4), masks, config=hi)["tcr_contact_loss"]
    assert abs(float(a.item()) - float(b.item())) < 1e-9


def test_natural_targets_are_wired_through():
    for chain, expect in (("alpha", 0.2953), ("beta", 0.3468)):
        assert abs(natural_r_pep(chain) - expect) < 5e-4
