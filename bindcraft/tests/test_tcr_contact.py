"""Tests for the TCR-specific contact loss (masks, gradients, diagnostics)."""

from __future__ import annotations

import pytest
import torch

from bindcraft.hallucination.tcr_contact import (
    A6_ALPHA,
    A6_BETA,
    TCRContactConfig,
    build_tcr_contact_masks,
    lookup_annotation,
    resolve_annotation,
    sequence_sha256,
    tcr_contact_losses,
)

A6_ALPHA_SEQ = (
    "KEVEQNSGPLSVPEGAIASLNCTYSDRGSQSFFWYRQYSGKSPELIMSIYSNGDKEDGRF"
    "TAQLNKASQYVSLLIRDSQPSDSATYLCAVTTDSWGKLQFGAGTQVVVTPDIQNPDPAVY"
    "QLRDSKSSDKSVCLFTDFDSQTNVSQSKDKDVYITDKTVLDMRSMDFKSNSAVAWSNKSD"
    "FACANAFNNSIIPEDTFFPS"
)
A6_BETA_SEQ = (
    "GVTQTPKFQVLKTGQSMTLQCAQDMNHEYMSWYRQDPGMGLRLIHYSVGAGITDQGEVPN"
    "GYNVSRSTTEDFPLRLLSAAPSQTSVYFCASRPGLAGGRPEQYFGPGTRLTVTEDLKNVF"
    "PPEVAVFEPSAEEISHTQKATLVCLATGFYPDHVELSWWVNGKEVHSGVSTDPQPLKEQP"
    "ALNDSRYALSSRLRVSATFWQNPRNHFRCQVQFYGLSENDEWTQDRAKPVTQIVSAEAWG"
    "RAD"
)

# peptide, HLA-A*0201, partner TCR chain -- the layout of the liubo 6 cases.
BETA_TARGET_LENGTHS = (9, 180, 200)
ALPHA_TARGET_LENGTHS = (9, 180, 243)
NUM_BINS = 64


def _config(**kwargs) -> TCRContactConfig:
    base = dict(enabled=True, peptide_chain_index=0, mhc_chain_indices=(1,))
    base.update(kwargs)
    return TCRContactConfig(**base)


def _masks(seq=A6_BETA_SEQ, lengths=BETA_TARGET_LENGTHS, config=None):
    annotation = resolve_annotation(seq)
    return (
        build_tcr_contact_masks(
            binder_length=len(seq),
            target_chain_lengths=lengths,
            annotation=annotation,
            config=config or _config(),
            device=torch.device("cpu"),
        ),
        annotation,
    )


def _distogram(full_len: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        1, full_len, full_len, NUM_BINS, generator=generator, dtype=torch.float32
    )


# -- annotation ------------------------------------------------------------


def test_a6_annotations_match_real_scaffolds():
    A6_ALPHA.validate(A6_ALPHA_SEQ)
    A6_BETA.validate(A6_BETA_SEQ)
    assert sequence_sha256(A6_ALPHA_SEQ) == A6_ALPHA.sha256
    assert sequence_sha256(A6_BETA_SEQ) == A6_BETA.sha256


def test_cdr3_spans_match_case_specs():
    # Exactly the spans recorded in the liubo case specs.
    assert A6_BETA.cdr3 == (89, 103)
    assert A6_BETA_SEQ[89:103] == "ASRPGLAGGRPEQY"
    assert A6_ALPHA.cdr3 == (88, 99)
    assert A6_ALPHA_SEQ[88:99] == "AVTTDSWGKLQ"


def test_lookup_by_digest():
    assert lookup_annotation(A6_BETA_SEQ) is A6_BETA
    assert lookup_annotation("PEPTIDE") is None


def test_unknown_scaffold_requires_explicit_cdr1_cdr2():
    with pytest.raises(ValueError, match="cdr1_span and cdr2_span are required"):
        resolve_annotation("ACDEFGHIKLMNPQRSTVWY", cdr3_span=(5, 9))


# -- masks -----------------------------------------------------------------


def test_framework_excludes_cdr1_and_cdr2():
    masks, _ = _masks()
    overlap_1 = (masks["framework"] * masks["cdr1"]).sum().item()
    overlap_2 = (masks["framework"] * masks["cdr2"]).sum().item()
    overlap_3 = (masks["framework"] * masks["cdr3"]).sum().item()
    assert overlap_1 == 0.0, "CDR1 must not be penalised as framework"
    assert overlap_2 == 0.0, "CDR2 must not be penalised as framework"
    assert overlap_3 == 0.0


def test_framework_plus_cdrs_covers_binder_exactly():
    masks, annotation = _masks()
    recombined = masks["framework"] + masks["cdr_all"]
    assert torch.equal(recombined, masks["binder"])
    n_cdr = sum(end - start for start, end in annotation.spans())
    assert masks["cdr_all"].sum().item() == float(n_cdr)
    assert masks["binder"].sum().item() == float(len(A6_BETA_SEQ))


def test_peptide_and_mhc_masks_are_the_right_chains():
    masks, _ = _masks()
    assert masks["peptide"].sum().item() == 9.0
    assert masks["mhc"].sum().item() == 180.0
    assert masks["pmhc"].sum().item() == 189.0
    # peptide occupies the first 9 positions of the complex
    assert masks["peptide"][:9].sum().item() == 9.0
    # the partner TCR chain (index 2) is excluded from pMHC
    assert masks["pmhc"][189:189 + 200].sum().item() == 0.0


def test_partner_tcr_chain_never_counted_as_mhc():
    # roles cannot distinguish HLA from the partner TCR chain, so selection is
    # by index; guard that the default config does not pull in chain 2.
    masks, _ = _masks()
    binder_start = sum(BETA_TARGET_LENGTHS)
    assert masks["pmhc"][:binder_start].sum().item() == 189.0
    assert masks["binder"][:binder_start].sum().item() == 0.0


def test_masks_work_for_alpha_case():
    masks, annotation = _masks(A6_ALPHA_SEQ, ALPHA_TARGET_LENGTHS)
    assert annotation is A6_ALPHA
    assert masks["binder"].sum().item() == 200.0
    assert (masks["framework"] * masks["cdr1"]).sum().item() == 0.0


def test_peptide_index_cannot_overlap_mhc():
    annotation = resolve_annotation(A6_BETA_SEQ)
    with pytest.raises(ValueError, match="must not also appear"):
        build_tcr_contact_masks(
            binder_length=len(A6_BETA_SEQ),
            target_chain_lengths=BETA_TARGET_LENGTHS,
            annotation=annotation,
            config=_config(peptide_chain_index=1, mhc_chain_indices=(1,)),
            device=torch.device("cpu"),
        )


# -- losses ----------------------------------------------------------------


def test_contact_loss_changes_across_candidates():
    config = _config()
    masks, _ = _masks(config=config)
    full_len = sum(BETA_TARGET_LENGTHS) + len(A6_BETA_SEQ)
    values = []
    for seed in (0, 1, 2):
        out = tcr_contact_losses(_distogram(full_len, seed=seed), masks, config=config)
        values.append(float(out["cdr3_peptide_contact_nll"].item()))
    assert len(set(values)) == len(values), "CDR3-peptide contact must vary"


def test_contact_loss_is_differentiable_wrt_distogram():
    config = _config()
    masks, _ = _masks(config=config)
    full_len = sum(BETA_TARGET_LENGTHS) + len(A6_BETA_SEQ)
    dgram = _distogram(full_len, seed=7).requires_grad_(True)
    out = tcr_contact_losses(dgram, masks, config=config)
    grad = torch.autograd.grad(out["tcr_contact_loss"].mean(), dgram)[0]
    assert torch.isfinite(grad).all()
    assert grad.norm().item() > 0.0


def test_gradient_touches_cdr3_rows_not_only_framework():
    config = _config()
    masks, _ = _masks(config=config)
    full_len = sum(BETA_TARGET_LENGTHS) + len(A6_BETA_SEQ)
    dgram = _distogram(full_len, seed=11).requires_grad_(True)
    out = tcr_contact_losses(dgram, masks, config=config)
    grad = torch.autograd.grad(out["cdr3_peptide_contact_nll"].mean(), dgram)[0]
    cdr3_rows = masks["cdr3"].bool()
    assert grad[0][cdr3_rows].norm().item() > 0.0


def test_framework_penalty_is_bounded_by_hinge():
    config = _config(fw_hinge=6.0)
    masks, _ = _masks(config=config)
    full_len = sum(BETA_TARGET_LENGTHS) + len(A6_BETA_SEQ)
    for seed in range(4):
        out = tcr_contact_losses(_distogram(full_len, seed=seed), masks, config=config)
        penalty = float(out["framework_penalty"].item())
        assert 0.0 <= penalty <= config.fw_hinge + 1e-6


def test_r_pep_is_a_ratio_in_unit_interval():
    config = _config()
    masks, _ = _masks(config=config)
    full_len = sum(BETA_TARGET_LENGTHS) + len(A6_BETA_SEQ)
    out = tcr_contact_losses(_distogram(full_len, seed=3), masks, config=config)
    r_pep = float(out["r_pep"].item())
    assert 0.0 <= r_pep <= 1.0


def test_r_pep_does_not_carry_gradient():
    config = _config()
    masks, _ = _masks(config=config)
    full_len = sum(BETA_TARGET_LENGTHS) + len(A6_BETA_SEQ)
    dgram = _distogram(full_len, seed=5).requires_grad_(True)
    out = tcr_contact_losses(dgram, masks, config=config)
    assert not out["r_pep"].requires_grad, "r_pep is diagnostic-only"
