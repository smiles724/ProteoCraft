"""TCR-specific contact loss for the ESMFold2 pipeline.

Implements a CDR3-localized interface objective on top of the existing
differentiable distogram contact primitive in ``esmfold2_runtime``:

    L_contact = Contact_nll(CDR3, peptide)
                + lambda_fw * relu(fw_hinge - Contact_nll(framework, pMHC))

``Contact_nll`` is the AF-style binned contact entropy already used for the
hotspot loss: *lower* means *more* contact.  The first term therefore rewards
the redesigned CDR3 for touching the peptide, and the hinge term penalises the
framework once it comes closer to the pMHC than ``fw_hinge`` allows.

The hinge is deliberate.  ``Contact_nll`` is unbounded above, so the literal
``-lambda_fw * Contact_nll(framework, pMHC)`` form would hand the optimiser an
unbounded reward for shoving the framework away from the target.  ``relu`` keeps
the penalty inside ``[0, fw_hinge]`` while preserving the intent.

CDR1/CDR2 are *excluded* from the framework mask: they contribute to MHC
recognition naturally and must not be penalised.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence

import torch


# --------------------------------------------------------------------------
# A6 scaffold CDR annotation
# --------------------------------------------------------------------------
# IMGT CDR1/CDR2 for the A6 TCR V domains (alpha = TRAV12-2, beta = TRBV6-5),
# anchored on the conserved Cys23 / Trp41 / Cys104 / Phe118 positions and
# cross-checked against the CDR3 spans in the liubo 6-case specs.  These two
# scaffolds are byte-identical across all three peptide targets, so the whole
# 6-case matrix is covered by exactly these two entries.


@dataclass(frozen=True)
class TCRChainAnnotation:
    """CDR spans for one fixed TCR scaffold chain (0-based, end-exclusive)."""

    name: str
    length: int
    sha256: str
    cdr1: tuple[int, int]
    cdr2: tuple[int, int]
    cdr3: tuple[int, int]
    cdr1_motif: str
    cdr2_motif: str
    cdr3_motif: str
    # End of the variable domain (exclusive): FR4 ends here, the constant
    # domain begins.  Framework contact is scored over the V domain only.
    v_domain_end: int = 0

    def spans(self) -> tuple[tuple[int, int], ...]:
        return (self.cdr1, self.cdr2, self.cdr3)

    def validate(self, sequence: str) -> None:
        if len(sequence) != self.length:
            raise ValueError(
                f"{self.name}: expected binder length {self.length}, got {len(sequence)}"
            )
        for label, span, motif in (
            ("cdr1", self.cdr1, self.cdr1_motif),
            ("cdr2", self.cdr2, self.cdr2_motif),
            ("cdr3", self.cdr3, self.cdr3_motif),
        ):
            observed = sequence[span[0] : span[1]]
            if observed != motif:
                raise ValueError(
                    f"{self.name}: {label} span {span} yields {observed!r}, "
                    f"expected {motif!r}"
                )


A6_ALPHA = TCRChainAnnotation(
    name="A6_alpha",
    length=200,
    sha256="f843ada5aca9cf7c50b5b32dca5bbea0daa11481df921c93129ca3711c0b895f",
    cdr1=(25, 31),
    cdr2=(48, 54),
    cdr3=(88, 99),
    cdr1_motif="DRGSQS",
    cdr2_motif="IYSNGD",
    cdr3_motif="AVTTDSWGKLQ",
    # FR4 = FGAGTQVVVTP [99,110); constant domain starts at DIQNPDPAVY.
    v_domain_end=110,
)

A6_BETA = TCRChainAnnotation(
    name="A6_beta",
    length=243,
    sha256="d845b588c6a76d1770a6f6abdb5a373f0f23ff54d2df1ab046fcd3b04b0ffdb3",
    cdr1=(24, 29),
    cdr2=(46, 52),
    cdr3=(89, 103),
    cdr1_motif="MNHEY",
    cdr2_motif="SVGAGI",
    cdr3_motif="ASRPGLAGGRPEQY",
    # FR4 = FGPGTRLTVT [103,113); constant domain starts at EDLKNVFPPE.
    v_domain_end=113,
)

KNOWN_SCAFFOLDS: tuple[TCRChainAnnotation, ...] = (A6_ALPHA, A6_BETA)


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()


def lookup_annotation(sequence: str) -> Optional[TCRChainAnnotation]:
    """Return the annotation whose sha256 matches ``sequence``, else ``None``.

    Matching on the digest of the *starting* scaffold is intentional: CDR3 is
    redesigned during the run, so we key on the original sequence supplied via
    ``--init_binder_seq`` and keep the spans fixed thereafter.
    """

    digest = sequence_sha256(sequence)
    for annotation in KNOWN_SCAFFOLDS:
        if annotation.sha256 == digest:
            return annotation
    return None


def resolve_annotation(
    sequence: str,
    *,
    cdr3_span: Optional[tuple[int, int]] = None,
    cdr1_span: Optional[tuple[int, int]] = None,
    cdr2_span: Optional[tuple[int, int]] = None,
    v_domain_end: Optional[int] = None,
    strict: bool = True,
) -> TCRChainAnnotation:
    """Resolve CDR spans for ``sequence``.

    Prefers a registered scaffold; otherwise builds an annotation from the
    explicitly supplied spans.  With ``strict`` (the default) an unknown
    scaffold without explicit CDR1/CDR2 spans is an error rather than a silent
    fallback that would sweep CDR1/CDR2 into the framework penalty.
    """

    known = lookup_annotation(sequence)
    if known is not None:
        known.validate(sequence)
        return known

    if cdr3_span is None:
        raise ValueError(
            "unknown TCR scaffold and no cdr3_span supplied; cannot build "
            "contact masks"
        )
    if cdr1_span is None or cdr2_span is None:
        if strict:
            raise ValueError(
                "unknown TCR scaffold: cdr1_span and cdr2_span are required so "
                "that CDR1/CDR2 are not penalised as framework. Register the "
                "scaffold in tcr_contact.KNOWN_SCAFFOLDS or pass the spans "
                "explicitly."
            )
        cdr1_span = cdr1_span or (0, 0)
        cdr2_span = cdr2_span or (0, 0)

    return TCRChainAnnotation(
        name=f"custom_{sequence_sha256(sequence)[:10]}",
        length=len(sequence),
        sha256=sequence_sha256(sequence),
        cdr1=tuple(cdr1_span),
        cdr2=tuple(cdr2_span),
        cdr3=tuple(cdr3_span),
        cdr1_motif=sequence[cdr1_span[0] : cdr1_span[1]],
        cdr2_motif=sequence[cdr2_span[0] : cdr2_span[1]],
        cdr3_motif=sequence[cdr3_span[0] : cdr3_span[1]],
        v_domain_end=int(v_domain_end or len(sequence)),
    )


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass
class TCRContactConfig:
    """Settings for the TCR-specific contact loss."""

    enabled: bool = False
    # Normalised-gradient mixing coefficient for the proposal path.  Like
    # lambda_esmc this scales a unit-norm direction, NOT a scalar loss weight.
    lambda_contact: float = 0.1
    # Weight of the framework hinge penalty inside L_contact.
    lambda_fw: float = 1.0
    # Framework penalty activates once framework contact nll drops below this.
    # Calibrated as a guardrail, not a driver: on the A6 scaffold the natural
    # framework-pMHC nll sits at ~8.0-8.8 in the proposal path and ~13.5-14.8
    # in the verifier path (the full model and the fast experimental model put
    # the distogram on different scales), so each threshold sits just below its
    # own natural range and fires only on a genuine excursion toward the pMHC.
    fw_hinge: float = 7.0
    # Verifier-path threshold; falls back to fw_hinge when None.
    fw_hinge_verifier: Optional[float] = None
    # Which non-binder chain is the 9-mer peptide, and which are MHC.
    peptide_chain_index: int = 0
    mhc_chain_indices: tuple[int, ...] = (1,)
    # Contact geometry.
    cdr3_cutoff: float = 8.0
    fw_cutoff: float = 8.0
    num_contacts_cdr3: int = 1
    num_contacts_fw: int = 1
    # Cutoff used for the r_pep diagnostic contact-probability mass.
    r_pep_cutoff: float = 8.0
    # Verifier-path weight for the hard contact loss.
    verifier_weight: float = 1.0
    use_gradient: bool = True
    use_verifier: bool = True
    # Explicit spans for scaffolds not in KNOWN_SCAFFOLDS.
    cdr1_span: Optional[tuple[int, int]] = None
    cdr2_span: Optional[tuple[int, int]] = None
    cdr3_span: Optional[tuple[int, int]] = None
    v_domain_end: Optional[int] = None
    strict_annotation: bool = True
    # The partner TCR chain (the alpha when redesigning beta, and vice versa).
    # Its FR1-FR4 also count as framework for the pMHC penalty.
    partner_tcr_chain_index: Optional[int] = None
    partner_cdr1_span: Optional[tuple[int, int]] = None
    partner_cdr2_span: Optional[tuple[int, int]] = None
    partner_cdr3_span: Optional[tuple[int, int]] = None
    partner_v_domain_end: Optional[int] = None

    @classmethod
    def from_mapping(cls, raw: Optional[Mapping]) -> "TCRContactConfig":
        if not raw:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in raw.items() if k in known}
        for key in ("mhc_chain_indices",):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = tuple(int(v) for v in kwargs[key])
        for key in (
            "cdr1_span", "cdr2_span", "cdr3_span",
            "partner_cdr1_span", "partner_cdr2_span", "partner_cdr3_span",
        ):
            if kwargs.get(key) is not None:
                kwargs[key] = tuple(int(v) for v in kwargs[key])
        return cls(**kwargs)


# --------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------


def build_tcr_contact_masks(
    *,
    binder_length: int,
    target_chain_lengths: Sequence[int],
    annotation: TCRChainAnnotation,
    config: TCRContactConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Build residue masks over the concatenated complex.

    Layout matches ``_structure_losses``: the non-binder chains come first, in
    order, and the binder occupies the final ``binder_length`` positions.

    Chains are selected by **index**, not by role: in the TCR cases the HLA and
    the partner TCR chain both carry role ``scaffold``, so roles cannot tell
    them apart.
    """

    chain_lengths = [int(v) for v in target_chain_lengths]
    target_length = sum(chain_lengths)
    full_len = target_length + int(binder_length)

    if annotation.length != int(binder_length):
        raise ValueError(
            f"annotation {annotation.name} length {annotation.length} != "
            f"binder_length {binder_length}"
        )

    n_chains = len(chain_lengths)
    peptide_index = int(config.peptide_chain_index)
    if not (0 <= peptide_index < n_chains):
        raise ValueError(
            f"peptide_chain_index {peptide_index} out of range for "
            f"{n_chains} non-binder chains"
        )
    mhc_indices = tuple(int(v) for v in (config.mhc_chain_indices or ()))
    for index in mhc_indices:
        if not (0 <= index < n_chains):
            raise ValueError(
                f"mhc_chain_indices entry {index} out of range for "
                f"{n_chains} non-binder chains"
            )
    if peptide_index in mhc_indices:
        raise ValueError(
            "peptide_chain_index must not also appear in mhc_chain_indices"
        )

    starts: list[int] = []
    offset = 0
    for length in chain_lengths:
        starts.append(offset)
        offset += length

    def _zeros() -> torch.Tensor:
        return torch.zeros(full_len, device=device, dtype=dtype)

    peptide_mask = _zeros()
    peptide_mask[starts[peptide_index] : starts[peptide_index] + chain_lengths[peptide_index]] = 1.0

    mhc_mask = _zeros()
    for index in mhc_indices:
        mhc_mask[starts[index] : starts[index] + chain_lengths[index]] = 1.0

    pmhc_mask = torch.clamp(peptide_mask + mhc_mask, max=1.0)

    binder_mask = _zeros()
    binder_mask[-int(binder_length) :] = 1.0

    binder_offset = target_length

    def _span_mask(span: tuple[int, int]) -> torch.Tensor:
        mask = _zeros()
        start, end = int(span[0]), int(span[1])
        if end > start:
            mask[binder_offset + start : binder_offset + end] = 1.0
        return mask

    cdr1_mask = _span_mask(annotation.cdr1)
    cdr2_mask = _span_mask(annotation.cdr2)
    cdr3_mask = _span_mask(annotation.cdr3)
    cdr_all_mask = torch.clamp(cdr1_mask + cdr2_mask + cdr3_mask, max=1.0)

    # Framework = FR1 u FR2 u FR3 u FR4 of the *variable* domain only.
    # Two exclusions matter:
    #   - the constant domain is dropped; it sits far from the pMHC and would
    #     dilute the per-residue average until the penalty never fires;
    #   - CDR1/CDR2 are dropped; they contribute to MHC recognition in natural
    #     TCRs and must not be penalised.
    binder_v_mask = _zeros()
    v_end = int(annotation.v_domain_end or binder_length)
    binder_v_mask[binder_offset : binder_offset + min(v_end, int(binder_length))] = 1.0
    framework_mask = torch.clamp(binder_v_mask - cdr_all_mask, min=0.0)

    # The partner TCR chain is fixed, but its framework can still mediate an
    # abnormal pMHC contact, so FR1-FR4 of both chains count (spec 8: alpha + beta).
    partner_index = config.partner_tcr_chain_index
    if partner_index is not None:
        partner_index = int(partner_index)
        if not (0 <= partner_index < n_chains):
            raise ValueError(
                f"partner_tcr_chain_index {partner_index} out of range for "
                f"{n_chains} non-binder chains"
            )
        if partner_index == peptide_index or partner_index in mhc_indices:
            raise ValueError(
                "partner_tcr_chain_index must not overlap the peptide or MHC chains"
            )
        p_start = starts[partner_index]
        p_len = chain_lengths[partner_index]
        p_v_end = int(config.partner_v_domain_end or p_len)
        partner_fr = _zeros()
        partner_fr[p_start : p_start + min(p_v_end, p_len)] = 1.0
        for span in (
            config.partner_cdr1_span,
            config.partner_cdr2_span,
            config.partner_cdr3_span,
        ):
            if span:
                partner_fr[p_start + int(span[0]) : p_start + int(span[1])] = 0.0
        framework_mask = torch.clamp(framework_mask + partner_fr, max=1.0)

    if float(cdr3_mask.sum().item()) <= 0.0:
        raise ValueError("CDR3 mask is empty")
    if float(framework_mask.sum().item()) <= 0.0:
        raise ValueError("framework mask is empty")
    if float(peptide_mask.sum().item()) <= 0.0:
        raise ValueError("peptide mask is empty")

    return {
        "peptide": peptide_mask,
        "mhc": mhc_mask,
        "pmhc": pmhc_mask,
        "binder": binder_mask,
        "cdr1": cdr1_mask,
        "cdr2": cdr2_mask,
        "cdr3": cdr3_mask,
        "cdr_all": cdr_all_mask,
        "framework": framework_mask,
    }


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------


def _soft_contact_mass(
    distogram_logits: torch.Tensor,
    bin_distance: torch.Tensor,
    *,
    cutoff: float,
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
) -> torch.Tensor:
    """Summed probability mass of ``d_ij < cutoff`` over ``i in A, j in B``.

    Used only for the ``r_pep`` diagnostic, where a genuine "amount of contact"
    is wanted rather than the entropy-style nll the optimiser sees.
    """

    probs = torch.softmax(distogram_logits, dim=-1)
    within = (bin_distance < float(cutoff)).to(probs.dtype)
    pair = (probs * within).sum(-1)
    pair = pair * mask_a[None, :, None] * mask_b[None, None, :]
    return pair.sum(dim=(-1, -2))


def tcr_contact_losses(
    distogram_logits: torch.Tensor,
    masks: Mapping[str, torch.Tensor],
    *,
    config: TCRContactConfig,
    verifier_path: bool = False,
) -> Dict[str, torch.Tensor]:
    """Compute the TCR contact objective and its diagnostics.

    Returns tensors (batch-shaped) so the caller can keep the autograd graph.
    """

    from bindcraft.hallucination.esmfold2_runtime import (
        _contact_loss,
        _midpoint_bins,
    )

    bin_distance = _midpoint_bins(
        distogram_logits.device,
        num_bins=distogram_logits.shape[-1],
        dtype=distogram_logits.dtype,
    )

    cdr3_peptide_nll = _contact_loss(
        distogram_logits,
        bin_distance,
        num_contacts=int(config.num_contacts_cdr3),
        min_sep=0,
        cutoff=float(config.cdr3_cutoff),
        chain_mask=masks["cdr3"],
        binder_mask=masks["peptide"],
    )
    framework_pmhc_nll = _contact_loss(
        distogram_logits,
        bin_distance,
        num_contacts=int(config.num_contacts_fw),
        min_sep=0,
        cutoff=float(config.fw_cutoff),
        chain_mask=masks["framework"],
        binder_mask=masks["pmhc"],
    )

    # Bounded framework penalty: zero while the framework stays farther than
    # fw_hinge, rising to at most fw_hinge as it closes in.  Keeps the term in
    # [0, fw_hinge] instead of the unbounded -Contact form.
    hinge = float(config.fw_hinge)
    if verifier_path and config.fw_hinge_verifier is not None:
        hinge = float(config.fw_hinge_verifier)
    framework_penalty = torch.relu(
        torch.as_tensor(
            hinge,
            device=framework_pmhc_nll.device,
            dtype=framework_pmhc_nll.dtype,
        )
        - framework_pmhc_nll
    )

    total = cdr3_peptide_nll + float(config.lambda_fw) * framework_penalty

    with torch.no_grad():
        cdr3_peptide_mass = _soft_contact_mass(
            distogram_logits,
            bin_distance,
            cutoff=float(config.r_pep_cutoff),
            mask_a=masks["cdr3"],
            mask_b=masks["peptide"],
        )
        tcr_pmhc_mass = _soft_contact_mass(
            distogram_logits,
            bin_distance,
            cutoff=float(config.r_pep_cutoff),
            mask_a=masks["binder"],
            mask_b=masks["pmhc"],
        )
        r_pep = cdr3_peptide_mass / (tcr_pmhc_mass + 1e-8)

    return {
        "tcr_contact_loss": total,
        "cdr3_peptide_contact_nll": cdr3_peptide_nll,
        "framework_pmhc_contact_nll": framework_pmhc_nll,
        "framework_penalty": framework_penalty,
        "framework_hinge_used": torch.as_tensor(
            hinge,
            device=framework_pmhc_nll.device,
            dtype=framework_pmhc_nll.dtype,
        ).expand_as(framework_penalty),
        "cdr3_peptide_contact_mass": cdr3_peptide_mass,
        "tcr_pmhc_contact_mass": tcr_pmhc_mass,
        "r_pep": r_pep,
    }


def annotation_from_config(
    config: TCRContactConfig,
    *,
    binder_length: int,
) -> TCRChainAnnotation:
    """Rebuild an annotation from spans already resolved into ``config``.

    The CDR3 mutates during design, so a digest lookup would stop matching
    after the first accepted mutation.  The spans are therefore resolved once
    from the *starting* scaffold (at CLI time) and carried in the config; the
    runtime only ever reads them back through this helper.
    """

    for key in ("cdr1_span", "cdr2_span", "cdr3_span"):
        if getattr(config, key) is None:
            raise ValueError(
                f"tcr_contact_config is missing {key}; resolve the scaffold "
                "annotation before starting the run"
            )
    placeholder = "X" * int(binder_length)
    return TCRChainAnnotation(
        name="resolved_from_config",
        length=int(binder_length),
        sha256="",
        cdr1=tuple(config.cdr1_span),
        cdr2=tuple(config.cdr2_span),
        cdr3=tuple(config.cdr3_span),
        cdr1_motif=placeholder[config.cdr1_span[0] : config.cdr1_span[1]],
        cdr2_motif=placeholder[config.cdr2_span[0] : config.cdr2_span[1]],
        cdr3_motif=placeholder[config.cdr3_span[0] : config.cdr3_span[1]],
        v_domain_end=int(config.v_domain_end or binder_length),
    )


def resolve_contact_config_spans(
    raw: Optional[Mapping],
    *,
    scaffold_sequence: str,
    partner_sequence: Optional[str] = None,
) -> Dict:
    """Fill CDR spans into a raw contact-config mapping from the scaffold.

    Called once at launch time, while the binder sequence is still the
    untouched natural scaffold.
    """

    payload = dict(raw or {})
    if not payload.get("enabled"):
        return payload
    explicit = payload.get("cdr3_span")
    annotation = resolve_annotation(
        scaffold_sequence,
        cdr3_span=tuple(explicit) if explicit else None,
        cdr1_span=(
            tuple(payload["cdr1_span"]) if payload.get("cdr1_span") else None
        ),
        cdr2_span=(
            tuple(payload["cdr2_span"]) if payload.get("cdr2_span") else None
        ),
        v_domain_end=payload.get("v_domain_end"),
        strict=bool(payload.get("strict_annotation", True)),
    )
    payload["cdr1_span"] = list(annotation.cdr1)
    payload["cdr2_span"] = list(annotation.cdr2)
    payload["cdr3_span"] = list(annotation.cdr3)
    payload["v_domain_end"] = int(annotation.v_domain_end)
    payload["annotation_name"] = annotation.name

    if partner_sequence is not None:
        partner = lookup_annotation(partner_sequence)
        if partner is None:
            if bool(payload.get("strict_annotation", True)):
                raise ValueError(
                    "partner TCR chain is not a registered scaffold; register it "
                    "in tcr_contact.KNOWN_SCAFFOLDS so its FR1-FR4 can be masked"
                )
        else:
            payload["partner_cdr1_span"] = list(partner.cdr1)
            payload["partner_cdr2_span"] = list(partner.cdr2)
            payload["partner_cdr3_span"] = list(partner.cdr3)
            payload["partner_v_domain_end"] = int(partner.v_domain_end)
            payload["partner_annotation_name"] = partner.name
    return payload
