"""Sequence constraints for the opt-in dual-target antibody mode.

The mask is explicit; this module never infers CDR numbering or scientific
protection regions. Immutable means amino-acid identity, not fixed coordinates.
"""
from __future__ import annotations


def design_mask(config, runtime):
    raw = getattr(runtime, "binder_mutable_mask", None)
    if config.design_mode == "miniprotein":
        if raw is not None:
            raise ValueError("Masked redesign requires explicit masked_antibody mode")
        return None
    if raw is None:
        raise ValueError("masked_antibody requires runtime.binder_mutable_mask")
    if any(type(value) not in (bool, int) or value not in (False, True) for value in raw):
        raise ValueError("binder_mutable_mask must contain booleans or integer 0/1")
    mask = tuple(bool(value) for value in raw)
    if not mask or not any(mask) or all(mask):
        raise ValueError("masked_antibody requires both mutable and immutable positions")
    return mask


def check_sequence(sequence, reference, mask):
    if len(sequence) != len(reference) or (mask is not None and len(mask) != len(reference)):
        raise ValueError("Proposal or mask length differs from its reference")
    for position, (aa, original) in enumerate(zip(sequence, reference), 1):
        mutable = mask is None or mask[position - 1]
        if not mutable and aa != original:
            raise ValueError(f"Immutable position {position} changed: {original} -> {aa}")
        if mutable and aa == "C":
            raise ValueError(f"Cys at mutable position {position}; fix this position explicitly")


def validate_design(config, runtime, seeds):
    if runtime is not None:
        if runtime.verifier_loss_mode != "aligned_common":
            raise ValueError("Use aligned_common for comparable human/mouse verifier losses")
        if runtime.template_structure_path is not None:
            raise ValueError("Dual-target search does not support coordinate templates")
        if runtime.one_step_diffusion_recycle_coordinates:
            raise ValueError("Cross-step coordinate recycling is not supported in dual-target mode")
    mask = design_mask(config, runtime)
    if mask is not None and seeds:
        reference = next(iter(seeds.values()))
        for sequence in seeds.values():
            check_sequence(sequence, reference, mask)
    return mask
