"""
ESMFold2 backend helpers for optional ProteoCraft integration.

This module intentionally contains only lightweight contracts and tensor
utilities.  It must not import ESMFold2/transformers at module import time, so
the existing Protenix V1/V2 paths stay usable in the current environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch


# ESMFold2 residue order from Biohub ESMFold2 constants:
# ALA, ARG, ASN, ASP, CYS, GLN, GLU, GLY, HIS, ILE,
# LEU, LYS, MET, PHE, PRO, SER, THR, TRP, TYR, VAL
ESM_AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
AA_TO_ESM_IDX = {aa: idx for idx, aa in enumerate(ESM_AA_ORDER)}
CYS_INDEX = AA_TO_ESM_IDX["C"]

PAIR_CHAIN_IPTM_KEYS = (
    "pair_chain_iptm",
    "pair_chain_i_ptm",
    "pair_chains_iptm",
    "chain_pair_iptm",
    "binder_target_pair_iptm",
)
IPTM_KEYS = ("iptm", "i_ptm")
ESMFOLD2_ACTION_GROUP_LOSS_KEYS = {
    "interface_formation": {
        "pair_chain_iptm_loss": 1.0,
        "iptm_loss": 0.5,
        "inter_contact_loss": 1.0,
    },
    "core_stabilization": {
        "binder_confidence_loss": 1.0,
        "intra_contact_loss": 1.0,
        "global_confidence_loss": 0.5,
    },
    "compactness_repair": {
        "glob_loss": 1.0,
    },
}
ESMFOLD2_ROUTING_LOSS_SCALES = {
    "pair_chain_iptm_loss": 1.0,
    "iptm_loss": 1.0,
    "binder_confidence_loss": 1.0,
    "global_confidence_loss": 1.0,
    # Contact losses are distogram entropies in nats, roughly bounded by
    # log(number_of_bins).  A scale of 5 maps the practical range to 0..1.
    "inter_contact_loss": 5.0,
    "intra_contact_loss": 5.0,
    # Globularity is an ELU-shifted radius-of-gyration penalty.  Keep the same
    # defensive scale so an outlier cannot dominate routing by itself.
    "glob_loss": 5.0,
}
AUTO_ESMC_LAMBDA_BY_TASK = {
    "denovo": 0.15,
    "de_novo": 0.15,
    "minibinder": 0.15,
    "binder": 0.15,
    "tcr": 0.05,
    "cdr": 0.05,
    "cdr_redesign": 0.05,
    "antibody": 0.05,
    "antibody_like": 0.05,
}


@dataclass(frozen=True)
class ESMFold2ProposalConfig:
    """Configuration for the first ESMFold2 Stage-A proposal backend."""

    experimental_model_name: str = "biohub/ESMFold2-Experimental"
    verifier_model_name: str = "biohub/ESMFold2"
    verifier_num_loops: int = 20
    verifier_num_sampling_steps: int = 200
    soft_steps: int = 3
    learning_rate: float = 0.1
    temperature_min: float = 1e-2
    temperature_max: float = 1.0
    lambda_esmc: Union[str, float] = "auto"
    esmc_n_passes: int = 4
    esmc_batch_size: int = 4
    num_loops: int = 1
    num_sampling_steps: int = 1
    stageb_group_weight: float = 1.0

    def __post_init__(self) -> None:
        if int(self.verifier_num_loops) <= 0:
            raise ValueError("verifier_num_loops must be positive")
        if int(self.verifier_num_sampling_steps) <= 0:
            raise ValueError("verifier_num_sampling_steps must be positive")
        if int(self.soft_steps) <= 0:
            raise ValueError("soft_steps must be positive")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive")
        if float(self.temperature_min) <= 0.0:
            raise ValueError("temperature_min must be positive")
        if float(self.temperature_max) < float(self.temperature_min):
            raise ValueError("temperature_max must be >= temperature_min")
        if float(self.stageb_group_weight) < 0.0:
            raise ValueError("stageb_group_weight must be non-negative")


@dataclass
class ESMFold2VerifierResult:
    """MCTS-compatible verifier result for ESMFold2 candidates."""

    binder_seq: str
    target_seq: Union[str, Sequence[str]]
    total_loss: float
    loss_dict: Dict[str, float]
    summary_confidence: Dict[str, Any]
    coordinate: Optional[torch.Tensor] = field(default=None, repr=False)
    plddt_logits: Optional[torch.Tensor] = field(default=None, repr=False)


def _as_bool_mask(
    mutable_mask: Optional[Sequence[bool]],
    length: int,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if mutable_mask is None:
        return torch.ones(length, dtype=torch.bool, device=device)
    mask = torch.as_tensor(mutable_mask, dtype=torch.bool, device=device)
    if mask.numel() != length:
        raise ValueError(
            f"mutable_mask length {mask.numel()} does not match sequence length {length}"
        )
    return mask.reshape(length)


def _sequence_indices(sequence: str) -> torch.Tensor:
    missing = sorted({aa for aa in sequence if aa not in AA_TO_ESM_IDX})
    if missing:
        raise ValueError(f"Unsupported amino acid(s) for ESMFold2 backend: {missing}")
    return torch.tensor([AA_TO_ESM_IDX[aa] for aa in sequence], dtype=torch.long)


def _finite_float(value: Any) -> Optional[float]:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(scalar):
        return None
    return scalar


def _first_finite_metric(
    summary_confidence: Mapping[str, Any],
    keys: Sequence[str],
) -> Tuple[Optional[float], Optional[str]]:
    for key in keys:
        if key not in summary_confidence:
            continue
        scalar = _finite_float(summary_confidence[key])
        if scalar is not None:
            return scalar, key
    return None, None


def resolve_esm_verifier_loss(
    summary_confidence: Mapping[str, Any],
) -> Tuple[float, str]:
    """Convert ESMFold2 confidence metrics to a scalar MCTS loss.

    First version uses binder-target pair-chain iPTM when available, otherwise
    falls back to global iPTM.  The value is turned into a minimization loss as
    ``1 - score``.
    """

    score, source = _first_finite_metric(summary_confidence, PAIR_CHAIN_IPTM_KEYS)
    if score is None:
        score, source = _first_finite_metric(summary_confidence, IPTM_KEYS)
    if score is None or source is None:
        raise ValueError(
            "ESMFold2 verifier summary must contain pair-chain iPTM or iPTM"
        )
    score = min(max(score, 0.0), 1.0)
    return 1.0 - score, source


def build_esm_verifier_result(
    *,
    binder_seq: str,
    target_seq: Union[str, Sequence[str]],
    summary_confidence: Mapping[str, Any],
    coordinate: Optional[torch.Tensor] = None,
    plddt_logits: Optional[torch.Tensor] = None,
    extra_loss_dict: Optional[Mapping[str, float]] = None,
) -> ESMFold2VerifierResult:
    """Build a verifier result object accepted by the existing MCTS cache."""

    total_loss, source = resolve_esm_verifier_loss(summary_confidence)
    loss_dict: Dict[str, float] = {
        "total": float(total_loss),
        "esm_verify_loss": float(total_loss),
    }
    if source in PAIR_CHAIN_IPTM_KEYS:
        loss_dict["pair_chain_iptm_loss"] = float(total_loss)
    elif source in IPTM_KEYS:
        loss_dict["iptm_loss"] = float(total_loss)
    plddt = _finite_float(summary_confidence.get("plddt"))
    if plddt is not None:
        plddt_unit = plddt / 100.0 if plddt > 1.0 else plddt
        loss_dict["binder_confidence_loss"] = 1.0 - min(max(plddt_unit, 0.0), 1.0)
    ptm = _finite_float(summary_confidence.get("ptm"))
    if ptm is not None:
        loss_dict["global_confidence_loss"] = 1.0 - min(max(ptm, 0.0), 1.0)
    if extra_loss_dict:
        for key, value in extra_loss_dict.items():
            scalar = _finite_float(value)
            if scalar is not None:
                loss_dict[str(key)] = scalar
    summary = dict(summary_confidence)
    summary["esm_verify_loss"] = float(total_loss)
    summary["esm_verify_source"] = source
    return ESMFold2VerifierResult(
        binder_seq=binder_seq,
        target_seq=target_seq,
        total_loss=float(total_loss),
        loss_dict=loss_dict,
        summary_confidence=summary,
        coordinate=coordinate,
        plddt_logits=plddt_logits,
    )


def _scaled_routing_value(key: str, value: float) -> Tuple[float, float]:
    raw = max(float(value), 0.0)
    scale = max(float(ESMFOLD2_ROUTING_LOSS_SCALES.get(key, 1.0)), 1.0e-8)
    normalized = min(raw / scale, 1.0)
    return raw, normalized


def _weighted_group_score(
    loss_vector: Mapping[str, Any],
    weights: Mapping[str, float],
) -> Tuple[Optional[float], Dict[str, Dict[str, float]]]:
    values = []
    weight_sum = 0.0
    terms: Dict[str, Dict[str, float]] = {}
    for key, weight in weights.items():
        if key not in loss_vector:
            continue
        value = _finite_float(loss_vector.get(key))
        if value is None:
            continue
        w = abs(float(weight))
        if w <= 0.0:
            continue
        raw, normalized = _scaled_routing_value(key, float(value))
        values.append(w * normalized)
        weight_sum += w
        terms[key] = {
            "raw": raw,
            "scale": float(ESMFOLD2_ROUTING_LOSS_SCALES.get(key, 1.0)),
            "normalized": normalized,
            "weight": w,
        }
    if not values or weight_sum <= 0.0:
        return None, {}
    return float(sum(values) / weight_sum), terms


def compute_esmfold2_action_group_routing(
    loss_vector: Mapping[str, Any],
    *,
    temperature: float = 1.0,
) -> Dict[str, object]:
    """Compute normalized ESMFold2 Stage-B routing scores and priors."""

    raw_scores: Dict[str, float] = {}
    term_details: Dict[str, Dict[str, Dict[str, float]]] = {}
    for group, weights in ESMFOLD2_ACTION_GROUP_LOSS_KEYS.items():
        score, terms = _weighted_group_score(loss_vector, weights)
        if score is not None:
            raw_scores[group] = score
            term_details[group] = terms
    if not raw_scores:
        return {
            "group_priors": {"unrouted": 1.0},
            "group_scores": {},
            "group_terms": {},
        }

    groups = list(raw_scores)
    scores = torch.tensor([raw_scores[group] for group in groups], dtype=torch.float32)
    priors = torch.softmax(scores / max(float(temperature), 1e-8), dim=0)
    return {
        "group_priors": {
            group: float(priors[idx].item())
            for idx, group in enumerate(groups)
        },
        "group_scores": raw_scores,
        "group_terms": term_details,
    }


def compute_esmfold2_action_group_priors(
    loss_vector: Mapping[str, Any],
    *,
    temperature: float = 1.0,
) -> Dict[str, float]:
    """Compute ESMFold2-native Stage-B routing priors from real loss terms."""

    return dict(
        compute_esmfold2_action_group_routing(
            loss_vector,
            temperature=temperature,
        )["group_priors"]
    )


def select_esmfold2_action_group(
    loss_vector: Mapping[str, Any],
    *,
    temperature: float = 1.0,
) -> Dict[str, object]:
    """Select the ESMFold2 Stage-B repair group for the current verified node."""

    routing = compute_esmfold2_action_group_routing(loss_vector, temperature=temperature)
    priors = dict(routing["group_priors"])
    action_group = max(priors, key=priors.get)
    return {
        "action_group": action_group,
        "group_prior": priors[action_group],
        "group_priors": priors,
        "group_scores": routing["group_scores"],
        "group_terms": routing["group_terms"],
    }


def init_binder_logits(
    sequence: str,
    mutable_mask: Optional[Sequence[bool]] = None,
    *,
    batch_size: int = 1,
    random_scale: float = 0.01,
    fixed_logit: float = 10.0,
    suppress_cys: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Initialize ESMFold2 soft binder logits for a current discrete sequence."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = torch.device(device) if device is not None else None
    length = len(sequence)
    current_idx = _sequence_indices(sequence).to(device=device)
    mutable = _as_bool_mask(mutable_mask, length, device=device)

    if random_scale > 0.0:
        logits = torch.randn(
            (batch_size, length, len(ESM_AA_ORDER)),
            device=device,
            dtype=dtype,
            generator=generator,
        ) * float(random_scale)
    else:
        logits = torch.zeros(
            (batch_size, length, len(ESM_AA_ORDER)),
            device=device,
            dtype=dtype,
        )

    fixed_positions = (~mutable).nonzero(as_tuple=False).flatten()
    if fixed_positions.numel() > 0:
        logits[:, fixed_positions, :] = 0.0
        logits[:, fixed_positions, current_idx[fixed_positions]] = float(fixed_logit)

    if suppress_cys:
        mutable_positions = mutable.nonzero(as_tuple=False).flatten()
        if mutable_positions.numel() > 0:
            logits[:, mutable_positions, CYS_INDEX] = float("-inf")

    logits.requires_grad_(requires_grad)
    return logits


def build_gradient_mask(
    sequence: str,
    mutable_mask: Optional[Sequence[bool]] = None,
    *,
    batch_size: int = 1,
    suppress_cys: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a [B, L, 20] mask for ESMFold2 binder-logit gradients."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = torch.device(device) if device is not None else None
    length = len(sequence)
    _sequence_indices(sequence)
    mutable = _as_bool_mask(mutable_mask, length, device=device)
    mask = torch.zeros(
        (batch_size, length, len(ESM_AA_ORDER)),
        device=device,
        dtype=dtype,
    )
    if mutable.any():
        mask[:, mutable, :] = 1.0
    if suppress_cys:
        mask[:, :, CYS_INDEX] = 0.0
    return mask


def normalized_gradient_tensor(
    gradient: torch.Tensor,
    gradient_mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize a masked gradient tensor in the style of ESMFold2 design code."""

    if gradient.shape != gradient_mask.shape:
        raise ValueError(
            f"gradient shape {tuple(gradient.shape)} does not match mask shape "
            f"{tuple(gradient_mask.shape)}"
        )
    masked = torch.nan_to_num(gradient) * gradient_mask
    output = torch.zeros_like(masked)
    if masked.numel() == 0:
        return output

    flat_masked = masked.reshape(masked.shape[0], -1)
    flat_mask = gradient_mask.reshape(gradient_mask.shape[0], -1)
    norms = torch.linalg.vector_norm(flat_masked, dim=1, keepdim=True)
    active = flat_mask.sum(dim=1, keepdim=True) > 0
    safe = active & (norms > eps)
    if safe.any():
        normalized = flat_masked / torch.clamp(norms, min=eps)
        output = normalized.reshape_as(masked)
        output = torch.where(safe.reshape(-1, 1, 1), output, torch.zeros_like(output))
    return output * gradient_mask


def combine_normalized_gradients(
    structure_grad: torch.Tensor,
    esmc_grad: Optional[torch.Tensor],
    *,
    gradient_mask: torch.Tensor,
    lambda_esmc: float,
) -> torch.Tensor:
    """Combine ESMFold2 structural gradient and optional ESMC sequence prior."""

    combined = normalized_gradient_tensor(structure_grad, gradient_mask)
    if esmc_grad is not None and float(lambda_esmc) != 0.0:
        combined = combined + float(lambda_esmc) * normalized_gradient_tensor(
            esmc_grad,
            gradient_mask,
        )
    return combined * gradient_mask


def resolve_lambda_esmc(
    value: Union[str, float, int, None],
    *,
    task_type: Optional[str] = None,
) -> float:
    """Resolve the ESMC sequence-prior weight.

    ``auto`` follows the Biohub binder-design convention: stronger sequence
    prior for de novo/minibinder design and weaker prior for antibody-like
    redesign tasks.
    """

    if value is None:
        return 0.0
    if isinstance(value, (float, int)):
        return float(value)
    normalized = str(value).strip().lower()
    if normalized in {"off", "none", "false", "0"}:
        return 0.0
    if normalized == "auto":
        task = str(task_type or "denovo").strip().lower()
        return AUTO_ESMC_LAMBDA_BY_TASK.get(task, AUTO_ESMC_LAMBDA_BY_TASK["denovo"])
    return float(normalized)


def esm_temperature(
    step: int,
    *,
    total_steps: int,
    temperature_min: float = 1e-2,
    temperature_max: float = 1.0,
) -> float:
    """Cosine temperature schedule used by the short ESMFold2 soft update."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    clamped_step = min(max(int(step), 0), int(total_steps) - 1)
    t = (clamped_step + 1) / max(int(total_steps), 1)
    remaining = 0.5 * (1.0 + math.cos(math.pi * t))
    return float(temperature_min + (temperature_max - temperature_min) * remaining)


def logit_delta_mutation_scores(
    logits_before: torch.Tensor,
    logits_after: torch.Tensor,
    *,
    current_seq: str,
    mutable_mask: Optional[Sequence[bool]] = None,
    suppress_cys: bool = True,
) -> torch.Tensor:
    """Score discrete mutations from a short ESMFold2/ESMC soft-logit update.

    score(i, aa) =
      [after(i, aa) - before(i, aa)]
      - [after(i, current_aa) - before(i, current_aa)]

    Higher score means the soft update pushed that amino acid up more than the
    current residue at the same position.
    """

    if logits_before.shape != logits_after.shape:
        raise ValueError(
            f"logits_before shape {tuple(logits_before.shape)} does not match "
            f"logits_after shape {tuple(logits_after.shape)}"
        )
    if logits_before.ndim != 2 or logits_before.shape[1] != len(ESM_AA_ORDER):
        raise ValueError("logits tensors must have shape [L, 20]")
    length = len(current_seq)
    if logits_before.shape[0] != length:
        raise ValueError(
            f"logits length {logits_before.shape[0]} does not match sequence length {length}"
        )

    current_idx = _sequence_indices(current_seq).to(device=logits_before.device)
    mutable = _as_bool_mask(mutable_mask, length, device=logits_before.device)
    delta = torch.nan_to_num(
        logits_after - logits_before,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    current_delta = delta[torch.arange(length, device=delta.device), current_idx]
    scores = delta - current_delta.unsqueeze(-1)

    invalid = torch.zeros_like(scores, dtype=torch.bool)
    invalid |= ~mutable.unsqueeze(-1)
    invalid[torch.arange(length, device=delta.device), current_idx] = True
    if suppress_cys:
        invalid[:, CYS_INDEX] = True
    scores = scores.masked_fill(invalid, float("-inf"))
    return scores


def _scores_to_gradient_like_matrix(scores: torch.Tensor) -> torch.Tensor:
    """Convert high-is-good mutation scores to legacy low-is-good delta matrix."""

    converted = torch.where(
        torch.isneginf(scores),
        torch.full_like(scores, 1.0e9),
        -scores,
    )
    return converted


def propose_sequences_from_logit_delta(
    logits_before: torch.Tensor,
    logits_after: torch.Tensor,
    *,
    current_seq: str,
    mutable_mask: Optional[Sequence[bool]] = None,
    K_top: int = 10,
    B_candidates: int = 8,
    T_sample: float = 1.0,
    max_mutations_per_seq: int = 3,
) -> list[str]:
    """Generate discrete candidate sequences from ESMFold2 soft-logit updates.

    The output side deliberately reuses the existing Stage-A Top-K + sampling
    contract: first select top single mutations, then sample up to
    ``max_mutations_per_seq`` edits per candidate.
    """

    from bindcraft.hallucination.mutation_proposer import (
        AA_1LETTER,
        PROTEIN_RESIDUE_NAMES,
        sample_candidate_sequences,
        select_top_k_mutations,
    )

    current_indices = [AA_TO_ESM_IDX[aa] for aa in current_seq]
    scores = logit_delta_mutation_scores(
        logits_before,
        logits_after,
        current_seq=current_seq,
        mutable_mask=mutable_mask,
    ).detach().cpu()
    delta_like = _scores_to_gradient_like_matrix(scores)
    positions, new_aas, benefits = select_top_k_mutations(
        delta_like,
        current_indices,
        K_top,
    )
    if positions.numel() == 0:
        return [current_seq] * int(B_candidates)

    candidate_indices = sample_candidate_sequences(
        current_indices,
        positions,
        new_aas,
        benefits,
        B_candidates,
        T_sample,
        max_mutations_per_seq,
    )

    def indices_to_sequence(indices: Sequence[int]) -> str:
        return "".join(AA_1LETTER[PROTEIN_RESIDUE_NAMES[int(idx)]] for idx in indices)

    return [indices_to_sequence(indices) for indices in candidate_indices]
