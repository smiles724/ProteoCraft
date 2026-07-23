"""
ESMFold2 backend helpers for optional ProteoCraft integration.

This module intentionally contains only lightweight contracts and tensor
utilities.  It must not import ESMFold2/transformers at module import time, so
the existing Protenix V1/V2 paths stay usable in the current environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple, Union

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
ESMFOLD2_ACTION_GROUP_ROUTING_KEYS = {
    "interface_formation": "pair_chain_iptm_loss",
    "core_stabilization": "binder_confidence_loss",
    "compactness_repair": "binder_rg_excess_loss",
}
ESMFOLD2_ACTION_GROUP_PROPOSAL_LOSS_KEYS = {
    "interface_formation": {"inter_contact_loss": 1.0},
    "core_stabilization": {"intra_contact_loss": 1.0},
    "compactness_repair": {"glob_loss": 1.0},
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
    soft_steps: int = 30
    learning_rate: float = 0.1
    temperature_min: float = 1e-2
    temperature_max: float = 1.0
    lambda_esmc: Union[str, float] = "auto"
    esmc_n_passes: int = 4
    esmc_batch_size: int = 4
    num_loops: int = 1
    num_sampling_steps: int = 1
    proposal_load_internal_esmc: bool = True
    proposal_lm_dropout: Optional[float] = 0.5
    force_proposal_lm_dropout: bool = True
    hotspot_weight: float = 0.0
    hotspot_cutoff: float = 8.0
    hotspot_interface_mode: str = "additive"
    hotspot_verify_weight: float = 0.0
    hotspot_verify_near_cutoff: float = 8.0
    hotspot_verify_far_cutoff: float = 20.0
    stageb_group_weight: float = 1.0
    track_proposal_confidence: bool = False
    proposal_confidence_temperature: float = 0.05
    proposal_confidence_interval: int = 1
    proposal_confidence_num_loops: int = 1
    proposal_confidence_num_sampling_steps: int = 50
    proposal_confidence_max_candidates: int = 8
    proposal_confidence_tail_steps: int = 0
    target_chain_roles: Optional[Tuple[str, ...]] = None
    binder_mutable_mask: Optional[Tuple[bool, ...]] = None

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
        if float(self.hotspot_weight) < 0.0:
            raise ValueError("hotspot_weight must be non-negative")
        if float(self.hotspot_cutoff) <= 0.0:
            raise ValueError("hotspot_cutoff must be positive")
        hotspot_interface_mode = str(self.hotspot_interface_mode).strip().lower()
        if hotspot_interface_mode not in {"additive", "localized"}:
            raise ValueError(
                "hotspot_interface_mode must be 'additive' or 'localized'"
            )
        if hotspot_interface_mode == "localized" and float(self.hotspot_weight) > 1.0:
            raise ValueError(
                "localized hotspot_interface_mode requires hotspot_weight in [0, 1]"
            )
        if float(self.hotspot_verify_weight) < 0.0:
            raise ValueError("hotspot_verify_weight must be non-negative")
        if float(self.hotspot_verify_near_cutoff) < 0.0:
            raise ValueError("hotspot_verify_near_cutoff must be non-negative")
        if float(self.hotspot_verify_far_cutoff) <= float(
            self.hotspot_verify_near_cutoff
        ):
            raise ValueError(
                "hotspot_verify_far_cutoff must be greater than "
                "hotspot_verify_near_cutoff"
            )
        if self.proposal_lm_dropout is not None:
            dropout = float(self.proposal_lm_dropout)
            if dropout < 0.0 or dropout >= 1.0:
                raise ValueError("proposal_lm_dropout must be in [0, 1)")
        if float(self.proposal_confidence_temperature) <= 0.0:
            raise ValueError("proposal_confidence_temperature must be positive")
        if int(self.proposal_confidence_interval) <= 0:
            raise ValueError("proposal_confidence_interval must be positive")
        if int(self.proposal_confidence_num_loops) <= 0:
            raise ValueError("proposal_confidence_num_loops must be positive")
        if int(self.proposal_confidence_num_sampling_steps) <= 0:
            raise ValueError("proposal_confidence_num_sampling_steps must be positive")
        if int(self.proposal_confidence_max_candidates) <= 0:
            raise ValueError("proposal_confidence_max_candidates must be positive")
        if int(self.proposal_confidence_tail_steps) < 0:
            raise ValueError("proposal_confidence_tail_steps must be non-negative")
        if self.target_chain_roles is not None:
            roles = tuple(
                str(role).strip().lower() for role in self.target_chain_roles
            )
            if not roles:
                raise ValueError("target_chain_roles cannot be empty")
            invalid_roles = sorted(set(roles) - {"target", "scaffold"})
            if invalid_roles:
                raise ValueError(
                    "target_chain_roles only supports target/scaffold; "
                    f"got {invalid_roles}"
                )
            if "target" not in roles:
                raise ValueError("target_chain_roles must contain at least one target")
        if self.binder_mutable_mask is not None:
            mutable_mask = tuple(bool(value) for value in self.binder_mutable_mask)
            if not mutable_mask:
                raise ValueError("binder_mutable_mask cannot be empty")
            if not any(mutable_mask):
                raise ValueError("binder_mutable_mask must select at least one residue")


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


@dataclass
class ESMFold2ProposalBatch:
    """Mutation proposal output plus optional persistent soft-logit state."""

    sequences: Sequence[str]
    logits_before: Optional[torch.Tensor] = field(default=None, repr=False)
    logits_after: Optional[torch.Tensor] = field(default=None, repr=False)
    confidence_candidates: Sequence[Mapping[str, Any]] = field(default_factory=list)
    soft_step_metrics: Sequence[Mapping[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.sequences = [str(sequence) for sequence in self.sequences]

    def __iter__(self) -> Iterator[str]:
        return iter(self.sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> str:
        return self.sequences[index]

    def __contains__(self, sequence: object) -> bool:
        return sequence in self.sequences


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

    The search value is strictly based on binder-target pair-chain iPTM.  A
    missing value indicates a broken chain mapping or incomplete verifier output
    and is therefore an error rather than a reason to silently change metrics.
    """

    score, source = _first_finite_metric(summary_confidence, PAIR_CHAIN_IPTM_KEYS)
    if score is None or source is None:
        raise ValueError(
            "ESMFold2 verifier summary must contain binder-target pair-chain iPTM"
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
    hotspot_verify_loss: Optional[float] = None,
    hotspot_verify_weight: float = 0.0,
) -> ESMFold2VerifierResult:
    """Build a verifier result object accepted by the existing MCTS cache."""

    global_loss, source = resolve_esm_verifier_loss(summary_confidence)
    total_loss = float(global_loss)
    hotspot_loss = _finite_float(hotspot_verify_loss)
    hotspot_weight = float(hotspot_verify_weight)
    if hotspot_weight < 0.0:
        raise ValueError("hotspot_verify_weight must be non-negative")
    if hotspot_weight > 0.0 and hotspot_loss is None:
        raise ValueError(
            "positive hotspot_verify_weight requires hotspot_verify_loss"
        )
    hotspot_enabled = hotspot_loss is not None and hotspot_weight > 0.0
    if hotspot_enabled:
        if hotspot_loss < 0.0 or hotspot_loss > 1.0:
            raise ValueError("hotspot_verify_loss must be in [0, 1]")
        total_loss += hotspot_weight * hotspot_loss
    loss_dict: Dict[str, float] = {
        "total": float(total_loss),
        "esm_verify_loss": float(total_loss),
    }
    loss_dict["pair_chain_iptm_loss"] = float(global_loss)
    if hotspot_enabled:
        loss_dict.update(
            {
                "esm_verify_global_loss": float(global_loss),
                "hotspot_verify_distance_loss": float(hotspot_loss),
                "hotspot_verify_weight": hotspot_weight,
                "hotspot_verify_weighted_contribution": hotspot_weight
                * float(hotspot_loss),
            }
        )
    plddt = _finite_float(summary_confidence.get("binder_plddt"))
    if plddt is not None:
        plddt_unit = plddt / 100.0 if plddt > 1.0 else plddt
        loss_dict["binder_confidence_loss"] = 1.0 - min(max(plddt_unit, 0.0), 1.0)
    rg_excess = _finite_float(summary_confidence.get("binder_rg_excess"))
    if rg_excess is not None:
        loss_dict["binder_rg_excess_loss"] = min(max(rg_excess, 0.0), 1.0)
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
    if hotspot_enabled:
        summary.update(
            {
                "esm_verify_global_loss": float(global_loss),
                "hotspot_verify_distance_loss": float(hotspot_loss),
                "hotspot_verify_weight": hotspot_weight,
                "hotspot_verify_weighted_contribution": hotspot_weight
                * float(hotspot_loss),
            }
        )
    return ESMFold2VerifierResult(
        binder_seq=binder_seq,
        target_seq=target_seq,
        total_loss=float(total_loss),
        loss_dict=loss_dict,
        summary_confidence=summary,
        coordinate=coordinate,
        plddt_logits=plddt_logits,
    )


def compute_esmfold2_action_group_routing(
    loss_vector: Mapping[str, Any],
    *,
    temperature: float = 1.0,
) -> Dict[str, object]:
    """Compute normalized ESMFold2 Stage-B routing scores and priors."""

    raw_scores: Dict[str, float] = {}
    term_details: Dict[str, Dict[str, Dict[str, float]]] = {}
    missing = []
    for group, key in ESMFOLD2_ACTION_GROUP_ROUTING_KEYS.items():
        value = _finite_float(loss_vector.get(key))
        if value is None:
            missing.append(key)
            continue
        score = min(max(float(value), 0.0), 1.0)
        raw_scores[group] = score
        term_details[group] = {
            key: {
                "raw": float(value),
                "scale": 1.0,
                "normalized": score,
                "weight": 1.0,
            }
        }
    if missing:
        raise ValueError(
            "ESMFold2 Stage-B routing requires full-verifier metrics: "
            + ", ".join(missing)
        )

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


def anchor_binder_logits_to_sequence(
    logits: torch.Tensor,
    sequence: str,
    mutable_mask: Optional[Sequence[bool]] = None,
    *,
    anchor_logit: float = 2.0,
    fixed_logit: float = 10.0,
    anchor_margin: float = 0.1,
    suppress_cys: bool = True,
    clone: bool = True,
) -> torch.Tensor:
    """Return node-local logits anchored to a verified discrete sequence.

    The ESMFold2 proposal state should remain logits, not probabilities.  This
    helper preserves the existing soft-logit landscape while making the node's
    verified sequence the row-wise argmax, so child nodes do not drift away from
    the sequence that was actually verifier-scored.
    """

    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError("node-local ESM logits must have batch size 1")
        logits_2d = logits[0]
    elif logits.ndim == 2:
        logits_2d = logits
    else:
        raise ValueError("node-local ESM logits must have shape [L, 20] or [1, L, 20]")

    length = len(sequence)
    if logits_2d.shape != (length, len(ESM_AA_ORDER)):
        raise ValueError(
            f"logits shape {tuple(logits_2d.shape)} does not match sequence shape "
            f"({length}, {len(ESM_AA_ORDER)})"
        )

    anchored = logits_2d.detach().clone() if clone else logits_2d
    current_idx = _sequence_indices(sequence).to(device=anchored.device)
    mutable = _as_bool_mask(mutable_mask, length, device=anchored.device)

    if suppress_cys:
        anchored[:, CYS_INDEX] = float("-inf")

    for pos in range(length):
        aa_idx = int(current_idx[pos].item())
        if not bool(mutable[pos].item()):
            anchored[pos, :] = 0.0
            if suppress_cys:
                anchored[pos, CYS_INDEX] = float("-inf")
            anchored[pos, aa_idx] = float(fixed_logit)
            continue

        row = anchored[pos]
        finite_row = row[torch.isfinite(row)]
        row_max = float(finite_row.max().item()) if finite_row.numel() else 0.0
        desired = max(float(row[aa_idx].item()), float(anchor_logit))
        if desired < row_max:
            desired = row_max + float(anchor_margin)
        row[aa_idx] = desired

    return anchored.detach().cpu().clone()


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

    norms = torch.linalg.vector_norm(masked, dim=(-1, -2))
    active_positions = torch.square(masked).sum(dim=-1) > 0
    effective_lengths = active_positions.sum(dim=-1).to(masked.dtype)
    safe = (effective_lengths > 0) & (norms > eps)
    if safe.any():
        scale = torch.sqrt(torch.clamp(effective_lengths, min=1.0))
        normalized = masked / torch.clamp(norms, min=eps).view(-1, 1, 1)
        normalized = normalized * scale.view(-1, 1, 1)
        output = torch.where(safe.view(-1, 1, 1), normalized, torch.zeros_like(masked))
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


def sequence_from_logits_argmax(
    logits: torch.Tensor,
    *,
    current_seq: str,
    mutable_mask: Optional[Sequence[bool]] = None,
    suppress_cys: bool = True,
) -> str:
    """Convert binder logits to a discrete sequence by argmax.

    Fixed positions are copied from ``current_seq`` so CDR/TCR masks stay
    model-independent and cannot leak edits into protected regions.
    """

    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError("batched logits must have batch size 1")
        logits_2d = logits[0]
    elif logits.ndim == 2:
        logits_2d = logits
    else:
        raise ValueError("logits must have shape [L, 20] or [1, L, 20]")

    length = len(current_seq)
    if logits_2d.shape != (length, len(ESM_AA_ORDER)):
        raise ValueError(
            f"logits shape {tuple(logits_2d.shape)} does not match "
            f"({length}, {len(ESM_AA_ORDER)})"
        )

    scores = torch.nan_to_num(
        logits_2d.detach().clone(),
        nan=float("-inf"),
        posinf=1.0e9,
        neginf=float("-inf"),
    )
    if suppress_cys:
        scores[:, CYS_INDEX] = float("-inf")
    indices = torch.argmax(scores, dim=-1).detach().cpu().tolist()
    mutable = [bool(item) for item in _as_bool_mask(mutable_mask, length).cpu().tolist()]

    chars = list(str(current_seq))
    for idx, aa_index in enumerate(indices):
        if mutable[idx]:
            chars[idx] = ESM_AA_ORDER[int(aa_index)]
    return "".join(chars)


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
    include_logit_candidates: bool = False,
) -> list[str]:
    """Generate discrete candidate sequences from ESMFold2 soft-logit updates.

    The output side deliberately reuses the existing Stage-A Top-K + sampling
    contract: first select top single mutations, then sample up to
    ``max_mutations_per_seq`` edits per candidate.  When requested, it also
    prepends the direct ``argmax(logits_after)`` sequence so Stage A can test
    the soft optimizer's full current sequence instead of only sampled local
    edits.
    """

    from bindcraft.hallucination.mutation_proposer import (
        AA_1LETTER,
        PROTEIN_RESIDUE_NAMES,
        sample_candidate_sequences,
        select_top_k_mutations,
    )

    current_indices = [AA_TO_ESM_IDX[aa] for aa in current_seq]
    candidates: list[str] = []
    if include_logit_candidates:
        argmax_seq = sequence_from_logits_argmax(
            logits_after,
            current_seq=current_seq,
            mutable_mask=mutable_mask,
        )
        if argmax_seq != current_seq:
            candidates.append(argmax_seq)

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
        while len(candidates) < int(B_candidates):
            candidates.append(current_seq)
        return candidates[: int(B_candidates)]

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

    sampled_sequences = [indices_to_sequence(indices) for indices in candidate_indices]
    if not include_logit_candidates:
        return sampled_sequences

    ordered: list[str] = []
    for sequence in [*candidates, *sampled_sequences]:
        if sequence not in ordered:
            ordered.append(sequence)
        if len(ordered) >= int(B_candidates):
            return ordered[: int(B_candidates)]
    for sequence in sampled_sequences:
        ordered.append(sequence)
        if len(ordered) >= int(B_candidates):
            return ordered[: int(B_candidates)]
    while len(ordered) < int(B_candidates):
        ordered.append(current_seq)
    if not ordered:
        return [current_seq] * int(B_candidates)
    return ordered[: int(B_candidates)]
