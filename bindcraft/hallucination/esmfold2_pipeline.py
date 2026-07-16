"""
MCTS loop for the optional ESMFold2 backend.

This file intentionally avoids importing the Protenix pipeline.  The first
ESMFold2 integration should be an explicit backend path, while the default
Protenix V1/V2 execution remains unchanged.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from bindcraft.hallucination.esmfold2_backend import (
    ESM_AA_ORDER,
    ESMFold2ProposalBatch,
    anchor_binder_logits_to_sequence,
    init_binder_logits,
    select_esmfold2_action_group,
    sequence_from_logits_argmax,
)
from bindcraft.hallucination.mcts_legacy import LegacyMCTSTree
from bindcraft.hallucination.mcts_controller import MCTSTree


logger = logging.getLogger(__name__)
VALID_ESM_BINDER_AAS = ESM_AA_ORDER.replace("C", "")
ESMFOLD2_LOGIT_INIT_MODES = {"legacy_anchor", "official_free", "official_exact"}


@dataclass(frozen=True)
class ESMFold2MutationProposal:
    sequence: str
    edits: List[Tuple[int, str, str]]
    prior: float
    benefit: float = 0.0
    action_group: str = "unrouted"
    group_prior: float = 1.0
    proposal_logits: Optional[torch.Tensor] = None


class _NoopArtifactRecorder:
    def write_step(self, **event: Any) -> None:
        return None

    def write_candidate(self, **event: Any) -> None:
        return None

    def write_mcts_node(self, **event: Any) -> None:
        return None

    def write_mcts_edge(self, **event: Any) -> None:
        return None

    def write_mcts_action(self, **event: Any) -> None:
        return None

    def write_soft_step(self, **event: Any) -> None:
        return None

    def write_mcts_handoff(self, **event: Any) -> None:
        return None

    def write_checkpoint(self, **event: Any) -> None:
        return None

    def write_summary(self, **summary: Any) -> None:
        return None


class _ESMArtifactRecorder(_NoopArtifactRecorder):
    """Append-only JSONL artifact writer for ESMFold2 MCTS diagnostics."""

    def __init__(self, artifact_dir: Union[str, os.PathLike[str]]) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_path = self.artifact_dir / "trajectory.jsonl"
        self.candidates_path = self.artifact_dir / "candidates.jsonl"
        self.nodes_path = self.artifact_dir / "mcts_nodes.jsonl"
        self.edges_path = self.artifact_dir / "mcts_edges.jsonl"
        self.actions_path = self.artifact_dir / "mcts_actions.jsonl"
        self.soft_steps_path = self.artifact_dir / "soft_steps.jsonl"
        self.handoff_path = self.artifact_dir / "mcts_handoff.jsonl"
        self.summary_path = self.artifact_dir / "run_summary.json"
        self.trajectory_path.write_text("", encoding="utf-8")
        self.candidates_path.write_text("", encoding="utf-8")
        self.nodes_path.write_text("", encoding="utf-8")
        self.edges_path.write_text("", encoding="utf-8")
        self.actions_path.write_text("", encoding="utf-8")
        self.soft_steps_path.write_text("", encoding="utf-8")
        self.handoff_path.write_text("", encoding="utf-8")

    def write_step(self, **event: Any) -> None:
        self._append_jsonl(self.trajectory_path, event)

    def write_candidate(self, **event: Any) -> None:
        self._append_jsonl(self.candidates_path, event)

    def write_mcts_node(self, **event: Any) -> None:
        self._append_jsonl(self.nodes_path, event)

    def write_mcts_edge(self, **event: Any) -> None:
        self._append_jsonl(self.edges_path, event)

    def write_mcts_action(self, **event: Any) -> None:
        self._append_jsonl(self.actions_path, event)

    def write_soft_step(self, **event: Any) -> None:
        self._append_jsonl(self.soft_steps_path, event)

    def write_mcts_handoff(self, **event: Any) -> None:
        self._append_jsonl(self.handoff_path, event)

    def write_checkpoint(self, **event: Any) -> None:
        stage = _safe_path_token(str(event.get("stage", "unknown")))
        step = event.get("step")
        label = _safe_path_token(str(event.get("label", "round_best")))
        step_index = 0 if step is None else int(step)
        step_name = f"step_{step_index:04d}"
        step_dir = self.artifact_dir / "checkpoints" / stage / step_name
        step_dir.mkdir(parents=True, exist_ok=True)

        result = event.get("result")
        if result is None:
            return
        sequence = str(getattr(result, "binder_seq", ""))
        summary = dict(getattr(result, "summary_confidence", {}) or {})
        mmcif = summary.get("mmcif")
        structure_file = None
        structure_status = "missing_mmcif"
        if isinstance(mmcif, str) and mmcif.strip():
            structure_file = f"{step_name}_{label}.cif"
            (step_dir / structure_file).write_text(mmcif, encoding="utf-8")
            structure_status = "ok"

        metadata = {
            "stage": event.get("stage"),
            "step": step,
            "label": label,
            "binder_seq": sequence,
            "binder_seq_hash": _sequence_hash(sequence),
            "target_seq": getattr(result, "target_seq", None),
            "total_loss": float(getattr(result, "total_loss")),
            "loss_dict": _loss_dict_for_artifact(result),
            "summary_confidence": _summary_without_structure(summary),
            "structure_file": structure_file,
            "structure_format": "mmcif" if structure_file else None,
            "structure_status": structure_status,
            "selected_sequence": event.get("selected_sequence"),
            "parent_sequence": event.get("parent_sequence"),
            "is_global_best": bool(event.get("is_global_best", False)),
        }
        meta_file = f"{step_name}_{label}_meta.json"
        (step_dir / meta_file).write_text(
            json.dumps(_json_ready(metadata), ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )

    def write_summary(self, **summary: Any) -> None:
        self.summary_path.write_text(
            json.dumps(_json_ready(summary), ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_ready(payload), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _make_artifact_recorder(
    artifact_dir: Optional[Union[str, os.PathLike[str]]],
) -> _NoopArtifactRecorder:
    if artifact_dir is None:
        return _NoopArtifactRecorder()
    return _ESMArtifactRecorder(artifact_dir)


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_ready(float(value.detach().cpu().item()))
        return {
            "tensor_shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    scalar = _finite_float(value)
    if scalar is not None:
        return scalar
    return str(value)


def _safe_path_token(text: str) -> str:
    safe = []
    for char in text:
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "item"


def _sequence_hash(sequence: Optional[str]) -> Optional[str]:
    if sequence is None:
        return None
    return hashlib.sha256(str(sequence).encode("utf-8")).hexdigest()[:16]


def _summary_without_structure(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        str(key): value
        for key, value in dict(summary or {}).items()
        if str(key) not in {"mmcif"}
    }


def _loss_dict_for_artifact(result: Any) -> Dict[str, float]:
    return {
        str(key): float(scalar)
        for key, value in dict(getattr(result, "loss_dict", {}) or {}).items()
        if (scalar := _finite_float(value)) is not None
    }


def _summary_for_artifact(result: Any) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key, value in dict(getattr(result, "summary_confidence", {}) or {}).items():
        if isinstance(value, torch.Tensor) and value.numel() != 1:
            continue
        summary[str(key)] = _json_ready(value)
    return summary


def _write_candidate_artifact(
    recorder: _NoopArtifactRecorder,
    *,
    stage: str,
    step: Optional[int],
    parent_sequence: Optional[str],
    result: Any,
    action_group: Optional[str] = None,
) -> None:
    sequence = str(result.binder_seq)
    recorder.write_candidate(
        stage=stage,
        step=step,
        parent_sequence=parent_sequence,
        sequence=sequence,
        edits=_sequence_edits(parent_sequence, sequence) if parent_sequence else [],
        action_group=action_group,
        total_loss=float(result.total_loss),
        loss_dict=_loss_dict_for_artifact(result),
        summary_confidence=_summary_for_artifact(result),
    )


def _write_soft_step_artifacts(
    recorder: _NoopArtifactRecorder,
    *,
    stage: str,
    step: Optional[int],
    selected_sequence: str,
    batch: ESMFold2ProposalBatch,
    action_group: Optional[str] = None,
) -> None:
    for metric in list(getattr(batch, "soft_step_metrics", []) or []):
        recorder.write_soft_step(
            stage=stage,
            step=step,
            selected_sequence=str(selected_sequence),
            action_group=action_group,
            **dict(metric),
        )


def _mean_legacy_loss(node: Any) -> Optional[float]:
    if not hasattr(node, "W") or not hasattr(node, "visit_count"):
        return None
    visits = max(int(getattr(node, "visit_count", 0)), 1)
    return float(getattr(node, "W")) / visits


def _mcts_node_artifact(
    *,
    stage: str,
    node: Any,
    step_created: Optional[int],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sequence = str(getattr(node, "sequence", ""))
    parent = getattr(node, "parent", None)
    payload: Dict[str, Any] = {
        "stage": stage,
        "step_created": step_created,
        "node_id": int(getattr(node, "node_id")),
        "node_uid": f"{stage}:{int(getattr(node, 'node_id'))}",
        "parent_node_id": None if parent is None else int(getattr(parent, "node_id")),
        "parent_node_uid": None if parent is None else f"{stage}:{int(getattr(parent, 'node_id'))}",
        "depth": int(getattr(node, "depth", 0)),
        "sequence": sequence,
        "sequence_hash": _sequence_hash(sequence),
        "parent_sequence_hash": _sequence_hash(getattr(parent, "sequence", None)) if parent is not None else None,
        "edits": list(getattr(node, "edit_from_parent", []) or []),
        "total_loss": float(getattr(node, "L_real")),
        "visit_count": int(getattr(node, "visit_count", 0)),
    }
    if parent is not None and not payload["edits"]:
        payload["edits"] = _sequence_edits(str(getattr(parent, "sequence", "")), sequence)
    if hasattr(node, "W"):
        payload["W"] = float(getattr(node, "W"))
        payload["mean_loss"] = _mean_legacy_loss(node)
    if hasattr(node, "tree_reward"):
        payload["tree_reward"] = float(getattr(node, "tree_reward"))
        payload["value_sum"] = float(getattr(node, "value_sum", 0.0))
        payload["q_value"] = float(getattr(node, "q_value", 0.0))
        payload["prior"] = float(getattr(node, "prior", 1.0))
        payload["group_id"] = str(getattr(node, "group_id", "ungrouped"))
        payload["loss_vector"] = dict(getattr(node, "loss_vector", {}) or {})
        payload["status_flags"] = dict(getattr(node, "status_flags", {}) or {})
        payload["structural_signature"] = dict(getattr(node, "structural_signature", {}) or {})
    if extra:
        payload.update(extra)
    return payload


def _mcts_edge_artifact(
    *,
    stage: str,
    parent: Any,
    child: Any,
    step_created: Optional[int],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    parent_seq = str(getattr(parent, "sequence", ""))
    child_seq = str(getattr(child, "sequence", ""))
    payload: Dict[str, Any] = {
        "stage": stage,
        "step_created": step_created,
        "parent_node_id": int(getattr(parent, "node_id")),
        "child_node_id": int(getattr(child, "node_id")),
        "parent_node_uid": f"{stage}:{int(getattr(parent, 'node_id'))}",
        "child_node_uid": f"{stage}:{int(getattr(child, 'node_id'))}",
        "parent_sequence_hash": _sequence_hash(parent_seq),
        "child_sequence_hash": _sequence_hash(child_seq),
        "edits": _sequence_edits(parent_seq, child_seq),
        "num_mutations": sum(1 for a, b in zip(parent_seq, child_seq) if a != b),
        "child_total_loss": float(getattr(child, "L_real")),
    }
    if extra:
        payload.update(extra)
    return payload


def _mcts_action_artifact(
    *,
    stage: str,
    step: Optional[int],
    action: Any,
    selected: bool,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sequence = str(getattr(action, "sequence", ""))
    payload: Dict[str, Any] = {
        "stage": stage,
        "step": step,
        "action_id": int(getattr(action, "action_id")),
        "parent_node_id": int(getattr(action, "parent_node_id")),
        "parent_node_uid": f"{stage}:{int(getattr(action, 'parent_node_id'))}",
        "child_node_id": getattr(action, "child_node_id", None),
        "child_node_uid": (
            None
            if getattr(action, "child_node_id", None) is None
            else f"{stage}:{int(getattr(action, 'child_node_id'))}"
        ),
        "sequence": sequence,
        "sequence_hash": _sequence_hash(sequence),
        "edits": list(getattr(action, "edits", []) or []),
        "prior": float(getattr(action, "prior", 0.0)),
        "benefit": float(getattr(action, "benefit", 0.0)),
        "action_group": str(getattr(action, "action_group", "unrouted")),
        "group_prior": float(getattr(action, "group_prior", 1.0)),
        "visit_count": int(getattr(action, "visit_count", 0)),
        "value_sum": float(getattr(action, "value_sum", 0.0)),
        "q_value": float(getattr(action, "q_value", 0.0)),
        "status": str(getattr(action, "status", "unknown")),
        "selected": bool(selected),
        "status_flags": dict(getattr(action, "status_flags", {}) or {}),
    }
    if extra:
        payload.update(extra)
    return payload


def _stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _normalize_target_key(target_seq: Union[str, Sequence[str]]) -> str:
    if isinstance(target_seq, str):
        return target_seq
    return "|".join(str(seq) for seq in target_seq)


def _placeholder_logits(sequence: str) -> torch.Tensor:
    return torch.zeros(len(sequence), len(ESM_AA_ORDER), dtype=torch.float32)


def _initial_proposal_logits(
    sequence: str,
    mutable_mask: Optional[Sequence[bool]],
) -> torch.Tensor:
    logits = init_binder_logits(
        sequence,
        mutable_mask=mutable_mask,
        batch_size=1,
        random_scale=0.0,
        requires_grad=False,
    )[0]
    with torch.no_grad():
        for pos, aa in enumerate(str(sequence)):
            aa_index = ESM_AA_ORDER.index(aa)
            if torch.isfinite(logits[pos, aa_index]):
                logits[pos, aa_index] = max(float(logits[pos, aa_index].item()), 2.0)
    return anchor_binder_logits_to_sequence(
        logits,
        str(sequence),
        mutable_mask=mutable_mask,
    )


def _anchor_proposal_logits(
    logits: Optional[torch.Tensor],
    sequence: str,
    mutable_mask: Optional[Sequence[bool]],
) -> torch.Tensor:
    if logits is None:
        return _initial_proposal_logits(sequence, mutable_mask)
    return anchor_binder_logits_to_sequence(
        logits,
        str(sequence),
        mutable_mask=mutable_mask,
    )


def _normalize_proposal_batch(raw: Any) -> ESMFold2ProposalBatch:
    if isinstance(raw, ESMFold2ProposalBatch):
        return raw
    return ESMFold2ProposalBatch(sequences=[str(sequence) for sequence in raw])


def _proposal_confidence_records(batch: ESMFold2ProposalBatch) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for item in list(getattr(batch, "confidence_candidates", []) or []):
        if not isinstance(item, dict):
            continue
        sequence = str(item.get("sequence", "")).strip().upper()
        if not sequence:
            continue
        record = dict(item)
        record["sequence"] = sequence
        records.append(record)
    records.sort(key=_proposal_confidence_score, reverse=True)
    return records


def _proposal_confidence_score(record: Dict[str, Any]) -> float:
    for key in ("confidence_score", "pair_chain_iptm", "iptm"):
        scalar = _finite_float(record.get(key))
        if scalar is not None:
            return float(scalar)
    return float("-inf")


def _proposal_sequences_from_batch(
    batch: ESMFold2ProposalBatch,
    *,
    include_confidence: bool = True,
) -> List[str]:
    confidence_sequences = []
    if include_confidence:
        confidence_sequences = [
            str(record["sequence"])
            for record in _proposal_confidence_records(batch)
        ]
    ordinary_sequences = [str(sequence) for sequence in list(batch.sequences or [])]
    return list(dict.fromkeys([*confidence_sequences, *ordinary_sequences]))


def _proposal_confidence_summary(batch: ESMFold2ProposalBatch) -> Dict[str, Any]:
    records = _proposal_confidence_records(batch)
    if not records:
        return {
            "proposal_confidence_candidate_count": 0,
            "proposal_confidence_best": None,
        }
    return {
        "proposal_confidence_candidate_count": len(records),
        "proposal_confidence_best": records[0],
        "proposal_confidence_candidates": records,
    }


def _call_backend_propose(
    backend: Any,
    *,
    proposal_logits: Optional[torch.Tensor],
    hotspot_params: Optional[Dict[str, Any]] = None,
    track_confidence_override: Optional[bool] = None,
    soft_steps_override: Optional[int] = None,
    **kwargs: Any,
) -> ESMFold2ProposalBatch:
    propose_kwargs = dict(kwargs)
    if _callable_accepts_argument(backend.propose, "proposal_logits"):
        propose_kwargs["proposal_logits"] = (
            proposal_logits.detach().clone() if proposal_logits is not None else None
        )
    if _callable_accepts_argument(backend.propose, "track_confidence_override"):
        propose_kwargs["track_confidence_override"] = track_confidence_override
    if _callable_accepts_argument(backend.propose, "soft_steps_override"):
        propose_kwargs["soft_steps_override"] = soft_steps_override
    if _callable_accepts_argument(backend.propose, "hotspot_params"):
        propose_kwargs["hotspot_params"] = (
            dict(hotspot_params) if hotspot_params is not None else None
        )
    raw = backend.propose(**propose_kwargs)
    return _normalize_proposal_batch(raw)


def _call_backend_continuous_design(
    backend: Any,
    *,
    proposal_logits: Optional[torch.Tensor],
    hotspot_params: Optional[Dict[str, Any]] = None,
    track_confidence_override: Optional[bool] = None,
    soft_steps_override: Optional[int] = None,
    **kwargs: Any,
) -> ESMFold2ProposalBatch:
    if not hasattr(backend, "continuous_design"):
        raise ValueError("ESMFold2 continuous_stage_a requires backend.continuous_design")
    design_kwargs = dict(kwargs)
    if _callable_accepts_argument(backend.continuous_design, "proposal_logits"):
        design_kwargs["proposal_logits"] = (
            proposal_logits.detach().clone() if proposal_logits is not None else None
        )
    if _callable_accepts_argument(backend.continuous_design, "track_confidence_override"):
        design_kwargs["track_confidence_override"] = track_confidence_override
    if _callable_accepts_argument(backend.continuous_design, "soft_steps_override"):
        design_kwargs["soft_steps_override"] = soft_steps_override
    if _callable_accepts_argument(backend.continuous_design, "hotspot_params"):
        design_kwargs["hotspot_params"] = (
            dict(hotspot_params) if hotspot_params is not None else None
        )
    raw = backend.continuous_design(**design_kwargs)
    return _normalize_proposal_batch(raw)


def _callable_accepts_argument(func: Any, name: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == name:
            return True
    return False


def _sequence_edits(parent_seq: str, child_seq: str) -> List[Tuple[int, str, str]]:
    return [
        (idx, old, new)
        for idx, (old, new) in enumerate(zip(parent_seq, child_seq))
        if old != new
    ]


def _allocate_group_budgets(group_priors: Dict[str, float], total_budget: int) -> Dict[str, int]:
    if total_budget <= 0:
        return {}
    valid = [
        (str(group), max(float(prior), 0.0))
        for group, prior in (group_priors or {}).items()
        if float(prior) > 0.0
    ]
    if not valid:
        return {}
    total_prior = sum(prior for _, prior in valid)
    if total_prior <= 0.0:
        return {}
    scaled = [(group, total_budget * prior / total_prior) for group, prior in valid]
    budgets = {group: int(math.floor(raw)) for group, raw in scaled}
    used = sum(budgets.values())
    remainders = sorted(
        ((raw - math.floor(raw), group) for group, raw in scaled),
        reverse=True,
    )
    for _, group in remainders:
        if used >= total_budget:
            break
        budgets[group] += 1
        used += 1
    return {group: count for group, count in budgets.items() if count > 0}


def _normalize_prior_mode(mode: Optional[str]) -> str:
    normalized = str(mode or "full").strip().lower()
    if normalized not in {"full", "uniform", "no_gradient", "no_group", "no_sequence"}:
        return "full"
    return normalized


def _apply_stageb_group_prior_mode(
    prior_mode: str,
    action_group: str,
    group_priors: Dict[str, float],
) -> Tuple[str, Dict[str, float]]:
    prior_mode = _normalize_prior_mode(prior_mode)
    if prior_mode in {"uniform", "no_group"}:
        return "unrouted", {"unrouted": 1.0}
    return action_group, group_priors or {action_group: 1.0}


def _make_phase0_seed_sequences(
    init_binder_seq: Optional[str],
    *,
    num_beams: int,
    L_binder: int,
    seed: Optional[int],
    mutable_mask: Optional[Sequence[bool]] = None,
) -> List[str]:
    num_beams = max(int(num_beams), 1)
    rng = random.Random(seed)
    mask = _normalize_mutable_mask(mutable_mask, L_binder)

    if init_binder_seq is not None:
        init = str(init_binder_seq).strip().upper()
        if len(init) != int(L_binder):
            raise ValueError(
                f"init_binder_seq length {len(init)} does not match L_binder {L_binder}"
            )
        invalid = sorted(set(init) - set(ESM_AA_ORDER))
        if invalid:
            raise ValueError(f"Invalid amino acid(s) in init_binder_seq: {invalid}")
        mutable_positions = [idx for idx, is_mutable in enumerate(mask) if is_mutable]
        seeds = [init]
        while len(seeds) < num_beams and mutable_positions:
            seq = list(init)
            pos = rng.choice(mutable_positions)
            aa_choices = [aa for aa in VALID_ESM_BINDER_AAS if aa != seq[pos]]
            seq[pos] = rng.choice(aa_choices)
            candidate = "".join(seq)
            if candidate not in seeds:
                seeds.append(candidate)
        return seeds

    seeds: List[str] = []
    while len(seeds) < num_beams:
        candidate = "".join(rng.choice(VALID_ESM_BINDER_AAS) for _ in range(L_binder))
        if candidate not in seeds:
            seeds.append(candidate)
    return seeds


def _normalize_logit_init_mode(mode: Optional[str]) -> str:
    normalized = str(mode or "legacy_anchor").strip().lower()
    if normalized not in ESMFOLD2_LOGIT_INIT_MODES:
        raise ValueError(
            "esmfold2_logit_init_mode must be one of: "
            + ", ".join(sorted(ESMFOLD2_LOGIT_INIT_MODES))
        )
    return normalized


@contextmanager
def _torch_seed_context(seed: Optional[int], device: Optional[Union[str, torch.device]] = None):
    devices: List[int] = []
    if device is not None:
        torch_device = torch.device(device)
        if torch_device.type == "cuda" and torch.cuda.is_available():
            devices = [torch_device.index if torch_device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        yield


def _make_official_free_phase0_seed_states(
    *,
    num_beams: int,
    L_binder: int,
    seed: Optional[int],
    mutable_mask: Sequence[bool],
    base_sequence: Optional[str] = None,
) -> List[Tuple[str, torch.Tensor]]:
    """Create official-style free mutable logits and matching argmax roots.

    Fixed scaffold positions come from ``base_sequence`` and are hard anchored
    by ``init_binder_logits``.  Mutable positions stay near-uniform/free and
    the initial discrete residue is selected by row-wise argmax, matching the
    ESMFold2 binder-design style more closely than anchoring mutable residues to
    an arbitrary starting sequence.
    """

    if base_sequence is not None:
        base_sequence = str(base_sequence).strip().upper()
        if len(base_sequence) != int(L_binder):
            raise ValueError(
                f"init_binder_seq length {len(base_sequence)} does not match L_binder {L_binder}"
            )
        invalid = sorted(set(base_sequence) - set(ESM_AA_ORDER))
        if invalid:
            raise ValueError(f"Invalid amino acid(s) in init_binder_seq: {invalid}")
    elif not all(bool(item) for item in mutable_mask):
        raise ValueError(
            "official_free ESMFold2 root initialization without init_binder_seq "
            "requires all binder positions to be mutable"
        )

    base_rng = random.Random(seed)
    states: List[Tuple[str, torch.Tensor]] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = max(int(num_beams), 1) * 100
    while len(states) < max(int(num_beams), 1) and attempts < max_attempts:
        attempts += 1
        base_seq = base_sequence or "".join(
            base_rng.choice(VALID_ESM_BINDER_AAS) for _ in range(L_binder)
        )
        generator = torch.Generator()
        generator.manual_seed(_stable_seed(seed, "esm_phase0_free_logits", attempts))
        logits = init_binder_logits(
            base_seq,
            mutable_mask=mutable_mask,
            batch_size=1,
            random_scale=0.01,
            requires_grad=False,
            generator=generator,
        )[0].detach().cpu()
        seq = sequence_from_logits_argmax(
            logits,
            current_seq=base_seq,
            mutable_mask=mutable_mask,
        )
        if seq in seen:
            continue
        seen.add(seq)
        states.append((seq, logits))
    if len(states) < max(int(num_beams), 1):
        raise RuntimeError("failed to generate unique official_free ESMFold2 phase0 roots")
    return states


def _make_official_exact_phase0_seed_states(
    *,
    num_beams: int,
    L_binder: int,
    seed: Optional[int],
    mutable_mask: Sequence[bool],
    base_sequence: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> List[Tuple[str, torch.Tensor]]:
    """Create official-seed logits for exact official-binder-design diagnosis."""

    if base_sequence is not None:
        base_sequence = str(base_sequence).strip().upper()
        if len(base_sequence) != int(L_binder):
            raise ValueError(
                f"init_binder_seq length {len(base_sequence)} does not match L_binder {L_binder}"
            )
        invalid = sorted(set(base_sequence) - set(ESM_AA_ORDER))
        if invalid:
            raise ValueError(f"Invalid amino acid(s) in init_binder_seq: {invalid}")
    elif not all(bool(item) for item in mutable_mask):
        raise ValueError(
            "official_exact ESMFold2 root initialization without init_binder_seq "
            "requires all binder positions to be mutable"
        )

    template_seq = base_sequence or ("A" * int(L_binder))
    batch_size = max(int(num_beams), 1)
    torch_device = torch.device(device) if device is not None else torch.device("cpu")
    with _torch_seed_context(seed, torch_device):
        logits_batch = init_binder_logits(
            template_seq,
            mutable_mask=mutable_mask,
            batch_size=batch_size,
            random_scale=0.01,
            requires_grad=False,
            device=torch_device,
        ).detach().cpu()
    states: List[Tuple[str, torch.Tensor]] = []
    for idx in range(batch_size):
        logits = logits_batch[idx]
        seq = sequence_from_logits_argmax(
            logits,
            current_seq=template_seq,
            mutable_mask=mutable_mask,
        )
        states.append((seq, logits))
    return states


def _make_phase0_seed_states(
    init_binder_seq: Optional[str],
    *,
    num_beams: int,
    L_binder: int,
    seed: Optional[int],
    mutable_mask: Sequence[bool],
    logit_init_mode: str,
    device: Optional[Union[str, torch.device]] = None,
) -> List[Tuple[str, Optional[torch.Tensor]]]:
    mode = _normalize_logit_init_mode(logit_init_mode)
    if mode == "official_free":
        return _make_official_free_phase0_seed_states(
            num_beams=num_beams,
            L_binder=L_binder,
            seed=seed,
            mutable_mask=mutable_mask,
            base_sequence=init_binder_seq,
        )
    if mode == "official_exact":
        return _make_official_exact_phase0_seed_states(
            num_beams=num_beams,
            L_binder=L_binder,
            seed=seed,
            mutable_mask=mutable_mask,
            base_sequence=init_binder_seq,
            device=device,
        )
    return [
        (seq, None)
        for seq in _make_phase0_seed_sequences(
            init_binder_seq,
            num_beams=num_beams,
            L_binder=L_binder,
            seed=seed,
            mutable_mask=mutable_mask,
        )
    ]


def _normalize_mutable_mask(
    mutable_mask: Optional[Sequence[bool]],
    length: int,
) -> List[bool]:
    if mutable_mask is None:
        return [True] * int(length)
    mask = [bool(item) for item in mutable_mask]
    if len(mask) != int(length):
        raise ValueError(f"mutable_mask length {len(mask)} does not match binder length {length}")
    return mask


def _clip_sequence_by_mutable_mask(
    sequence: str,
    mutable_mask: Sequence[bool],
    reference_sequence: str,
) -> str:
    if len(sequence) != len(reference_sequence):
        raise ValueError("candidate sequence length must match reference sequence length")
    if len(mutable_mask) != len(reference_sequence):
        raise ValueError("mutable_mask length must match reference sequence length")
    chars = list(str(sequence))
    for idx, is_mutable in enumerate(mutable_mask):
        if not is_mutable:
            chars[idx] = reference_sequence[idx]
    return "".join(chars)


def _verify_with_cache(
    backend: Any,
    verifier_cache: Dict[str, Any],
    *,
    binder_seq: str,
    target_seq: Union[str, Sequence[str]],
    seed: Optional[int],
) -> Any:
    if binder_seq in verifier_cache:
        return verifier_cache[binder_seq]
    result = backend.verify(
        binder_seq=binder_seq,
        target_seq=target_seq,
        seed=seed,
    )
    verifier_cache[str(result.binder_seq)] = result
    return result


def _finite_float(value: Any) -> Optional[float]:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(scalar):
        return None
    return scalar


def _merge_preview_losses_for_routing(
    backend: Any,
    preview_cache: Dict[str, Dict[str, float]],
    loss_vector: Dict[str, Any],
    *,
    binder_seq: str,
    target_seq: Union[str, Sequence[str]],
    seed: Optional[int],
    hotspot_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Merge optional lightweight ESMFold2 preview losses into a node vector."""

    if not hasattr(backend, "preview_loss_dict"):
        return {}
    required_keys = ["inter_contact_loss", "intra_contact_loss", "glob_loss"]
    if hotspot_params:
        required_keys.append("hotspot_contact_loss")
    if all(key in loss_vector for key in required_keys):
        return {}
    if binder_seq not in preview_cache:
        try:
            preview_kwargs = dict(
                binder_seq=binder_seq,
                target_seq=target_seq,
                seed=seed,
            )
            if _callable_accepts_argument(backend.preview_loss_dict, "hotspot_params"):
                preview_kwargs["hotspot_params"] = dict(hotspot_params) if hotspot_params else None
            preview = backend.preview_loss_dict(**preview_kwargs)
        except Exception as exc:  # pragma: no cover - diagnostic fallback path
            logger.warning(
                "ESM StageB preview routing loss failed for %s: %s",
                binder_seq[:20],
                exc,
            )
            preview_cache[binder_seq] = {}
        else:
            preview_cache[binder_seq] = {
                str(key): float(scalar)
                for key, value in dict(preview or {}).items()
                if (scalar := _finite_float(value)) is not None
            }
    for key, value in preview_cache[binder_seq].items():
        loss_vector.setdefault(key, value)
    return preview_cache[binder_seq]


def _format_float_map(values: Dict[str, float], *, precision: int = 3) -> str:
    if not values:
        return "-"
    return ",".join(
        f"{key}:{float(values[key]):.{precision}f}"
        for key in sorted(values)
    )


def _format_count_map(values: Dict[str, int]) -> str:
    if not values:
        return "-"
    return ",".join(f"{key}:{int(values[key])}" for key in sorted(values))


def _count_items(items: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        key = str(item)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _top_handoff_results(
    verifier_cache: Dict[str, Any],
    *,
    top_k: int,
    min_distance: int = 1,
) -> List[Any]:
    ordered = sorted(verifier_cache.values(), key=lambda result: float(result.total_loss))
    selected: List[Any] = []
    for result in ordered:
        seq = str(result.binder_seq)
        if any(
            sum(a != b for a, b in zip(seq, str(existing.binder_seq))) < min_distance
            for existing in selected
        ):
            continue
        selected.append(result)
        if len(selected) >= max(int(top_k), 1):
            break
    if not selected and ordered:
        selected.append(ordered[0])
    return selected


def _make_stageb_proposals(
    *,
    current_seq: str,
    candidates: Sequence[str],
    action_group: str,
    group_prior: float,
    proposal_logits: Optional[torch.Tensor] = None,
) -> List[ESMFold2MutationProposal]:
    unique = [seq for seq in dict.fromkeys(str(item) for item in candidates) if seq != current_seq]
    if not unique:
        return []
    base_prior = max(float(group_prior), 1.0e-12) / max(len(unique), 1)
    proposals: List[ESMFold2MutationProposal] = []
    for seq in unique:
        proposals.append(
            ESMFold2MutationProposal(
                sequence=seq,
                edits=_sequence_edits(current_seq, seq),
                prior=base_prior,
                benefit=0.0,
                action_group=action_group,
                group_prior=max(float(group_prior), 1.0e-12),
                proposal_logits=proposal_logits.detach().clone() if proposal_logits is not None else None,
            )
        )
    return proposals


def run_esmfold2_mcts_hallucination(
    *,
    backend: Any,
    target_seq: Union[str, Sequence[str]],
    L_binder: int = 40,
    N_steps: int = 100,
    K_top: int = 10,
    B_candidates: int = 8,
    T_sample: float = 1.0,
    max_mutations_per_seq: int = 3,
    stagea_max_mutations_per_seq: Optional[int] = None,
    stageb_max_mutations_per_seq: Optional[int] = None,
    num_beams: int = 8,
    init_binder_seq: Optional[str] = None,
    mcts_C: float = 1.0,
    mcts_max_nodes: int = 5000,
    mcts_strategy: str = "legacy",
    mcts_bootstrap_steps: int = 50,
    mcts_bootstrap_patience: int = 10,
    mcts_bootstrap_min_delta: float = 0.02,
    mcts_bootstrap_max_steps: int = 80,
    mcts_handoff_top_k: Optional[int] = None,
    mcts_expand_actions_per_step: int = 2,
    mcts_prior_mode: str = "full",
    seed: Optional[int] = 42,
    mutable_mask: Optional[Sequence[bool]] = None,
    hotspot_params: Optional[Dict[str, Any]] = None,
    artifact_dir: Optional[Union[str, os.PathLike[str]]] = None,
    esmfold2_logit_init_mode: str = "legacy_anchor",
    esmfold2_stagea_soft_steps: Optional[int] = None,
    esmfold2_stageb_soft_steps: Optional[int] = None,
    esmfold2_final_polish: bool = False,
) -> Tuple[str, str, List[float]]:
    """
    Run the first ESMFold2 closed-loop path.

    The backend supplies two operations:
      - verify(binder_seq, target_seq, seed) -> MCTS-compatible result
      - propose(current_seq, target_seq, global_step, K/B/T args, seed) -> sequences

    ``stage_a_only`` keeps the ESMFold2 Stage-A proposal/verify loop for all
    steps.  ``legacy`` is kept as a backwards-compatible alias for that mode.
    ``two_stage`` uses Stage A as bootstrap, then hands verified candidates to
    the action-level strict MCTS tree and uses ESMFold2-native group routing for
    Stage B.  ``continuous_stage_a`` is a diagnostic mode: run one continuous
    soft-logit update from the best phase0 root, then verify the argmax sequence.
    """

    if backend is None:
        raise ValueError("backend is required for ESMFold2 MCTS")
    L_binder = int(L_binder)
    N_steps = max(int(N_steps), 0)
    K_top = max(int(K_top), 1)
    B_candidates = max(int(B_candidates), 1)
    max_mutations_per_seq = max(int(max_mutations_per_seq), 1)
    stagea_max_mutations = (
        max(int(stagea_max_mutations_per_seq), 1)
        if stagea_max_mutations_per_seq is not None
        else max_mutations_per_seq
    )
    stageb_max_mutations = (
        max(int(stageb_max_mutations_per_seq), 1)
        if stageb_max_mutations_per_seq is not None
        else max_mutations_per_seq
    )
    strategy = str(mcts_strategy or "legacy").strip().lower()
    if strategy == "strict":
        strategy = "two_stage"
    if strategy == "legacy":
        strategy = "stage_a_only"
    if strategy not in {"stage_a_only", "two_stage", "continuous_stage_a"}:
        raise ValueError(
            "ESMFold2 backend supports mcts_strategy stage_a_only, "
            "two_stage, or continuous_stage_a"
        )
    target_key = _normalize_target_key(target_seq)
    mutable_mask = _normalize_mutable_mask(mutable_mask, L_binder)
    hotspot_params = dict(hotspot_params) if hotspot_params else None
    hotspot_indices = [
        int(index)
        for index in (hotspot_params or {}).get("target_hotspot_concat_indices", [])
    ]
    backend_config = getattr(backend, "config", None)
    hotspot_weight = float(getattr(backend_config, "hotspot_weight", 0.0))
    hotspot_cutoff = float(getattr(backend_config, "hotspot_cutoff", 8.0))
    if hotspot_weight > 0.0 and not hotspot_indices:
        raise ValueError(
            "positive ESMFold2 hotspot_weight requires target hotspot indices"
        )
    hotspot_summary = {
        "hotspot_indices_0based": hotspot_indices,
        "hotspot_weight": hotspot_weight,
        "hotspot_cutoff": hotspot_cutoff,
        "hotspot_enabled": bool(hotspot_indices and hotspot_weight > 0.0),
    }
    logit_init_mode = _normalize_logit_init_mode(esmfold2_logit_init_mode)
    stagea_soft_steps = (
        int(esmfold2_stagea_soft_steps)
        if esmfold2_stagea_soft_steps is not None
        else None
    )
    stageb_soft_steps = (
        int(esmfold2_stageb_soft_steps)
        if esmfold2_stageb_soft_steps is not None
        else stagea_soft_steps
    )
    if stagea_soft_steps is not None and stagea_soft_steps <= 0:
        raise ValueError("esmfold2_stagea_soft_steps must be positive")
    if stageb_soft_steps is not None and stageb_soft_steps <= 0:
        raise ValueError("esmfold2_stageb_soft_steps must be positive")
    artifact_recorder = _make_artifact_recorder(artifact_dir)

    tree = LegacyMCTSTree(C=mcts_C, max_nodes=mcts_max_nodes)
    verifier_cache: Dict[str, Any] = {}

    logger.info("=" * 70)
    logger.info("ESMFold2 MCTS closed loop")
    logger.info("Target length key=%s | binder length=%d | steps=%d", target_key, L_binder, N_steps)
    logger.info(
        "K_top=%d B=%d T_sample=%.3f max_mut=%d stagea_max_mut=%d stageb_max_mut=%d",
        K_top,
        B_candidates,
        T_sample,
        max_mutations_per_seq,
        stagea_max_mutations,
        stageb_max_mutations,
    )
    logger.info(
        "strategy=%s bootstrap_steps=%d max=%d logit_init=%s",
        strategy,
        mcts_bootstrap_steps,
        mcts_bootstrap_max_steps,
        logit_init_mode,
    )
    logger.info(
        "ESMFold2 soft_steps: stage_a=%s stage_b=%s",
        str(stagea_soft_steps or "backend_default"),
        str(stageb_soft_steps or "backend_default"),
    )
    logger.info(
        "ESMFold2 hotspot: indices_0based=%s weight=%.4f cutoff=%.2fA enabled=%s",
        hotspot_indices or "none",
        hotspot_weight,
        hotspot_cutoff,
        hotspot_summary["hotspot_enabled"],
    )
    logger.info("=" * 70)

    phase0_states = _make_phase0_seed_states(
        init_binder_seq,
        num_beams=num_beams,
        L_binder=L_binder,
        seed=seed,
        mutable_mask=mutable_mask,
        logit_init_mode=logit_init_mode,
        device=getattr(backend, "device", None) if logit_init_mode == "official_exact" else None,
    )
    phase0_logits_by_sequence = {
        str(seq): logits.detach().cpu().clone()
        for seq, logits in phase0_states
        if logits is not None
    }
    phase0_results = []
    for idx, (binder_seq, _) in enumerate(phase0_states):
        result = _verify_with_cache(
            backend,
            verifier_cache,
            binder_seq=binder_seq,
            target_seq=target_seq,
            seed=_stable_seed(seed, "esm_phase0", target_key, binder_seq),
        )
        phase0_results.append(result)
        _write_candidate_artifact(
            artifact_recorder,
            stage="phase0",
            step=None,
            parent_sequence=None,
            result=result,
        )
        logger.info(
            "ESM Phase0 %d/%d | seq=%s L=%.4f",
            idx + 1,
            len(phase0_states),
            binder_seq[:20],
            result.total_loss,
        )

    phase0_results.sort(key=lambda result: result.total_loss)
    proposal_logits_by_sequence: Dict[str, torch.Tensor] = {}
    stagea_node_by_sequence: Dict[str, Any] = {}
    for result in phase0_results[: max(int(num_beams), 1)]:
        root_logits = phase0_logits_by_sequence.get(str(result.binder_seq))
        if root_logits is None:
            root_logits = _initial_proposal_logits(result.binder_seq, mutable_mask)
        root_node = tree.add_root_node(
            sequence=result.binder_seq,
            logits=root_logits,
            L_real=result.total_loss,
            coordinate=result.coordinate,
        )
        stagea_node_by_sequence[str(root_node.sequence)] = root_node
        proposal_logits_by_sequence[str(result.binder_seq)] = root_logits
        artifact_recorder.write_mcts_node(
            **_mcts_node_artifact(
                stage="stage_a",
                node=root_node,
                step_created=None,
                extra={"source": "phase0_root"},
            )
        )

    best_result = phase0_results[0]
    best_seq = str(best_result.binder_seq)
    best_loss = float(best_result.total_loss)
    best_logits = _anchor_proposal_logits(
        proposal_logits_by_sequence.get(best_seq),
        best_seq,
        mutable_mask,
    ).detach().cpu().clone()
    proposal_logits_by_sequence[best_seq] = best_logits
    losses_history: List[float] = []

    def _remember_global_best(result: Any, logits: Optional[torch.Tensor] = None) -> None:
        nonlocal best_result, best_seq, best_loss, best_logits
        seq = str(result.binder_seq)
        source_logits = logits
        if source_logits is None:
            source_logits = proposal_logits_by_sequence.get(seq)
        best_result = result
        best_seq = seq
        best_loss = float(result.total_loss)
        best_logits = _anchor_proposal_logits(
            source_logits,
            seq,
            mutable_mask,
        ).detach().cpu().clone()
        proposal_logits_by_sequence[seq] = best_logits

    artifact_recorder.write_step(
        stage="phase0",
        step=None,
        candidate_count=len(phase0_results),
        round_best_loss=float(best_result.total_loss),
        global_best_loss=best_loss,
        best_sequence=best_seq,
        selected_sequence=None,
        logit_init_mode=logit_init_mode,
    )
    artifact_recorder.write_checkpoint(
        stage="phase0",
        step=0,
        label="round_best",
        result=best_result,
        is_global_best=True,
    )

    def _run_final_polish_if_enabled(
        *,
        current_best_result: Any,
        current_best_seq: str,
        current_best_loss: float,
    ) -> Tuple[Any, str, str, float]:
        if not bool(esmfold2_final_polish):
            return current_best_result, current_best_seq, current_best_seq, current_best_loss

        polish_step = len(losses_history)
        polish_seed = _stable_seed(seed, "esm_final_polish", target_key, current_best_seq)
        polish_logits = (
            best_logits
            if current_best_seq == best_seq
            else proposal_logits_by_sequence.get(current_best_seq)
        )
        polish_logits = _anchor_proposal_logits(
            polish_logits,
            current_best_seq,
            mutable_mask,
        ).detach().cpu().clone()

        proposal_batch = _call_backend_continuous_design(
            backend,
            current_seq=current_best_seq,
            target_seq=target_seq,
            global_step=polish_step,
            seed=polish_seed,
            mutable_mask=mutable_mask,
            proposal_logits=polish_logits,
            hotspot_params=hotspot_params,
            track_confidence_override=True,
            soft_steps_override=stagea_soft_steps,
        )
        _write_soft_step_artifacts(
            artifact_recorder,
            stage="final_polish",
            step=polish_step,
            selected_sequence=current_best_seq,
            batch=proposal_batch,
        )
        if proposal_batch.logits_after is not None:
            proposal_logits_by_sequence[current_best_seq] = proposal_batch.logits_after.detach().cpu().clone()

        candidates = [
            _clip_sequence_by_mutable_mask(str(seq), mutable_mask, current_best_seq)
            for seq in _proposal_sequences_from_batch(proposal_batch, include_confidence=True)
        ]
        candidates = list(dict.fromkeys(seq for seq in candidates if seq))
        if not candidates:
            candidates = [current_best_seq]

        candidate_results = []
        for cand_seq in candidates:
            result = _verify_with_cache(
                backend,
                verifier_cache,
                binder_seq=cand_seq,
                target_seq=target_seq,
                seed=_stable_seed(seed, "esm_final_polish_verify", target_key, cand_seq),
            )
            candidate_results.append(result)
            _write_candidate_artifact(
                artifact_recorder,
                stage="final_polish",
                step=polish_step,
                parent_sequence=current_best_seq,
                result=result,
            )

        round_best = min(candidate_results, key=lambda result: float(result.total_loss))
        final_best_result = current_best_result
        final_best_seq = current_best_seq
        final_best_loss = current_best_loss
        polish_improved = float(round_best.total_loss) < current_best_loss
        if polish_improved:
            final_best_result = round_best
            final_best_seq = str(round_best.binder_seq)
            final_best_loss = float(round_best.total_loss)
            _remember_global_best(round_best, proposal_batch.logits_after)

        losses_history.append(final_best_loss)
        artifact_recorder.write_step(
            stage="final_polish",
            step=polish_step,
            selected_sequence=current_best_seq,
            candidate_count=len(candidate_results),
            round_best_loss=float(round_best.total_loss),
            global_best_loss=final_best_loss,
            best_sequence=final_best_seq,
            polish_sequence=str(round_best.binder_seq),
            polish_improved=polish_improved,
            **_proposal_confidence_summary(proposal_batch),
        )
        artifact_recorder.write_checkpoint(
            stage="final_polish",
            step=polish_step,
            label="round_best",
            result=round_best,
            selected_sequence=current_best_seq,
            parent_sequence=current_best_seq,
            is_global_best=(str(round_best.binder_seq) == final_best_seq),
        )
        logger.info(
            "ESM final polish | selected=%s K=%d round_best=%.4f global_best=%.4f",
            current_best_seq[:20],
            len(candidate_results),
            float(round_best.total_loss),
            final_best_loss,
        )
        return final_best_result, final_best_seq, final_best_seq, final_best_loss

    if strategy == "continuous_stage_a":
        selected_node = stagea_node_by_sequence.get(best_seq) or tree.select()
        current_seq = str(selected_node.sequence)
        proposal_batch = _call_backend_continuous_design(
            backend,
            current_seq=current_seq,
            target_seq=target_seq,
            global_step=0,
            seed=_stable_seed(seed, "esm_continuous_stage_a", target_key, current_seq),
            mutable_mask=mutable_mask,
            proposal_logits=selected_node.logits,
            hotspot_params=hotspot_params,
            track_confidence_override=None,
            soft_steps_override=stagea_soft_steps,
        )
        _write_soft_step_artifacts(
            artifact_recorder,
            stage="continuous_stage_a",
            step=0,
            selected_sequence=current_seq,
            batch=proposal_batch,
        )
        if proposal_batch.logits_after is not None:
            selected_node.logits = proposal_batch.logits_after.detach().cpu().clone()
            proposal_logits_by_sequence[current_seq] = selected_node.logits
        candidates = [
            _clip_sequence_by_mutable_mask(str(seq), mutable_mask, current_seq)
            for seq in _proposal_sequences_from_batch(proposal_batch)
        ]
        candidates = list(dict.fromkeys(seq for seq in candidates if seq))
        if not candidates:
            candidates = [current_seq]

        candidate_results = []
        for cand_seq in candidates:
            result = _verify_with_cache(
                backend,
                verifier_cache,
                binder_seq=cand_seq,
                target_seq=target_seq,
                seed=_stable_seed(seed, "esm_continuous_verify", target_key, cand_seq),
            )
            candidate_results.append(result)
            _write_candidate_artifact(
                artifact_recorder,
                stage="continuous_stage_a",
                step=0,
                parent_sequence=current_seq,
                result=result,
            )

        round_best = min(candidate_results, key=lambda result: float(result.total_loss))
        expandable_results = [
            result for result in candidate_results
            if str(result.binder_seq) != current_seq
        ]
        if expandable_results:
            new_nodes = tree.expand(selected_node, expandable_results)
            for node in new_nodes:
                node.logits = _anchor_proposal_logits(
                    proposal_batch.logits_after,
                    node.sequence,
                    mutable_mask,
                )
                proposal_logits_by_sequence[str(node.sequence)] = node.logits
                artifact_recorder.write_mcts_node(
                    **_mcts_node_artifact(
                        stage="continuous_stage_a",
                        node=node,
                        step_created=0,
                        extra={"source": "continuous_argmax"},
                    )
                )
                artifact_recorder.write_mcts_edge(
                    **_mcts_edge_artifact(
                        stage="continuous_stage_a",
                        parent=selected_node,
                        child=node,
                        step_created=0,
                    )
                )
            tree.backpropagate(selected_node, [float(result.total_loss) for result in expandable_results])
        if float(round_best.total_loss) < best_loss:
            _remember_global_best(round_best, proposal_batch.logits_after)
        losses_history.append(best_loss)
        artifact_recorder.write_step(
            stage="continuous_stage_a",
            step=0,
            selected_node_id=int(selected_node.node_id),
            selected_sequence=current_seq,
            candidate_count=len(candidate_results),
            round_best_loss=float(round_best.total_loss),
            global_best_loss=best_loss,
            best_sequence=best_seq,
            **_proposal_confidence_summary(proposal_batch),
        )
        artifact_recorder.write_checkpoint(
            stage="continuous_stage_a",
            step=0,
            label="round_best",
            result=round_best,
            selected_sequence=current_seq,
            parent_sequence=current_seq,
            is_global_best=(str(round_best.binder_seq) == best_seq),
        )
        artifact_recorder.write_summary(
            best_sequence=best_seq,
            final_sequence=str(round_best.binder_seq),
            best_loss=best_loss,
            n_steps=len(losses_history),
            strategy=strategy,
            artifact_dir=str(artifact_dir) if artifact_dir is not None else None,
            **hotspot_summary,
        )
        logger.info(
            "ESM continuous StageA | selected=%s round_best=%s round_best_loss=%.4f best=%.4f",
            current_seq[:20],
            str(round_best.binder_seq)[:20],
            float(round_best.total_loss),
            best_loss,
        )
        return best_seq, str(round_best.binder_seq), losses_history

    if strategy == "stage_a_only":
        stagea_steps = N_steps
    else:
        min_bootstrap = max(int(mcts_bootstrap_steps), 0)
        max_bootstrap = max(int(mcts_bootstrap_max_steps), min_bootstrap)
        stagea_steps = min(N_steps, max_bootstrap)
        if min_bootstrap > 0:
            stagea_steps = min(N_steps, max(stagea_steps, min_bootstrap))
        if N_steps > 0 and stagea_steps >= N_steps:
            stagea_steps = max(N_steps - 1, 0)

    stagea_best = best_loss
    stagea_stale = 0
    for step in range(stagea_steps):
        selected_node = tree.select()
        current_seq = selected_node.sequence
        proposal_seed = _stable_seed(seed, "esm_propose", target_key, current_seq, step)
        proposal_batch = _call_backend_propose(
            backend,
            current_seq=current_seq,
            target_seq=target_seq,
            global_step=step,
            K_top=K_top,
            B_candidates=B_candidates,
            T_sample=T_sample,
            max_mutations_per_seq=stagea_max_mutations,
            seed=proposal_seed,
            action_group=None,
            mutable_mask=mutable_mask,
            proposal_logits=selected_node.logits,
            hotspot_params=hotspot_params,
            track_confidence_override=False,
            soft_steps_override=stagea_soft_steps,
        )
        _write_soft_step_artifacts(
            artifact_recorder,
            stage="stage_a",
            step=step,
            selected_sequence=current_seq,
            batch=proposal_batch,
        )
        if proposal_batch.logits_after is not None:
            selected_node.logits = proposal_batch.logits_after.detach().cpu().clone()
            proposal_logits_by_sequence[current_seq] = selected_node.logits
        candidates = _proposal_sequences_from_batch(proposal_batch, include_confidence=False)
        candidates = [
            _clip_sequence_by_mutable_mask(str(seq), mutable_mask, current_seq)
            for seq in candidates
        ]
        candidates = [seq for seq in candidates if seq != current_seq]
        candidate_limit = max(
            int(B_candidates),
            0,
        )
        candidates = list(dict.fromkeys(candidates))[:candidate_limit]
        if not candidates:
            logger.warning("ESM StageA step %d: no candidates from backend", step)
            losses_history.append(best_loss)
            continue

        round_results = []
        for cand_seq in candidates:
            result = _verify_with_cache(
                backend,
                verifier_cache,
                binder_seq=cand_seq,
                target_seq=target_seq,
                seed=_stable_seed(seed, "esm_verify", target_key, cand_seq),
            )
            round_results.append(result)
            _write_candidate_artifact(
                artifact_recorder,
                stage="stage_a",
                step=step,
                parent_sequence=current_seq,
                result=result,
            )

        new_nodes = tree.expand(selected_node, round_results)
        for node in new_nodes:
            node.logits = _anchor_proposal_logits(
                proposal_batch.logits_after,
                node.sequence,
                mutable_mask,
            )
            proposal_logits_by_sequence[str(node.sequence)] = node.logits
            stagea_node_by_sequence[str(node.sequence)] = node
        tree.backpropagate(selected_node, [result.total_loss for result in round_results])
        for node in new_nodes:
            artifact_recorder.write_mcts_node(
                **_mcts_node_artifact(
                    stage="stage_a",
                    node=node,
                    step_created=step,
                    extra={"source": "stage_a_expand"},
                )
            )
            artifact_recorder.write_mcts_edge(
                **_mcts_edge_artifact(
                    stage="stage_a",
                    parent=selected_node,
                    child=node,
                    step_created=step,
                )
            )
        tree.prune()

        round_best = min(round_results, key=lambda result: result.total_loss)
        if float(round_best.total_loss) < best_loss:
            _remember_global_best(round_best)
        if best_loss < stagea_best - float(mcts_bootstrap_min_delta):
            stagea_best = best_loss
            stagea_stale = 0
        else:
            stagea_stale += 1
        losses_history.append(best_loss)
        artifact_recorder.write_step(
            stage="stage_a",
            step=step,
            selected_node_id=int(selected_node.node_id),
            selected_sequence=current_seq,
            candidate_count=len(round_results),
            round_best_loss=float(round_best.total_loss),
            global_best_loss=best_loss,
            best_sequence=best_seq,
            **_proposal_confidence_summary(proposal_batch),
        )
        artifact_recorder.write_checkpoint(
            stage="stage_a",
            step=step,
            label="round_best",
            result=round_best,
            selected_sequence=current_seq,
            parent_sequence=current_seq,
            is_global_best=(str(round_best.binder_seq) == best_seq),
        )

        logger.info(
            "ESM StageA step %3d/%d | selected=%s K=%d round_best=%.4f global_best=%.4f",
            step,
            N_steps,
            current_seq[:20],
            len(round_results),
            round_best.total_loss,
            best_loss,
        )
        if (
            strategy == "two_stage"
            and step + 1 >= max(int(mcts_bootstrap_steps), 0)
            and stagea_stale >= max(int(mcts_bootstrap_patience), 1)
        ):
            logger.info("ESM StageA bootstrap stopped by patience at step %d", step)
            break

    if strategy == "stage_a_only" or len(losses_history) >= N_steps:
        best_result, best_seq, final_seq, best_loss = _run_final_polish_if_enabled(
            current_best_result=best_result,
            current_best_seq=best_seq,
            current_best_loss=best_loss,
        )
        artifact_recorder.write_summary(
            best_sequence=best_seq,
            final_sequence=final_seq,
            best_loss=best_loss,
            n_steps=len(losses_history),
            strategy=strategy,
            artifact_dir=str(artifact_dir) if artifact_dir is not None else None,
            **hotspot_summary,
        )
        return best_seq, final_seq, losses_history

    handoff_k = int(mcts_handoff_top_k or num_beams)
    handoff_results = _top_handoff_results(
        verifier_cache,
        top_k=handoff_k,
        min_distance=1,
    )
    strict_tree = MCTSTree(
        C=mcts_C,
        max_nodes=mcts_max_nodes,
        max_children_per_node=max(int(B_candidates), 1),
        expand_actions_per_step=max(int(mcts_expand_actions_per_step), 1),
        use_diversity_bonus=False,
        use_structural_grouping=False,
    )
    for idx, result in enumerate(handoff_results):
        strict_tree.store_verifier_result(result)
        handoff_logits = proposal_logits_by_sequence.get(str(result.binder_seq))
        root_node = strict_tree.add_root_node(
            sequence=result.binder_seq,
            logits=_anchor_proposal_logits(handoff_logits, result.binder_seq, mutable_mask),
            L_real=result.total_loss,
            coordinate=result.coordinate,
            loss_vector=result.loss_dict,
            prior=1.0 / max(len(handoff_results), 1),
            status_flags={"esm_stage": "handoff", "handoff_rank": idx + 1},
        )
        source_node = stagea_node_by_sequence.get(str(result.binder_seq))
        artifact_recorder.write_mcts_node(
            **_mcts_node_artifact(
                stage="stage_b",
                node=root_node,
                step_created=len(losses_history),
                extra={
                    "source": "handoff_root",
                    "handoff_rank": idx + 1,
                    "source_stage": "stage_a",
                    "source_node_id": None if source_node is None else int(source_node.node_id),
                    "source_sequence_hash": _sequence_hash(str(result.binder_seq)),
                },
            )
        )
        artifact_recorder.write_mcts_handoff(
            source_stage="stage_a",
            target_stage="stage_b",
            handoff_rank=idx + 1,
            source_node_id=None if source_node is None else int(source_node.node_id),
            target_node_id=int(root_node.node_id),
            source_node_uid=None if source_node is None else f"stage_a:{int(source_node.node_id)}",
            target_node_uid=f"stage_b:{int(root_node.node_id)}",
            sequence=str(result.binder_seq),
            sequence_hash=_sequence_hash(str(result.binder_seq)),
            total_loss=float(result.total_loss),
            prior=1.0 / max(len(handoff_results), 1),
        )

    preview_loss_cache: Dict[str, Dict[str, float]] = {}
    for step in range(len(losses_history), N_steps):
        selection = strict_tree.select_decision()
        selected_node = selection.node
        current_seq = selected_node.sequence

        if selected_node.loss_vector is None:
            selected_node.loss_vector = {}
        _merge_preview_losses_for_routing(
            backend,
            preview_loss_cache,
            selected_node.loss_vector,
            binder_seq=current_seq,
            target_seq=target_seq,
            seed=_stable_seed(seed, "esm_stageb_preview", target_key, current_seq),
            hotspot_params=hotspot_params,
        )
        routing = select_esmfold2_action_group(selected_node.loss_vector, temperature=1.0)
        action_group = str(routing["action_group"])
        group_priors = {
            str(group): float(prior)
            for group, prior in dict(routing.get("group_priors", {})).items()
        }
        action_group, group_priors = _apply_stageb_group_prior_mode(
            mcts_prior_mode,
            action_group,
            group_priors,
        )
        group_prior = float(
            group_priors.get(
                action_group,
                max(group_priors.values()) if group_priors else 1.0,
            )
        )

        expand_budget = min(
            max(int(mcts_expand_actions_per_step), 1),
            strict_tree.remaining_expansion_slots(selected_node),
        )
        if expand_budget <= 0:
            losses_history.append(best_loss)
            continue

        proposal_pool_budget = max(B_candidates, expand_budget)
        group_budgets = _allocate_group_budgets(group_priors, proposal_pool_budget)
        proposal_pool: List[ESMFold2MutationProposal] = []
        selected_update_logits: Optional[torch.Tensor] = None
        for group_name, group_budget in group_budgets.items():
            proposal_batch = _call_backend_propose(
                backend,
                current_seq=current_seq,
                target_seq=target_seq,
                global_step=step,
                K_top=K_top,
                B_candidates=group_budget,
                T_sample=T_sample,
                max_mutations_per_seq=stageb_max_mutations,
                seed=_stable_seed(seed, "esm_stageb_propose", target_key, current_seq, step, group_name),
                action_group=group_name,
                mutable_mask=mutable_mask,
                proposal_logits=selected_node.logits,
                hotspot_params=hotspot_params,
                track_confidence_override=False,
                soft_steps_override=stageb_soft_steps,
            )
            _write_soft_step_artifacts(
                artifact_recorder,
                stage="stage_b",
                step=step,
                selected_sequence=current_seq,
                batch=proposal_batch,
                action_group=group_name,
            )
            if proposal_batch.logits_after is not None and (
                selected_update_logits is None or group_name == action_group
            ):
                selected_update_logits = proposal_batch.logits_after.detach().cpu().clone()
            candidates = _proposal_sequences_from_batch(proposal_batch, include_confidence=False)
            candidates = [
                _clip_sequence_by_mutable_mask(str(seq), mutable_mask, current_seq)
                for seq in candidates
            ]
            proposal_pool.extend(
                _make_stageb_proposals(
                    current_seq=current_seq,
                    candidates=candidates,
                    action_group=group_name,
                    group_prior=group_priors.get(group_name, group_prior),
                    proposal_logits=proposal_batch.logits_after,
                )
            )
        if selected_update_logits is not None:
            selected_node.logits = selected_update_logits
            proposal_logits_by_sequence[current_seq] = selected_node.logits

        if not proposal_pool:
            logger.warning("ESM StageB step %d: no candidates from backend", step)
            losses_history.append(best_loss)
            continue

        registered_actions = strict_tree.register_action_proposals(selected_node, proposal_pool)
        selected_actions = strict_tree.select_unexpanded_actions(
            selected_node,
            max_actions=expand_budget,
            candidates=registered_actions,
        )
        if not selected_actions:
            losses_history.append(best_loss)
            continue
        action_by_sequence = {
            str(action.sequence): action
            for action in selected_actions
        }

        expand_proposals: List[ESMFold2MutationProposal] = []
        expand_results: List[Any] = []
        reward_by_sequence: Dict[str, float] = {}
        for action in selected_actions:
            if strict_tree.has_sequence(action.sequence):
                existing = strict_tree.get_node_by_sequence(action.sequence)
                if existing is not None:
                    reward_by_sequence[action.sequence] = existing.tree_reward
                    action.status = "duplicate"
                continue
            cached = strict_tree.get_cached_verifier_result(action.sequence)
            if cached is None:
                cached = _verify_with_cache(
                    backend,
                    verifier_cache,
                    binder_seq=action.sequence,
                    target_seq=target_seq,
                    seed=_stable_seed(seed, "esm_stageb_verify", target_key, action.sequence),
                )
                strict_tree.store_verifier_result(cached)
            _write_candidate_artifact(
                artifact_recorder,
                stage="stage_b",
                step=step,
                parent_sequence=current_seq,
                result=cached,
                action_group=str(action.action_group),
            )
            expand_proposals.append(action.proposal)
            expand_results.append(cached)

        new_nodes = strict_tree.expand(selected_node, expand_proposals, expand_results)
        proposal_by_sequence = {
            str(proposal.sequence): proposal
            for proposal in expand_proposals
        }
        for node in new_nodes:
            proposal = proposal_by_sequence.get(str(node.sequence))
            node.logits = _anchor_proposal_logits(
                getattr(proposal, "proposal_logits", None),
                node.sequence,
                mutable_mask,
            )
            proposal_logits_by_sequence[str(node.sequence)] = node.logits
            reward_by_sequence[node.sequence] = node.tree_reward
            action = action_by_sequence.get(str(node.sequence))
            artifact_recorder.write_mcts_node(
                **_mcts_node_artifact(
                    stage="stage_b",
                    node=node,
                    step_created=step,
                    extra={
                        "source": "stage_b_expand",
                        "action_id": None if action is None else int(action.action_id),
                        "action_group": None if action is None else str(action.action_group),
                    },
                )
            )
            artifact_recorder.write_mcts_edge(
                **_mcts_edge_artifact(
                    stage="stage_b",
                    parent=selected_node,
                    child=node,
                    step_created=step,
                    extra={
                        "action_id": None if action is None else int(action.action_id),
                        "action_group": None if action is None else str(action.action_group),
                    },
                )
            )

        for result in expand_results:
            reward_by_sequence.setdefault(
                result.binder_seq,
                strict_tree.result_reward(result),
            )

        for action in selected_actions:
            if action.sequence in reward_by_sequence:
                strict_tree.backpropagate_action(
                    selection,
                    action,
                    reward_by_sequence[action.sequence],
                )
            artifact_recorder.write_mcts_action(
                **_mcts_action_artifact(
                    stage="stage_b",
                    step=step,
                    action=action,
                    selected=True,
                    extra={
                        "selection_decision_type": selection.decision_type,
                        "selection_best_action_score": selection.best_action_score,
                        "selection_frontier_score": selection.frontier_score,
                        "parent_sequence_hash": _sequence_hash(current_seq),
                    },
                )
            )

        strict_tree.prune()
        round_results = expand_results
        if round_results:
            round_best = min(round_results, key=lambda result: float(result.total_loss))
            if float(round_best.total_loss) < best_loss:
                _remember_global_best(round_best)
        losses_history.append(best_loss)
        artifact_recorder.write_step(
            stage="stage_b",
            step=step,
            selected_node_id=int(selected_node.node_id),
            selected_action_ids=[int(action.action_id) for action in selected_actions],
            selected_sequence=current_seq,
            candidate_count=len(round_results),
            round_best_loss=float(round_best.total_loss) if round_results else None,
            global_best_loss=best_loss,
            best_sequence=best_seq,
            primary_action_group=action_group,
            group_prior=group_prior,
            group_scores=dict(routing.get("group_scores", {})),
            group_priors=group_priors,
            group_budgets=group_budgets,
            selected_group_counts=_count_items([str(action.action_group) for action in selected_actions]),
            selected_action_count=len(selected_actions),
            loss_vector_keys=sorted(str(key) for key in selected_node.loss_vector.keys()),
        )
        if round_results:
            artifact_recorder.write_checkpoint(
                stage="stage_b",
                step=step,
                label="round_best",
                result=round_best,
                selected_sequence=current_seq,
                parent_sequence=current_seq,
                is_global_best=(str(round_best.binder_seq) == best_seq),
            )

        logger.info(
            "ESM StageB step %3d/%d | selected=%s primary_group=%s Pg=%.3f scores=%s priors=%s budgets=%s selected_groups=%s keys=%s actions=%d global_best=%.4f",
            step,
            N_steps,
            current_seq[:20],
            action_group,
            group_prior,
            _format_float_map(dict(routing.get("group_scores", {}))),
            _format_float_map(group_priors),
            _format_count_map(group_budgets),
            _format_count_map(_count_items([str(action.action_group) for action in selected_actions])),
            ",".join(sorted(str(key) for key in selected_node.loss_vector.keys())),
            len(selected_actions),
            best_loss,
        )

    best_result, best_seq, final_seq, best_loss = _run_final_polish_if_enabled(
        current_best_result=best_result,
        current_best_seq=best_seq,
        current_best_loss=best_loss,
    )
    artifact_recorder.write_summary(
        best_sequence=best_seq,
        final_sequence=final_seq,
        best_loss=best_loss,
        n_steps=len(losses_history),
        strategy=strategy,
        artifact_dir=str(artifact_dir) if artifact_dir is not None else None,
        **hotspot_summary,
    )
    return best_seq, final_seq, losses_history
