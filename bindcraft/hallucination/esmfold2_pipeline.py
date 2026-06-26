"""
MCTS loop for the optional ESMFold2 backend.

This file intentionally avoids importing the Protenix pipeline.  The first
ESMFold2 integration should be an explicit backend path, while the default
Protenix V1/V2 execution remains unchanged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from bindcraft.hallucination.esmfold2_backend import (
    ESM_AA_ORDER,
    select_esmfold2_action_group,
)
from bindcraft.hallucination.mcts_legacy import LegacyMCTSTree
from bindcraft.hallucination.mcts_controller import MCTSTree


logger = logging.getLogger(__name__)
VALID_ESM_BINDER_AAS = ESM_AA_ORDER.replace("C", "")


@dataclass(frozen=True)
class ESMFold2MutationProposal:
    sequence: str
    edits: List[Tuple[int, str, str]]
    prior: float
    benefit: float = 0.0
    action_group: str = "unrouted"
    group_prior: float = 1.0


class _NoopArtifactRecorder:
    def write_step(self, **event: Any) -> None:
        return None

    def write_candidate(self, **event: Any) -> None:
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
        self.summary_path = self.artifact_dir / "run_summary.json"
        self.trajectory_path.write_text("", encoding="utf-8")
        self.candidates_path.write_text("", encoding="utf-8")

    def write_step(self, **event: Any) -> None:
        self._append_jsonl(self.trajectory_path, event)

    def write_candidate(self, **event: Any) -> None:
        self._append_jsonl(self.candidates_path, event)

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
) -> Dict[str, float]:
    """Merge optional lightweight ESMFold2 preview losses into a node vector."""

    if not hasattr(backend, "preview_loss_dict"):
        return {}
    if all(key in loss_vector for key in ("inter_contact_loss", "intra_contact_loss", "glob_loss")):
        return {}
    if binder_seq not in preview_cache:
        try:
            preview = backend.preview_loss_dict(
                binder_seq=binder_seq,
                target_seq=target_seq,
                seed=seed,
            )
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
    artifact_dir: Optional[Union[str, os.PathLike[str]]] = None,
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
    Stage B.
    """

    if backend is None:
        raise ValueError("backend is required for ESMFold2 MCTS")
    L_binder = int(L_binder)
    N_steps = max(int(N_steps), 0)
    K_top = max(int(K_top), 1)
    B_candidates = max(int(B_candidates), 1)
    max_mutations_per_seq = max(int(max_mutations_per_seq), 1)
    strategy = str(mcts_strategy or "legacy").strip().lower()
    if strategy == "strict":
        strategy = "two_stage"
    if strategy == "legacy":
        strategy = "stage_a_only"
    if strategy not in {"stage_a_only", "two_stage"}:
        raise ValueError("ESMFold2 backend supports mcts_strategy stage_a_only or two_stage")
    target_key = _normalize_target_key(target_seq)
    mutable_mask = _normalize_mutable_mask(mutable_mask, L_binder)
    artifact_recorder = _make_artifact_recorder(artifact_dir)

    tree = LegacyMCTSTree(C=mcts_C, max_nodes=mcts_max_nodes)
    verifier_cache: Dict[str, Any] = {}

    logger.info("=" * 70)
    logger.info("ESMFold2 MCTS closed loop")
    logger.info("Target length key=%s | binder length=%d | steps=%d", target_key, L_binder, N_steps)
    logger.info("K_top=%d B=%d T_sample=%.3f max_mut=%d", K_top, B_candidates, T_sample, max_mutations_per_seq)
    logger.info("strategy=%s bootstrap_steps=%d max=%d", strategy, mcts_bootstrap_steps, mcts_bootstrap_max_steps)
    logger.info("=" * 70)

    phase0_seqs = _make_phase0_seed_sequences(
        init_binder_seq,
        num_beams=num_beams,
        L_binder=L_binder,
        seed=seed,
        mutable_mask=mutable_mask,
    )
    phase0_results = []
    for idx, binder_seq in enumerate(phase0_seqs):
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
            len(phase0_seqs),
            binder_seq[:20],
            result.total_loss,
        )

    phase0_results.sort(key=lambda result: result.total_loss)
    for result in phase0_results[: max(int(num_beams), 1)]:
        tree.add_root_node(
            sequence=result.binder_seq,
            logits=_placeholder_logits(result.binder_seq),
            L_real=result.total_loss,
            coordinate=result.coordinate,
        )

    best_result = phase0_results[0]
    best_seq = str(best_result.binder_seq)
    best_loss = float(best_result.total_loss)
    losses_history: List[float] = []
    artifact_recorder.write_step(
        stage="phase0",
        step=None,
        candidate_count=len(phase0_results),
        round_best_loss=float(best_result.total_loss),
        global_best_loss=best_loss,
        best_sequence=best_seq,
        selected_sequence=None,
    )
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
        candidates = backend.propose(
            current_seq=current_seq,
            target_seq=target_seq,
            global_step=step,
            K_top=K_top,
            B_candidates=B_candidates,
            T_sample=T_sample,
            max_mutations_per_seq=max_mutations_per_seq,
            seed=proposal_seed,
            action_group=None,
            mutable_mask=mutable_mask,
        )
        candidates = [
            _clip_sequence_by_mutable_mask(str(seq), mutable_mask, current_seq)
            for seq in candidates
        ]
        candidates = [seq for seq in candidates if seq != current_seq]
        candidates = list(dict.fromkeys(candidates))[:B_candidates]
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
            node.logits = _placeholder_logits(node.sequence)
        tree.backpropagate(selected_node, [result.total_loss for result in round_results])
        tree.prune()

        round_best = min(round_results, key=lambda result: result.total_loss)
        if float(round_best.total_loss) < best_loss:
            best_result = round_best
            best_seq = str(round_best.binder_seq)
            best_loss = float(round_best.total_loss)
        if best_loss < stagea_best - float(mcts_bootstrap_min_delta):
            stagea_best = best_loss
            stagea_stale = 0
        else:
            stagea_stale += 1
        losses_history.append(best_loss)
        artifact_recorder.write_step(
            stage="stage_a",
            step=step,
            selected_sequence=current_seq,
            candidate_count=len(round_results),
            round_best_loss=float(round_best.total_loss),
            global_best_loss=best_loss,
            best_sequence=best_seq,
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
        artifact_recorder.write_summary(
            best_sequence=best_seq,
            final_sequence=best_seq,
            best_loss=best_loss,
            n_steps=len(losses_history),
            strategy=strategy,
            artifact_dir=str(artifact_dir) if artifact_dir is not None else None,
        )
        return best_seq, best_seq, losses_history

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
        strict_tree.add_root_node(
            sequence=result.binder_seq,
            logits=_placeholder_logits(result.binder_seq),
            L_real=result.total_loss,
            coordinate=result.coordinate,
            loss_vector=result.loss_dict,
            prior=1.0 / max(len(handoff_results), 1),
            status_flags={"esm_stage": "handoff", "handoff_rank": idx + 1},
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
        for group_name, group_budget in group_budgets.items():
            candidates = backend.propose(
                current_seq=current_seq,
                target_seq=target_seq,
                global_step=step,
                K_top=K_top,
                B_candidates=group_budget,
                T_sample=T_sample,
                max_mutations_per_seq=max_mutations_per_seq,
                seed=_stable_seed(seed, "esm_stageb_propose", target_key, current_seq, step, group_name),
                action_group=group_name,
                mutable_mask=mutable_mask,
            )
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
                )
            )

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
        for node in new_nodes:
            node.logits = _placeholder_logits(node.sequence)
            reward_by_sequence[node.sequence] = node.tree_reward

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

        strict_tree.prune()
        round_results = expand_results
        if round_results:
            round_best = min(round_results, key=lambda result: float(result.total_loss))
            if float(round_best.total_loss) < best_loss:
                best_result = round_best
                best_seq = str(round_best.binder_seq)
                best_loss = float(round_best.total_loss)
        losses_history.append(best_loss)
        artifact_recorder.write_step(
            stage="stage_b",
            step=step,
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

    artifact_recorder.write_summary(
        best_sequence=best_seq,
        final_sequence=best_seq,
        best_loss=best_loss,
        n_steps=len(losses_history),
        strategy=strategy,
        artifact_dir=str(artifact_dir) if artifact_dir is not None else None,
    )
    return best_seq, best_seq, losses_history
