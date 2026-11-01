"""Dual-target adaptation of the native action-level ESMFold2 Stage B.

The existing single-target pipeline and its acceleration adapters are reused,
never patched. Both soft states, preview contexts and quality floors stay
separate; only the discrete sequence and the action tree are shared.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import logging
import math
from pathlib import Path
import random
from types import SimpleNamespace

import torch

from .esmfold2_backend import minimally_align_binder_logits_to_sequence
from .esmfold2_pipeline import (
    _ESMFold2StageBMCTSTree, _make_stageb_proposals,
    _rerank_stageb_candidates_with_preview, _select_stageb_actions_by_group_quota,
)
from .esmfold2_dual_target import _plain, _write_json, validate_sequence, validate_target_context
from .esmfold2_dual_target_constraints import design_mask, validate_design, check_sequence

logger = logging.getLogger(__name__)
SIDES = ("human", "mouse")
FLOOR = "stageb_pair_chain_quality_floor"
REQUIRED = ("pair_chain_iptm_loss", "binder_confidence_loss", "binder_rg_excess_loss")


class DualTargetStageBTree(_ESMFold2StageBMCTSTree):
    """Keep the native per-target reward, protect the weaker target."""

    def _compute_base_node_reward(self, total_loss, loss_vector, base_reward=None, observe=False):
        rewards, statuses = {}, {}
        for side in SIDES:
            losses = {key.split("/", 1)[1]: value for key, value in loss_vector.items()
                      if key.startswith(side + "/")}
            if not losses:
                raise ValueError(f"Stage-B reward requires separate {side} verifier metrics")
            rewards[side], statuses[side] = super()._compute_base_node_reward(
                total_loss, losses, base_reward=base_reward, observe=observe)
        weaker = min(SIDES, key=lambda side: rewards[side])
        status = {**statuses[weaker], "reward_mode": "esmfold2_stageb_dual_worst",
                  "target_rewards": rewards, "target_status": statuses,
                  "pair_chain_floor_violated": any(
                      item["pair_chain_floor_violated"] for item in statuses.values())}
        return rewards[weaker], status


def side_losses(record, side):
    raw = record["targets"][side]["loss_dict"]
    losses = {str(key): float(value) for key, value in raw.items()
              if isinstance(value, (float, int)) and math.isfinite(value)}
    missing = [key for key in REQUIRED if key not in losses]
    if missing:
        raise ValueError(f"Stage B requires {side} full-verifier metrics: {', '.join(missing)}")
    return losses


def pair_iptm(record, side):
    return 1.0 - min(max(side_losses(record, side)["pair_chain_iptm_loss"], 0.0), 1.0)


def joint_result(record, floors):
    """Create a strict-tree result without mutating cached per-target records."""
    per_side = {side: {**side_losses(record, side), FLOOR: float(floors[side])}
                for side in SIDES}
    common = set.intersection(*(set(losses) for losses in per_side.values()))
    vector = {key: max(per_side[side][key] for side in SIDES) for key in common}
    vector.update({f"{side}/{key}": value for side, losses in per_side.items()
                   for key, value in losses.items()})
    return SimpleNamespace(binder_seq=record["sequence"], target_seq="dual_target",
                           total_loss=record["worst_loss"], loss_dict=vector,
                           summary_confidence={}, coordinate=None)


def guarded_rank(record, tree, floors):
    quality = all(pair_iptm(record, side) >= floors[side] for side in SIDES)
    reward = tree.result_reward(joint_result(record, floors))
    tail = (reward, min(pair_iptm(record, side) for side in SIDES),
            -record["worst_loss"], -record["mean_loss"])
    hotspot = []
    for side in SIDES:
        target = record["targets"][side]
        coverage = target["summary_confidence"].get("hotspot_heavy_atom_coverage",
                                                    target["loss_dict"].get("hotspot_heavy_atom_coverage"))
        if coverage is not None:
            coverage = float(coverage)
            if not math.isfinite(coverage):
                raise ValueError("Nonfinite hotspot coverage in Stage B")
            distance = target["summary_confidence"].get("hotspot_mean_min_heavy_atom_distance")
            distance = float(distance) if distance is not None else float("inf")
            hotspot.append((min(max(coverage, 0.0), 1.0), distance))
    if not hotspot:
        # Neutral hotspot slots keep ranks comparable even if optional hotspot
        # diagnostics are absent from one result. Ordering still matches the
        # native no-hotspot quality -> reward -> pair-iPTM -> loss order.
        return (int(quality), 1, 1.0, *tail[:2], 0.0, *tail[2:])
    coverage = min(item[0] for item in hotspot)
    distance = max(item[1] for item in hotspot)
    fully_covered = int(coverage >= 1.0 - 1e-8)
    if fully_covered:
        return (int(quality), fully_covered, coverage, *tail[:2], -distance, *tail[2:])
    return (int(quality), fully_covered, coverage, -distance, *tail)


def select_expansions(actions, records, tree, floors, limit):
    pairs = list(zip(actions, records))
    eligible = [pair for pair in pairs if all(
        pair_iptm(pair[1], side) >= floors[side] for side in SIDES)]
    eligible.sort(key=lambda pair: guarded_rank(pair[1], tree, floors), reverse=True)
    selected = eligible[:limit]
    # Native Stage B allows one quality exploration when guarded slots remain.
    if len(selected) < limit:
        selected_ids = {pair[0].action_id for pair in selected}
        remaining = [pair for pair in pairs if pair[0].action_id not in selected_ids]
        remaining.sort(key=lambda pair: (
            min(pair_iptm(pair[1], side) - floors[side] for side in SIDES),
            tree.result_reward(joint_result(pair[1], floors))), reverse=True)
        selected.extend(remaining[:1])
    return selected


def stageb_seed(seed, label, step, sequence):
    return int(hashlib.sha256(f"{seed}:{label}:{step}:{sequence}".encode()).hexdigest()[:8], 16)


def joint_preview(backend, targets, parent, candidates, config, step):
    if not config.stageb_discrete_rerank:
        return {}, "disabled"
    scores = {}
    for side in SIDES:
        # Local caches are scoped to side + this parent/step's paired LM seed.
        # A same-length other target can never reuse a preview from this side.
        scores[side] = _rerank_stageb_candidates_with_preview(
            backend, {}, current_seq=parent, candidates=candidates,
            target_seq=targets[side]["sequence"], action_group="unrouted",
            seed=stageb_seed(config.seed, "preview:" + side, step, parent),
            hotspot_params=targets[side].get("hotspot_params"))
    if not candidates or any(set(scores[side]) != set(candidates) for side in SIDES):
        # Never mix one-target preview, partial preview and gradient scores.
        return {}, "gradient_fallback_whole_pool"
    combined = {}
    for sequence in candidates:
        parent_loss = max(scores[side][sequence]["parent_loss"] for side in SIDES)
        loss = max(scores[side][sequence]["loss"] for side in SIDES)
        combined[sequence] = {"parent_loss": parent_loss, "loss": loss,
                              "benefit": parent_loss - loss,
                              "target_scores": {side: scores[side][sequence] for side in SIDES}}
    return combined, "dual_worst_preview"


def save_handoff(out, roots, paired_states, manifest, completed_steps):
    payload = {"schema_version": 1, "stage": "stage_a_complete",
               "completed_stagea_steps": completed_steps,
               "targets": manifest["targets"], "search": manifest["search"],
               "runtime": manifest["runtime"], "source_sha256": manifest["source_sha256"],
               "roots": [{"record": record, "states": {
                   side: paired_states[record["sequence"]][side].detach().cpu().clone()
                   for side in SIDES}} for record in roots]}
    path = Path(out) / "stagea_handoff.pt"
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    _write_json(Path(out) / "stagea_handoff.json", {
        "schema_version": 1, "completed_stagea_steps": completed_steps,
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "state_origin": "paired_soft_logits", "roots": [record["sequence"] for record in roots]})


def load_stagea_source(directory, config, targets, runtime):
    """Read completed Stage A; legacy results explicitly use sequence handoff.

    All roots are reverified in the new run. Old scores/contexts are not loaded
    into its verifier cache, and legacy results are never called exact resume.
    """
    import json
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "completed":
        raise ValueError("Stage-A source must have a completed manifest")
    source_runtime = manifest.get("runtime") or {}
    def contexts(value, context_runtime):
        if not isinstance(value, dict) or set(value) != set(SIDES):
            raise ValueError("Stage-A source requires human/mouse target contexts")
        checked = {side: validate_target_context(value[side], context_runtime, side=side)
                   for side in SIDES}
        return {side: {**checked[side], "hotspot_params": checked[side].get("hotspot_params")}
                for side in SIDES}
    if contexts(manifest.get("targets"), source_runtime) != contexts(targets, runtime):
        raise ValueError("Stage-A source target/hotspot contexts do not match")
    for key in ("seed", "verifier_seed"):
        if manifest.get("search", {}).get(key) != getattr(config, key):
            raise ValueError(f"Stage-A source {key} does not match")
    if manifest.get("search", {}).get("design_mode", "miniprotein") != config.design_mode:
        raise ValueError("Stage-A source design mode does not match")
    mutable_mask = design_mask(config, runtime)
    if mutable_mask is not None:
        if list(source_runtime.get("binder_mutable_mask") or []) != list(mutable_mask):
            raise ValueError("Stage-A source mutation mask differs")
    if runtime is not None:
        for key in ("experimental_model_name", "verifier_model_name", "verifier_loss_mode",
                    "verifier_num_loops", "verifier_num_sampling_steps", "target_chain_roles"):
            if _plain(source_runtime.get(key)) != _plain(getattr(runtime, key)):
                raise ValueError(f"Stage-A source verifier/proposal contract differs: {key}")
        # Only the newly introduced opt-in fields receive compatibility defaults.
        # Missing older runtime fields remain mismatches, as before.
        new_defaults = {"target_assembly": "monomer", "target_interface_mode": "all",
                        "target_interface_contact_cutoff": 5.0,
                        "target_interface_min_contacts": 5}
        for key, default in new_defaults.items():
            if source_runtime.get(key, default) != getattr(runtime, key, default):
                raise ValueError(f"Stage-A source target interface contract differs: {key}")
    checkpoint = directory / "stagea_handoff.pt"
    states = {}
    if checkpoint.exists():
        receipt = json.loads((directory / "stagea_handoff.json").read_text())
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if receipt.get("checkpoint_sha256") != digest:
            raise ValueError("Stage-A handoff checkpoint hash mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (payload.get("schema_version") != 1 or payload.get("stage") != "stage_a_complete"
                or contexts(payload.get("targets"), payload.get("runtime")) != contexts(targets, runtime)
                or _plain(payload.get("runtime")) != manifest.get("runtime")
                or payload.get("search") != manifest.get("search")):
            raise ValueError("Stage-A handoff checkpoint contract mismatch")
        if runtime is not None:
            ignored = {"optimization_report_path", "optimization_root", "optimization_timing"}
            current = {key: value for key, value in asdict(runtime).items() if key not in ignored}
            previous = {key: value for key, value in source_runtime.items() if key not in ignored}
            for key, default in new_defaults.items():
                previous.setdefault(key, default)
            if _plain(current) != _plain(previous):
                raise ValueError("Paired soft-state handoff requires the same runtime configuration")
        records = [item["record"] for item in payload["roots"]]
        for item in payload["roots"]:
            sequence = validate_sequence(item["record"]["sequence"], binder=True)
            pair = item["states"]
            if set(pair) != set(SIDES):
                raise ValueError("Handoff requires both target soft states")
            for side, tensor in pair.items():
                if (not isinstance(tensor, torch.Tensor) or tensor.shape != (len(sequence), 20)
                        or torch.isnan(tensor).any() or torch.isposinf(tensor).any()
                        or not torch.isfinite(tensor).any(dim=-1).all()):
                    raise ValueError(f"Invalid {side} handoff logits")
            states[sequence] = pair
        completed = int(payload["completed_stagea_steps"])
        origin = "paired_soft_logits"
    else:
        if manifest.get("mode") != "dual_target_4plus4":
            raise ValueError("Legacy sequence handoff requires a completed dual_target_4plus4 run")
        records = json.loads((directory / "ranked.json").read_text())
        records = sorted((item for item in records if item.get("valid")),
                         key=lambda item: (item["worst_loss"], item["mean_loss"], item["sequence"]))
        completed = int(manifest["completed_outer_steps"])
        origin = "sequence_only_reinitialized"
    records = records[:config.handoff_top_k]
    if not records or completed <= 0:
        raise ValueError("Stage-A source contains no valid handoff roots")
    for record in records:
        validate_sequence(record["sequence"], binder=True)
        if mutable_mask is None:
            if "C" in record["sequence"]:
                raise ValueError("Stage-A handoff must not change a Cys-bearing sequence")
        else:
            source_seeds = manifest.get("input_seeds") or manifest["seeds"]
            validate_design(config, runtime, source_seeds)
            check_sequence(record["sequence"], next(iter(source_seeds.values())), mutable_mask)
    return {"directory": str(directory), "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "state_origin": origin, "completed_stagea_steps": completed,
            "roots": records, "states": states}


def run_stage_b(*, backend, targets, roots, paired_states, verify, config, out,
                start_step, optimizer_offset, stageb_steps, write_step, verify_many=None):
    mutable_mask = design_mask(config, getattr(backend, "config", None))
    mask_kwargs = {"mutable_mask": mutable_mask} if mutable_mask is not None else {}
    tree = DualTargetStageBTree(
        C=config.exploration, max_nodes=config.stageb_max_nodes,
        max_children_per_node=config.stageb_verifier_candidates,
        expand_actions_per_step=config.tree_children, use_pareto_pruning=False,
        use_diversity_bonus=False, use_structural_grouping=False,
        hotspot_distance_weight=config.stageb_hotspot_distance_weight,
        hotspot_coverage_weight=config.stageb_hotspot_coverage_weight,
        hotspot_miss_penalty=config.stageb_hotspot_miss_penalty,
        min_pair_chain_iptm=config.stageb_min_pair_chain_iptm)
    states, branch_floors, evaluated = {}, {}, {}
    final_floors = {side: max(config.stageb_min_pair_chain_iptm,
                             max(pair_iptm(record, side) for record in roots)
                             - config.stageb_handoff_quality_tolerance) for side in SIDES}
    for record in roots:
        sequence = record["sequence"]
        floors = {side: max(config.stageb_min_pair_chain_iptm,
                            pair_iptm(record, side) - config.stageb_handoff_quality_tolerance)
                  for side in SIDES}
        pair = {side: minimally_align_binder_logits_to_sequence(
            paired_states[sequence][side].clone(), sequence, **mask_kwargs).detach().cpu() for side in SIDES}
        result = joint_result(record, floors)
        tree.store_verifier_result(result)
        root = tree.add_root_node(sequence, pair["human"], record["worst_loss"],
                                  loss_vector=result.loss_dict, prior=1.0 / len(roots),
                                  status_flags={"esm_stage": "dual_stage_b_handoff"})
        # The generic tree carries one tensor; this map is the authoritative
        # paired state, including when only one target proposed the child.
        states[root.node_id], branch_floors[root.node_id] = pair, floors
        evaluated[sequence] = record
    _write_json(Path(out) / "stageb_handoff.json", {
        "roots": [{"sequence": root.sequence, "quality_floors": branch_floors[root.node_id]}
                  for root in tree.roots], "final_quality_floors": final_floors,
        "reward": "min(native_human_stageb_reward, native_mouse_stageb_reward)"})
    budgets = {"human": (config.stageb_rerank_pool_size + 1) // 2,
               "mouse": config.stageb_rerank_pool_size // 2}
    for local_step in range(stageb_steps):
        step = start_step + local_step
        optimizer_step = optimizer_offset + local_step
        selection = tree.select_decision()
        parent = selection.node
        floors = branch_floors[parent.node_id]
        limit = min(config.tree_children, tree.remaining_expansion_slots(parent))
        event = {"stage": "stage_b", "step": step, "optimizer_step": optimizer_step,
                 "parent_id": parent.node_id, "parent_sequence": parent.sequence,
                 "selection": selection.decision_type, "quality_floors": floors,
                 "proposals": {}, "candidates": [], "admitted": []}
        if limit <= 0:
            event["skip_reason"] = "no_tree_slots"
            write_step(event)
            continue
        batches, sequences, origins = {}, {}, {}
        for side in SIDES:
            batch = backend.propose(
                current_seq=parent.sequence, target_seq=targets[side]["sequence"],
                global_step=optimizer_step, K_top=10, B_candidates=budgets[side],
                T_sample=config.sampling_temperature, max_mutations_per_seq=1,
                action_group="unrouted", include_argmax_candidate=False,
                seed=stageb_seed(config.seed, "proposal:" + side, optimizer_step, parent.sequence),
                proposal_logits=states[parent.node_id][side].clone(),
                soft_steps_override=config.stageb_soft_steps,
                track_confidence_override=False, hotspot_params=targets[side].get("hotspot_params"), **mask_kwargs)
            if batch.logits_before is None or batch.logits_after is None:
                raise ValueError("Stage B requires before/after proposal logits")
            for tensor in (batch.logits_before, batch.logits_after):
                if (tensor.shape not in ((len(parent.sequence), 20), (1, len(parent.sequence), 20))
                        or torch.isnan(tensor).any() or torch.isposinf(tensor).any()
                        or not torch.isfinite(tensor).any(dim=-1).all()):
                    raise ValueError("Invalid Stage-B soft logits")
            # Do not retain a CUDA/autograd graph for each local proposal pool.
            batches[side] = (batch.logits_before.detach().cpu(), batch.logits_after.detach().cpu())
            candidates = []
            for sequence in dict.fromkeys(batch.sequences):
                validate_sequence(sequence, binder=True)
                if len(sequence) != len(parent.sequence):
                    raise ValueError("Stage B cannot change binder length")
                if mutable_mask is None:
                    if "C" in sequence:
                        raise ValueError("Stage-B proposal introduced suppressed Cys")
                else:
                    check_sequence(sequence, parent.sequence, mutable_mask)
                if sum(a != b for a, b in zip(sequence, parent.sequence)) == 1:
                    candidates.append(sequence)
                    origins.setdefault(sequence, []).append(side)
                if len(candidates) == budgets[side]:
                    break
            sequences[side] = candidates
            event["proposals"][side] = candidates
        pool = list(origins)
        preview, preview_mode = joint_preview(backend, targets, parent.sequence, pool, config, optimizer_step)
        proposal_lists = {side: _make_stageb_proposals(
            current_seq=parent.sequence, candidates=sequences[side], action_group="unrouted",
            group_prior=1.0, logits_before=batches[side][0], logits_after=batches[side][1],
            use_gradient_prior=True, rerank_scores=preview) for side in SIDES}
        grouped = {}
        for side in SIDES:
            for proposal in proposal_lists[side]:
                grouped.setdefault(proposal.sequence, []).append(proposal)
        total_prior = sum(proposal.prior for items in grouped.values() for proposal in items)
        total_gradient = sum(proposal.gradient_prior for items in grouped.values() for proposal in items)
        proposals = [replace(items[0],
            prior=sum(item.prior for item in items) / max(total_prior, 1e-12),
            gradient_prior=sum(item.gradient_prior for item in items) / max(total_gradient, 1e-12),
            logit_delta_benefit=max(item.logit_delta_benefit for item in items),
            proposal_logits=None) for items in grouped.values()]
        actions = tree.register_action_proposals(parent, proposals)
        chosen, _ = _select_stageb_actions_by_group_quota(
            tree, parent, actions, group_priors={"unrouted": 1.0},
            max_actions=config.stageb_verifier_candidates,
            exploration_actions=config.stageb_exploration_actions if preview else 0,
            rng=random.Random(stageb_seed(config.seed, "explore", optimizer_step, parent.sequence)),
            limit_to_tree_slots=False)
        event.update(unique_candidates=len(pool), preview_mode=preview_mode,
                     preview=preview, verifier_candidates=[action.sequence for action in chosen])
        valid_actions, records = [], []
        paired_records = (verify_many(action.sequence for action in chosen)
                          if verify_many is not None else [verify(action.sequence) for action in chosen])
        for action, record in zip(chosen, paired_records, strict=True):
            event["candidates"].append({**record, "origins": origins.get(action.sequence, [])})
            if not record["valid"]:
                action.status = "invalid"
                continue
            # Missing full-verifier metrics abort rather than silently weaken
            # the native Stage-B reward/quality contract.
            result = joint_result(record, floors)
            tree.store_verifier_result(result)
            valid_actions.append(action)
            records.append(record)
            evaluated[action.sequence] = record
        selected = select_expansions(valid_actions, records, tree, floors, limit)
        selected_ids = {action.action_id for action, _ in selected}
        for action in valid_actions:
            if action.action_id not in selected_ids:
                action.status = "evaluated_pruned"
        results = [joint_result(record, floors) for _, record in selected]
        children = tree.expand(parent, [action.proposal for action, _ in selected], results)
        child_by_sequence = {child.sequence: child for child in children}
        for action, record in selected:
            child = child_by_sequence.get(action.sequence)
            if child is None:
                continue
            states[child.node_id] = {side: minimally_align_binder_logits_to_sequence(
                batches[side][1].clone(), child.sequence, **mask_kwargs).detach().cpu() for side in SIDES}
            child.logits = states[child.node_id]["human"].clone()
            branch_floors[child.node_id] = dict(floors)
            tree.backpropagate_action(selection, action, child.tree_reward)
        event["admitted"] = list(child_by_sequence)
        event["tree_size"] = tree.total_nodes
        tree.prune()
        write_step(event)
        best = max(evaluated.values(), key=lambda record: guarded_rank(record, tree, final_floors))
        _write_json(Path(out) / "best.json", annotate(best, tree, final_floors))
        logger.info("dual_target stage_b step=%d/%d pool=%d verified=%d admitted=%d",
                    local_step + 1, stageb_steps, len(pool), len(chosen), len(children))
    ranked = sorted(evaluated.values(),
                    key=lambda record: (guarded_rank(record, tree, final_floors), record["sequence"]),
                    reverse=True)
    return [annotate(record, tree, final_floors) for record in ranked]


def annotate(record, tree, floors):
    return {**record, "stageb": {"quality_floors": dict(floors),
             "quality_ok": all(pair_iptm(record, side) >= floors[side] for side in SIDES),
             "target_quality_ok": {side: pair_iptm(record, side) >= floors[side] for side in SIDES},
             "target_pair_chain_iptm": {side: pair_iptm(record, side) for side in SIDES},
             "tree_reward": tree.result_reward(joint_result(record, floors)),
             "guarded_rank": _plain(guarded_rank(record, tree, floors))}}
