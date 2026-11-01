"""Opt-in, independent 4+4 proposals with joint verification.

This module is deliberately not imported by any existing launch path. Both
targets use the same runtime/configuration but independent node-local logits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from .esmfold2_backend import (
    ESM_AA_ORDER, build_logit_delta_candidate_batch, init_binder_logits, anchor_binder_logits_to_sequence,
    minimally_align_binder_logits_to_sequence, sequence_from_logits_argmax,
)
from .mcts_legacy import LegacyMCTSTree

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DualTargetConfig:
    outer_steps: int = 70
    soft_steps: int = 30
    mutation_fraction: float = 0.5
    candidate_strategy: str = "positive_score_guarded"
    dominance_gap: float = 0.5
    protected_mutation_candidates: int = 2
    sampling_temperature: float = 1.0
    refill_attempts: int = 8
    tree_children: int = 2
    exploration: float = 1.0
    min_root_expansions: int = 2
    seed: int = 43
    verifier_seed: int = 1043
    # Opt-in scheduling only; existing callers keep candidate-major execution.
    verification_order: str = "candidate"
    design_mode: str = "miniprotein"
    seed_cysteine_policy: str = "reject"
    initialization_seed: int | None = None
    # Old JSON/API callers retain their single-stage contract. The shipped
    # profile explicitly selects two_stage and a 110-step total budget.
    strategy: str = "stage_a_only"
    stagea_steps: int = 70
    handoff_top_k: int = 2
    stageb_soft_steps: int = 20
    stageb_rerank_pool_size: int = 20
    stageb_verifier_candidates: int = 8
    stageb_exploration_actions: int = 1
    stageb_max_mutations_per_seq: int = 1
    stageb_discrete_rerank: bool = True
    stageb_prior_mode: str = "no_group"
    stageb_max_nodes: int = 5000
    stageb_min_pair_chain_iptm: float = 0.0
    stageb_handoff_quality_tolerance: float = 0.05
    stageb_hotspot_coverage_weight: float = 0.25
    stageb_hotspot_distance_weight: float = 0.25
    stageb_hotspot_miss_penalty: float = 0.0

    def __post_init__(self):
        if self.verification_order not in {"candidate", "target"}:
            raise ValueError("verification_order must be candidate or target")
        for name in ("outer_steps", "soft_steps", "refill_attempts", "tree_children"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.mutation_fraction <= 1:
            raise ValueError("mutation_fraction must be in (0, 1]")
        if self.candidate_strategy not in {"positive_score", "positive_score_guarded"}:
            raise ValueError("dual target supports positive_score or positive_score_guarded")
        if not math.isfinite(self.sampling_temperature) or self.sampling_temperature <= 0:
            raise ValueError("sampling_temperature must be positive and finite")
        if not math.isfinite(self.exploration) or self.exploration < 0:
            raise ValueError("exploration must be finite and nonnegative")
        if not math.isfinite(self.dominance_gap) or self.dominance_gap < 0:
            raise ValueError("dominance_gap must be finite and nonnegative")
        if self.protected_mutation_candidates not in range(4):
            raise ValueError("protected_mutation_candidates must be between 0 and 3")
        if self.tree_children > 8:
            raise ValueError("tree_children cannot exceed the eight proposal slots")
        for name in ("seed", "verifier_seed", "min_root_expansions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.design_mode not in {"miniprotein", "masked_antibody"}:
            raise ValueError("design_mode must be miniprotein or masked_antibody")
        allowed_cys = {"preserve_fixed"} if self.design_mode == "masked_antibody" else {"reject", "random_replace"}
        if self.seed_cysteine_policy not in allowed_cys:
            raise ValueError(f"{self.design_mode} seed_cysteine_policy must be one of {sorted(allowed_cys)}")
        if self.initialization_seed is not None and (isinstance(self.initialization_seed, bool)
                or not isinstance(self.initialization_seed, int) or self.initialization_seed < 0):
            raise ValueError("initialization_seed must be a nonnegative integer or null")
        if self.strategy not in {"stage_a_only", "two_stage", "stage_b_only"}:
            raise ValueError("strategy must be stage_a_only, two_stage, or stage_b_only")
        for name in ("stagea_steps", "handoff_top_k", "stageb_soft_steps",
                     "stageb_rerank_pool_size", "stageb_verifier_candidates", "stageb_max_nodes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.strategy == "two_stage" and self.stagea_steps >= self.outer_steps:
            raise ValueError("two_stage requires outer_steps > stagea_steps")
        if self.stageb_rerank_pool_size < 2:
            raise ValueError("Stage B requires at least one proposal slot per target")
        if (isinstance(self.stageb_exploration_actions, bool)
                or not isinstance(self.stageb_exploration_actions, int)
                or not 0 <= self.stageb_exploration_actions < self.stageb_verifier_candidates):
            raise ValueError("stageb_exploration_actions must be below the verifier budget")
        if isinstance(self.stageb_max_mutations_per_seq, bool) or self.stageb_max_mutations_per_seq != 1:
            raise ValueError("Stage B implements the native single-mutation contract")
        if not isinstance(self.stageb_discrete_rerank, bool):
            raise ValueError("stageb_discrete_rerank must be boolean")
        if self.stageb_prior_mode != "no_group":
            raise ValueError("Dual-target Stage B currently supports the default no_group prior")
        for name in ("stageb_min_pair_chain_iptm", "stageb_handoff_quality_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        for name in ("stageb_hotspot_coverage_weight", "stageb_hotspot_distance_weight",
                     "stageb_hotspot_miss_penalty"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


def validate_sequence(sequence: str, *, binder: bool = False) -> str:
    """Validate the alphabet, independently of experiment length categories.

    ``binder`` remains accepted for compatibility with existing callers. Model
    resource limits and competition length rules belong to the run plan, not
    this shared sequence validator.
    """
    if not isinstance(sequence, str) or not sequence or set(sequence) - set(ESM_AA_ORDER):
        raise ValueError("Sequences must contain only uppercase canonical amino acids")
    return sequence


def validate_target_context(target, runtime_config=None, *, side="target"):
    """Validate explicitly opted-in homotrimers without changing monomer inputs.

    The pipe-delimited sequence is only a runtime transport representation; the
    three original FASTA identifiers remain part of cache and handoff identity.
    No protomer is silently replicated, dropped, or relabelled.
    """
    if not isinstance(target, dict) or "sequence" not in target:
        raise ValueError(f"{side} target context requires a sequence")
    if set(target) - {"sequence", "hotspot_params", "assembly", "chain_ids"}:
        raise ValueError("Unknown target context fields")
    def setting(name, default):
        if isinstance(runtime_config, dict):
            return runtime_config.get(name, default)
        return getattr(runtime_config, name, default)
    assembly = setting("target_assembly", "monomer")
    if assembly == "monomer":
        if set(target) & {"assembly", "chain_ids"}:
            raise ValueError("Monomer target contexts must not contain assembly or chain_ids metadata")
        validate_sequence(target["sequence"])
        return dict(target)
    if assembly != "homotrimer":
        raise ValueError(f"Unsupported target assembly: {assembly}")
    if setting("target_interface_mode", "all") != "contacted":
        raise ValueError("Homotrimer dual-target search requires target_interface_mode=contacted")
    if target.get("assembly") != "homotrimer":
        raise ValueError(f"{side} target must explicitly declare assembly=homotrimer")
    sequence = target["sequence"]
    if not isinstance(sequence, str):
        raise ValueError(f"{side} homotrimer sequence must be a pipe-delimited string")
    chains = sequence.split("|")
    if len(chains) != 3:
        raise ValueError(f"{side} homotrimer must contain exactly three chains")
    for chain in chains:
        validate_sequence(chain)
    if len(set(chains)) != 1:
        raise ValueError(f"{side} homotrimer chains must have identical canonical sequences")
    chain_ids = target.get("chain_ids")
    if (not isinstance(chain_ids, (list, tuple)) or len(chain_ids) != 3
            or any(not isinstance(name, str) or not name.strip() for name in chain_ids)
            or len(set(chain_ids)) != 3):
        raise ValueError(f"{side} homotrimer requires three nonempty unique chain_ids")
    roles = setting("target_chain_roles", None)
    if roles is not None and tuple(roles) != ("target", "target", "target"):
        raise ValueError("Homotrimer protomers must all have the target chain role")
    return {**target, "chain_ids": list(chain_ids)}


def prepare_seed_sequences(seeds, config, mutable_mask=None):
    """Replace input Cys once, before root verification, independently of order.

    Historical affinity labels belong to the originals, never these sequences.
    """
    if not seeds or len(set(seeds.values())) != len(seeds):
        raise ValueError("Provide unique starting sequences")
    processed, changes = {}, {}
    alphabet = ESM_AA_ORDER.replace("C", "")
    for name, original in seeds.items():
        validate_sequence(original, binder=True)
        if config.design_mode == "masked_antibody":
            from .esmfold2_dual_target_constraints import check_sequence
            if mutable_mask is None:
                raise ValueError("masked_antibody initialization requires an explicit mask")
            check_sequence(original, original, mutable_mask)
            processed[name] = original
            changes[name] = {"original_sequence": original, "initialized_sequence": original,
                             "replacements": [], "policy": "preserve_fixed"}
            continue
        if "C" in original and config.seed_cysteine_policy == "reject":
            raise ValueError("Cys-bearing seeds require explicit seed_cysteine_policy=random_replace")
        initialization_seed = config.seed if config.initialization_seed is None else config.initialization_seed
        local_seed = int(hashlib.sha256(f"{initialization_seed}:{original}".encode()).hexdigest(), 16)
        rng = random.Random(local_seed)
        sequence, replacements = [], []
        for position, aa in enumerate(original, 1):
            new_aa = rng.choice(alphabet) if aa == "C" else aa
            sequence.append(new_aa)
            if aa != new_aa:
                replacements.append({"position_1based": position, "from": aa, "to": new_aa})
        processed[name] = "".join(sequence)
        changes[name] = {"original_sequence": original, "initialized_sequence": processed[name],
                         "replacements": replacements, "rng_seed_sha256": f"{local_seed:064x}"}
    if len(set(processed.values())) != len(processed):
        raise ValueError("Cys replacement produced duplicate seeds; choose a different search seed")
    return processed, changes


def candidates_from_logits(parent, before, after, config, seed, mutable_mask=None):
    """Reserve exactly one raw slot and up to three non-raw mutation slots.

    Refill only resamples discrete edits; it never runs another soft update.
    Raw no-ops remain recorded, but do not become an extra mutation slot.
    """
    for tensor in (before, after):
        # -inf is an intentional amino-acid mask in existing ESM logits.
        if torch.isnan(tensor).any() or torch.isposinf(tensor).any() or not torch.isfinite(tensor).any(dim=-1).all():
            raise ValueError("Invalid soft logits: NaN, +inf, or fully masked residue")
    raw = sequence_from_logits_argmax(after, current_seq=parent, mutable_mask=mutable_mask)
    candidates = [{"sequence": raw, "kind": "raw_argmax", "noop": raw == parent}]
    seen = {parent, raw}
    # CPU tensors and a forked RNG make refill deterministic without perturbing
    # proposal-model RNG state or retaining GPU graphs.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for _ in range(config.refill_attempts):
            batch = build_logit_delta_candidate_batch(
                before.detach().cpu(), after.detach().cpu(), current_seq=parent, mutable_mask=mutable_mask,
                B_candidates=3, include_logit_candidates=False,
                mutation_fraction=config.mutation_fraction,
                candidate_strategy=("positive_score" if config.protected_mutation_candidates == 0
                                    else config.candidate_strategy),
                dominance_gap=config.dominance_gap,
                protected_mutation_candidates=config.protected_mutation_candidates,
                T_sample=config.sampling_temperature,
            )
            for sequence, metadata in zip(batch.sequences, batch.candidate_metadata):
                if sequence not in seen:
                    seen.add(sequence)
                    candidates.append({"sequence": sequence, "kind": "mutation", "noop": False,
                                       "mutation_metadata": dict(metadata)})
                    if len(candidates) == 4:
                        return candidates
    return candidates


def _plain(value):
    if isinstance(value, torch.Tensor):
        return _plain(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_plain(payload), ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def run_dual_target(*, backend: Any, targets: dict, seeds: dict[str, str],
                    config: DualTargetConfig, output_dir: str | Path,
                    stagea_results: str | Path | None = None):
    """Run shared-tree search; all successful verification results are eligible
    for final ranking, even if they are not admitted as the two tree children.

    targets = {"human": {"sequence": ..., "hotspot_params": ...}, "mouse": ...}
    The runtime must remain fixed for this run. No cross-run cache is loaded.
    """
    if set(targets) != {"human", "mouse"}:
        raise ValueError("Exactly human and mouse target contexts are required")
    # Freeze caller-owned target dictionaries before cache keys are constructed.
    runtime_config = getattr(backend, "config", None)
    targets = json.loads(json.dumps({side: targets[side] for side in ("human", "mouse")}))
    targets = {side: validate_target_context(target, runtime_config, side=side)
               for side, target in targets.items()}
    if targets["human"]["sequence"] == targets["mouse"]["sequence"]:
        raise ValueError("Human and mouse targets must be distinct")
    from .esmfold2_dual_target_constraints import validate_design, check_sequence
    mutable_mask = validate_design(config, runtime_config, seeds)
    if mutable_mask is not None and getattr(backend, "task_type", None) != "antibody":
        raise ValueError("masked_antibody requires an antibody runtime (task_type='antibody')")
    mask_kwargs = {"mutable_mask": mutable_mask} if mutable_mask is not None else {}
    from .esmfold2_dual_target_stageb import load_stagea_source, run_stage_b, save_handoff
    source = None
    if config.strategy == "stage_b_only":
        if stagea_results is None or seeds:
            raise ValueError("stage_b_only requires stagea_results and no new seeds")
        source = load_stagea_source(stagea_results, config, targets, runtime_config)
        seeds = {f"handoff_{i}": record["sequence"] for i, record in enumerate(source["roots"])}
    elif stagea_results is not None:
        raise ValueError("stagea_results requires explicit stage_b_only strategy")
    input_seeds = dict(seeds)
    validate_design(config, runtime_config, input_seeds)
    seeds, seed_changes = prepare_seed_sequences(input_seeds, config, mutable_mask)
    fixed_reference = next(iter(seeds.values()))
    stagea_budget = (config.stagea_steps if config.strategy == "two_stage"
                    else 0 if config.strategy == "stage_b_only" else config.outer_steps)
    stageb_budget = config.outer_steps - stagea_budget
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Output directory must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    # Exclusive manifest creation prevents silent reuse/overwrite of a run.
    manifest = {"mode": "dual_target_4plus4" if config.strategy == "stage_a_only" else "dual_target_two_stage",
                "status": "running", "phase": "root_verification",
                "stage_budgets": {"stage_a": stagea_budget, "stage_b": stageb_budget},
                "completed_stagea_steps": 0, "completed_stageb_steps": 0,
                "search": asdict(config), "targets": targets, "seeds": seeds,
                "input_seeds": input_seeds, "seed_initialization": seed_changes,
                "runtime": asdict(runtime_config) if runtime_config is not None else None,
                "torch_version": str(torch.__version__),
                "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ("esmfold2_dual_target.py", "esmfold2_runtime.py",
                                               "esmfold2_backend.py", "esmfold2_one_step.py", "esmfold2_interface.py",
                                               "esmfold2_optimizations.py", "esmfold2_memory.py", "mcts_legacy.py",
                                               "esmfold2_dual_target_stageb.py", "esmfold2_dual_target_constraints.py", "esmfold2_pipeline.py",
                                               "mcts_controller.py")},
                "optimization_report": str(out / "optimization_report.json"),
                "score": "lexicographic(max(human_loss,mouse_loss), mean_loss)",
                "state": "separate target logits; minimal alignment to shared child sequence"}
    if mutable_mask is not None:
        manifest["task_type"] = backend.task_type
        manifest["sequence_constraints"] = {
            "mutable_mask": list(mutable_mask), "reference_sequence": fixed_reference,
            "fixed_positions_1based": [i + 1 for i, flag in enumerate(mutable_mask) if not flag],
            "fixed_cysteines_1based": [i + 1 for i, aa in enumerate(fixed_reference) if aa == "C"],
            "scope": "sequence identities only; no coordinate template"}
    if source is not None:
        manifest["stagea_source"] = {key: value for key, value in source.items()
                                     if key not in {"states", "roots"}}
        manifest["stagea_source"]["root_sequences"] = [record["sequence"] for record in source["roots"]]
    with (out / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2)
    (out / "initialized_seeds.fasta").write_text("".join(
        f">{name}\n{sequence}\n" for name, sequence in seeds.items()))
    manager = getattr(backend, "optimizations", None)
    if manager is not None:
        # Keep acceleration receipts with this run for both CLI and API callers.
        manager.report_path = out / "optimization_report.json"
    context_keys = {side: hashlib.sha256(json.dumps({"target": target, "runtime": manifest["runtime"],
                        "seed": config.verifier_seed}, sort_keys=True).encode()).hexdigest()
                    for side, target in targets.items()}
    cache, verified, states, sequence_states = {}, {}, {}, {}
    tree = LegacyMCTSTree(C=config.exploration, selection_mode="hierarchical_ucb",
                          min_root_expansions=config.min_root_expansions)

    def verify_side(sequence, side):
        if mutable_mask is not None:
            check_sequence(sequence, fixed_reference, mutable_mask)
        target = targets[side]
        key = (side, context_keys[side], sequence)
        if key not in cache:
            try:
                result = backend.verify(binder_seq=sequence, target_seq=target["sequence"],
                                        seed=config.verifier_seed,
                                        hotspot_params=target.get("hotspot_params"))
                loss = float(result.total_loss)
                if not math.isfinite(loss):
                    raise ValueError("Nonfinite verifier loss")
                if result.binder_seq != sequence:
                    raise ValueError("Verifier returned the wrong binder sequence")
                if result.target_seq != target["sequence"]:
                    raise ValueError("Verifier returned the wrong target sequence")
                cache[key] = {"status": "ok", "loss": loss,
                              "loss_dict": _plain(result.loss_dict),
                              "summary_confidence": _plain(result.summary_confidence)}
            except (RuntimeError, ValueError) as error:
                # Failures are reported, never replaced by the other side's
                # score. Programming errors and interrupts still abort.
                cache[key] = {"status": "failed", "error": str(error)}
            with (out / "verifications.jsonl").open("a") as handle:
                handle.write(json.dumps({"side": side, "context": context_keys[side],
                                         "sequence": sequence, **cache[key]}, allow_nan=False) + "\n")
        return cache[key]

    def assemble(sequence):
        results = {side: cache[(side, context_keys[side], sequence)] for side in targets}
        valid = all(r["status"] == "ok" for r in results.values())
        losses = [r["loss"] for r in results.values()] if valid else []
        record = {"sequence": sequence, "valid": valid, "targets": results,
                  "worst_loss": max(losses) if valid else None,
                  "mean_loss": sum(losses) / 2 if valid else None}
        verified[sequence] = record
        return record


    def verify_many(sequences):
        # Materialize once, retain caller order (including duplicates), and only
        # expose joint records after every requested candidate has both sides.
        sequences = list(sequences)
        unique = list(dict.fromkeys(sequences))
        if config.verification_order == "target":
            for side in targets:
                for sequence in unique:
                    verify_side(sequence, side)
        else:
            for sequence in unique:
                for side in targets:
                    verify_side(sequence, side)
        records = {sequence: assemble(sequence) for sequence in unique}
        return [records[sequence] for sequence in sequences]

    def verify(sequence):
        return verify_many([sequence])[0]

    def rank(record):
        return record["worst_loss"], record["mean_loss"], record["sequence"]

    def write_step(event):
        with (out / "steps.jsonl").open("a") as handle:
            handle.write(json.dumps(_plain(event), allow_nan=False) + "\n")
        stage = event["stage"]
        manifest["phase"] = stage
        manifest["completed_stagea_steps" if stage == "stage_a" else "completed_stageb_steps"] += 1
        _write_json(out / "manifest.json", manifest)

    try:
        seed_records = verify_many(seeds.values())
        for (name, sequence), record in zip(seeds.items(), seed_records, strict=True):
            if not record["valid"]:
                continue
            logits = init_binder_logits(sequence, random_scale=0, suppress_cys=True,
                                         requires_grad=False, **mask_kwargs)
            # Root initialization is stronger than child-state reconciliation:
            # use the existing seed anchor (2.0), not a nearly uniform 0.1 tie break.
            logits = anchor_binder_logits_to_sequence(logits, sequence, **mask_kwargs)
            root = tree.add_root_node(sequence, logits, record["worst_loss"])
            restored = source["states"].get(sequence) if source is not None else None
            states[root.node_id] = {side: minimally_align_binder_logits_to_sequence(
                restored[side].clone(), sequence, **mask_kwargs).detach().cpu() if restored is not None
                else logits.clone() for side in targets}
            sequence_states[sequence] = states[root.node_id]
        if not tree.roots:
            raise RuntimeError("No starting sequence passed both verifiers; see verifications.jsonl")
        manifest["phase"] = "stage_a" if stagea_budget else "stage_b_handoff"
        _write_json(out / "manifest.json", manifest)
        for step in range(stagea_budget):
            parent = tree.select()
            batches, proposals, merged = {}, {}, {}
            for index, (side, target) in enumerate(targets.items()):
                proposal_seed = config.seed + step * 2 + index
                batch = backend.propose(
                    current_seq=parent.sequence, target_seq=target["sequence"], global_step=step,
                    K_top=10, B_candidates=1, T_sample=config.sampling_temperature,
                    max_mutations_per_seq=3, mutation_fraction=config.mutation_fraction,
                    candidate_strategy=config.candidate_strategy, include_argmax_candidate=True,
                    dominance_gap=config.dominance_gap,
                    protected_mutation_candidates=config.protected_mutation_candidates,
                    seed=proposal_seed, proposal_logits=states[parent.node_id][side].clone(),
                    preserve_proposal_logits=True, soft_steps_override=config.soft_steps,
                    track_confidence_override=False, hotspot_params=target.get("hotspot_params"), **mask_kwargs,
                )
                if batch.logits_before is None or batch.logits_after is None:
                    raise ValueError("Runtime must return before/after logits")
                # Runtime performs exactly one soft chunk; discrete generation
                # here enforces the raw + three mutation quota, including no-ops.
                proposals[side] = candidates_from_logits(parent.sequence, batch.logits_before,
                                                          batch.logits_after, config, proposal_seed, **mask_kwargs)
                batches[side] = batch.logits_after.detach().cpu().clone()
                for candidate in proposals[side]:
                    sequence = candidate["sequence"]
                    validate_sequence(sequence, binder=True)
                    merged.setdefault(sequence, []).append({"side": side, **candidate})
            records = verify_many(merged)
            for record in records:
                if record["valid"] and record["sequence"] not in sequence_states:
                    sequence_states[record["sequence"]] = {
                        side: minimally_align_binder_logits_to_sequence(
                            logits.clone(), record["sequence"], **mask_kwargs).detach().cpu()
                        for side, logits in batches.items()}
            eligible = sorted([r for r in records if r["valid"] and r["sequence"] != parent.sequence
                               and r["sequence"] not in {c.sequence for c in parent.children}], key=rank)
            admitted = eligible[:config.tree_children]
            children = tree.expand(parent, [SimpleNamespace(binder_seq=r["sequence"],
                                  total_loss=r["worst_loss"], coordinate=None) for r in admitted])
            for child in children:
                states[child.node_id] = {side: minimally_align_binder_logits_to_sequence(
                    logits.clone(), child.sequence, **mask_kwargs).detach().cpu()
                    for side, logits in batches.items()}
                sequence_states[child.sequence] = states[child.node_id]
            tree.backpropagate(parent, [r["worst_loss"] for r in admitted] or [parent.L_real])
            event = {"stage": "stage_a", "step": step, "parent_id": parent.node_id, "parent_sequence": parent.sequence,
                     "proposals": proposals, "unique_candidates": len(merged),
                     "candidates": [{**r, "origins": merged[r["sequence"]]} for r in records],
                     "admitted": [c.sequence for c in children]}
            write_step(event)
            best = min((r for r in verified.values() if r["valid"]), key=rank)
            _write_json(out / "best.json", best)
            logger.info("dual_target step=%d/%d candidates=%d admitted=%d best_worst_loss=%.6f",
                        step + 1, config.outer_steps, len(merged), len(children), best["worst_loss"])
        ranked = sorted((r for r in verified.values() if r["valid"]), key=rank)
        if stagea_budget:
            _write_json(out / "stagea_ranked.json", ranked)
            save_handoff(out, ranked[:config.handoff_top_k], sequence_states, manifest, stagea_budget)
        if stageb_budget:
            manifest.update(phase="stage_b_handoff", final_selection="stageb_guarded_both_targets")
            _write_json(out / "manifest.json", manifest)
            ranked = run_stage_b(
                backend=backend, targets=targets, roots=ranked[:config.handoff_top_k],
                paired_states=sequence_states, verify=verify, verify_many=verify_many, config=config, out=out,
                start_step=stagea_budget,
                optimizer_offset=source["completed_stagea_steps"] if source is not None else stagea_budget,
                stageb_steps=stageb_budget, write_step=write_step)
            _write_json(out / "best_by_stageb_guarded.json", ranked[0])
        _write_json(out / "best_by_worst_loss.json", min(
            (record for record in verified.values() if record["valid"]), key=rank))
        _write_json(out / "best.json", ranked[0])
        _write_json(out / "ranked.json", ranked)
        (out / "ranked.fasta").write_text("".join(
            f">rank_{i}|worst_loss={r['worst_loss']:.8g}|mean_loss={r['mean_loss']:.8g}\n{r['sequence']}\n"
            for i, r in enumerate(ranked, 1)))
        manifest.update(status="completed", phase="completed", completed_outer_steps=config.outer_steps,
                        unique_verified_sequences=len(verified), valid_sequences=len(ranked))
        _write_json(out / "manifest.json", manifest)
        return ranked
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        _write_json(out / "manifest.json", manifest)
        raise
    finally:
        if manager is not None:
            search_failed = sys.exc_info()[0] is not None
            try:
                manager.finalize()
            except Exception as error:
                logger.exception("Could not finalize dual-target optimization diagnostics")
                manifest.update(status="failed", optimization_report_error=str(error))
                _write_json(out / "manifest.json", manifest)
                if not search_failed:
                    raise
