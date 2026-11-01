import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bindcraft.hallucination.esmfold2_backend import (
    AA_TO_ESM_IDX, ESMFold2ProposalConfig, init_binder_logits,
    minimally_align_binder_logits_to_sequence, sequence_from_logits_argmax,
)
from bindcraft.hallucination.esmfold2_runtime import ESMFold2RuntimeBackend
from bindcraft.hallucination import esmfold2_dual_target as dual

PARENT = "A" * 40
TARGETS = {"human": {"sequence": "G" * 50}, "mouse": {"sequence": "S" * 51}}


def logits():
    return minimally_align_binder_logits_to_sequence(
        init_binder_logits(PARENT, random_scale=0, requires_grad=False), PARENT)


def result(sequence, target, loss):
    return SimpleNamespace(binder_seq=sequence, target_seq=target, total_loss=loss,
                           loss_dict={"total": loss}, summary_confidence={})


def test_raw_noop_does_not_create_four_mutation_slots_and_rng_is_preserved():
    before = logits()
    after = before.clone()
    after[:8, AA_TO_ESM_IDX["Y"]] = .05  # positive delta, still below current A
    state = torch.random.get_rng_state().clone()
    cfg = dual.DualTargetConfig(mutation_fraction=.025, candidate_strategy="positive_score")
    candidates = dual.candidates_from_logits(PARENT, before, after, cfg, 11)
    assert torch.equal(state, torch.random.get_rng_state())
    assert candidates[0] == {"sequence": PARENT, "kind": "raw_argmax", "noop": True}
    assert len(candidates) == 4
    assert all(sum(a != b for a, b in zip(PARENT, c["sequence"])) == 1 for c in candidates[1:])
    assert candidates == dual.candidates_from_logits(PARENT, before, after, cfg, 11)


def test_raw_and_mutation_dedup_shortfall_is_bounded():
    before = logits()
    after = before.clone()
    after[0, AA_TO_ESM_IDX["Y"]] = 2
    candidates = dual.candidates_from_logits(PARENT, before, after,
                    dual.DualTargetConfig(mutation_fraction=.025, refill_attempts=2), 11)
    assert len(candidates) == 1  # only available mutation is the raw argmax
    assert candidates[0]["sequence"] == "Y" + PARENT[1:]


class Backend:
    def __init__(self, *, failing=False):
        self.verify_calls, self.propose_calls = [], []
        self.failing = failing

    def verify(self, *, binder_seq, target_seq, **kwargs):
        self.verify_calls.append((binder_seq, target_seq, kwargs["seed"]))
        if self.failing and binder_seq.startswith("Y") and target_seq.startswith("S"):
            raise RuntimeError("synthetic mouse failure")
        loss = .9 if binder_seq == PARENT else (.1 if target_seq.startswith("G") else .4)
        return result(binder_seq, target_seq, loss)

    def propose(self, **kwargs):
        self.propose_calls.append({**kwargs, "proposal_logits": kwargs["proposal_logits"].clone()})
        before = kwargs["proposal_logits"].clone()
        after = before.clone()
        after[0, AA_TO_ESM_IDX["Y"]] += 3 if kwargs["target_seq"].startswith("G") else 4
        return SimpleNamespace(logits_before=before, logits_after=after)


def test_both_targets_verify_once_shared_candidates_and_separate_child_states(tmp_path):
    backend = Backend()
    ranked = dual.run_dual_target(backend=backend, targets=TARGETS, seeds={"seed": PARENT},
             config=dual.DualTargetConfig(outer_steps=2, exploration=0, mutation_fraction=.025),
             output_dir=tmp_path)
    assert len(backend.propose_calls) == 4  # exactly one soft chunk per target per step
    assert all(c["current_seq"] == PARENT for c in backend.propose_calls[:2])
    assert torch.equal(backend.propose_calls[0]["proposal_logits"], backend.propose_calls[1]["proposal_logits"])
    assert torch.all(backend.propose_calls[0]["proposal_logits"][:, AA_TO_ESM_IDX["A"]] == 2.0)
    assert backend.propose_calls[2]["current_seq"] == ranked[0]["sequence"]
    assert not torch.equal(backend.propose_calls[2]["proposal_logits"], backend.propose_calls[3]["proposal_logits"])
    for call in backend.propose_calls:
        assert sequence_from_logits_argmax(call["proposal_logits"], current_seq=call["current_seq"]) == call["current_seq"]
    assert len(backend.verify_calls) == 4  # parent + child, each human and mouse once
    assert len(set(backend.verify_calls)) == 4
    assert ranked[0]["worst_loss"] == .4
    events = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert len(events[0]["candidates"][0]["origins"]) == 2
    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "completed"


def test_one_side_failure_excludes_candidate_and_does_not_replace_with_human_score(tmp_path):
    ranked = dual.run_dual_target(backend=Backend(failing=True), targets=TARGETS,
              seeds={"seed": PARENT}, config=dual.DualTargetConfig(outer_steps=1), output_dir=tmp_path)
    assert [r["sequence"] for r in ranked] == [PARENT]
    event = json.loads((tmp_path / "steps.jsonl").read_text())
    assert event["admitted"] == []
    assert event["candidates"][0]["worst_loss"] is None


def test_worst_side_wins_and_mean_breaks_ties(tmp_path, monkeypatch):
    sequences = [aa + PARENT[1:] for aa in "YFW"]
    def candidates(*args):
        return [{"sequence": seq, "kind": "raw_argmax" if i == 0 else "mutation", "noop": False}
                for i, seq in enumerate(sequences)]
    monkeypatch.setattr(dual, "candidates_from_logits", candidates)
    backend = Backend()
    scores = {PARENT: (.9, .9), sequences[0]: (.01, .8), sequences[1]: (.4, .4), sequences[2]: (.3, .4)}
    backend.verify = lambda binder_seq, target_seq, **kw: result(binder_seq, target_seq,
                               scores[binder_seq][int(target_seq.startswith("S"))])
    ranked = dual.run_dual_target(backend=backend, targets=TARGETS, seeds={"seed": PARENT},
                                  config=dual.DualTargetConfig(outer_steps=1), output_dir=tmp_path)
    assert [r["sequence"] for r in ranked] == [sequences[2], sequences[1], sequences[0], PARENT]


def test_real_runtime_propose_adapter_without_loading_models(tmp_path):
    calls = []
    def soft_update(current_seq, target_seq, global_step, seed, action_group, mutable_mask,
                    proposal_logits=None, **kwargs):
        calls.append((current_seq, target_seq))
        after = proposal_logits.clone()
        after[:8, AA_TO_ESM_IDX["Y"]] += 2
        after[:8, AA_TO_ESM_IDX["F"]] += 1
        return proposal_logits, after
    backend = ESMFold2RuntimeBackend(config=ESMFold2ProposalConfig(verifier_loss_mode="aligned_common"),
                                    device="cpu", soft_update_fn=soft_update)
    backend.verify = lambda binder_seq, target_seq, **kw: result(binder_seq, target_seq, .5)
    dual.run_dual_target(backend=backend, targets=TARGETS, seeds={"seed": PARENT},
                        config=dual.DualTargetConfig(outer_steps=1, mutation_fraction=.025), output_dir=tmp_path)
    assert len(calls) == 2
    event = json.loads((tmp_path / "steps.jsonl").read_text())
    assert all(len(event["proposals"][side]) == 4 for side in TARGETS)
    assert event["unique_candidates"] <= 8
    assert backend.experimental_model is None and backend.verifier_model is None
    report = json.loads((tmp_path / "optimization_report.json").read_text())
    assert report["requested"]["proposal"] == "off"  # library fixture, not CLI defaults
    assert report["counters"]["proposal_update.successes"] == 2


def test_no_valid_root_fails_run_and_existing_run_cannot_be_overwritten(tmp_path):
    backend = Backend()
    backend.verify = lambda **kw: result(kw["binder_seq"], kw["target_seq"], float("nan"))
    with pytest.raises(RuntimeError, match="No starting"):
        dual.run_dual_target(backend=backend, targets=TARGETS, seeds={"seed": PARENT},
                            config=dual.DualTargetConfig(outer_steps=1), output_dir=tmp_path)
    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "failed"
    with pytest.raises(FileExistsError):
        dual.run_dual_target(backend=Backend(), targets=TARGETS, seeds={"seed": PARENT},
                            config=dual.DualTargetConfig(outer_steps=1), output_dir=tmp_path)


def test_target_and_seed_validation_precede_model_calls(tmp_path):
    with pytest.raises(ValueError, match="distinct"):
        dual.run_dual_target(backend=Backend(), targets={"human": TARGETS["human"], "mouse": TARGETS["human"]},
                            seeds={"seed": PARENT}, config=dual.DualTargetConfig(), output_dir=tmp_path)
    assert not (tmp_path / "manifest.json").exists()


def test_eight_distinct_candidates_all_receive_two_verifications(tmp_path, monkeypatch):
    human, mouse = [aa + PARENT[1:] for aa in "YFWD"], [aa + PARENT[1:] for aa in "ENQR"]
    counter = iter((human, mouse))
    monkeypatch.setattr(dual, "candidates_from_logits", lambda *args: [
        {"sequence": s, "kind": "raw_argmax" if i == 0 else "mutation", "noop": False}
        for i, s in enumerate(next(counter))])
    backend = Backend()
    dual.run_dual_target(backend=backend, targets=TARGETS, seeds={"seed": PARENT},
                        config=dual.DualTargetConfig(outer_steps=1), output_dir=tmp_path)
    assert len(backend.verify_calls) == 18  # eight candidates + one parent, twice each
    for seq in human + mouse:
        assert {t for s, t, _ in backend.verify_calls if s == seq} == {"G" * 50, "S" * 51}
    event = json.loads((tmp_path / "steps.jsonl").read_text())
    assert event["unique_candidates"] == 8


def test_multiple_roots_of_different_lengths_receive_warmup(tmp_path):
    backend = Backend()
    second = "A" * 50
    dual.run_dual_target(backend=backend, targets=TARGETS, seeds={"one": PARENT, "two": second},
                        config=dual.DualTargetConfig(outer_steps=2), output_dir=tmp_path)
    calls = [c["current_seq"] for c in backend.propose_calls]
    assert calls[0] == calls[1] and calls[2] == calls[3]
    assert set(calls) == {PARENT, second}  # warmup visits both, ordered by root score


def test_invalid_logits_are_not_converted_into_candidates():
    before = logits()
    after = before.clone()
    after[0, 0] = float("nan")
    with pytest.raises(ValueError, match="Invalid soft logits"):
        dual.candidates_from_logits(PARENT, before, after, dual.DualTargetConfig(), 1)


@pytest.mark.parametrize("length", [40, 110, 250, 301])
def test_cli_validate_only_does_not_create_run(tmp_path, length):
    root = Path(__file__).resolve().parents[2]
    for name, seq in (("human", "G" * 50), ("mouse", "S" * 51), ("seeds", "A" * length)):
        (tmp_path / f"{name}.fasta").write_text(f">{name}\n{seq}\n")
    output = tmp_path / "run"
    command = [sys.executable, str(root / "scripts/run_esmfold2_dual_target.py"),
               "--config", str(root / "configs/esmfold2_dual_target.json"),
               "--output-dir", str(output), "--validate-only"]
    for name in ("human", "mouse", "seeds"):
        command += [f"--{name}-fasta", str(tmp_path / f"{name}.fasta")]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["models_loaded"] is False
    assert json.loads(proc.stdout)["optimization_requested"] == {
        "proposal": "fast", "verifier": "fast", "esmc": "exact"}
    assert not output.exists()


@pytest.mark.parametrize("length", [10, 110, 250, 301])
def test_variable_length_search_preserves_paired_verification_and_child_logits(tmp_path, length):
    parent = "A" * length
    backend = Backend()
    backend.verify = lambda binder_seq, target_seq, **kw: result(
        binder_seq, target_seq, .9 if binder_seq == parent else .4)
    dual.run_dual_target(backend=backend, targets=TARGETS, seeds={"seed": parent},
                        config=dual.DualTargetConfig(outer_steps=2, exploration=0),
                        output_dir=tmp_path)
    assert len(backend.propose_calls) == 4
    assert all(call["proposal_logits"].shape == (length, 20) for call in backend.propose_calls)
    events = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert events[0]["admitted"]
    assert all(len(c["sequence"]) == length for event in events for c in event["candidates"])
    for sequence in {seq for seq, _, _ in backend.verify_calls}:
        assert {target for seq, target, _ in backend.verify_calls if seq == sequence} == {
            TARGETS["human"]["sequence"], TARGETS["mouse"]["sequence"]}


@pytest.mark.parametrize("sequence", ["", "a", "AX", "A A", None])
def test_sequence_validation_still_rejects_empty_or_invalid_alphabet(sequence):
    with pytest.raises(ValueError, match="canonical"):
        dual.validate_sequence(sequence, binder=True)


def test_equal_length_targets_do_not_share_sequence_cache(tmp_path):
    targets = {"mouse": {"sequence": "S" * 50}, "human": {"sequence": "G" * 50}}
    backend = Backend()
    ranked = dual.run_dual_target(backend=backend, targets=targets, seeds={"seed": PARENT},
                                 config=dual.DualTargetConfig(outer_steps=1), output_dir=tmp_path)
    assert len(backend.verify_calls) == 4
    assert ranked[0]["targets"]["human"]["loss"] == .1
    assert ranked[0]["targets"]["mouse"]["loss"] == .4
    assert [c["target_seq"][0] for c in backend.propose_calls] == ["G", "S"]


@pytest.mark.parametrize("fail_search", [False, True])
def test_optimization_report_finalizes_on_success_and_failure(tmp_path, fail_search):
    backend = Backend()
    calls = []
    backend.optimizations = SimpleNamespace(report_path=None, finalize=lambda: calls.append("finalized"))
    if fail_search:
        def fail(**kw):
            raise RuntimeError("proposal failure")
        backend.propose = fail
        with pytest.raises(RuntimeError, match="proposal failure"):
            dual.run_dual_target(backend=backend, targets=TARGETS, seeds={"seed": PARENT},
                                config=dual.DualTargetConfig(outer_steps=1), output_dir=tmp_path)
    else:
        dual.run_dual_target(backend=backend, targets=TARGETS, seeds={"seed": PARENT},
                            config=dual.DualTargetConfig(outer_steps=1), output_dir=tmp_path)
    assert calls == ["finalized"]
    assert backend.optimizations.report_path == tmp_path / "optimization_report.json"


def test_report_failure_cannot_mask_search_error_or_claim_completion(tmp_path):
    backend = Backend()
    def fail_report():
        raise OSError("report write failure")
    def fail_propose(**kw):
        raise RuntimeError("original search error")
    backend.optimizations = SimpleNamespace(report_path=None, finalize=fail_report)
    backend.propose = fail_propose
    with pytest.raises(RuntimeError, match="original search error"):
        dual.run_dual_target(backend=backend, targets=TARGETS, seeds={"seed": PARENT},
                            config=dual.DualTargetConfig(outer_steps=1), output_dir=tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["optimization_report_error"] == "report write failure"


def test_cli_fast_without_root_fails_before_loading_models(tmp_path):
    import os
    root = Path(__file__).resolve().parents[2]
    for name, seq in (("human", "G" * 50), ("mouse", "S" * 51), ("seeds", PARENT)):
        (tmp_path / f"{name}.fasta").write_text(f">{name}\n{seq}\n")
    command = [sys.executable, str(root / "scripts/run_esmfold2_dual_target.py"),
               "--config", str(root / "configs/esmfold2_dual_target.json"),
               "--output-dir", str(tmp_path / "run")]
    for name in ("human", "mouse", "seeds"):
        command += [f"--{name}-fasta", str(tmp_path / f"{name}.fasta")]
    env = dict(os.environ)
    env.pop("PROTEOCRAFT_UPLIFT_ROOT", None)
    proc = subprocess.run(command, capture_output=True, text=True, timeout=30, env=env)
    assert proc.returncode != 0
    assert "Acceleration requires --optimization-root" in proc.stderr
    assert "esmfold2_runtime.py" not in proc.stderr
    assert not (tmp_path / "run").exists()


def test_cysteine_replacement_is_reproducible_and_order_independent():
    import random
    original = "CACA" + "A" * 36
    second = "ACAC" + "A" * 36
    cfg = dual.DualTargetConfig(seed_cysteine_policy="random_replace", seed=43)
    state = random.getstate()
    first, changes = dual.prepare_seed_sequences({"one": original, "two": second}, cfg)
    reverse, _ = dual.prepare_seed_sequences({"two": second, "one": original}, cfg)
    assert first == reverse
    assert random.getstate() == state
    for name, source in (("one", original), ("two", second)):
        assert "C" not in first[name] and len(first[name]) == len(source)
        assert all(a == b for a, b in zip(source, first[name]) if a != "C")
        assert len(changes[name]["replacements"]) == 2
    assert [r["position_1based"] for r in changes["one"]["replacements"]] == [1, 3]


def test_cysteine_seeds_are_transformed_before_both_root_verifications(tmp_path):
    original = "C" * 6 + "A" * 34
    cfg = dual.DualTargetConfig(outer_steps=1, seed_cysteine_policy="random_replace")
    expected, _ = dual.prepare_seed_sequences({"seed": original}, cfg)
    backend = Backend()
    dual.run_dual_target(backend=backend, targets=TARGETS, seeds={"seed": original},
                        config=cfg, output_dir=tmp_path)
    assert [call[0] for call in backend.verify_calls[:2]] == [expected["seed"]] * 2
    assert all("C" not in call[0] for call in backend.verify_calls)
    assert all("C" not in call["current_seq"] for call in backend.propose_calls)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["input_seeds"]["seed"] == original
    assert manifest["seeds"]["seed"] == expected["seed"]
    assert (tmp_path / "initialized_seeds.fasta").exists()


def test_library_rejects_cysteine_seed_without_explicit_policy():
    with pytest.raises(ValueError, match="explicit seed_cysteine_policy"):
        dual.prepare_seed_sequences({"seed": "C" + PARENT[1:]}, dual.DualTargetConfig())


@pytest.mark.parametrize('failing', [False, True])
def test_target_major_keeps_joint_results_cache_and_candidate_order(tmp_path, monkeypatch, failing):
    sequences = [aa + PARENT[1:] for aa in 'YFWDENQR']
    def candidates(parent, before, after, config, seed):
        return [{'sequence': s, 'kind': 'mutation', 'noop': False}
                for s in sequences[:4] if seed % 2] or [
                    {'sequence': s, 'kind': 'mutation', 'noop': False} for s in sequences[4:]]
    monkeypatch.setattr(dual, 'candidates_from_logits', candidates)
    outputs, histories, calls = {}, {}, {}
    for order in ('candidate', 'target'):
        backend = Backend(failing=failing)
        path = tmp_path / order
        outputs[order] = dual.run_dual_target(backend=backend, targets=TARGETS,
            seeds={'seed': PARENT}, config=dual.DualTargetConfig(
                outer_steps=2, verification_order=order), output_dir=path)
        histories[order] = [json.loads(line) for line in (path / 'steps.jsonl').read_text().splitlines()]
        calls[order] = backend.verify_calls
    assert outputs['candidate'] == outputs['target']
    assert histories['candidate'] == histories['target']
    assert set(calls['candidate']) == set(calls['target'])
    assert len(calls['target']) == len(set(calls['target'])) == 18
    assert [target for _, target, _ in calls['target'][2:]] == [TARGETS['human']['sequence']] * 8 + [TARGETS['mouse']['sequence']] * 8
    assert [target for _, target, _ in calls['candidate'][2:]] == [TARGETS['human']['sequence'], TARGETS['mouse']['sequence']] * 8


def test_unknown_verification_order_is_rejected():
    with pytest.raises(ValueError, match='verification_order'):
        dual.DualTargetConfig(verification_order='automatic')
