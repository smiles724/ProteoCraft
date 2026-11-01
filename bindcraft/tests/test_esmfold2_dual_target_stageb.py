import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from bindcraft.hallucination import esmfold2_dual_target as dual
from bindcraft.hallucination import esmfold2_dual_target_stageb as stageb
from bindcraft.hallucination.esmfold2_backend import AA_TO_ESM_IDX, ESMFold2ProposalConfig
from bindcraft.hallucination.esmfold2_pipeline import _ESMFold2StageBMCTSTree
from bindcraft.hallucination.esmfold2_runtime import ESMFold2RuntimeBackend

PARENT = "A" * 40
TARGETS = {"human": {"sequence": "G" * 50}, "mouse": {"sequence": "S" * 50}}


class FullBackend:
    def __init__(self):
        self.config = ESMFold2ProposalConfig(verifier_loss_mode="aligned_common",
            verifier_num_loops=4, verifier_num_sampling_steps=50,
            proposal_optimization="fast", verifier_optimization="fast", esmc_optimization="exact")
        self.propose_calls, self.verify_calls, self.preview_calls = [], [], []

    def propose(self, **kwargs):
        self.propose_calls.append({**kwargs, "proposal_logits": kwargs["proposal_logits"].clone()})
        before = kwargs["proposal_logits"].clone()
        after = before.clone()
        human = kwargs["target_seq"].startswith("G")
        aa = "Y" if human else "F"
        if kwargs.get("action_group"):
            count = kwargs["B_candidates"]
            start = 5 if human else 20
            candidates = []
            for index in range(count):
                position = (start + index) % len(kwargs["current_seq"])
                new = aa if kwargs["current_seq"][position] != aa else "W"
                sequence = kwargs["current_seq"][:position] + new + kwargs["current_seq"][position + 1:]
                candidates.append(sequence)
                after[position, AA_TO_ESM_IDX[new]] += 3.0 if human else 4.0
        else:
            for position in range(4):
                after[position, AA_TO_ESM_IDX[aa]] += 3.0 if human else 4.0
            candidates = []  # Stage A assembles its own raw + mutation slots.
        return SimpleNamespace(logits_before=before, logits_after=after, sequences=candidates)

    def verify(self, *, binder_seq, target_seq, seed, **kwargs):
        self.verify_calls.append((binder_seq, target_seq, seed))
        mutations = sum(aa != "A" for aa in binder_seq)
        iptm = min(.6 + mutations * .02, .95)
        loss = max(.8 - mutations * .03, .1)
        return SimpleNamespace(binder_seq=binder_seq, target_seq=target_seq, total_loss=loss,
            loss_dict={"pair_chain_iptm_loss": 1 - iptm, "binder_confidence_loss": .15,
                       "global_confidence_loss": .2, "binder_rg_excess_loss": .1},
            summary_confidence={})

    def preview_loss_dict(self, *, binder_seq, target_seq, seed, **kwargs):
        self.preview_calls.append((binder_seq, target_seq, seed))
        mutations = sum(aa != "A" for aa in binder_seq)
        value = .8 - mutations * .01
        return {"intra_contact_loss": value, "inter_contact_loss": value, "glob_loss": value}


def full_config(**kwargs):
    return dual.DualTargetConfig(strategy="two_stage", outer_steps=5, stagea_steps=2, **kwargs)


def events(path):
    return [json.loads(line) for line in (path / "steps.jsonl").read_text().splitlines()]


def test_real_two_stage_budget_pool_paired_verification_and_soft_state(tmp_path):
    backend = FullBackend()
    ranked = dual.run_dual_target(backend=backend, targets=TARGETS, seeds={"one": PARENT},
                                 config=full_config(), output_dir=tmp_path)
    assert len(backend.propose_calls) == 10
    assert [call["soft_steps_override"] for call in backend.propose_calls] == [30] * 4 + [20] * 6
    assert all(call["B_candidates"] == 10 and call["max_mutations_per_seq"] == 1
               and call["action_group"] == "unrouted" for call in backend.propose_calls[4:])
    logs = events(tmp_path)
    assert [event["stage"] for event in logs] == ["stage_a"] * 2 + ["stage_b"] * 3
    assert [event["step"] for event in logs] == list(range(5))
    for event in logs[2:]:
        assert event["unique_candidates"] <= 20
        assert len(event["verifier_candidates"]) <= 8
        assert len(event["admitted"]) <= 2
        for sequence in event["verifier_candidates"]:
            assert sum(a != b for a, b in zip(sequence, event["parent_sequence"])) == 1
            assert {target for seq, target, _ in backend.verify_calls if seq == sequence} == {
                TARGETS["human"]["sequence"], TARGETS["mouse"]["sequence"]}
    assert len(backend.verify_calls) == len(set(backend.verify_calls))
    checkpoint = torch.load(tmp_path / "stagea_handoff.pt", weights_only=True)
    assert checkpoint["completed_stagea_steps"] == 2
    assert len(checkpoint["roots"]) == 2
    first_b = backend.propose_calls[4:6]
    inherited = next(root for root in checkpoint["roots"] if root["record"]["sequence"] == first_b[0]["current_seq"])
    assert torch.equal(first_b[0]["proposal_logits"], inherited["states"]["human"])
    assert torch.equal(first_b[1]["proposal_logits"], inherited["states"]["mouse"])
    assert not torch.equal(first_b[0]["proposal_logits"], first_b[1]["proposal_logits"])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["completed_stagea_steps"] == 2 and manifest["completed_stageb_steps"] == 3
    assert manifest["completed_outer_steps"] == 5
    assert manifest["final_selection"] == "stageb_guarded_both_targets"
    assert ranked[0]["stageb"]["quality_ok"]
    assert backend.config.proposal_optimization == "fast" and backend.config.verifier_optimization == "fast"


def test_both_preview_contexts_and_seeds_are_paired_even_equal_target_lengths(tmp_path):
    backend = FullBackend()
    dual.run_dual_target(backend=backend, targets=TARGETS, seeds={"one": PARENT},
                        config=full_config(), output_dir=tmp_path)
    for event in events(tmp_path)[2:]:
        assert event["preview_mode"] == "dual_worst_preview"
        for side in TARGETS:
            target = TARGETS[side]["sequence"]
            seed = stageb.stageb_seed(43, "preview:" + side, event["optimizer_step"], event["parent_sequence"])
            calls = [seq for seq, t, s in backend.preview_calls if t == target and s == seed]
            assert event["parent_sequence"] in calls
            assert set(event["preview"]) <= set(calls)
            assert len(calls) == len(set(calls))
    assert set(t for _, t, _ in backend.preview_calls) == {"G" * 50, "S" * 50}


def test_one_side_preview_failure_falls_back_for_whole_pool(tmp_path):
    backend = FullBackend()
    original = backend.preview_loss_dict
    def preview(**kwargs):
        if kwargs["target_seq"].startswith("S"):
            raise ValueError("mouse preview failed")
        return original(**kwargs)
    backend.preview_loss_dict = preview
    ranked = dual.run_dual_target(backend=backend, targets=TARGETS, seeds={"one": PARENT},
                                  config=full_config(), output_dir=tmp_path)
    assert ranked
    for event in events(tmp_path)[2:]:
        assert event["preview_mode"] == "gradient_fallback_whole_pool"
        assert event["preview"] == {}
        assert event["verifier_candidates"]


def record(human_pair, mouse_pair, loss=.2):
    sides = {side: {"status": "ok", "loss": loss,
                   "loss_dict": {"pair_chain_iptm_loss": 1 - pair,
                                 "binder_confidence_loss": .1, "binder_rg_excess_loss": .1},
                   "summary_confidence": {}}
             for side, pair in zip(stageb.SIDES, (human_pair, mouse_pair))}
    return {"sequence": PARENT, "valid": True, "worst_loss": loss, "mean_loss": loss, "targets": sides}


def test_quality_floors_are_per_target_and_native_rewards_are_preserved():
    floors = {"human": .75, "mouse": .55}
    good = record(.8, .6)
    bad = record(.95, .54, loss=.01)
    native = _ESMFold2StageBMCTSTree()
    tree = stageb.DualTargetStageBTree()
    joint = stageb.joint_result(good, floors)
    assert stageb.annotate(good, tree, floors)["stageb"]["quality_ok"]
    assert not stageb.annotate(bad, tree, floors)["stageb"]["quality_ok"]
    assert stageb.guarded_rank(good, tree, floors) > stageb.guarded_rank(bad, tree, floors)
    expected = []
    for side in stageb.SIDES:
        loss_dict = {**good["targets"][side]["loss_dict"], stageb.FLOOR: floors[side]}
        reward, _ = native._compute_base_node_reward(.2, loss_dict)
        expected.append(reward)
    assert tree.result_reward(joint) == min(expected)
    assert stageb.FLOOR not in good["targets"]["human"]["loss_dict"]
    assert len(stageb.guarded_rank(good, tree, floors)) == 8


def test_one_quality_exploration_matches_native_policy():
    floors = {"human": .75, "mouse": .55}
    good, bad, worse = record(.8, .6), record(.8, .54), record(.8, .1)
    tree = stageb.DualTargetStageBTree()
    actions = [SimpleNamespace(action_id=index) for index in range(3)]
    chosen = stageb.select_expansions(actions, [good, bad, worse], tree, floors, 2)
    assert [action.action_id for action, _ in chosen] == [0, 1]


def test_both_preview_scores_protect_weaker_side():
    backend = FullBackend()
    a, b = "Y" + PARENT[1:], "F" + PARENT[1:]
    scores = {PARENT: (.8, .8), a: (.1, .95), b: (.6, .6)}
    def preview(binder_seq, target_seq, **kwargs):
        value = scores[binder_seq][int(target_seq.startswith("S"))]
        return {"intra_contact_loss": value, "inter_contact_loss": value, "glob_loss": value}
    backend.preview_loss_dict = preview
    combined, mode = stageb.joint_preview(backend, TARGETS, PARENT, [a, b], full_config(), 2)
    assert mode == "dual_worst_preview"
    assert combined[b]["benefit"] > combined[a]["benefit"]


@pytest.mark.parametrize("legacy", [False, True])
def test_stageb_only_reuses_source_without_rerunning_stagea_or_overwriting_it(tmp_path, legacy):
    source, output = tmp_path / "source", tmp_path / "stageb"
    dual.run_dual_target(backend=FullBackend(), targets=TARGETS, seeds={"one": PARENT},
                        config=dual.DualTargetConfig(outer_steps=2), output_dir=source)
    if legacy:
        (source / "stagea_handoff.pt").unlink()
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    backend = FullBackend()
    config = dual.DualTargetConfig(strategy="stage_b_only", outer_steps=2)
    ranked = dual.run_dual_target(backend=backend, targets=TARGETS, seeds={}, config=config,
                                 output_dir=output, stagea_results=source)
    assert len(backend.propose_calls) == 4
    assert all(call["soft_steps_override"] == 20 for call in backend.propose_calls)
    assert [event["step"] for event in events(output)] == [0, 1]
    assert [event["optimizer_step"] for event in events(output)] == [2, 3]
    manifest = json.loads((output / "manifest.json").read_text())
    expected = "sequence_only_reinitialized" if legacy else "paired_soft_logits"
    assert manifest["stagea_source"]["state_origin"] == expected
    assert manifest["completed_stagea_steps"] == 0
    assert manifest["completed_stageb_steps"] == 2 and ranked
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    source_roots = json.loads((source / "ranked.json").read_text())[:2]
    for root in source_roots:
        assert {target for seq, target, _ in backend.verify_calls if seq == root["sequence"]} == {
            "G" * 50, "S" * 50}  # Roots are freshly scored, never loaded as cache hits.


def test_stagea_handoff_source_rejects_tampering_context_seed_and_partial_run(tmp_path):
    source = tmp_path / "source"
    backend = FullBackend()
    dual.run_dual_target(backend=backend, targets=TARGETS, seeds={"one": PARENT},
                        config=dual.DualTargetConfig(outer_steps=1), output_dir=source)
    config = dual.DualTargetConfig(strategy="stage_b_only", outer_steps=1)
    with pytest.raises(ValueError, match="target/hotspot"):
        stageb.load_stagea_source(source, config, {**TARGETS, "mouse": {"sequence": "T" * 50}}, backend.config)
    with pytest.raises(ValueError, match="seed"):
        stageb.load_stagea_source(source, replace(config, seed=44), TARGETS, backend.config)
    with pytest.raises(ValueError, match="same runtime"):
        stageb.load_stagea_source(source, config, TARGETS, replace(backend.config, learning_rate=.1))
    with (source / "stagea_handoff.pt").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        stageb.load_stagea_source(source, config, TARGETS, backend.config)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["status"] = "running"
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="completed"):
        stageb.load_stagea_source(source, config, TARGETS, backend.config)


def test_stageb_missing_full_verifier_metrics_fails_instead_of_downgrading(tmp_path):
    backend = FullBackend()
    original = backend.verify
    def verify(**kwargs):
        result = original(**kwargs)
        result.loss_dict.pop("binder_rg_excess_loss")
        return result
    backend.verify = verify
    with pytest.raises(ValueError, match="full-verifier metrics"):
        dual.run_dual_target(backend=backend, targets=TARGETS, seeds={"one": PARENT},
                            config=full_config(), output_dir=tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["completed_stageb_steps"] == 0


def test_stageb_verifier_failure_does_not_admit_one_target_only(tmp_path):
    backend = FullBackend()
    original = backend.verify
    def verify(**kwargs):
        if kwargs["binder_seq"][5] == "Y" and kwargs["target_seq"].startswith("S"):
            raise RuntimeError("mouse verifier failed")
        return original(**kwargs)
    backend.verify = verify
    ranked = dual.run_dual_target(backend=backend, targets=TARGETS, seeds={"one": PARENT},
                                 config=full_config(), output_dir=tmp_path)
    for event in events(tmp_path)[2:]:
        invalid = {candidate["sequence"] for candidate in event["candidates"] if not candidate["valid"]}
        assert not invalid.intersection(event["admitted"])
    assert all(record["valid"] for record in ranked)


def test_shipped_two_stage_configuration_matches_native_default_budget_and_acceleration():
    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / "configs/esmfold2_dual_target.json").read_text())
    defaults = yaml.safe_load((root / "configs/proteocraft_profiles/esmfold2_d.yaml").read_text())["defaults"]
    config = dual.DualTargetConfig(**payload["search"])
    pairs = {"outer_steps": "n_steps", "strategy": "mcts_strategy", "stagea_steps": "mcts_bootstrap_steps",
             "handoff_top_k": "mcts_handoff_top_k", "soft_steps": "esmfold2_stagea_soft_steps",
             "stageb_soft_steps": "esmfold2_stageb_soft_steps", "tree_children": "mcts_expand_actions_per_step",
             "stageb_rerank_pool_size": "esmfold2_stageb_rerank_pool_size",
             "stageb_verifier_candidates": "esmfold2_stageb_verifier_candidates",
             "stageb_exploration_actions": "esmfold2_stageb_exploration_actions",
             "stageb_max_mutations_per_seq": "stageb_max_mutations_per_seq",
             "stageb_discrete_rerank": "esmfold2_stageb_discrete_rerank", "stageb_prior_mode": "mcts_prior_mode"}
    for dual_key, native_key in pairs.items():
        assert getattr(config, dual_key) == defaults[native_key]
    assert (config.outer_steps, config.stagea_steps) == (110, 70)
    for component in ("proposal", "verifier", "esmc"):
        assert payload["runtime"][component + "_optimization"] == defaults["esmfold2_" + component + "_optimization"]


@pytest.mark.parametrize("changes", [
    {"outer_steps": 70}, {"stageb_rerank_pool_size": 1}, {"stageb_max_mutations_per_seq": 2},
    {"stageb_exploration_actions": 8}, {"stageb_prior_mode": "full"},
    {"stageb_handoff_quality_tolerance": float("nan")}, {"stageb_max_nodes": False},
])
def test_invalid_stageb_settings_are_rejected(changes):
    arguments = {"strategy": "two_stage", "outer_steps": 110, **changes}
    with pytest.raises(ValueError):
        dual.DualTargetConfig(**arguments)


def test_stageb_preview_can_be_disabled_explicitly(tmp_path):
    backend = FullBackend()
    dual.run_dual_target(backend=backend, targets=TARGETS, seeds={"one": PARENT},
                        config=full_config(stageb_discrete_rerank=False), output_dir=tmp_path)
    assert not backend.preview_calls
    assert all(event["preview_mode"] == "disabled" for event in events(tmp_path)[2:])


def test_full_default_110_steps_actually_execute_70_plus_40(tmp_path):
    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / "configs/esmfold2_dual_target.json").read_text())
    backend = FullBackend()
    dual.run_dual_target(backend=backend, targets=TARGETS, seeds={"one": PARENT},
                        config=dual.DualTargetConfig(**payload["search"]), output_dir=tmp_path)
    logs = events(tmp_path)
    assert len(logs) == 110
    assert [event["step"] for event in logs] == list(range(110))
    assert sum(event["stage"] == "stage_a" for event in logs) == 70
    assert sum(event["stage"] == "stage_b" for event in logs) == 40
    assert [call["soft_steps_override"] for call in backend.propose_calls] == [30] * 140 + [20] * 80
    assert [event["optimizer_step"] for event in logs[70:]] == list(range(70, 110))


def test_two_stage_uses_real_runtime_proposal_and_existing_report_adapter(tmp_path):
    calls = []
    def soft_update(current_seq, target_seq, global_step, seed, action_group,
                    mutable_mask, proposal_logits=None, **kwargs):
        calls.append((target_seq, kwargs["soft_steps_override"], action_group))
        before = proposal_logits.clone()
        after = before.clone()
        for position in range(min(12, len(current_seq))):
            aa = "Y" if target_seq.startswith("G") else "F"
            if aa == current_seq[position]:
                aa = "W"
            after[position, AA_TO_ESM_IDX[aa]] += 4
        return before, after
    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(verifier_loss_mode="aligned_common"),
        device="cpu", soft_update_fn=soft_update)
    fake = FullBackend()
    backend.verify = fake.verify
    backend.preview_loss_dict = fake.preview_loss_dict
    dual.run_dual_target(backend=backend, targets=TARGETS, seeds={"one": PARENT},
                        config=full_config(), output_dir=tmp_path)
    assert len(calls) == 10
    assert [item[1] for item in calls] == [30] * 4 + [20] * 6
    assert backend.experimental_model is None and backend.verifier_model is None
    report = json.loads((tmp_path / "optimization_report.json").read_text())
    assert report["counters"]["proposal_update.successes"] == 10
    assert events(tmp_path)[2]["verifier_candidates"]


def test_stageb_error_finalizes_acceleration_report_and_cannot_claim_completion(tmp_path):
    backend = FullBackend()
    finalized = []
    backend.optimizations = SimpleNamespace(report_path=None, finalize=lambda: finalized.append(True))
    original = backend.propose
    def propose(**kwargs):
        if kwargs.get("action_group"):
            raise RuntimeError("Stage-B proposal failed")
        return original(**kwargs)
    backend.propose = propose
    with pytest.raises(RuntimeError, match="Stage-B proposal failed"):
        dual.run_dual_target(backend=backend, targets=TARGETS, seeds={"one": PARENT},
                            config=full_config(), output_dir=tmp_path)
    assert finalized == [True]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["completed_stagea_steps"] == 2 and manifest["completed_stageb_steps"] == 0


@pytest.mark.parametrize("legacy", [False, True])
def test_cli_stagea_handoff_validates_remaining_40_steps_without_loading_models(tmp_path, legacy):
    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / "configs/esmfold2_dual_target.json").read_text())
    backend = FullBackend()
    backend.config = ESMFold2ProposalConfig(**payload["runtime"])
    source = tmp_path / "source"
    dual.run_dual_target(backend=backend, targets=TARGETS, seeds={"one": PARENT},
                        config=dual.DualTargetConfig(outer_steps=70), output_dir=source)
    if legacy:
        (source / "stagea_handoff.pt").unlink()
    for side in TARGETS:
        (tmp_path / f"{side}.fasta").write_text(f">{side}\n{TARGETS[side]['sequence']}\n")
    output = tmp_path / "stageb"
    command = [sys.executable, str(root / "scripts/run_esmfold2_dual_target.py"),
        "--config", str(root / "configs/esmfold2_dual_target.json"),
        "--human-fasta", str(tmp_path / "human.fasta"), "--mouse-fasta", str(tmp_path / "mouse.fasta"),
        "--stagea-results", str(source), "--output-dir", str(output), "--validate-only"]
    process = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result["strategy"] == "stage_b_only" and result["outer_steps"] == 40
    assert result["stage_budgets"] == {"stage_a": 0, "stage_b": 40}
    assert result["models_loaded"] is False and not output.exists()
    assert result["handoff_state_origin"] == ("sequence_only_reinitialized" if legacy else "paired_soft_logits")


def test_nonfinite_stageb_logits_abort_instead_of_generating_candidates(tmp_path):
    backend = FullBackend()
    original = backend.propose
    def propose(**kwargs):
        batch = original(**kwargs)
        if kwargs.get("action_group"):
            batch.logits_after[0, 0] = float("nan")
        return batch
    backend.propose = propose
    with pytest.raises(ValueError, match="Invalid Stage-B soft logits"):
        dual.run_dual_target(backend=backend, targets=TARGETS, seeds={"one": PARENT},
                            config=full_config(), output_dir=tmp_path)
    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "failed"


def test_grouped_verification_preserves_entire_two_stage_trajectory(tmp_path):
    outputs, histories, backends = {}, {}, {}
    for order in ('candidate', 'target'):
        backend = FullBackend()
        backends[order] = backend
        path = tmp_path / order
        outputs[order] = dual.run_dual_target(backend=backend, targets=TARGETS,
            seeds={'one': PARENT}, config=full_config(verification_order=order), output_dir=path)
        histories[order] = events(path)
    assert outputs['candidate'] == outputs['target']
    assert histories['candidate'] == histories['target']
    assert set(backends['candidate'].verify_calls) == set(backends['target'].verify_calls)
    assert backends['candidate'].preview_calls == backends['target'].preview_calls
    assert len(backends['target'].verify_calls) == len(set(backends['target'].verify_calls))
    target_calls = backends['target'].verify_calls
    switch_count = sum(a[1] != b[1] for a, b in zip(target_calls, target_calls[1:]))
    assert switch_count <= 2 * 5 + 1
