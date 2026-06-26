import torch
import pytest
import json

from bindcraft.hallucination.esmfold2_backend import build_esm_verifier_result
from bindcraft.hallucination.esmfold2_pipeline import run_esmfold2_mcts_hallucination


class FakeESMFold2Backend:
    def __init__(self):
        self.verify_calls = []
        self.propose_calls = []
        self.preview_calls = []
        self.scores = {
            "AAA": 0.10,
            "AAY": 0.80,
            "AAF": 0.40,
            "AYY": 0.90,
        }

    def verify(self, *, binder_seq, target_seq, seed=None):
        self.verify_calls.append((binder_seq, target_seq, seed))
        score = self.scores.get(binder_seq, 0.05)
        extra = {
        }
        return build_esm_verifier_result(
            binder_seq=binder_seq,
            target_seq=target_seq,
            summary_confidence={"pair_chain_iptm": score, "iptm": score},
            extra_loss_dict=extra,
            coordinate=torch.zeros(1, len(binder_seq) + len(str(target_seq)), 3),
        )

    def preview_loss_dict(self, *, binder_seq, target_seq, seed=None):
        self.preview_calls.append((binder_seq, target_seq, seed))
        return {
            "inter_contact_loss": 0.6,
            "intra_contact_loss": 0.1,
            "glob_loss": 0.05,
        }

    def propose(
        self,
        *,
        current_seq,
        target_seq,
        global_step,
        K_top,
        B_candidates,
        T_sample,
        max_mutations_per_seq,
        seed=None,
        action_group=None,
        mutable_mask=None,
    ):
        self.propose_calls.append(
            {
                "current_seq": current_seq,
                "target_seq": target_seq,
                "global_step": global_step,
                "seed": seed,
                "action_group": action_group,
                "mutable_mask": mutable_mask,
                "max_mutations_per_seq": max_mutations_per_seq,
            }
        )
        if current_seq == "AAA":
            return ["AAY", "AAF"][:B_candidates]
        if current_seq == "AAY":
            return ["AYY"][:B_candidates]
        return []


def test_esmfold2_stagea_pipeline_uses_backend_for_verify_and_proposal():
    backend = FakeESMFold2Backend()

    best_seq, final_seq, losses = run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=2,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=2,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
    )

    assert best_seq == "AYY"
    assert final_seq == "AYY"
    assert losses[-1] == pytest.approx(0.10)
    assert [call[0] for call in backend.verify_calls] == ["AAA", "AAY", "AAF", "AYY"]
    assert [call["current_seq"] for call in backend.propose_calls] == ["AAA", "AAY"]
    assert [call["action_group"] for call in backend.propose_calls] == [None, None]


def test_esmfold2_stage_a_only_strategy_never_enters_stageb():
    backend = FakeESMFold2Backend()

    best_seq, final_seq, losses = run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=2,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=2,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="stage_a_only",
        mcts_bootstrap_steps=1,
        mcts_bootstrap_max_steps=1,
    )

    assert best_seq == "AYY"
    assert final_seq == "AYY"
    assert losses[-1] == pytest.approx(0.10)
    assert [call["action_group"] for call in backend.propose_calls] == [None, None]
    assert backend.preview_calls == []


def test_esmfold2_pipeline_passes_mutable_mask_and_clips_fixed_positions():
    backend = FakeESMFold2Backend()

    best_seq, final_seq, losses = run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=1,
        num_beams=1,
        init_binder_seq="AAA",
        mutable_mask=[False, True, True],
        K_top=4,
        B_candidates=2,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
    )

    assert best_seq == "AAY"
    assert final_seq == "AAY"
    assert losses[-1] == pytest.approx(0.20)
    assert backend.propose_calls[0]["mutable_mask"] == [False, True, True]
    assert all(seq[0] == "A" for seq, *_ in backend.verify_calls)


def test_esmfold2_two_stage_pipeline_routes_stageb_proposals_by_group():
    backend = FakeESMFold2Backend()

    best_seq, final_seq, losses = run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=3,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=2,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="two_stage",
        mcts_bootstrap_steps=1,
        mcts_bootstrap_max_steps=1,
        mcts_expand_actions_per_step=1,
    )

    assert best_seq == "AYY"
    assert final_seq == "AYY"
    assert losses[-1] == pytest.approx(0.10)
    assert backend.propose_calls[0]["action_group"] is None
    assert backend.propose_calls[1]["action_group"] == "interface_formation"
    assert backend.preview_calls


def test_esmfold2_two_stage_pipeline_passes_mutable_mask_to_stageb():
    backend = FakeESMFold2Backend()

    run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=3,
        num_beams=1,
        init_binder_seq="AAA",
        mutable_mask=[False, True, True],
        K_top=4,
        B_candidates=2,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="two_stage",
        mcts_bootstrap_steps=1,
        mcts_bootstrap_max_steps=1,
        mcts_expand_actions_per_step=1,
    )

    assert backend.propose_calls[0]["mutable_mask"] == [False, True, True]
    assert backend.propose_calls[1]["mutable_mask"] == [False, True, True]


def test_esmfold2_two_stage_uses_separate_stagea_and_stageb_mutation_budgets():
    backend = FakeESMFold2Backend()

    run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=3,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=2,
        T_sample=1.0,
        max_mutations_per_seq=3,
        stagea_max_mutations_per_seq=6,
        stageb_max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="two_stage",
        mcts_bootstrap_steps=1,
        mcts_bootstrap_max_steps=1,
        mcts_expand_actions_per_step=1,
    )

    assert backend.propose_calls[0]["action_group"] is None
    assert backend.propose_calls[0]["max_mutations_per_seq"] == 6
    assert backend.propose_calls[1]["action_group"] == "interface_formation"
    assert backend.propose_calls[1]["max_mutations_per_seq"] == 1


def test_esmfold2_pipeline_writes_structured_step_artifacts(tmp_path):
    backend = FakeESMFold2Backend()
    artifact_dir = tmp_path / "esmfold2_artifacts"

    best_seq, final_seq, losses = run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=3,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=2,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="two_stage",
        mcts_bootstrap_steps=1,
        mcts_bootstrap_max_steps=1,
        mcts_expand_actions_per_step=1,
        artifact_dir=artifact_dir,
    )

    assert best_seq == final_seq == "AYY"
    assert losses[-1] == pytest.approx(0.10)

    trajectory_path = artifact_dir / "trajectory.jsonl"
    candidates_path = artifact_dir / "candidates.jsonl"
    summary_path = artifact_dir / "run_summary.json"
    assert trajectory_path.exists()
    assert candidates_path.exists()
    assert summary_path.exists()

    trajectory = [json.loads(line) for line in trajectory_path.read_text().splitlines()]
    candidates = [json.loads(line) for line in candidates_path.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())

    assert [event["stage"] for event in trajectory] == ["phase0", "stage_a", "stage_b"]
    assert trajectory[1]["candidate_count"] == 2
    assert trajectory[2]["primary_action_group"] == "interface_formation"
    assert trajectory[2]["selected_group_counts"] == {"interface_formation": 1}
    assert any(event["stage"] == "stage_b" and event["sequence"] == "AYY" for event in candidates)
    assert candidates[-1]["loss_dict"]["total"] == pytest.approx(0.10)
    assert summary["best_sequence"] == "AYY"
    assert summary["best_loss"] == pytest.approx(0.10)
