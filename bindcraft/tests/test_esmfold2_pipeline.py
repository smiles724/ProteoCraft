import torch
import pytest
import json

from bindcraft.hallucination.esmfold2_backend import (
    AA_TO_ESM_IDX,
    ESM_AA_ORDER,
    ESMFold2ProposalBatch,
    build_esm_verifier_result,
)
from bindcraft.hallucination.mcts_legacy import LegacyMCTSTree
from bindcraft.hallucination.esmfold2_pipeline import run_esmfold2_mcts_hallucination
from bindcraft.hallucination.esmfold2_pipeline import (
    _allocate_group_budgets,
    _make_phase0_seed_states,
    _make_stageb_proposals,
)


class FakeESMFold2Backend:
    def __init__(self):
        self.verify_calls = []
        self.propose_calls = []
        self.continuous_calls = []
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
            summary_confidence={
                "pair_chain_iptm": score,
                "iptm": score,
                "binder_plddt": 0.8,
                "binder_rg_excess": 0.1,
                "mmcif": f"data_{binder_seq}\n#\n",
            },
            extra_loss_dict=extra,
            coordinate=torch.zeros(1, len(binder_seq) + len(str(target_seq)), 3),
        )

    def preview_loss_dict(self, *, binder_seq, target_seq, seed=None, hotspot_params=None):
        self.preview_calls.append((binder_seq, target_seq, seed, hotspot_params))
        losses = {
            "inter_contact_loss": 0.6,
            "intra_contact_loss": 0.1,
            "glob_loss": 0.05,
        }
        if hotspot_params:
            losses["hotspot_contact_loss"] = 0.4
        return losses

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
        proposal_logits=None,
        hotspot_params=None,
        track_confidence_override=None,
        soft_steps_override=None,
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
                "proposal_logits": proposal_logits.detach().clone() if proposal_logits is not None else None,
                "hotspot_params": hotspot_params,
                "track_confidence_override": track_confidence_override,
                "soft_steps_override": soft_steps_override,
            }
        )
        if current_seq == "AAA":
            return ["AAY", "AAF"][:B_candidates]
        if current_seq == "AAY":
            return ["AYY"][:B_candidates]
        return []

    def continuous_design(
        self,
        *,
        current_seq,
        target_seq,
        global_step,
        seed=None,
        mutable_mask=None,
        proposal_logits=None,
        hotspot_params=None,
        track_confidence_override=None,
        soft_steps_override=None,
    ):
        self.continuous_calls.append(
            {
                "current_seq": current_seq,
                "target_seq": target_seq,
                "global_step": global_step,
                "seed": seed,
                "mutable_mask": mutable_mask,
                "proposal_logits": proposal_logits.detach().clone() if proposal_logits is not None else None,
                "hotspot_params": hotspot_params,
                "track_confidence_override": track_confidence_override,
                "soft_steps_override": soft_steps_override,
            }
        )
        logits = (
            proposal_logits.detach().clone()
            if proposal_logits is not None
            else torch.zeros(len(current_seq), 20)
        )
        return ESMFold2ProposalBatch(
            sequences=["AYY"],
            logits_after=logits,
        )


class FakeHotspotVerifyBackend(FakeESMFold2Backend):
    def __init__(self):
        super().__init__()
        self.hotspot_verify_calls = []

    def verify(self, *, binder_seq, target_seq, seed=None, hotspot_params=None):
        self.hotspot_verify_calls.append(hotspot_params)
        return super().verify(
            binder_seq=binder_seq,
            target_seq=target_seq,
            seed=seed,
        )


def test_stageb_group_budget_keeps_every_repair_group_represented():
    budgets = _allocate_group_budgets(
        {
            "interface_formation": 0.8,
            "core_stabilization": 0.15,
            "compactness_repair": 0.05,
        },
        8,
    )

    assert sum(budgets.values()) == 8
    assert all(budgets[group] >= 1 for group in budgets)
    assert set(budgets) == {
        "interface_formation",
        "core_stabilization",
        "compactness_repair",
    }


def test_stageb_proposals_are_strict_single_mutations_with_gradient_priors():
    before = torch.zeros(3, 20)
    after = before.clone()
    after[1, AA_TO_ESM_IDX["Y"]] = 3.0
    after[1, AA_TO_ESM_IDX["F"]] = 1.0

    proposals = _make_stageb_proposals(
        current_seq="AAA",
        candidates=["AYA", "AFA", "AYY"],
        action_group="interface_formation",
        group_prior=0.6,
        logits_before=before,
        logits_after=after,
        mutable_mask=[True, True, True],
    )

    assert [proposal.sequence for proposal in proposals] == ["AYA", "AFA"]
    assert all(len(proposal.edits) == 1 for proposal in proposals)
    assert proposals[0].benefit > proposals[1].benefit
    assert proposals[0].gradient_prior > proposals[1].gradient_prior
    assert sum(proposal.prior for proposal in proposals) == pytest.approx(0.6)

    uniform = _make_stageb_proposals(
        current_seq="AAA",
        candidates=["AYA", "AFA"],
        action_group="interface_formation",
        group_prior=0.6,
        logits_before=before,
        logits_after=after,
        mutable_mask=[True, True, True],
        use_gradient_prior=False,
    )
    assert [proposal.gradient_prior for proposal in uniform] == pytest.approx([0.5, 0.5])


def test_stageb_soft_updates_do_not_overwrite_parent_node_logits():
    class StatefulStageBBackend(FakeESMFold2Backend):
        def __init__(self):
            super().__init__()
            self.stageb_inputs = []
            self.scores.update({"FAY": 0.01, "GAY": 0.01, "IAY": 0.01})

        def propose(self, **kwargs):
            if kwargs.get("action_group") is None:
                return super().propose(**kwargs)
            before = kwargs["proposal_logits"].detach().clone()
            self.stageb_inputs.append(
                (kwargs["global_step"], kwargs["action_group"], before.clone())
            )
            group_to_candidate = {
                "interface_formation": ("FAY", "F"),
                "core_stabilization": ("GAY", "G"),
                "compactness_repair": ("IAY", "I"),
            }
            candidate, amino_acid = group_to_candidate[kwargs["action_group"]]
            after = before.clone()
            after[0, AA_TO_ESM_IDX[amino_acid]] += 5.0
            return ESMFold2ProposalBatch(
                sequences=[candidate],
                logits_before=before,
                logits_after=after,
            )

    backend = StatefulStageBBackend()
    run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=3,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=3,
        T_sample=1.0,
        max_mutations_per_seq=3,
        seed=7,
        mcts_strategy="two_stage",
        mcts_bootstrap_steps=1,
        mcts_bootstrap_max_steps=1,
        mcts_expand_actions_per_step=1,
    )

    step_one = [entry[2] for entry in backend.stageb_inputs if entry[0] == 1]
    step_two = [entry[2] for entry in backend.stageb_inputs if entry[0] == 2]
    assert step_one and step_two
    assert all(torch.equal(logits, step_one[0]) for logits in step_one + step_two)


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


def test_stagea_legacy_tree_selects_lower_average_loss_path():
    tree = LegacyMCTSTree(C=0.0)
    low_loss = tree.add_root_node(
        sequence="AAA",
        logits=torch.zeros(3, 20),
        L_real=0.2,
    )
    tree.add_root_node(
        sequence="BBB",
        logits=torch.zeros(3, 20),
        L_real=0.8,
    )

    tree.backpropagate(low_loss, [0.1])

    assert tree.select().sequence == "AAA"


def test_esmfold2_continuous_stagea_designs_once_then_verifies_argmax_sequence():
    backend = FakeESMFold2Backend()

    best_seq, final_seq, losses = run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=150,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=2,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="continuous_stage_a",
    )

    assert best_seq == "AYY"
    assert final_seq == "AYY"
    assert losses == [pytest.approx(0.10)]
    assert [call[0] for call in backend.verify_calls] == ["AAA", "AYY"]
    assert [call["current_seq"] for call in backend.continuous_calls] == ["AAA"]
    assert backend.propose_calls == []
    assert backend.preview_calls == []


def test_esmfold2_pipeline_propagates_hotspot_without_preview_routing():
    backend = FakeESMFold2Backend()
    hotspot_params = {"target_hotspot_concat_indices": [1, 2]}

    run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq=["TT", "GG"],
        L_binder=3,
        N_steps=2,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=1,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="two_stage",
        mcts_bootstrap_steps=1,
        mcts_bootstrap_max_steps=1,
        mcts_expand_actions_per_step=1,
        hotspot_params=hotspot_params,
    )

    assert backend.propose_calls
    assert all(call["hotspot_params"] == hotspot_params for call in backend.propose_calls)
    assert backend.preview_calls == []


def test_esmfold2_pipeline_propagates_hotspots_to_verify_when_supported():
    backend = FakeHotspotVerifyBackend()
    hotspot_params = {"target_hotspot_concat_indices": [1, 2]}

    run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=1,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=1,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="stage_a_only",
        hotspot_params=hotspot_params,
    )

    assert backend.hotspot_verify_calls
    assert all(call == hotspot_params for call in backend.hotspot_verify_calls)


def test_esmfold2_pipeline_propagates_hotspot_to_continuous_and_final_polish():
    hotspot_params = {"target_hotspot_concat_indices": [0]}
    continuous_backend = FakeESMFold2Backend()

    run_esmfold2_mcts_hallucination(
        backend=continuous_backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=1,
        num_beams=1,
        init_binder_seq="AAA",
        B_candidates=1,
        mcts_strategy="continuous_stage_a",
        hotspot_params=hotspot_params,
    )
    assert continuous_backend.continuous_calls[0]["hotspot_params"] == hotspot_params

    polish_backend = FakeESMFold2Backend()
    run_esmfold2_mcts_hallucination(
        backend=polish_backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=1,
        num_beams=1,
        init_binder_seq="AAA",
        B_candidates=1,
        mcts_strategy="stage_a_only",
        esmfold2_final_polish=True,
        hotspot_params=hotspot_params,
    )
    assert polish_backend.continuous_calls[0]["hotspot_params"] == hotspot_params


def test_esmfold2_pipeline_rejects_positive_backend_weight_without_hotspots():
    backend = FakeESMFold2Backend()
    backend.config = type("Config", (), {"hotspot_weight": 0.5})()

    with pytest.raises(
        ValueError,
        match="positive ESMFold2 hotspot_weight requires target hotspot indices",
    ):
        run_esmfold2_mcts_hallucination(
            backend=backend,
            target_seq="TTTT",
            L_binder=3,
            N_steps=1,
            num_beams=1,
            init_binder_seq="AAA",
        )


def test_esmfold2_pipeline_rejects_positive_hotspot_verify_weight_without_hotspots():
    backend = FakeESMFold2Backend()
    backend.config = type(
        "Config",
        (),
        {"hotspot_weight": 0.0, "hotspot_verify_weight": 0.2},
    )()

    with pytest.raises(
        ValueError,
        match="positive ESMFold2 hotspot_verify_weight requires target hotspot indices",
    ):
        run_esmfold2_mcts_hallucination(
            backend=backend,
            target_seq="TTTT",
            L_binder=3,
            N_steps=1,
            num_beams=1,
            init_binder_seq="AAA",
        )


def test_esmfold2_official_free_roots_keep_unanchored_mutable_logits():
    backend = FakeESMFold2Backend()

    run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=150,
        num_beams=1,
        init_binder_seq=None,
        K_top=4,
        B_candidates=2,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="continuous_stage_a",
        esmfold2_logit_init_mode="official_free",
    )

    assert len(backend.continuous_calls) == 1
    current_seq = backend.continuous_calls[0]["current_seq"]
    proposal_logits = backend.continuous_calls[0]["proposal_logits"]
    assert proposal_logits is not None
    assert len(current_seq) == 3
    for pos, aa in enumerate(current_seq):
        aa_idx = AA_TO_ESM_IDX[aa]
        assert proposal_logits[pos, aa_idx].item() < 0.5
        assert torch.argmax(proposal_logits[pos]).item() == aa_idx


def test_esmfold2_official_exact_roots_use_direct_torch_seed():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        expected = torch.randn((1, 3, len(ESM_AA_ORDER))) * 0.01
        expected[:, :, AA_TO_ESM_IDX["C"]] = float("-inf")
    expected_logits = expected[0]
    expected_seq = "".join(
        ESM_AA_ORDER[int(torch.argmax(expected_logits[pos]).item())]
        for pos in range(3)
    )

    states = _make_phase0_seed_states(
        None,
        num_beams=1,
        L_binder=3,
        seed=7,
        mutable_mask=[True, True, True],
        logit_init_mode="official_exact",
    )

    seq, logits = states[0]
    assert seq == expected_seq
    assert torch.allclose(logits, expected_logits, equal_nan=False)


def test_esmfold2_official_free_scaffold_roots_anchor_only_fixed_positions():
    class SameLengthContinuousBackend(FakeESMFold2Backend):
        def continuous_design(self, **kwargs):
            self.continuous_calls.append(
                {
                    "current_seq": kwargs["current_seq"],
                    "target_seq": kwargs["target_seq"],
                    "global_step": kwargs["global_step"],
                    "seed": kwargs.get("seed"),
                    "mutable_mask": kwargs.get("mutable_mask"),
                    "proposal_logits": kwargs["proposal_logits"].detach().clone()
                    if kwargs.get("proposal_logits") is not None else None,
                }
            )
            return ESMFold2ProposalBatch(sequences=[kwargs["current_seq"]])

    backend = SameLengthContinuousBackend()

    run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=4,
        N_steps=150,
        num_beams=1,
        init_binder_seq="AAAA",
        K_top=4,
        B_candidates=2,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mutable_mask=[False, True, True, False],
        mcts_strategy="continuous_stage_a",
        esmfold2_logit_init_mode="official_free",
    )

    assert len(backend.continuous_calls) == 1
    current_seq = backend.continuous_calls[0]["current_seq"]
    proposal_logits = backend.continuous_calls[0]["proposal_logits"]
    assert proposal_logits is not None
    assert current_seq[0] == "A"
    assert current_seq[3] == "A"

    fixed_idx = AA_TO_ESM_IDX["A"]
    assert proposal_logits[0, fixed_idx].item() >= 9.0
    assert proposal_logits[3, fixed_idx].item() >= 9.0
    for pos in (1, 2):
        aa_idx = AA_TO_ESM_IDX[current_seq[pos]]
        assert torch.argmax(proposal_logits[pos]).item() == aa_idx
        assert proposal_logits[pos, aa_idx].item() < 0.5


def test_esmfold2_continuous_stagea_verifies_tracked_confidence_candidates():
    class ConfidenceContinuousBackend(FakeESMFold2Backend):
        def continuous_design(self, **kwargs):
            super().continuous_design(**kwargs)
            return ESMFold2ProposalBatch(
                sequences=["AAF"],
                confidence_candidates=[
                    {
                        "sequence": "AAY",
                        "local_step": 140,
                        "confidence_score": 0.80,
                        "confidence_source": "iptm",
                    },
                    {
                        "sequence": "AYY",
                        "local_step": 145,
                        "confidence_score": 0.70,
                        "confidence_source": "iptm",
                    },
                ],
            )

    backend = ConfidenceContinuousBackend()

    best_seq, final_seq, losses = run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=150,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=2,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="continuous_stage_a",
    )

    assert best_seq == "AYY"
    assert final_seq == "AYY"
    assert losses == [pytest.approx(0.10)]
    assert [call[0] for call in backend.verify_calls] == ["AAA", "AAY", "AYY", "AAF"]


def test_esmfold2_stagea_ignores_tracked_confidence_candidate_during_normal_search():
    class ConfidenceProposalBackend(FakeESMFold2Backend):
        def propose(self, **kwargs):
            super().propose(**kwargs)
            return ESMFold2ProposalBatch(
                sequences=["AAF"],
                confidence_candidates=[
                    {
                        "sequence": "AAY",
                        "local_step": 4,
                        "confidence_score": 0.80,
                        "confidence_source": "iptm",
                    }
                ],
            )

    backend = ConfidenceProposalBackend()

    best_seq, final_seq, losses = run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=1,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=1,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="stage_a_only",
    )

    assert best_seq == "AAF"
    assert final_seq == "AAF"
    assert losses == [pytest.approx(0.60)]
    assert [call[0] for call in backend.verify_calls] == ["AAA", "AAF"]
    assert backend.propose_calls[0]["track_confidence_override"] is False


def test_esmfold2_stagea_does_not_use_tracked_confidence_candidates_without_final_polish():
    class ManyConfidenceProposalBackend(FakeESMFold2Backend):
        def propose(self, **kwargs):
            super().propose(**kwargs)
            return ESMFold2ProposalBatch(
                sequences=[],
                confidence_candidates=[
                    {
                        "sequence": "AAY",
                        "local_step": 10,
                        "confidence_score": 0.80,
                        "confidence_source": "iptm",
                    },
                    {
                        "sequence": "AAF",
                        "local_step": 11,
                        "confidence_score": 0.70,
                        "confidence_source": "iptm",
                    },
                    {
                        "sequence": "AYY",
                        "local_step": 12,
                        "confidence_score": 0.60,
                        "confidence_source": "iptm",
                    },
                ],
            )

    backend = ManyConfidenceProposalBackend()

    best_seq, final_seq, losses = run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=1,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=1,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="stage_a_only",
    )

    assert best_seq == "AAA"
    assert final_seq == "AAA"
    assert losses == [pytest.approx(0.90)]
    assert [call[0] for call in backend.verify_calls] == ["AAA"]
    assert backend.propose_calls[0]["track_confidence_override"] is False


def test_esmfold2_final_polish_is_the_only_stage_using_tracked_confidence_candidates():
    class ConfidenceProposalAndPolishBackend(FakeESMFold2Backend):
        def propose(self, **kwargs):
            super().propose(**kwargs)
            return ESMFold2ProposalBatch(
                sequences=["AAF"],
                confidence_candidates=[
                    {
                        "sequence": "AYY",
                        "local_step": 28,
                        "confidence_score": 0.90,
                        "confidence_source": "pair_chain_iptm",
                    }
                ],
            )

        def continuous_design(self, **kwargs):
            super().continuous_design(**kwargs)
            return ESMFold2ProposalBatch(
                sequences=["AAF"],
                confidence_candidates=[
                    {
                        "sequence": "AYY",
                        "local_step": 28,
                        "confidence_score": 0.90,
                        "confidence_source": "pair_chain_iptm",
                    }
                ],
            )

    backend = ConfidenceProposalAndPolishBackend()

    best_seq, final_seq, losses = run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=1,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=1,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="stage_a_only",
        esmfold2_final_polish=True,
    )

    assert best_seq == "AYY"
    assert final_seq == "AYY"
    assert losses == [pytest.approx(0.60), pytest.approx(0.10)]
    assert [call[0] for call in backend.verify_calls] == ["AAA", "AAF", "AYY"]
    assert backend.propose_calls[0]["track_confidence_override"] is False
    assert backend.continuous_calls[0]["current_seq"] == "AAF"
    assert backend.continuous_calls[0]["track_confidence_override"] is True


def test_esmfold2_final_polish_keeps_global_best_when_polish_is_worse(tmp_path):
    class WorsePolishBackend(FakeESMFold2Backend):
        def continuous_design(self, **kwargs):
            super().continuous_design(**kwargs)
            return ESMFold2ProposalBatch(
                sequences=["AAF"],
                confidence_candidates=[
                    {
                        "sequence": "AAF",
                        "local_step": 28,
                        "confidence_score": 0.40,
                        "confidence_source": "pair_chain_iptm",
                    }
                ],
            )

    backend = WorsePolishBackend()
    artifact_dir = tmp_path / "final_polish_worse"

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
        esmfold2_final_polish=True,
        artifact_dir=artifact_dir,
    )

    assert best_seq == "AYY"
    assert final_seq == "AYY"
    assert losses[-1] == pytest.approx(0.10)

    summary = json.loads((artifact_dir / "run_summary.json").read_text())
    assert summary["best_sequence"] == "AYY"
    assert summary["final_sequence"] == "AYY"

    trajectory = [json.loads(line) for line in (artifact_dir / "trajectory.jsonl").read_text().splitlines()]
    final_polish = trajectory[-1]
    assert final_polish["stage"] == "final_polish"
    assert final_polish["selected_sequence"] == "AYY"
    assert final_polish["round_best_loss"] == pytest.approx(0.60)
    assert final_polish["global_best_loss"] == pytest.approx(0.10)
    assert final_polish["best_sequence"] == "AYY"


def test_esmfold2_final_polish_starts_from_global_best_logits():
    class GlobalBestLogitsBackend(FakeESMFold2Backend):
        def propose(self, **kwargs):
            proposal_logits = kwargs["proposal_logits"]
            updated = proposal_logits.detach().clone()
            updated[1, AA_TO_ESM_IDX["F"]] = 7.0
            updated[2, AA_TO_ESM_IDX["Y"]] = 5.0
            self.propose_calls.append(
                {
                    "current_seq": kwargs["current_seq"],
                    "proposal_logits": proposal_logits.detach().clone(),
                    "track_confidence_override": kwargs.get("track_confidence_override"),
                }
            )
            return ESMFold2ProposalBatch(
                sequences=["AYY"],
                logits_after=updated,
            )

        def continuous_design(self, **kwargs):
            proposal_logits = kwargs["proposal_logits"]
            self.continuous_calls.append(
                {
                    "current_seq": kwargs["current_seq"],
                    "proposal_logits": proposal_logits.detach().clone(),
                    "track_confidence_override": kwargs.get("track_confidence_override"),
                }
            )
            return ESMFold2ProposalBatch(
                sequences=["AYY"],
                logits_after=proposal_logits.detach().clone(),
            )

    backend = GlobalBestLogitsBackend()

    best_seq, final_seq, losses = run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=1,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=1,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="stage_a_only",
        esmfold2_final_polish=True,
    )

    assert best_seq == final_seq == "AYY"
    assert losses[-1] == pytest.approx(0.10)
    polish_logits = backend.continuous_calls[0]["proposal_logits"]
    assert polish_logits[1, AA_TO_ESM_IDX["F"]].item() == pytest.approx(7.0)
    assert polish_logits[2, AA_TO_ESM_IDX["Y"]].item() >= 5.0


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
    assert backend.preview_calls == []


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


def test_esmfold2_two_stage_uses_separate_stagea_and_stageb_soft_steps():
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
        seed=7,
        mcts_strategy="two_stage",
        mcts_bootstrap_steps=1,
        mcts_bootstrap_max_steps=1,
        mcts_expand_actions_per_step=1,
        esmfold2_stagea_soft_steps=30,
        esmfold2_stageb_soft_steps=5,
    )

    assert backend.propose_calls[0]["action_group"] is None
    assert backend.propose_calls[0]["soft_steps_override"] == 30
    assert backend.propose_calls[1]["action_group"] == "interface_formation"
    assert backend.propose_calls[1]["soft_steps_override"] == 5


def test_esmfold2_pipeline_persists_node_local_logits_between_stagea_visits():
    class StatefulLogitBackend(FakeESMFold2Backend):
        def propose(self, **kwargs):
            proposal_logits = kwargs.get("proposal_logits")
            self.propose_calls.append(
                {
                    "current_seq": kwargs["current_seq"],
                    "proposal_logits": (
                        proposal_logits.detach().clone()
                        if proposal_logits is not None
                        else None
                    ),
                }
            )
            if kwargs["current_seq"] == "AAA":
                assert proposal_logits is not None
                updated = proposal_logits.detach().clone()
                updated[1, AA_TO_ESM_IDX["F"]] = 4.0
                updated[2, AA_TO_ESM_IDX["Y"]] = 3.0
                return ESMFold2ProposalBatch(
                    sequences=["AAY"],
                    logits_after=updated,
                )
            return ESMFold2ProposalBatch(
                sequences=[],
                logits_after=proposal_logits,
            )

    backend = StatefulLogitBackend()

    run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=2,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=1,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="stage_a_only",
    )

    assert [call["current_seq"] for call in backend.propose_calls] == ["AAA", "AAY"]
    second_logits = backend.propose_calls[1]["proposal_logits"]
    assert second_logits is not None
    assert second_logits[1, AA_TO_ESM_IDX["F"]].item() == pytest.approx(4.0)
    assert second_logits[2, AA_TO_ESM_IDX["Y"]].item() >= 3.0


def test_esmfold2_pipeline_remains_compatible_with_list_only_proposal_backend():
    class ListOnlyBackend(FakeESMFold2Backend):
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
                    "action_group": action_group,
                }
            )
            return ["AAY"][:B_candidates] if current_seq == "AAA" else []

    backend = ListOnlyBackend()

    best_seq, final_seq, losses = run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=1,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=1,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        hotspot_params={"target_hotspot_concat_indices": [1]},
    )

    assert best_seq == final_seq == "AAY"
    assert losses[-1] == pytest.approx(0.20)


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
    nodes_path = artifact_dir / "mcts_nodes.jsonl"
    edges_path = artifact_dir / "mcts_edges.jsonl"
    actions_path = artifact_dir / "mcts_actions.jsonl"
    soft_steps_path = artifact_dir / "soft_steps.jsonl"
    handoff_path = artifact_dir / "mcts_handoff.jsonl"
    summary_path = artifact_dir / "run_summary.json"
    assert trajectory_path.exists()
    assert candidates_path.exists()
    assert nodes_path.exists()
    assert edges_path.exists()
    assert actions_path.exists()
    assert soft_steps_path.exists()
    assert handoff_path.exists()
    assert summary_path.exists()

    trajectory = [json.loads(line) for line in trajectory_path.read_text().splitlines()]
    candidates = [json.loads(line) for line in candidates_path.read_text().splitlines()]
    nodes = [json.loads(line) for line in nodes_path.read_text().splitlines()]
    edges = [json.loads(line) for line in edges_path.read_text().splitlines()]
    actions = [json.loads(line) for line in actions_path.read_text().splitlines()]
    soft_steps = [json.loads(line) for line in soft_steps_path.read_text().splitlines()]
    handoff = [json.loads(line) for line in handoff_path.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())

    assert [event["stage"] for event in trajectory] == ["phase0", "stage_a", "stage_b"]
    assert trajectory[1]["candidate_count"] == 2
    assert trajectory[2]["primary_action_group"] == "interface_formation"
    assert trajectory[2]["selected_group_counts"] == {"interface_formation": 1}
    assert soft_steps == []
    assert any(event["stage"] == "stage_b" and event["sequence"] == "AYY" for event in candidates)
    assert candidates[-1]["loss_dict"]["total"] == pytest.approx(0.10)
    assert any(node["stage"] == "stage_a" and node["parent_node_id"] is None for node in nodes)
    assert any(node["stage"] == "stage_b" and node["sequence"] == "AYY" for node in nodes)
    assert any(edge["stage"] == "stage_a" and edge["edits"] for edge in edges)
    assert any(action["stage"] == "stage_b" and action["selected"] for action in actions)
    assert any(event["source_stage"] == "stage_a" and event["target_stage"] == "stage_b" for event in handoff)

    stage_b_ckpt_dir = artifact_dir / "checkpoints" / "stage_b" / "step_0001"
    stage_b_meta = json.loads((stage_b_ckpt_dir / "step_0001_round_best_meta.json").read_text())
    assert (stage_b_ckpt_dir / "step_0001_round_best.cif").exists()
    assert stage_b_meta["binder_seq"] == "AYY"
    assert stage_b_meta["total_loss"] == pytest.approx(0.10)
    assert stage_b_meta["structure_format"] == "mmcif"
    assert "mmcif" not in stage_b_meta["summary_confidence"]
    assert summary["best_sequence"] == "AYY"
    assert summary["best_loss"] == pytest.approx(0.10)
