from dataclasses import dataclass

import pytest
import torch

from bindcraft.hallucination.mcts_controller import MCTSTree, RunningLossNormalizer
from bindcraft.hallucination.mcts_controller import HARD_CONSTRAINT_INVALID_REWARD
from bindcraft.hallucination.mcts_legacy import LegacyMCTSTree
from bindcraft.hallucination.mutation_proposer import (
    AA_1_TO_IDX,
    MutationProposal,
    apply_position_weight_bias,
    apply_action_group_mutation_bias,
    compute_mutation_benefit,
    compute_mutation_benefit_legacy,
    compute_de_novo_action_group_priors,
    extract_action_group_position_weights,
    propose_mutation_actions,
    propose_mutations,
    propose_mutations_legacy,
    normalize_mcts_prior_mode,
    normalize_mcts_proposal_method,
    sample_mutation_actions,
    select_de_novo_action_group,
    should_use_gradient_prior,
    should_use_sequence_prior,
    apply_position_weight_mask,
)
from bindcraft.hallucination.zeroth_order import compute_mcts_structural_metadata
from bindcraft.hallucination.pipeline import (
    _allocate_action_group_budgets,
    _apply_stageb_prior_mode_to_groups,
    _resolve_proposal_amp_dtype,
    _stageb_proposal_tau,
)


@dataclass
class DummyZerothResult:
    binder_seq: str
    total_loss: float
    loss_dict: dict
    coordinate: torch.Tensor
    summary_confidence: dict = None


def _result(sequence: str, total_loss: float, loss_dict: dict = None) -> DummyZerothResult:
    return DummyZerothResult(
        binder_seq=sequence,
        total_loss=total_loss,
        loss_dict=loss_dict or {"total": total_loss, "i_pae": total_loss / 2.0},
        coordinate=torch.zeros(1, len(sequence), 3),
        summary_confidence={},
    )


class TestLegacyMCTSBootstrap:
    def test_legacy_tree_uses_loss_ucb_expand_and_backprop(self):
        tree = LegacyMCTSTree(C=1.0, max_nodes=20)
        root = tree.add_root_node("AAAA", torch.zeros(4, 20), L_real=10.0)

        assert tree.select() is root
        children = tree.expand(root, [_result("AAAC", 5.0), _result("AAAG", 7.0)])
        assert [child.sequence for child in children] == ["AAAC", "AAAG"]
        assert len(root.children) == 2
        assert children[0].depth == 1

        tree.backpropagate(root, [5.0, 7.0])
        assert root.W == pytest.approx(22.0)
        assert root.visit_count == 3
        assert tree.N_total == 3
        assert tree.get_global_best().sequence == "AAAC"

    def test_legacy_mutation_proposal_keeps_plain_sequence_api(self):
        torch.manual_seed(0)
        grad = torch.zeros(4, 20)
        grad[0, AA_1_TO_IDX["S"]] = -2.0
        grad[1, AA_1_TO_IDX["T"]] = -1.5

        candidates = propose_mutations_legacy(
            grad,
            "AAAA",
            K_top=4,
            B_candidates=3,
            T_sample=0.5,
            max_mutations_per_seq=2,
            use_prior=False,
        )

        assert len(candidates) == 3
        assert all(isinstance(seq, str) for seq in candidates)
        assert all(len(seq) == 4 for seq in candidates)

    def test_legacy_and_document_blosum_directions_are_separate(self):
        current_indices = [AA_1_TO_IDX["A"]]
        grad = torch.zeros(1, 20)
        ser = AA_1_TO_IDX["S"]

        legacy = compute_mutation_benefit_legacy(
            grad, current_indices, use_prior=True, prior_strength=0.1
        )
        document = compute_mutation_benefit(
            grad, current_indices, use_prior=True, prior_strength=0.1
        )

        assert -legacy[0, ser].item() < 0.0
        assert -document[0, ser].item() > 0.0


class TestStage1GradientPriorPuct:
    def test_blosum_prior_increases_conservative_substitution_benefit(self):
        current_indices = [AA_1_TO_IDX["A"]]
        grad = torch.zeros(1, 20)

        no_prior = compute_mutation_benefit(
            grad, current_indices, use_prior=False, prior_strength=0.1
        )
        with_prior = compute_mutation_benefit(
            grad, current_indices, use_prior=True, prior_strength=0.1
        )

        ser = AA_1_TO_IDX["S"]  # BLOSUM62 S<-A is positive.
        trp = AA_1_TO_IDX["W"]  # BLOSUM62 W<-A is negative.

        assert -no_prior[0, ser].item() == pytest.approx(0.0)
        assert -with_prior[0, ser].item() > -no_prior[0, ser].item()
        assert -with_prior[0, trp].item() < -no_prior[0, trp].item()
        assert with_prior[0, current_indices[0]].item() == pytest.approx(0.0)

    def test_prior_mode_helpers_preserve_full_default_and_disable_sequence_prior(self):
        assert normalize_mcts_prior_mode(None) == "full"
        assert normalize_mcts_prior_mode("NO_GRADIENT") == "no_gradient"
        assert should_use_gradient_prior("full")
        assert should_use_gradient_prior("no_group")
        assert should_use_gradient_prior("no_sequence")
        assert not should_use_gradient_prior("no_gradient")
        assert not should_use_gradient_prior("uniform")
        assert should_use_sequence_prior("full", True)
        assert not should_use_sequence_prior("no_sequence", True)
        assert not should_use_sequence_prior("uniform", True)
        assert not should_use_sequence_prior("full", False)
        with pytest.raises(ValueError):
            normalize_mcts_prior_mode("bad_mode")

    def test_proposal_method_helper_preserves_current_default(self):
        assert normalize_mcts_proposal_method(None) == "one_step_fixed"
        assert normalize_mcts_proposal_method("ONE_STEP_FIXED") == "one_step_fixed"
        assert normalize_mcts_proposal_method("one_step_adaptive") == "one_step_adaptive"
        assert normalize_mcts_proposal_method("two_step_truncated") == "two_step_truncated"
        with pytest.raises(ValueError):
            normalize_mcts_proposal_method("bad_method")

    def test_stageb_adaptive_tau_anneals_from_high_noise_to_positive_floor(self):
        assert _stageb_proposal_tau(
            "one_step_fixed", 4.0, local_step=0, total_steps=120
        ) == pytest.approx(4.0)
        assert _stageb_proposal_tau(
            "two_step_truncated", 4.0, local_step=0, total_steps=120
        ) == pytest.approx(4.0)

        start_tau = _stageb_proposal_tau(
            "one_step_adaptive",
            4.0,
            local_step=0,
            total_steps=120,
            adaptive_tau_start=160.0,
            adaptive_tau_end=0.1,
        )
        mid_tau = _stageb_proposal_tau(
            "one_step_adaptive",
            4.0,
            local_step=60,
            total_steps=120,
            adaptive_tau_start=160.0,
            adaptive_tau_end=0.1,
        )
        end_tau = _stageb_proposal_tau(
            "one_step_adaptive",
            4.0,
            local_step=119,
            total_steps=120,
            adaptive_tau_start=160.0,
            adaptive_tau_end=0.1,
        )

        assert start_tau == pytest.approx(160.0)
        assert 0.1 < mid_tau < start_tau
        assert end_tau == pytest.approx(0.1)

    def test_proposal_amp_auto_is_v2_only(self):
        assert _resolve_proposal_amp_dtype("protenix-v2", "auto") is torch.bfloat16
        assert _resolve_proposal_amp_dtype("protenix_base_default_v1.0.0", "auto") is None
        assert _resolve_proposal_amp_dtype("protenix-v2", "off") is None
        assert _resolve_proposal_amp_dtype("protenix_base_default_v1.0.0", "bf16") is torch.bfloat16
        with pytest.raises(ValueError):
            _resolve_proposal_amp_dtype("protenix-v2", "fp16")

    def test_no_gradient_prior_keeps_actions_uniform_within_group(self):
        torch.manual_seed(0)
        current_indices = [AA_1_TO_IDX["A"], AA_1_TO_IDX["A"], AA_1_TO_IDX["A"]]
        positions = torch.tensor([0, 1, 2], dtype=torch.long)
        new_aas = torch.tensor(
            [AA_1_TO_IDX["S"], AA_1_TO_IDX["T"], AA_1_TO_IDX["G"]],
            dtype=torch.long,
        )
        benefits = torch.tensor([10.0, 1.0, 0.1])

        proposals = sample_mutation_actions(
            current_indices,
            positions,
            new_aas,
            benefits,
            B_candidates=3,
            T_sample=1.0,
            max_mutations_per_seq=1,
            group_prior=0.6,
            prior_mode="no_gradient",
        )

        assert proposals
        expected = 0.6 / len(proposals)
        assert all(p.prior == pytest.approx(expected) for p in proposals)

    def test_no_gradient_proposal_ignores_gradient_matrix(self):
        seq = "AAAA"
        grad_a = torch.zeros(4, 20)
        grad_a[0, AA_1_TO_IDX["S"]] = -10.0
        grad_b = torch.zeros(4, 20)
        grad_b[1, AA_1_TO_IDX["T"]] = -10.0

        torch.manual_seed(7)
        proposals_a = propose_mutation_actions(
            grad_a,
            seq,
            K_top=4,
            B_candidates=6,
            T_sample=0.5,
            max_mutations_per_seq=1,
            use_prior=False,
            prior_mode="no_gradient",
            action_group="unrouted",
            group_prior=1.0,
        )
        torch.manual_seed(7)
        proposals_b = propose_mutation_actions(
            grad_b,
            seq,
            K_top=4,
            B_candidates=6,
            T_sample=0.5,
            max_mutations_per_seq=1,
            use_prior=False,
            prior_mode="no_gradient",
            action_group="unrouted",
            group_prior=1.0,
        )

        assert [(p.sequence, p.edits) for p in proposals_a] == [
            (p.sequence, p.edits) for p in proposals_b
        ]

    def test_uniform_proposal_samples_beyond_gradient_top_k(self):
        torch.manual_seed(3)
        grad = torch.zeros(4, 20)
        grad[0, AA_1_TO_IDX["S"]] = -10.0

        proposals = propose_mutation_actions(
            grad,
            "AAAA",
            K_top=1,
            B_candidates=12,
            T_sample=1.0,
            max_mutations_per_seq=1,
            use_prior=True,
            prior_mode="uniform",
            action_group="unrouted",
            group_prior=1.0,
        )

        assert len({p.sequence for p in proposals}) > 1
        assert any(p.sequence != "SAAA" for p in proposals)
        assert all(p.prior == pytest.approx(1.0 / len(proposals)) for p in proposals)

    def test_full_proposal_still_uses_gradient_top_k(self):
        torch.manual_seed(0)
        grad = torch.zeros(4, 20)
        grad[0, AA_1_TO_IDX["S"]] = -10.0

        proposals = propose_mutation_actions(
            grad,
            "AAAA",
            K_top=1,
            B_candidates=4,
            T_sample=0.2,
            max_mutations_per_seq=1,
            use_prior=False,
            prior_mode="full",
            action_group="unrouted",
            group_prior=1.0,
        )

        assert proposals
        assert {p.sequence for p in proposals} == {"SAAA"}

    def test_mcts_tree_puct_reward_backup_dedup_and_global_best(self):
        logits = torch.zeros(4, 20, requires_grad=True)
        coordinate = torch.ones(1, 4, 3, requires_grad=True)
        tree = MCTSTree(
            C=1.0,
            max_nodes=20,
            max_children_per_node=2,
            use_reward_normalization=False,
        )

        root = tree.add_root_node(
            sequence="AAAA",
            logits=logits,
            coordinate=coordinate,
            L_real=10.0,
            loss_vector={"total": 10.0},
            tree_reward=-10.0,
            prior=1.0,
        )

        assert root.logits.device.type == "cpu"
        assert root.coordinate.device.type == "cpu"
        assert not root.logits.requires_grad
        assert not root.coordinate.requires_grad
        assert tree.has_sequence("AAAA")
        assert tree.can_expand(root)
        assert tree.remaining_expansion_slots(root) == 2
        assert tree.select() is root

        proposals = [
            MutationProposal("AAAS", [(3, "A", "S")], prior=0.01, benefit=1.0),
            MutationProposal("AAAT", [(3, "A", "T")], prior=1.0, benefit=0.5),
            MutationProposal("AAAA", [], prior=0.5, benefit=0.0),
        ]
        results = [
            _result("AAAS", 5.0),
            _result("AAAT", 6.0),
            _result("AAAA", 4.0),
        ]

        children = tree.expand(root, proposals, results)
        assert [child.sequence for child in children] == ["AAAS", "AAAT"]
        assert tree.total_nodes == 3
        assert len(root.children) == 2
        assert not tree.can_expand(root)
        assert tree.remaining_expansion_slots(root) == 0
        assert tree.has_sequence("AAAS")
        assert tree.has_sequence("AAAT")
        assert children[0].status_flags["action_group"] == "unrouted"
        assert children[0].status_flags["proposal_prior"] == pytest.approx(0.01)
        assert children[1].status_flags["group_prior"] == pytest.approx(1.0)

        root.visit_count = 100
        selected = tree.select()
        assert selected.sequence == "AAAT"

        before_q = root.q_value
        tree.backpropagate(root, [-5.0, -6.0])
        assert root.visit_count == 102
        assert root.value_sum == pytest.approx(-21.0)
        assert root.q_value < before_q
        assert tree.N_total == 3

        best = tree.get_global_best()
        assert best.sequence == "AAAS"
        assert best.tree_reward == pytest.approx(-5.0)

    def test_mcts_progressive_widening_caps_width_but_does_not_block_descent(self):
        tree = MCTSTree(
            C=1.0,
            max_nodes=20,
            progressive_widening_k=2.0,
            progressive_widening_alpha=0.0,
            max_children_per_node=2,
            use_reward_normalization=False,
        )
        root = tree.add_root_node(
            sequence="AAAA",
            logits=torch.zeros(4, 20),
            L_real=10.0,
            loss_vector={"total": 10.0},
            tree_reward=-10.0,
            prior=1.0,
        )

        tree.expand(
            root,
            [MutationProposal("AAAC", [(3, "A", "C")], prior=1.0, benefit=1.0)],
            [_result("AAAC", 1.0)],
        )
        assert tree.can_expand(root)
        assert len(root.children) == 1
        assert tree.select().sequence == "AAAC"

        tree.expand(
            root,
            [MutationProposal("AAAG", [(3, "A", "G")], prior=0.5, benefit=0.5)],
            [_result("AAAG", 2.0)],
        )
        assert not tree.can_expand(root)
        assert len(root.children) == 2
        assert tree.select().sequence == "AAAC"

    def test_mcts_select_can_descend_after_one_verified_child(self):
        tree = MCTSTree(
            C=1.0,
            max_nodes=30,
            progressive_widening_k=10.0,
            progressive_widening_alpha=0.0,
            max_children_per_node=10,
            min_children_before_descend=4,
            use_reward_normalization=False,
        )
        root = tree.add_root_node(
            sequence="AAAA",
            logits=torch.zeros(4, 20),
            L_real=10.0,
            loss_vector={"total": 10.0},
            tree_reward=-10.0,
            prior=1.0,
        )

        tree.expand(
            root,
            [MutationProposal("AAAC", [(3, "A", "C")], prior=1.0, benefit=1.0)],
            [_result("AAAC", 1.0)],
        )

        assert tree.can_expand(root)
        assert len(root.children) == 1
        decision = tree.select_decision()
        assert decision.decision_type == "descend_expand_frontier"
        assert decision.node.sequence == "AAAC"
        assert decision.best_action_score is not None
        assert decision.frontier_score is not None
        assert decision.best_action_score >= decision.frontier_score

    def test_mcts_select_can_descend_before_widening_limit_after_multiple_children(self):
        tree = MCTSTree(
            C=1.0,
            max_nodes=30,
            progressive_widening_k=10.0,
            progressive_widening_alpha=0.0,
            max_children_per_node=10,
            min_children_before_descend=2,
            use_reward_normalization=False,
        )
        root = tree.add_root_node(
            sequence="AAAA",
            logits=torch.zeros(4, 20),
            L_real=10.0,
            loss_vector={"total": 10.0},
            tree_reward=-10.0,
            prior=1.0,
        )

        tree.expand(
            root,
            [
                MutationProposal("AAAC", [(3, "A", "C")], prior=1.0, benefit=1.0),
                MutationProposal("AAAG", [(3, "A", "G")], prior=0.5, benefit=0.5),
            ],
            [_result("AAAC", 1.0), _result("AAAG", 2.0)],
        )

        assert tree.can_expand(root)
        decision = tree.select_decision()
        assert decision.decision_type == "descend_expand_frontier"
        assert decision.node.sequence == "AAAC"

    def test_action_level_puct_selects_fresh_unexpanded_action_and_backs_up_edge(self):
        tree = MCTSTree(
            C=1.0,
            max_nodes=20,
            max_children_per_node=4,
            expand_actions_per_step=1,
            use_reward_normalization=False,
        )
        root = tree.add_root_node(
            sequence="AAAA",
            logits=torch.zeros(4, 20),
            L_real=10.0,
            loss_vector={"total": 10.0},
            tree_reward=-10.0,
            prior=1.0,
        )
        proposals = [
            MutationProposal("AAAC", [(3, "A", "C")], prior=0.9, benefit=1.0),
            MutationProposal("AAAG", [(3, "A", "G")], prior=0.1, benefit=0.2),
        ]

        actions = tree.register_action_proposals(root, proposals)
        assert len(actions) == 2
        decision = tree.select_decision()
        assert decision.node is root
        assert decision.action is None
        assert decision.decision_type == "expand_frontier"

        selected_actions = tree.select_unexpanded_actions(
            root,
            max_actions=1,
            candidates=actions,
        )
        assert len(selected_actions) == 1
        assert selected_actions[0].sequence == "AAAC"

        children = tree.expand(root, [selected_actions[0].proposal], [_result("AAAC", 4.0)])
        assert len(children) == 1
        tree.backpropagate_action(decision, selected_actions[0], children[0].tree_reward)

        assert selected_actions[0].status == "expanded"
        assert selected_actions[0].child_node_id == children[0].node_id
        assert selected_actions[0].visit_count == 1
        assert selected_actions[0].q_value == pytest.approx(children[0].tree_reward)
        assert tree.get_node_stats()["num_actions"] == 2
        assert tree.get_node_stats()["num_expanded_actions"] == 1
        assert tree.get_node_stats()["num_unexpanded_actions"] == 1

    def test_mcts_expand_old_interface_remains_compatible(self):
        tree = MCTSTree(C=1.0, use_reward_normalization=False)
        root = tree.add_root_node("AAAA", torch.zeros(4, 20), L_real=10.0)

        children = tree.expand(root, [_result("AAAC", 7.0)])

        assert len(children) == 1
        assert children[0].sequence == "AAAC"
        assert children[0].prior == pytest.approx(1.0)


class TestStage2ActionRouting:
    def test_mcts_mutation_proposals_include_metadata_and_keep_legacy_api(self):
        torch.manual_seed(0)
        grad = torch.zeros(4, 20)
        grad[0, AA_1_TO_IDX["S"]] = -2.0
        grad[1, AA_1_TO_IDX["T"]] = -1.5
        grad[2, AA_1_TO_IDX["V"]] = -1.0
        grad[3, AA_1_TO_IDX["G"]] = -0.5

        proposals = propose_mutation_actions(
            grad,
            "AAAA",
            K_top=4,
            B_candidates=3,
            T_sample=0.5,
            max_mutations_per_seq=2,
            use_prior=False,
            action_group="interface_formation",
            group_prior=0.25,
        )

        assert len(proposals) == 3
        assert all(isinstance(p, MutationProposal) for p in proposals)
        assert all(isinstance(p.sequence, str) and len(p.sequence) == 4 for p in proposals)
        assert all(p.edits for p in proposals)
        assert all(p.action_group == "interface_formation" for p in proposals)
        assert all(p.group_prior == pytest.approx(0.25) for p in proposals)
        assert sum(p.prior for p in proposals) == pytest.approx(0.25)

        torch.manual_seed(0)
        legacy = propose_mutations(
            grad,
            "AAAA",
            K_top=4,
            B_candidates=3,
            T_sample=0.5,
            max_mutations_per_seq=2,
            use_prior=False,
        )
        assert len(legacy) == 3
        assert all(isinstance(seq, str) for seq in legacy)

    def test_de_novo_action_group_routing_prefers_largest_residual_group(self):
        interface_losses = {
            "plddt": 0.1,
            "pae": 0.2,
            "con": 0.2,
            "i_pae": 5.0,
            "i_pde": 2.0,
            "i_con": 3.0,
            "clash": 0.0,
            "i_clash": 0.0,
        }
        priors = compute_de_novo_action_group_priors(interface_losses)
        selected = select_de_novo_action_group(interface_losses)

        assert selected["action_group"] == "interface_formation"
        assert selected["group_prior"] == pytest.approx(priors["interface_formation"])
        assert sum(priors.values()) == pytest.approx(1.0)

        clash_losses = {
            "plddt": 0.1,
            "pae": 0.2,
            "con": 0.2,
            "i_pae": 0.2,
            "i_pde": 0.1,
            "i_con": 0.1,
            "clash": 4.0,
            "i_clash": 3.0,
        }
        assert select_de_novo_action_group(clash_losses)["action_group"] == "clash_repair"

    def test_action_group_routing_uses_component_scale_normalization(self):
        losses = {
            "plddt": 0.20,
            "pae": 0.10,
            "con": 0.0,
            "i_pae": 0.0,
            "i_pde": 0.0,
            "i_con": 1.0,
            "clash": 0.0,
            "i_clash": 0.0,
        }

        selected = select_de_novo_action_group(losses)

        assert selected["action_group"] == "interface_formation"
        assert selected["group_priors"]["interface_formation"] > selected["group_priors"]["core_stabilization"]

    def test_action_group_routing_uses_loss_weights_and_group_mean(self):
        losses = {
            "plddt": 0.20,
            "pae": 0.10,
            "con": 0.50,
            "pde": 0.03,
            "i_pae": 0.10,
            "i_con": 0.20,
            "i_pde": 0.05,
        }
        loss_weights = {
            "plddt": 1.0,
            "pae": 1.0,
            "con": 1.0,
            "pde": 1.0,
            "i_pae": 1.0,
            "i_con": 3.0,
            "i_pde": 1.0,
        }

        selected = select_de_novo_action_group(losses, loss_weights=loss_weights)

        assert selected["action_group"] == "interface_formation"
        assert selected["group_priors"]["interface_formation"] > selected["group_priors"]["core_stabilization"]

    def test_action_group_routing_uses_current_normalized_loss_scales(self):
        losses = {
            "plddt": 0.20,
            "pae": 0.10,
            "con": 1.0,
            "pde": 0.03,
            "i_pae": 0.60,
            "i_con": 0.04,
            "i_pde": 0.20,
            "clash": 0.0,
            "i_clash": 0.0,
        }
        tuned_weights = {
            "plddt": 0.1,
            "pae": 0.4,
            "con": 0.2,
            "pde": 0.05,
            "i_pae": 0.2,
            "i_con": 3.0,
            "i_pde": 0.03,
            "clash": 0.1,
            "i_clash": 0.1,
        }

        selected = select_de_novo_action_group(losses, loss_weights=tuned_weights)

        assert selected["action_group"] == "interface_formation"
        assert selected["group_priors"]["interface_formation"] > selected["group_priors"]["core_stabilization"]

    def test_rg_routing_normalizes_by_binder_length(self):
        losses = {
            "rg": 1.0,
            "i_pae": 0.10,
            "i_con": 0.08,
            "i_pde": 0.10,
        }
        tuned_weights = {
            "rg": 0.3,
            "i_pae": 0.2,
            "i_con": 3.0,
            "i_pde": 0.03,
        }

        selected = select_de_novo_action_group(
            losses,
            loss_weights=tuned_weights,
            binder_length=80,
        )

        assert selected["action_group"] == "interface_formation"
        assert selected["group_priors"]["interface_formation"] > selected["group_priors"]["compactness_repair"]

    def test_action_group_routing_drops_zero_score_groups(self):
        losses = {
            "plddt": 0.2,
            "pae": 0.1,
            "con": 0.5,
            "i_pae": 0.0,
            "i_con": 0.0,
            "i_pde": 0.0,
            "clash": 0.0,
            "i_clash": 0.0,
        }
        loss_weights = {
            "plddt": 1.0,
            "pae": 1.0,
            "con": 1.0,
            "i_pae": 1.0,
            "i_con": 1.0,
            "i_pde": 1.0,
            "clash": 1.0,
            "i_clash": 1.0,
        }

        priors = compute_de_novo_action_group_priors(losses, loss_weights=loss_weights)

        assert "clash_repair" not in priors
        assert "interface_formation" not in priors
        assert set(priors) == {"core_stabilization"}

    def test_action_group_routing_skips_zero_weight_reserved_groups(self):
        losses = {
            "plddt": 0.2,
            "i_con": 0.4,
            "specificity": 50.0,
            "solubility": 50.0,
        }
        loss_weights = {
            "plddt": 0.1,
            "i_con": 3.0,
            "specificity": 0.0,
            "solubility": 0.0,
        }

        priors = compute_de_novo_action_group_priors(losses, loss_weights=loss_weights)
        selected = select_de_novo_action_group(losses, loss_weights=loss_weights)

        assert "specificity_repair" not in priors
        assert "solubility_repair" not in priors
        assert selected["action_group"] == "interface_formation"

    def test_no_group_and_uniform_prior_modes_route_only_unrouted_actions(self):
        for prior_mode in ("no_group", "uniform"):
            action_group, group_priors = _apply_stageb_prior_mode_to_groups(
                prior_mode,
                "interface_formation",
                {"interface_formation": 0.9, "core_stabilization": 0.1},
            )

            assert action_group == "unrouted"
            assert group_priors == {"unrouted": 1.0}
            assert _allocate_action_group_budgets(group_priors, 4) == {"unrouted": 4}

    def test_action_group_bias_changes_mutation_search_space(self):
        current_indices = [AA_1_TO_IDX["F"], AA_1_TO_IDX["A"]]
        delta = torch.zeros(2, 20)

        unrouted, unrouted_stats = apply_action_group_mutation_bias(
            delta,
            current_indices,
            action_group=None,
        )
        assert torch.equal(unrouted, delta)
        assert unrouted_stats["routing_bias_max"] == pytest.approx(0.0)

        clash_delta, clash_stats = apply_action_group_mutation_bias(
            delta,
            current_indices,
            action_group="clash_repair",
            strength=0.25,
        )
        gly = AA_1_TO_IDX["G"]
        trp = AA_1_TO_IDX["W"]

        assert clash_stats["routing_bias_max"] > 0.0
        assert -clash_delta[0, gly].item() > -clash_delta[0, trp].item()
        assert -clash_delta[0, gly].item() > -clash_delta[1, gly].item()

    def test_structural_position_weights_bias_group_specific_routing(self):
        weights = extract_action_group_position_weights(
            "AAAA",
            "interface_formation",
            structural_signature={"target_facing_residue_indices": [3]},
        )
        delta = torch.zeros(4, 20)
        adjusted, stats = apply_position_weight_bias(delta, weights, strength=1.0)

        ser = AA_1_TO_IDX["S"]
        assert weights.tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0])
        assert stats["position_bias_max"] == pytest.approx(1.0)
        assert -adjusted[3, ser].item() > -adjusted[0, ser].item()

    def test_structural_position_mask_restricts_group_specific_routing(self):
        weights = extract_action_group_position_weights(
            "AAAA",
            "interface_formation",
            structural_signature={"target_facing_residue_indices": [3]},
        )
        delta = torch.zeros(4, 20)
        masked, stats = apply_position_weight_mask(delta, weights)

        assert stats["position_mask_active"] == pytest.approx(1.0)
        assert masked[0, AA_1_TO_IDX["S"]].item() > 1e5
        assert masked[3, AA_1_TO_IDX["S"]].item() == pytest.approx(0.0)

    def test_mcts_proposals_use_structural_position_weights_when_available(self):
        grad = torch.zeros(4, 20)
        grad[0, AA_1_TO_IDX["S"]] = -0.4
        grad[3, AA_1_TO_IDX["T"]] = -0.1

        proposals = propose_mutation_actions(
            grad,
            "AAAA",
            K_top=1,
            B_candidates=1,
            T_sample=0.01,
            max_mutations_per_seq=1,
            use_prior=False,
            action_group="interface_formation",
            structural_signature={"target_facing_residue_indices": [3]},
            position_bias_strength=1.0,
            routing_strength=0.0,
        )

        assert len(proposals) == 1
        assert proposals[0].edits[0][0] == 3


class TestStage3LiteGrouping:
    def test_loss_bin_grouping_limits_representatives_and_replaces_weak_leaf(self):
        tree = MCTSTree(
            C=1.0,
            max_nodes=20,
            max_children_per_node=10,
            representatives_per_group=2,
            loss_group_bin_width=10.0,
            use_reward_normalization=False,
        )
        root = tree.add_root_node("AAAA", torch.zeros(4, 20), L_real=20.0)
        proposals = [
            MutationProposal(
                "AAAS",
                [(3, "A", "S")],
                prior=0.3,
                benefit=1.0,
                action_group="interface_formation",
            ),
            MutationProposal(
                "AAAT",
                [(3, "A", "T")],
                prior=0.3,
                benefit=1.0,
                action_group="interface_formation",
            ),
            MutationProposal(
                "AAAV",
                [(3, "A", "V")],
                prior=0.3,
                benefit=1.0,
                action_group="interface_formation",
            ),
        ]
        same_group_losses = {"total": 5.0, "i_pae": 1.0}

        children = tree.expand(
            root,
            proposals,
            [
                _result("AAAS", 5.0, same_group_losses),
                _result("AAAT", 6.0, same_group_losses),
                _result("AAAV", 7.0, same_group_losses),
            ],
        )

        assert [child.sequence for child in children] == ["AAAS", "AAAT"]
        assert {child.sequence for child in root.children} == {"AAAS", "AAAT"}
        assert tree.group_rejections == 1
        assert tree.get_node_stats()["num_groups"] == 2  # root group + child group
        assert all(child.group_id == children[0].group_id for child in children)

        better = tree.expand(
            root,
            [
                MutationProposal(
                    "AAAG",
                    [(3, "A", "G")],
                    prior=0.3,
                    benefit=2.0,
                    action_group="interface_formation",
                )
            ],
            [_result("AAAG", 4.0, same_group_losses)],
        )

        assert [child.sequence for child in better] == ["AAAG"]
        assert {child.sequence for child in root.children} == {"AAAS", "AAAG"}
        assert tree.group_pruned == 1
        assert not tree.has_sequence("AAAT")
        assert tree.get_global_best().sequence == "AAAG"


class TestParetoPruning:
    def test_pareto_pruning_rejects_dominated_same_group_candidate(self):
        tree = MCTSTree(
            C=1.0,
            max_nodes=20,
            max_children_per_node=10,
            representatives_per_group=10,
            loss_group_bin_width=10.0,
            use_reward_normalization=False,
        )
        root = tree.add_root_node("AAAA", torch.zeros(4, 20), L_real=20.0)
        good_losses = {"total": 5.0, "plddt": 1.0, "i_pae": 1.0, "clash": 0.0}
        dominated_losses = {"total": 6.0, "plddt": 2.0, "i_pae": 2.0, "clash": 0.5}

        first = tree.expand(
            root,
            [
                MutationProposal(
                    "AAAS",
                    [(3, "A", "S")],
                    prior=1.0,
                    benefit=1.0,
                    action_group="interface_formation",
                )
            ],
            [_result("AAAS", 5.0, good_losses)],
        )
        second = tree.expand(
            root,
            [
                MutationProposal(
                    "AAAT",
                    [(3, "A", "T")],
                    prior=1.0,
                    benefit=1.0,
                    action_group="interface_formation",
                )
            ],
            [_result("AAAT", 6.0, dominated_losses)],
        )

        assert [child.sequence for child in first] == ["AAAS"]
        assert second == []
        assert tree.has_sequence("AAAS")
        assert not tree.has_sequence("AAAT")
        assert tree.get_node_stats()["pareto_rejections"] == 1

    def test_pareto_pruning_removes_dominated_existing_leaf(self):
        tree = MCTSTree(
            C=1.0,
            max_nodes=20,
            max_children_per_node=10,
            representatives_per_group=10,
            loss_group_bin_width=10.0,
            use_reward_normalization=False,
        )
        root = tree.add_root_node("AAAA", torch.zeros(4, 20), L_real=20.0)
        worse_losses = {"total": 6.0, "plddt": 2.0, "i_pae": 2.0, "clash": 0.5}
        better_losses = {"total": 5.0, "plddt": 1.0, "i_pae": 1.0, "clash": 0.0}

        tree.expand(
            root,
            [
                MutationProposal(
                    "AAAT",
                    [(3, "A", "T")],
                    prior=1.0,
                    benefit=1.0,
                    action_group="interface_formation",
                )
            ],
            [_result("AAAT", 6.0, worse_losses)],
        )
        better = tree.expand(
            root,
            [
                MutationProposal(
                    "AAAS",
                    [(3, "A", "S")],
                    prior=1.0,
                    benefit=1.0,
                    action_group="interface_formation",
                )
            ],
            [_result("AAAS", 5.0, better_losses)],
        )

        assert [child.sequence for child in better] == ["AAAS"]
        assert tree.has_sequence("AAAS")
        assert not tree.has_sequence("AAAT")
        assert {child.sequence for child in root.children} == {"AAAS"}
        assert tree.get_node_stats()["pareto_pruned"] == 1


class TestEliteArchive:
    def test_elite_archive_keeps_top_candidates_per_group(self):
        tree = MCTSTree(
            C=1.0,
            max_nodes=20,
            max_children_per_node=10,
            representatives_per_group=10,
            loss_group_bin_width=10.0,
            use_reward_normalization=False,
            use_pareto_pruning=False,
            elite_archive_size_per_group=2,
            elite_min_sequence_distance=0,
        )
        root = tree.add_root_node("AAAA", torch.zeros(4, 20), L_real=20.0)
        proposals = [
            MutationProposal("AAAS", [(3, "A", "S")], prior=0.3, benefit=1.0, action_group="interface_formation"),
            MutationProposal("AATA", [(2, "A", "T")], prior=0.3, benefit=1.0, action_group="interface_formation"),
            MutationProposal("AVAA", [(1, "A", "V")], prior=0.3, benefit=1.0, action_group="interface_formation"),
        ]
        same_group_losses = {"total": 5.0, "i_pae": 1.0}

        children = tree.expand(
            root,
            proposals,
            [
                _result("AAAS", 5.0, same_group_losses),
                _result("AATA", 4.0, same_group_losses),
                _result("AVAA", 6.0, same_group_losses),
            ],
        )
        archive = tree.get_elite_archive(children[0].group_id)[children[0].group_id]

        assert [entry.sequence for entry in archive] == ["AATA", "AAAS"]
        assert [entry.tree_reward for entry in archive] == pytest.approx([-4.0, -5.0])
        assert tree.get_node_stats()["elite_archive_size"] == 3  # root basin + two child elites

    def test_elite_archive_replaces_close_weaker_candidate(self):
        tree = MCTSTree(
            C=1.0,
            max_nodes=20,
            max_children_per_node=10,
            representatives_per_group=10,
            loss_group_bin_width=10.0,
            use_reward_normalization=False,
            use_pareto_pruning=False,
            elite_archive_size_per_group=3,
            elite_min_sequence_distance=1,
        )
        root = tree.add_root_node("AAAA", torch.zeros(4, 20), L_real=20.0)
        same_group_losses = {"total": 5.0, "i_pae": 1.0}
        first = tree.expand(
            root,
            [
                MutationProposal(
                    "AAAS",
                    [(3, "A", "S")],
                    prior=1.0,
                    benefit=1.0,
                    action_group="interface_formation",
                )
            ],
            [_result("AAAS", 5.0, same_group_losses)],
        )[0]
        second = tree.expand(
            root,
            [
                MutationProposal(
                    "AAAT",
                    [(3, "A", "T")],
                    prior=1.0,
                    benefit=1.0,
                    action_group="interface_formation",
                )
            ],
            [_result("AAAT", 4.0, same_group_losses)],
        )[0]
        archive = tree.get_elite_archive(first.group_id)[first.group_id]

        assert first.group_id == second.group_id
        assert [entry.sequence for entry in archive] == ["AAAT"]
        assert tree.get_node_stats()["elite_archive_updates"] == 3  # root + first + replacement


class TestStructuralGrouping:
    def test_structural_grouping_uses_contact_patch_secondary_and_length_fields(self):
        tree = MCTSTree(
            C=1.0,
            max_nodes=20,
            max_children_per_node=10,
            representatives_per_group=10,
            loss_group_bin_width=10.0,
            use_reward_normalization=False,
            use_pareto_pruning=False,
        )
        root = tree.add_root_node("AAAA", torch.zeros(4, 20), L_real=20.0)
        same_losses = {"total": 5.0, "i_pae": 1.0}
        first_result = _result("AAAS", 5.0, same_losses)
        first_result.summary_confidence = {
            "hotspot_contact_bitset": [1, 0, 1],
            "interface_patch_id": "patch_2",
            "secondary_structure_pattern": "helix-rich",
            "contact_fingerprint": {"h1": 1, "h2": 0},
        }
        second_result = _result("AATA", 5.0, same_losses)
        second_result.summary_confidence = {
            "hotspot_contact_bitset": [0, 1, 0],
            "interface_patch_id": "patch_3",
            "secondary_structure_pattern": "extended",
            "contact_fingerprint": {"h1": 0, "h2": 1},
        }

        children = tree.expand(
            root,
            [
                MutationProposal("AAAS", [(3, "A", "S")], prior=0.5, benefit=1.0, action_group="interface_formation"),
                MutationProposal("AATA", [(2, "A", "T")], prior=0.5, benefit=1.0, action_group="interface_formation"),
            ],
            [first_result, second_result],
        )

        assert len(children) == 2
        assert children[0].group_id != children[1].group_id
        assert "hotspot_contact_bitset:101" in children[0].group_id
        assert "interface_patch_id:patch_2" in children[0].group_id
        assert "secondary_structure_pattern:helix-rich" in children[0].group_id
        assert "contact_fingerprint:h1:1,h2:0" in children[0].group_id
        assert "length_bin:0-9" in children[0].group_id
        assert children[0].structural_signature["interface_patch_id"] == "patch_2"

    def test_root_node_accepts_structural_signature_for_grouping(self):
        tree = MCTSTree(
            C=1.0,
            max_nodes=20,
            use_reward_normalization=False,
        )
        root = tree.add_root_node(
            "A" * 12,
            torch.zeros(12, 20),
            L_real=10.0,
            loss_vector={"total": 10.0},
            structural_signature={
                "hotspot_contact_bitset": "111",
                "interface_patch_id": "patch_1",
                "secondary_structure_pattern": "mixed",
                "contact_fingerprint": "fp_a",
            },
        )

        assert "hotspot_contact_bitset:111" in root.group_id
        assert "interface_patch_id:patch_1" in root.group_id
        assert "secondary_structure_pattern:mixed" in root.group_id
        assert "contact_fingerprint:fp_a" in root.group_id
        assert "length_bin:10-19" in root.group_id

    def test_structural_signature_preserves_routing_residue_indices(self):
        tree = MCTSTree(C=1.0)
        result = _result("AAAA", 5.0, {"total": 5.0})
        result.summary_confidence = {
            "interface_patch_id": "patch_1",
            "core_residue_indices": [1, 2],
            "surface_residue_indices": [0, 3],
            "clash_residue_indices": [2],
        }

        signature = tree.extract_structural_signature(
            result=result,
            sequence=result.binder_seq,
            loss_vector=result.loss_dict,
        )

        assert signature["interface_patch_id"] == "patch_1"
        assert signature["core_residue_indices"] == [1, 2]
        assert signature["surface_residue_indices"] == [0, 3]
        assert signature["clash_residue_indices"] == [2]


class TestZerothOrderStructuralMetadata:
    def test_compute_mcts_structural_metadata_extracts_contact_and_routing_maps(self):
        x0_flat = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [8.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
                [4.5, 0.0, 0.0],
                [12.5, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        contact_probs = torch.zeros(6, 6)
        contact_probs[1, 4] = 0.35
        contact_probs[4, 1] = 0.35
        contact_probs[3, 5] = 0.45
        contact_probs[5, 3] = 0.45
        input_feature_dict = {
            "asym_id": torch.tensor([1, 1, 1, 1, 2, 2]),
            "atom_to_token_idx": torch.arange(6),
            "distogram_rep_atom_mask": torch.ones(6, dtype=torch.bool),
        }

        metadata = compute_mcts_structural_metadata(
            x0_flat=x0_flat,
            contact_probs=contact_probs,
            input_feature_dict=input_feature_dict,
            L_binder=4,
            binder_seq="AVKD",
            hotspot_params={"target_hotspot_concat_indices": [1]},
            contact_prob_threshold=0.2,
            patch_size=1,
            clash_distance=1.0,
        )

        assert metadata["contact_fingerprint"] == "11"
        assert metadata["hotspot_contact_bitset"] == "1"
        assert metadata["interface_patch_id"] == "patch_1"
        assert metadata["interface_residue_indices"] == [1, 3]
        assert metadata["target_facing_residue_indices"] == [1, 3]
        assert metadata["hotspot_facing_residue_indices"] == [3]
        assert metadata["background_contact_residue_indices"] == [1]
        assert metadata["clash_residue_indices"] == [1, 3]
        assert metadata["core_residue_indices"]
        assert metadata["buried_residue_indices"] == metadata["core_residue_indices"]
        assert metadata["surface_residue_indices"]
        assert metadata["non_interface_residue_indices"] == [0, 2]
        assert set(metadata["exposed_hydrophobic_residue_indices"]).issubset(
            set(metadata["surface_residue_indices"])
        )
        assert metadata["secondary_structure_pattern"] in {
            "helix-rich",
            "extended",
            "collapsed",
            "mixed",
        }

    def test_structural_metadata_uses_target_roles_for_interface_only(self):
        x0_flat = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [20.5, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        contact_probs = torch.zeros(5, 5)
        contact_probs[0, 3] = 0.40  # binder-target peptide contact
        contact_probs[3, 0] = 0.40
        contact_probs[2, 4] = 0.95  # binder-scaffold contact, not target interface
        contact_probs[4, 2] = 0.95
        input_feature_dict = {
            "asym_id": torch.tensor([1, 1, 1, 2, 3]),
            "atom_to_token_idx": torch.arange(5),
            "distogram_rep_atom_mask": torch.ones(5, dtype=torch.bool),
        }

        metadata = compute_mcts_structural_metadata(
            x0_flat=x0_flat,
            contact_probs=contact_probs,
            input_feature_dict=input_feature_dict,
            L_binder=3,
            binder_seq="AVK",
            target_chain_roles=["target", "scaffold"],
            contact_prob_threshold=0.2,
            patch_size=1,
            clash_distance=0.6,
        )

        assert metadata["contact_fingerprint"] == "1"
        assert metadata["interface_patch_id"] == "patch_0"
        assert metadata["interface_residue_indices"] == [0]
        assert metadata["target_facing_residue_indices"] == [0]
        assert metadata["non_interface_residue_indices"] == [1, 2]
        assert metadata["clash_residue_indices"] == [0, 2]


class TestHardConstraintGate:
    def test_hard_constraint_marks_invalid_and_assigns_low_reward(self):
        tree = MCTSTree(
            C=1.0,
            max_nodes=20,
            max_children_per_node=4,
            use_reward_normalization=False,
        )
        root = tree.add_root_node("AAAA", torch.zeros(4, 20), L_real=10.0)

        children = tree.expand(
            root,
            [MutationProposal("AAAC", [(3, "A", "C")], prior=1.0, benefit=1.0)],
            [_result("AAAC", 1.0, {"total": 1.0, "clash": 7.0})],
        )

        assert len(children) == 1
        child = children[0]
        assert child.status_flags["valid"] is False
        assert child.status_flags["hard_invalid"] is True
        assert child.status_flags["hard_penalty"] > 1.0
        assert "clash" in child.status_flags["hard_violations"]
        assert child.tree_reward == pytest.approx(HARD_CONSTRAINT_INVALID_REWARD)
        assert tree.result_reward(_result("AAAC", 1.0, {"total": 1.0, "clash": 7.0})) == pytest.approx(
            HARD_CONSTRAINT_INVALID_REWARD
        )

    def test_hard_constraint_penalizes_minor_violation_without_invalidating(self):
        tree = MCTSTree(C=1.0, use_reward_normalization=False)
        root = tree.add_root_node(
            "AAAA",
            torch.zeros(4, 20),
            L_real=4.0,
            loss_vector={"total": 4.0, "clash": 5.5},
        )

        assert root.status_flags["valid"] is True
        assert root.status_flags["hard_invalid"] is False
        assert root.status_flags["hard_penalty"] == pytest.approx(0.25)
        assert root.tree_reward == pytest.approx(-6.5)


class TestRewardNormalization:
    def test_normalized_loss_vector_reward_replaces_raw_total_loss(self):
        tree = MCTSTree(C=1.0, max_nodes=20, max_children_per_node=4)
        root = tree.add_root_node(
            "AAAA",
            torch.zeros(4, 20),
            L_real=1.0,
            loss_vector={"total": 1.0, "plddt": 0.2, "i_pae": 0.6, "clash": 0.0},
        )

        assert root.status_flags["reward_mode"] == "normalized_loss_vector"
        assert root.status_flags["normalized_loss"] == pytest.approx((0.2 + 0.6 + 0.0) / 3.0)
        assert root.tree_reward == pytest.approx(-root.status_flags["normalized_loss"])
        assert root.tree_reward != pytest.approx(-1.0)
        assert tree.get_node_stats()["reward_stats_count"] == 1

        before = tree.reward_normalizer.observations
        reward = tree.result_reward(
            _result("AAAC", 0.8, {"total": 0.8, "plddt": 0.1, "i_pae": 0.3, "clash": 0.0})
        )
        assert tree.reward_normalizer.observations == before
        assert reward == pytest.approx(-((0.1 + 0.3 + 0.0) / 3.0))

    def test_default_reward_scales_match_current_loss_units(self):
        normalizer = RunningLossNormalizer()

        for key in ("plddt", "pae", "pde", "i_pae", "i_pde", "rg", "clash", "i_clash"):
            assert normalizer.default_scales[key] == pytest.approx(1.0)

    def test_running_normalizer_switches_to_component_z_score(self):
        tree = MCTSTree(C=1.0, max_nodes=20)
        for seq, plddt_loss in [
            ("AAAA", 10.0),
            ("AAAC", 20.0),
            ("AAAD", 30.0),
            ("AAAE", 40.0),
        ]:
            tree.add_root_node(
                seq,
                torch.zeros(4, 20),
                L_real=plddt_loss,
                loss_vector={"total": plddt_loss, "plddt": plddt_loss},
            )

        last = tree.get_node_by_sequence("AAAE")
        assert last is not None
        component = last.status_flags["normalized_components"]["plddt"]
        assert component != pytest.approx(40.0 / 100.0)
        assert component == pytest.approx((40.0 - 25.0) / torch.tensor([10.0, 20.0, 30.0, 40.0]).std().item())


class TestDiversityBonus:
    def test_diversity_bonus_rewards_novel_sequences_without_changing_first_root(self):
        tree = MCTSTree(
            C=1.0,
            max_nodes=20,
            max_children_per_node=10,
            use_reward_normalization=False,
            use_pareto_pruning=False,
            use_diversity_bonus=True,
            diversity_bonus_weight=0.1,
            diversity_bonus_max=1.0,
        )
        root = tree.add_root_node(
            "AAAA",
            torch.zeros(4, 20),
            L_real=10.0,
            loss_vector={"total": 10.0},
        )

        assert root.tree_reward == pytest.approx(-10.0)
        assert root.status_flags["diversity_bonus"] == pytest.approx(0.0)

        child = tree.expand(
            root,
            [
                MutationProposal(
                    "TTTT",
                    [(0, "A", "T"), (1, "A", "T"), (2, "A", "T"), (3, "A", "T")],
                    prior=1.0,
                    benefit=1.0,
                    action_group="interface_formation",
                )
            ],
            [_result("TTTT", 10.0, {"total": 10.0})],
        )[0]

        assert child.status_flags["diversity_sequence_score"] == pytest.approx(1.0)
        assert child.status_flags["diversity_group_score"] == pytest.approx(1.0)
        assert child.status_flags["diversity_score"] > 0.0
        assert child.status_flags["diversity_bonus"] == pytest.approx(0.075)
        assert child.tree_reward == pytest.approx(-9.925)

    def test_diversity_bonus_can_be_disabled(self):
        tree = MCTSTree(
            C=1.0,
            max_nodes=20,
            max_children_per_node=10,
            use_reward_normalization=False,
            use_pareto_pruning=False,
            use_diversity_bonus=False,
            diversity_bonus_weight=0.1,
        )
        root = tree.add_root_node("AAAA", torch.zeros(4, 20), L_real=10.0)
        child = tree.expand(
            root,
            [MutationProposal("TTTT", [(0, "A", "T")], prior=1.0, benefit=1.0)],
            [_result("TTTT", 10.0, {"total": 10.0})],
        )[0]

        assert child.status_flags["diversity_bonus"] == pytest.approx(0.0)
        assert child.tree_reward == pytest.approx(-10.0)


class TestVerifierCacheAndDuplicateMerge:
    def test_verifier_result_cache_is_cpu_lightweight_and_counted(self):
        tree = MCTSTree(C=1.0, use_reward_normalization=False)
        result = _result("AAAC", 3.0, {"total": 3.0})
        result.coordinate = torch.ones(1, 4, 3, requires_grad=True)

        cached = tree.store_verifier_result(result)

        assert cached.binder_seq == "AAAC"
        assert cached.coordinate.device.type == "cpu"
        assert not cached.coordinate.requires_grad
        assert cached.from_verifier_cache is True
        assert tree.get_node_stats()["verifier_cache_stores"] == 1

        assert tree.get_cached_verifier_result("AAAC") is cached
        assert tree.get_node_stats()["verifier_cache_hits"] == 1

    def test_duplicate_merge_updates_existing_node_without_new_node(self):
        tree = MCTSTree(C=1.0, use_reward_normalization=False)
        root = tree.add_root_node("AAAA", torch.zeros(4, 20), L_real=10.0)
        proposal = MutationProposal("AAAC", [(3, "A", "C")], prior=1.0, benefit=1.0)
        result = _result("AAAC", 3.0, {"total": 3.0})
        tree.store_verifier_result(result)
        child = tree.expand(root, [proposal], [result])[0]

        before_total_nodes = tree.total_nodes
        before_child_visits = child.visit_count
        duplicate_reward = tree.merge_duplicate(root, proposal, tree.get_cached_verifier_result("AAAC"))
        tree.backpropagate(root, [duplicate_reward])

        assert tree.total_nodes == before_total_nodes
        assert child.visit_count == before_child_visits + 1
        assert child.status_flags["duplicate_hits"] == 1
        assert tree.get_node_stats()["duplicate_merges"] == 1
        assert root.visit_count == 2
