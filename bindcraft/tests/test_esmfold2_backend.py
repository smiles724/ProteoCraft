import pytest
import torch

from bindcraft.hallucination.esmfold2_backend import (
    CYS_INDEX,
    ESM_AA_ORDER,
    ESMFold2ProposalConfig,
    build_esm_verifier_result,
    build_gradient_mask,
    combine_normalized_gradients,
    compute_esmfold2_action_group_priors,
    esm_temperature,
    init_binder_logits,
    logit_delta_mutation_scores,
    normalized_gradient_tensor,
    propose_sequences_from_logit_delta,
    resolve_esm_verifier_loss,
    resolve_lambda_esmc,
    select_esmfold2_action_group,
)
from bindcraft.hallucination.mcts_controller import MCTSTree


def test_verifier_loss_prefers_pair_chain_iptm_and_falls_back_to_iptm():
    loss, source = resolve_esm_verifier_loss(
        {"pair_chain_iptm": 0.82, "iptm": 0.10}
    )
    assert loss == pytest.approx(0.18)
    assert source == "pair_chain_iptm"

    fallback_loss, fallback_source = resolve_esm_verifier_loss({"iptm": 0.73})
    assert fallback_loss == pytest.approx(0.27)
    assert fallback_source == "iptm"

    with pytest.raises(ValueError):
        resolve_esm_verifier_loss({"ptm": 0.8})


def test_verifier_result_is_compatible_with_mcts_cache_contract():
    result = build_esm_verifier_result(
        binder_seq="ALY",
        target_seq="TARGET",
        summary_confidence={"pair_chain_iptm": 0.75, "plddt": 0.91},
        coordinate=torch.zeros(1, 8, 3),
    )

    assert result.total_loss == pytest.approx(0.25)
    assert result.loss_dict["esm_verify_loss"] == pytest.approx(0.25)
    assert result.loss_dict["pair_chain_iptm_loss"] == pytest.approx(0.25)
    assert "interface_loss" not in result.loss_dict
    assert result.summary_confidence["esm_verify_source"] == "pair_chain_iptm"

    tree = MCTSTree()
    cached = tree.store_verifier_result(result)
    assert cached.binder_seq == "ALY"
    assert cached.total_loss == pytest.approx(0.25)
    assert cached.coordinate.shape == (1, 8, 3)


def test_init_logits_and_gradient_mask_keep_fixed_positions_out_of_optimization():
    logits = init_binder_logits(
        "ALD",
        mutable_mask=[False, True, True],
        batch_size=2,
        random_scale=0.0,
    )
    aa_to_idx = {aa: idx for idx, aa in enumerate(ESM_AA_ORDER)}

    assert logits.shape == (2, 3, 20)
    assert logits.requires_grad
    assert logits[0, 0, aa_to_idx["A"]].item() == pytest.approx(10.0)
    assert torch.isneginf(logits[:, 1:, CYS_INDEX]).all()

    mask = build_gradient_mask("ALD", mutable_mask=[False, True, True], batch_size=2)
    assert mask.shape == (2, 3, 20)
    assert mask[:, 0, :].sum().item() == pytest.approx(0.0)
    assert mask[:, 1:, CYS_INDEX].sum().item() == pytest.approx(0.0)
    assert mask[:, 1:, :].sum().item() > 0.0


def test_logit_delta_scores_rank_mutations_from_short_soft_update():
    aa_to_idx = {aa: idx for idx, aa in enumerate(ESM_AA_ORDER)}
    before = torch.zeros(2, 20)
    after = before.clone()
    after[1, aa_to_idx["Y"]] = 2.0
    after[1, aa_to_idx["L"]] = 0.5
    after[1, aa_to_idx["V"]] = 1.1

    scores = logit_delta_mutation_scores(
        before,
        after,
        current_seq="AL",
        mutable_mask=[False, True],
    )

    assert torch.isneginf(scores[0]).all()
    assert torch.isneginf(scores[1, aa_to_idx["L"]])
    assert torch.isneginf(scores[1, CYS_INDEX])
    assert scores[1, aa_to_idx["Y"]].item() == pytest.approx(1.5)
    assert scores[1, aa_to_idx["V"]].item() == pytest.approx(0.6)


def test_logit_delta_scores_do_not_nan_when_current_mutable_residue_is_cys():
    aa_to_idx = {aa: idx for idx, aa in enumerate(ESM_AA_ORDER)}
    before = torch.zeros(2, 20)
    after = before.clone()
    before[1, CYS_INDEX] = float("-inf")
    after[1, CYS_INDEX] = float("-inf")
    after[1, aa_to_idx["Y"]] = 2.0

    scores = logit_delta_mutation_scores(
        before,
        after,
        current_seq="AC",
        mutable_mask=[False, True],
    )

    assert not torch.isnan(scores).any()
    assert torch.isneginf(scores[1, CYS_INDEX])
    assert scores[1, aa_to_idx["Y"]].item() == pytest.approx(2.0)


def test_combine_normalized_gradients_respects_mask_and_task_lambda():
    mask = torch.zeros(1, 2, 20)
    mask[:, 1, :] = 1.0
    mask[:, 1, CYS_INDEX] = 0.0

    structure_grad = torch.zeros_like(mask)
    structure_grad[0, 1, 0] = 3.0
    esmc_grad = torch.zeros_like(mask)
    esmc_grad[0, 1, 1] = 4.0

    combined = combine_normalized_gradients(
        structure_grad,
        esmc_grad,
        gradient_mask=mask,
        lambda_esmc=resolve_lambda_esmc("auto", task_type="denovo"),
    )

    assert combined[:, 0, :].sum().item() == pytest.approx(0.0)
    assert combined[0, 1, 0].item() > 0.0
    assert combined[0, 1, 1].item() > 0.0
    assert resolve_lambda_esmc("auto", task_type="cdr") < resolve_lambda_esmc(
        "auto", task_type="denovo"
    )


def test_normalized_gradient_tensor_matches_official_eff_length_scaling():
    mask = torch.zeros(1, 3, 20)
    mask[0, 0, 0] = 1.0
    mask[0, 2, 1] = 1.0
    grad = torch.zeros_like(mask)
    grad[0, 0, 0] = 3.0
    grad[0, 2, 1] = 4.0

    normalized = normalized_gradient_tensor(grad, mask)

    expected_scale = (2.0 ** 0.5) / 5.0
    assert normalized[0, 0, 0].item() == pytest.approx(3.0 * expected_scale)
    assert normalized[0, 2, 1].item() == pytest.approx(4.0 * expected_scale)
    assert torch.linalg.vector_norm(normalized).item() == pytest.approx(2.0 ** 0.5)
    assert normalized[0, 1, :].sum().item() == pytest.approx(0.0)


def test_temperature_schedule_and_config_match_short_esm_stage_a_contract():
    default_config = ESMFold2ProposalConfig(lambda_esmc="auto")
    assert default_config.soft_steps == 30
    assert default_config.proposal_lm_dropout == pytest.approx(0.5)
    assert default_config.force_proposal_lm_dropout is True
    assert default_config.experimental_model_name.endswith("ESMFold2-Experimental")

    config = ESMFold2ProposalConfig(soft_steps=3, lambda_esmc="auto")
    assert config.soft_steps == 3
    first = esm_temperature(0, total_steps=3)
    last = esm_temperature(2, total_steps=3)
    assert first > last
    assert last >= 1e-2


def test_logit_delta_candidates_reuse_stage_a_topk_sampling_contract():
    aa_to_idx = {aa: idx for idx, aa in enumerate(ESM_AA_ORDER)}
    before = torch.zeros(3, 20)
    after = before.clone()
    after[1, aa_to_idx["Y"]] = 3.0
    after[2, aa_to_idx["F"]] = 2.5

    torch.manual_seed(0)
    candidates = propose_sequences_from_logit_delta(
        before,
        after,
        current_seq="ALD",
        mutable_mask=[False, True, True],
        K_top=2,
        B_candidates=3,
        T_sample=0.01,
        max_mutations_per_seq=1,
    )

    assert len(candidates) == 3
    assert "AYD" in candidates
    assert all(seq[0] == "A" for seq in candidates)


def test_logit_delta_candidates_prepend_full_argmax_sequence():
    aa_to_idx = {aa: idx for idx, aa in enumerate(ESM_AA_ORDER)}
    before = torch.zeros(3, 20)
    after = before.clone()
    after[1, aa_to_idx["Y"]] = 3.0
    after[2, aa_to_idx["F"]] = 2.5

    torch.manual_seed(0)
    candidates = propose_sequences_from_logit_delta(
        before,
        after,
        current_seq="ALD",
        mutable_mask=[False, True, True],
        K_top=2,
        B_candidates=3,
        T_sample=0.01,
        max_mutations_per_seq=1,
        include_logit_candidates=True,
    )

    assert candidates[0] == "AYF"
    assert len(candidates) == 3
    assert all(seq[0] == "A" for seq in candidates)


def test_esmfold2_action_group_routing_uses_native_loss_terms():
    interface = select_esmfold2_action_group(
        {
            "pair_chain_iptm_loss": 0.7,
            "inter_contact_loss": 0.6,
            "intra_contact_loss": 0.1,
            "glob_loss": 0.05,
        }
    )
    assert interface["action_group"] == "interface_formation"
    assert interface["group_priors"]["interface_formation"] == pytest.approx(
        max(interface["group_priors"].values())
    )

    compact = select_esmfold2_action_group(
        {
            "pair_chain_iptm_loss": 0.1,
            "inter_contact_loss": 0.1,
            "intra_contact_loss": 0.1,
            "glob_loss": 1.2,
        }
    )
    assert compact["action_group"] == "compactness_repair"

    priors = compute_esmfold2_action_group_priors({}, temperature=1.0)
    assert priors == {"unrouted": 1.0}


def test_esmfold2_routing_normalizes_large_contact_losses_before_group_mean():
    routing = select_esmfold2_action_group(
        {
            "pair_chain_iptm_loss": 0.95,
            "inter_contact_loss": 5.0,
            "binder_confidence_loss": 0.2,
            "intra_contact_loss": 0.2,
            "glob_loss": 0.2,
        }
    )

    assert routing["group_scores"]["interface_formation"] <= 1.0
    assert routing["group_scores"]["core_stabilization"] <= 1.0
    assert routing["group_scores"]["compactness_repair"] <= 1.0
    assert routing["group_terms"]["interface_formation"]["inter_contact_loss"]["normalized"] == pytest.approx(1.0)
