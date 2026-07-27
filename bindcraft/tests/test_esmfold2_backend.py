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
    softly_anchor_mutation_logits,
)
from bindcraft.hallucination.mcts_controller import MCTSTree


def test_soft_mutation_anchor_changes_only_explicit_edit_positions():
    logits = torch.arange(3 * len(ESM_AA_ORDER), dtype=torch.float32).reshape(
        3, len(ESM_AA_ORDER)
    )
    logits[:, CYS_INDEX] = float("-inf")

    anchored = softly_anchor_mutation_logits(
        logits,
        parent_sequence="AAA",
        child_sequence="AYA",
        mutable_mask=[True, True, True],
        anchor_margin=0.1,
    )

    assert torch.equal(anchored[0], logits[0])
    assert torch.equal(anchored[2], logits[2])
    y_idx = ESM_AA_ORDER.index("Y")
    finite_before = logits[1][torch.isfinite(logits[1])]
    assert anchored[1, y_idx] == pytest.approx(float(finite_before.max()) + 0.1)


def test_verifier_loss_requires_pair_chain_iptm_without_global_fallback():
    loss, source = resolve_esm_verifier_loss(
        {"pair_chain_iptm": 0.82, "iptm": 0.10}
    )
    assert loss == pytest.approx(0.18)
    assert source == "pair_chain_iptm"

    with pytest.raises(ValueError, match="pair-chain iPTM"):
        resolve_esm_verifier_loss({"iptm": 0.73})


def test_verifier_result_is_compatible_with_mcts_cache_contract():
    result = build_esm_verifier_result(
        binder_seq="ALY",
        target_seq="TARGET",
        summary_confidence={
            "pair_chain_iptm": 0.75,
            "plddt": 0.91,
            "binder_plddt": 0.88,
            "binder_rg_excess": 0.12,
        },
        coordinate=torch.zeros(1, 8, 3),
    )

    assert result.total_loss == pytest.approx(0.25)
    assert result.loss_dict["esm_verify_loss"] == pytest.approx(0.25)
    assert result.loss_dict["pair_chain_iptm_loss"] == pytest.approx(0.25)
    assert result.loss_dict["binder_confidence_loss"] == pytest.approx(0.12)
    assert result.loss_dict["binder_rg_excess_loss"] == pytest.approx(0.12)
    assert "interface_loss" not in result.loss_dict
    assert result.summary_confidence["esm_verify_source"] == "pair_chain_iptm"

    tree = MCTSTree()
    cached = tree.store_verifier_result(result)
    assert cached.binder_seq == "ALY"
    assert cached.total_loss == pytest.approx(0.25)
    assert cached.coordinate.shape == (1, 8, 3)


def test_verifier_result_combines_global_and_hotspot_losses_without_overwriting_global_metric():
    result = build_esm_verifier_result(
        binder_seq="ALY",
        target_seq="TARGET",
        summary_confidence={"pair_chain_iptm": 0.75},
        hotspot_verify_loss=0.6,
        hotspot_verify_weight=0.2,
        extra_loss_dict={"hotspot_mean_min_heavy_atom_distance": 14.0},
    )

    assert result.total_loss == pytest.approx(0.37)
    assert result.loss_dict["pair_chain_iptm_loss"] == pytest.approx(0.25)
    assert result.loss_dict["esm_verify_global_loss"] == pytest.approx(0.25)
    assert result.loss_dict["hotspot_verify_distance_loss"] == pytest.approx(0.6)
    assert result.loss_dict["hotspot_verify_weighted_contribution"] == pytest.approx(0.12)
    assert result.loss_dict["hotspot_mean_min_heavy_atom_distance"] == pytest.approx(14.0)
    assert result.summary_confidence["esm_verify_loss"] == pytest.approx(0.37)


def test_verifier_result_can_use_aligned_common_loss_without_losing_pair_metric():
    result = build_esm_verifier_result(
        binder_seq="ALY",
        target_seq="TARGET",
        summary_confidence={"pair_chain_iptm": 0.75},
        aligned_loss_dict={
            "total": 0.42,
            "plddt": 0.12,
            "i_pae": 0.20,
        },
        hotspot_verify_loss=0.5,
        hotspot_verify_weight=0.2,
    )

    assert result.total_loss == pytest.approx(0.52)
    assert result.loss_dict["pair_chain_iptm_loss"] == pytest.approx(0.25)
    assert result.loss_dict["esm_verify_global_loss"] == pytest.approx(0.42)
    assert result.loss_dict["plddt"] == pytest.approx(0.12)
    assert result.summary_confidence["esm_verify_source"] == "esmfold2_aligned_common"


def test_verifier_result_adds_hotspot_too_close_penalty():
    result = build_esm_verifier_result(
        binder_seq="ALY",
        target_seq="TARGET",
        summary_confidence={"pair_chain_iptm": 0.75},
        hotspot_verify_loss=0.0,
        hotspot_verify_weight=0.2,
        hotspot_too_close_loss=0.4,
        hotspot_too_close_weight=0.25,
    )

    assert result.total_loss == pytest.approx(0.35)
    assert result.loss_dict["pair_chain_iptm_loss"] == pytest.approx(0.25)
    assert result.loss_dict["hotspot_too_close_loss"] == pytest.approx(0.4)
    assert result.loss_dict[
        "hotspot_too_close_weighted_contribution"
    ] == pytest.approx(0.1)


def test_zero_hotspot_verify_weight_preserves_legacy_verifier_result_shape():
    result = build_esm_verifier_result(
        binder_seq="ALY",
        target_seq="TARGET",
        summary_confidence={"pair_chain_iptm": 0.75},
        hotspot_verify_loss=0.6,
        hotspot_verify_weight=0.0,
    )

    assert result.total_loss == pytest.approx(0.25)
    assert set(result.loss_dict) == {
        "total",
        "esm_verify_loss",
        "pair_chain_iptm_loss",
    }
    assert "hotspot_verify_distance_loss" not in result.summary_confidence

    with pytest.raises(
        ValueError,
        match="positive hotspot_verify_weight requires hotspot_verify_loss",
    ):
        build_esm_verifier_result(
            binder_seq="ALY",
            target_seq="TARGET",
            summary_confidence={"pair_chain_iptm": 0.75},
            hotspot_verify_weight=0.2,
        )


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
    assert default_config.hotspot_weight == pytest.approx(0.0)
    assert default_config.hotspot_satisfied_proposal_weight == pytest.approx(0.0)
    assert default_config.hotspot_cutoff == pytest.approx(8.0)
    assert default_config.hotspot_interface_mode == "additive"
    assert default_config.hotspot_verify_weight == pytest.approx(0.0)
    assert default_config.hotspot_verify_near_cutoff == pytest.approx(8.0)
    assert default_config.hotspot_verify_far_cutoff == pytest.approx(20.0)
    assert default_config.hotspot_verify_min_distance == pytest.approx(2.5)
    assert default_config.hotspot_verify_too_close_weight == pytest.approx(0.0)
    assert default_config.one_step_diffusion_enabled is False
    assert default_config.one_step_diffusion_tau == pytest.approx(4.0)
    assert default_config.one_step_diffusion_weight == pytest.approx(1.0)
    assert default_config.one_step_diffusion_rg_weight == pytest.approx(0.3)
    assert default_config.one_step_diffusion_interval == 0
    assert default_config.one_step_diffusion_recycle_coordinates is False
    assert default_config.one_step_diffusion_hotspot_enabled is False
    assert default_config.distogram_hotspot_enabled is True
    assert default_config.verifier_loss_mode == "pair_chain_iptm"

    config = ESMFold2ProposalConfig(soft_steps=3, lambda_esmc="auto")
    assert config.soft_steps == 3
    first = esm_temperature(0, total_steps=3)
    last = esm_temperature(2, total_steps=3)
    assert first > last
    assert last >= 1e-2


def test_proposal_config_rejects_negative_hotspot_weight():
    with pytest.raises(ValueError, match="hotspot_weight must be non-negative"):
        ESMFold2ProposalConfig(hotspot_weight=-0.1)


def test_proposal_config_rejects_nonpositive_hotspot_cutoff():
    with pytest.raises(ValueError, match="hotspot_cutoff must be positive"):
        ESMFold2ProposalConfig(hotspot_cutoff=0.0)


def test_proposal_config_validates_hotspot_interface_mode():
    with pytest.raises(ValueError, match="hotspot_interface_mode"):
        ESMFold2ProposalConfig(hotspot_interface_mode="unknown")
    with pytest.raises(ValueError, match="hotspot_weight in"):
        ESMFold2ProposalConfig(
            hotspot_interface_mode="localized",
            hotspot_weight=1.1,
        )


def test_proposal_config_validates_hotspot_verifier_parameters():
    with pytest.raises(ValueError, match="hotspot_verify_weight must be non-negative"):
        ESMFold2ProposalConfig(hotspot_verify_weight=-0.1)
    with pytest.raises(
        ValueError,
        match="hotspot_verify_far_cutoff must be greater",
    ):
        ESMFold2ProposalConfig(
            hotspot_verify_near_cutoff=5.0,
            hotspot_verify_far_cutoff=5.0,
        )


def test_proposal_config_validates_one_step_diffusion_parameters():
    with pytest.raises(ValueError, match="one_step_diffusion_tau"):
        ESMFold2ProposalConfig(one_step_diffusion_tau=0.0)
    with pytest.raises(ValueError, match="one_step_diffusion_weight"):
        ESMFold2ProposalConfig(one_step_diffusion_weight=-0.1)
    with pytest.raises(ValueError, match="one_step_diffusion_rg_weight"):
        ESMFold2ProposalConfig(one_step_diffusion_rg_weight=-0.1)
    with pytest.raises(ValueError, match="one_step_diffusion_interval"):
        ESMFold2ProposalConfig(one_step_diffusion_interval=-1)
    with pytest.raises(
        ValueError,
        match="positive one_step_diffusion_interval",
    ):
        ESMFold2ProposalConfig(one_step_diffusion_interval=5)
    with pytest.raises(
        ValueError,
        match="one_step_diffusion_recycle_coordinates",
    ):
        ESMFold2ProposalConfig(
            one_step_diffusion_recycle_coordinates=True,
        )
    with pytest.raises(ValueError, match="verifier_loss_mode"):
        ESMFold2ProposalConfig(verifier_loss_mode="unknown")


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
            "binder_confidence_loss": 0.1,
            "binder_rg_excess_loss": 0.05,
        }
    )
    assert interface["action_group"] == "interface_formation"
    assert interface["group_priors"]["interface_formation"] == pytest.approx(
        max(interface["group_priors"].values())
    )

    compact = select_esmfold2_action_group(
        {
            "pair_chain_iptm_loss": 0.1,
            "binder_confidence_loss": 0.1,
            "binder_rg_excess_loss": 0.9,
        }
    )
    assert compact["action_group"] == "compactness_repair"

    with pytest.raises(ValueError, match="full-verifier metrics"):
        compute_esmfold2_action_group_priors({}, temperature=1.0)


def test_esmfold2_routing_uses_only_bounded_full_verifier_residuals():
    routing = select_esmfold2_action_group(
        {
            "pair_chain_iptm_loss": 0.95,
            "binder_confidence_loss": 0.2,
            "binder_rg_excess_loss": 1.2,
        }
    )

    assert routing["group_scores"]["interface_formation"] <= 1.0
    assert routing["group_scores"]["core_stabilization"] <= 1.0
    assert routing["group_scores"]["compactness_repair"] <= 1.0
    assert routing["group_terms"]["interface_formation"]["pair_chain_iptm_loss"]["normalized"] == pytest.approx(0.95)
    assert routing["group_terms"]["compactness_repair"]["binder_rg_excess_loss"]["normalized"] == pytest.approx(1.0)
