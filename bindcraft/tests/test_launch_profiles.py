from pathlib import Path

import pytest

from bindcraft.hallucination.profile_loader import (
    LaunchProfileError,
    apply_launch_profile,
    available_launch_profiles,
)
from scripts.run_hallucination import (
    build_arg_parser,
    parse_args_with_profile,
    resolve_model_backend,
)


def _parse_profile(profile: str, *extra_args: str):
    args, parser = parse_args_with_profile(
        ["--profile", profile, "--target", "ACDEFG", *extra_args]
    )
    resolve_model_backend(args, parser)
    return args


def test_bundled_launch_profiles_are_present():
    assert available_launch_profiles() == (
        "esmfold2_d",
        "protenix_v1",
        "protenix_v2",
    )


def test_protenix_profiles_share_algorithm_defaults_and_switch_model():
    v1 = _parse_profile("protenix_v1")
    v2 = _parse_profile("protenix_v2")

    assert v1.model_backend == v2.model_backend == "protenix"
    assert v1.model_name == "protenix_base_default_v1.0.0"
    assert v2.model_name == "protenix-v2"
    for field in (
        "mode",
        "binder_length",
        "n_steps",
        "mcts_strategy",
        "mcts_prior_mode",
        "mcts_proposal_method",
        "mcts_num_beams",
        "mcts_handoff_top_k",
        "mcts_expand_actions_per_step",
        "loss_preset",
    ):
        assert getattr(v1, field) == getattr(v2, field)


def test_esmfold2_d_profile_matches_formal_method_defaults():
    args = _parse_profile("esmfold2_d")

    assert args.launch_profile_name == "esmfold2_d"
    assert Path(args.launch_profile_path).name == "esmfold2_d.yaml"
    assert args.model_backend == "esmfold2"
    assert args.esmfold2_experimental_model == (
        "biohub/ESMFold2-Experimental-Fast"
    )
    assert args.mcts_strategy == "two_stage"
    assert args.n_steps == 110
    assert args.mcts_bootstrap_steps == 70
    assert args.mcts_bootstrap_max_steps == 70
    assert args.mcts_handoff_top_k == 2
    assert args.mcts_prior_mode == "no_group_matched_compute"
    assert args.mcts_num_beams == 8
    assert args.b_candidates == 8
    assert args.esmfold2_stagea_soft_steps == 30
    assert args.esmfold2_stagea_tree_children == 2
    assert args.esmfold2_stagea_root_selection_mode == "hierarchical_ucb"
    assert args.esmfold2_stagea_min_root_expansions == 2
    assert args.esmfold2_stagea_preserve_soft_state is True
    assert args.esmfold2_stagea_cumulative_annealing is False
    assert args.esmfold2_stagea_mutation_fraction == pytest.approx(0.5)
    assert args.esmfold2_stagea_candidate_strategy == "positive_score"
    assert args.esmfold2_stagea_include_argmax_candidate is True
    assert args.stagea_max_mutations_per_seq is None
    assert args.stageb_max_mutations_per_seq == 1
    assert args.esmfold2_stageb_discrete_rerank is True
    assert args.esmfold2_stageb_rerank_pool_size == 20
    assert args.esmfold2_stageb_exploration_actions == 1
    assert args.esmfold2_stageb_verifier_candidates == 8
    assert args.esmfold2_one_step_diffusion is True
    assert args.esmfold2_one_step_diffusion_mode == "truncated_trajectory"
    assert args.esmfold2_one_step_diffusion_interval == 5
    assert args.esmfold2_one_step_diffusion_tau == pytest.approx(1.0)
    assert args.esmfold2_one_step_diffusion_weight == pytest.approx(0.25)
    assert args.esmfold2_one_step_diffusion_rg_weight == pytest.approx(0.1)
    assert args.esmfold2_one_step_recycle_coordinates is False
    assert args.esmfold2_verifier_num_loops == 20
    assert args.esmfold2_verifier_num_sampling_steps == 200
    assert args.esmfold2_verifier_loss_mode == "aligned_common"
    assert args.esmfold2_hotspot_verify_weight == pytest.approx(0.0)
    assert args.esmfold2_hotspot_verify_too_close_weight == pytest.approx(0.0)


def test_explicit_cli_value_overrides_launch_profile():
    args = _parse_profile(
        "esmfold2_d",
        "--n_steps",
        "5",
        "--esmfold2_one_step_diffusion_weight",
        "0.5",
    )

    assert args.n_steps == 5
    assert args.esmfold2_one_step_diffusion_weight == pytest.approx(0.5)


def test_tcr_uses_low_precision_verifier_defaults():
    args = _parse_profile(
        "esmfold2_d",
        "--esmfold2_task_type",
        "tcr",
    )

    assert args.esmfold2_verifier_num_loops == 4
    assert args.esmfold2_verifier_num_sampling_steps == 50


def test_tcr_explicit_verifier_precision_overrides_task_defaults():
    args = _parse_profile(
        "esmfold2_d",
        "--esmfold2_task_type",
        "tcr",
        "--esmfold2_verifier_num_loops",
        "20",
        "--esmfold2_verifier_num_sampling_steps",
        "200",
    )

    assert args.esmfold2_verifier_num_loops == 20
    assert args.esmfold2_verifier_num_sampling_steps == 200


@pytest.mark.parametrize(
    ("task_type", "verifier_precision"),
    (
        ("denovo", (20, 200)),
        ("tcr", (4, 50)),
        ("cdr_redesign", (20, 200)),
    ),
)
def test_esmfold2_profile_covers_all_formal_task_modes(
    task_type,
    verifier_precision,
):
    args = _parse_profile(
        "esmfold2_d",
        "--esmfold2_task_type",
        task_type,
    )

    assert args.esmfold2_task_type == task_type
    assert args.mcts_strategy == "two_stage"
    assert args.n_steps == 110
    assert args.mcts_bootstrap_steps == 70
    assert args.mcts_bootstrap_max_steps == 70
    assert args.mcts_prior_mode == "no_group_matched_compute"
    assert args.b_candidates == 8
    assert args.esmfold2_stagea_candidate_strategy == "positive_score"
    assert args.esmfold2_stagea_mutation_fraction == pytest.approx(0.5)
    assert args.esmfold2_stagea_include_argmax_candidate is True
    assert args.esmfold2_one_step_diffusion_mode == "truncated_trajectory"
    assert args.esmfold2_one_step_diffusion_tau == pytest.approx(1.0)
    assert args.esmfold2_one_step_recycle_coordinates is False
    assert (
        args.esmfold2_verifier_num_loops,
        args.esmfold2_verifier_num_sampling_steps,
    ) == verifier_precision


def test_no_profile_preserves_existing_cli_defaults():
    args, parser = parse_args_with_profile(["--target", "ACDEFG"])

    assert args.launch_profile_name is None
    assert args.launch_profile_path is None
    assert args.model_backend == "protenix"
    assert resolve_model_backend(args, parser) == "protenix"
    assert args.model_name == "protenix_base_default_v1.0.0"


def test_profile_rejects_unknown_cli_option(tmp_path):
    profile_path = tmp_path / "invalid.yaml"
    profile_path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "name: invalid",
                "defaults:",
                "  nonexistent_option: 1",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        LaunchProfileError,
        match="unknown launch profile options: nonexistent_option",
    ):
        apply_launch_profile(build_arg_parser(), profile_path)
