import pytest
import subprocess
import sys

from bindcraft.hallucination.esmfold2_backend import build_esm_verifier_result
import scripts.run_hallucination as run_hallucination_script
from scripts.run_hallucination import build_arg_parser, resolve_model_backend


def _parse(extra_args):
    parser = build_arg_parser()
    return parser.parse_args(["--target", "ACDEFG"] + list(extra_args)), parser


def test_default_model_backend_preserves_protenix_path():
    args, parser = _parse([])

    assert args.model_backend == "protenix"
    assert resolve_model_backend(args, parser) == "protenix"
    assert args.seed == 42
    assert args.mcts_strategy == "two_stage"
    assert args.mcts_prior_mode == "full"
    assert args.n_steps == 200
    assert args.mcts_bootstrap_steps == 50
    assert args.mcts_bootstrap_max_steps == 80
    assert args.mcts_expand_actions_per_step == 2


def test_esmfold2_backend_is_explicit_and_mcts_only_for_first_version():
    args, parser = _parse(["--model_backend", "esmfold2", "--mode", "mcts"])

    assert args.model_backend == "esmfold2"
    assert resolve_model_backend(args, parser) == "esmfold2"
    assert args.esmfold2_experimental_model == (
        "biohub/ESMFold2-Experimental-Fast"
    )
    assert args.mcts_strategy == "two_stage"
    assert args.mcts_prior_mode == "no_group"
    assert args.seed == 43
    assert args.n_steps == 110
    assert args.mcts_bootstrap_steps == 70
    assert args.mcts_bootstrap_max_steps == 70
    assert args.esmfold2_soft_steps is None
    assert args.esmfold2_stagea_soft_steps == 30
    assert args.esmfold2_stagea_tree_children == 2
    assert args.esmfold2_stagea_root_selection_mode == "hierarchical_ucb"
    assert args.esmfold2_stagea_min_root_expansions == 2
    assert args.esmfold2_stagea_preserve_soft_state is True
    assert args.esmfold2_stagea_cumulative_annealing is False
    assert args.esmfold2_stagea_annealing_horizon == 150
    assert args.esmfold2_stagea_soft_anchor_margin == pytest.approx(0.1)
    assert args.esmfold2_stagea_paired_verifier_seed is True
    assert args.esmfold2_stagea_verifier_seed is None
    assert args.esmfold2_temperature_min == pytest.approx(0.01)
    assert args.esmfold2_temperature_max == pytest.approx(1.0)
    assert args.esmfold2_raw_lineage_annealing_horizon == 150
    assert args.esmfold2_stageb_soft_steps == 20
    assert args.mcts_expand_actions_per_step == 2
    assert args.mcts_handoff_top_k == 2
    assert args.esmfold2_stageb_discrete_rerank is True
    assert args.esmfold2_stageb_rerank_pool_size == 20
    assert args.esmfold2_stageb_exploration_actions == 1
    assert args.esmfold2_stageb_verifier_candidates == 8
    assert args.esmfold2_stageb_hotspot_coverage_weight == pytest.approx(0.25)
    assert args.esmfold2_stageb_hotspot_distance_weight == pytest.approx(0.25)
    assert args.esmfold2_stageb_hotspot_miss_penalty == pytest.approx(0.0)
    assert args.esmfold2_stageb_min_pair_chain_iptm == pytest.approx(0.0)
    assert args.esmfold2_stageb_handoff_quality_tolerance == pytest.approx(0.05)
    assert args.esmfold2_stageb_paired_verifier_seed is True
    assert args.esmfold2_stageb_verifier_seed is None
    assert args.esmfold2_stagea_mutation_fraction == pytest.approx(0.5)
    assert args.esmfold2_stagea_candidate_strategy == "positive_score"
    assert args.esmfold2_stagea_include_argmax_candidate is True
    assert args.stageb_max_mutations_per_seq == 1
    assert args.esmfold2_logit_init_mode == "official_exact"
    assert args.esmfold2_initial_logits_checkpoint is None
    assert args.esmfold2_save_verified_candidate_cif is False
    assert args.esmfold2_save_visualization_trace is False
    assert args.esmfold2_proposal_load_internal_esmc is True
    assert args.esmfold2_proposal_lm_dropout == pytest.approx(0.5)
    assert args.esmfold2_force_proposal_lm_dropout is True
    assert args.esmfold2_one_step_diffusion_hotspot is False
    assert args.esmfold2_one_step_diffusion is True
    assert args.esmfold2_one_step_diffusion_mode == "truncated_trajectory"
    assert args.esmfold2_one_step_diffusion_tau == pytest.approx(1.0)
    assert args.esmfold2_one_step_truncated_num_sampling_steps == 50
    assert args.esmfold2_one_step_diffusion_weight == pytest.approx(0.25)
    assert args.esmfold2_one_step_diffusion_rg_weight == pytest.approx(0.1)
    assert args.esmfold2_one_step_diffusion_interval == 5
    assert args.esmfold2_one_step_recycle_coordinates is False
    assert args.esmfold2_distogram_hotspot is True
    assert args.esmfold2_stagea_handoff_checkpoint_interval == 10
    assert args.esmfold2_resume_handoff_checkpoint is None
    assert args.esmfold2_resume_reverify_roots is False
    assert args.esmfold2_hotspot_weight is None
    assert args.esmfold2_hotspot_satisfied_proposal_weight == pytest.approx(0.0)
    assert args.esmfold2_hotspot_interface_mode is None
    assert args.esmfold2_hotspot_verify_weight is None
    assert args.esmfold2_hotspot_verify_min_distance == pytest.approx(2.5)
    assert args.esmfold2_hotspot_verify_too_close_weight is None
    run_hallucination_script.resolve_esmfold2_hotspot_defaults(
        args,
        model_backend="esmfold2",
        hotspot_params=None,
    )
    assert args.esmfold2_hotspot_weight == pytest.approx(0.0)
    assert args.esmfold2_hotspot_cutoff == pytest.approx(8.0)
    assert args.esmfold2_hotspot_interface_mode == "additive"
    assert args.esmfold2_hotspot_verify_weight == pytest.approx(0.0)
    assert args.esmfold2_hotspot_verify_near_cutoff == pytest.approx(8.0)
    assert args.esmfold2_hotspot_verify_far_cutoff == pytest.approx(20.0)
    assert args.esmfold2_hotspot_verify_too_close_weight == pytest.approx(0.0)


def test_esmfold2_stagea_mutation_fraction_is_configurable_and_validated():
    args, parser = _parse(
        [
            "--model_backend",
            "esmfold2",
            "--mode",
            "mcts",
            "--esmfold2_stagea_mutation_fraction",
            "0.2",
            "--esmfold2_stagea_candidate_strategy",
            "positive_score",
            "--esmfold2_stagea_include_argmax_candidate",
            "false",
        ]
    )
    assert resolve_model_backend(args, parser) == "esmfold2"
    assert args.esmfold2_stagea_mutation_fraction == pytest.approx(0.2)
    assert args.esmfold2_stagea_candidate_strategy == "positive_score"
    assert args.esmfold2_stagea_include_argmax_candidate is False

    bad_args, bad_parser = _parse(
        [
            "--model_backend",
            "esmfold2",
            "--mode",
            "mcts",
            "--esmfold2_stagea_mutation_fraction",
            "1.1",
        ]
    )
    with pytest.raises(SystemExit):
        resolve_model_backend(bad_args, bad_parser)

def test_load_esmfold2_backend_forwards_cli_learning_rate(monkeypatch):
    args, parser = _parse(
        [
            "--model_backend",
            "esmfold2",
            "--mode",
            "mcts",
            "--learning_rate",
            "0.037",
        ]
    )
    resolve_model_backend(args, parser)
    run_hallucination_script.resolve_esmfold2_verifier_precision_defaults(args)
    run_hallucination_script.resolve_esmfold2_hotspot_defaults(
        args,
        model_backend="esmfold2",
        hotspot_params=None,
    )
    captured = {}

    class FakeRuntimeBackend:
        def __init__(self, *, config, device, task_type):
            captured["config"] = config
            captured["device"] = device
            captured["task_type"] = task_type

    monkeypatch.setattr(
        "bindcraft.hallucination.esmfold2_runtime.ESMFold2RuntimeBackend",
        FakeRuntimeBackend,
    )

    backend = run_hallucination_script.load_esmfold2_backend(args, device="cpu")

    assert isinstance(backend, FakeRuntimeBackend)
    assert captured["config"].learning_rate == pytest.approx(0.037)
    assert captured["device"] == "cpu"
    assert captured["task_type"] == "denovo"


def test_load_esmfold2_backend_forwards_esmc_activation_checkpointing(monkeypatch):
    args, parser = _parse(
        [
            "--model_backend",
            "esmfold2",
            "--mode",
            "mcts",
            "--esmfold2_esmc_activation_checkpointing",
            "true",
        ]
    )
    resolve_model_backend(args, parser)
    run_hallucination_script.resolve_esmfold2_verifier_precision_defaults(args)
    run_hallucination_script.resolve_esmfold2_hotspot_defaults(
        args,
        model_backend="esmfold2",
        hotspot_params=None,
    )
    captured = {}

    class FakeRuntimeBackend:
        def __init__(self, *, config, device, task_type):
            captured["config"] = config
            captured["device"] = device
            captured["task_type"] = task_type

    monkeypatch.setattr(
        "bindcraft.hallucination.esmfold2_runtime.ESMFold2RuntimeBackend",
        FakeRuntimeBackend,
    )

    run_hallucination_script.load_esmfold2_backend(args, device="cpu")

    assert captured["config"].esmc_activation_checkpointing is True


def test_esmfold2_one_step_interval_options_parse_explicitly():
    args, _ = _parse(
        [
            "--model_backend",
            "esmfold2",
            "--mode",
            "mcts",
            "--esmfold2_one_step_diffusion",
            "true",
            "--esmfold2_one_step_diffusion_interval",
            "5",
            "--esmfold2_one_step_recycle_coordinates",
            "true",
        ]
    )

    assert args.esmfold2_one_step_diffusion is True
    assert args.esmfold2_one_step_diffusion_interval == 5
    assert args.esmfold2_one_step_recycle_coordinates is True


def test_esmfold2_hotspot_defaults_to_localized_half_weight():
    args, parser = _parse(
        [
            "--model_backend",
            "esmfold2",
            "--mode",
            "mcts",
            "--target_hotspots",
            "2-3",
        ]
    )
    hotspot_params = run_hallucination_script._build_hotspot_params(args, parser)

    run_hallucination_script.resolve_esmfold2_hotspot_defaults(
        args,
        model_backend="esmfold2",
        hotspot_params=hotspot_params,
    )

    assert args.esmfold2_hotspot_weight == pytest.approx(0.5)
    assert args.esmfold2_hotspot_interface_mode == "localized"
    assert args.esmfold2_hotspot_verify_weight == pytest.approx(0.25)
    assert args.esmfold2_hotspot_satisfied_proposal_weight == pytest.approx(0.0)
    assert args.esmfold2_hotspot_verify_min_distance == pytest.approx(2.5)
    assert args.esmfold2_hotspot_verify_too_close_weight == pytest.approx(0.25)


def test_explicit_esmfold2_hotspot_settings_override_conditional_defaults():
    args, parser = _parse(
        [
            "--model_backend",
            "esmfold2",
            "--mode",
            "mcts",
            "--target_hotspots",
            "2",
            "--esmfold2_hotspot_weight",
            "0.8",
            "--esmfold2_hotspot_interface_mode",
            "additive",
            "--esmfold2_hotspot_verify_weight",
            "0.3",
        ]
    )
    hotspot_params = run_hallucination_script._build_hotspot_params(args, parser)

    run_hallucination_script.resolve_esmfold2_hotspot_defaults(
        args,
        model_backend="esmfold2",
        hotspot_params=hotspot_params,
    )

    assert args.esmfold2_hotspot_weight == pytest.approx(0.8)
    assert args.esmfold2_hotspot_interface_mode == "additive"
    assert args.esmfold2_hotspot_verify_weight == pytest.approx(0.3)


def test_protenix_hotspots_do_not_enable_esmfold2_hotspot_defaults():
    args, parser = _parse(["--target_hotspots", "2"])
    hotspot_params = run_hallucination_script._build_hotspot_params(args, parser)

    run_hallucination_script.resolve_esmfold2_hotspot_defaults(
        args,
        model_backend="protenix",
        hotspot_params=hotspot_params,
    )

    assert args.esmfold2_hotspot_weight == pytest.approx(0.0)
    assert args.esmfold2_hotspot_interface_mode == "additive"
    assert args.esmfold2_hotspot_verify_weight == pytest.approx(0.0)


def test_explicit_zero_disables_default_esmfold2_hotspot_verifier_weight():
    args, parser = _parse(
        [
            "--model_backend",
            "esmfold2",
            "--mode",
            "mcts",
            "--target_hotspots",
            "2",
            "--esmfold2_hotspot_verify_weight",
            "0",
        ]
    )
    hotspot_params = run_hallucination_script._build_hotspot_params(args, parser)

    run_hallucination_script.resolve_esmfold2_hotspot_defaults(
        args,
        model_backend="esmfold2",
        hotspot_params=hotspot_params,
    )

    assert args.esmfold2_hotspot_weight == pytest.approx(0.5)
    assert args.esmfold2_hotspot_interface_mode == "localized"
    assert args.esmfold2_hotspot_verify_weight == pytest.approx(0.0)


def test_esmfold2_backend_accepts_explicit_stage_a_only_strategy():
    args, parser = _parse(
        [
            "--model_backend",
            "esmfold2",
            "--mode",
            "mcts",
            "--mcts_strategy",
            "stage_a_only",
        ]
    )

    assert resolve_model_backend(args, parser) == "esmfold2"
    assert args.mcts_strategy == "stage_a_only"


def test_esmfold2_backend_preserves_explicit_two_stage_strategy():
    args, parser = _parse(
        [
            "--model_backend",
            "esmfold2",
            "--mode",
            "mcts",
            "--mcts_strategy",
            "two_stage",
        ]
    )

    assert resolve_model_backend(args, parser) == "esmfold2"
    assert args.mcts_strategy == "two_stage"


def test_esmfold2_best_of_n_resolves_strategy_specific_defaults():
    args, parser = _parse(
        [
            "--model_backend",
            "esmfold2",
            "--mode",
            "mcts",
            "--mcts_strategy",
            "best_of_n",
        ]
    )

    assert resolve_model_backend(args, parser) == "esmfold2"
    assert args.esmfold2_best_of_n == 8
    assert args.esmfold2_best_of_n_verify_candidates_per_outer_step == 0
    assert args.esmfold2_best_of_n_continuation_policy == "raw_argmax"
    assert args.esmfold2_stagea_soft_steps == 150
    assert args.esmfold2_stageb_soft_steps == 20
    assert args.esmfold2_track_proposal_confidence is True
    assert args.esmfold2_proposal_confidence_tail_steps == 20


def test_esmfold2_raw_lineage_resolves_tail_confidence_defaults():
    args, parser = _parse(
        [
            "--model_backend",
            "esmfold2",
            "--mode",
            "mcts",
            "--mcts_strategy",
            "raw_lineage_stage_a",
        ]
    )

    assert resolve_model_backend(args, parser) == "esmfold2"
    assert args.esmfold2_stagea_soft_steps == 30
    assert args.esmfold2_track_proposal_confidence is False
    assert args.esmfold2_proposal_confidence_tail_steps == 20


def test_esmfold2_best_of_n_explicit_overrides_win_over_defaults():
    args, parser = _parse(
        [
            "--model_backend",
            "esmfold2",
            "--mode",
            "mcts",
            "--mcts_strategy",
            "best_of_n",
            "--esmfold2_best_of_n",
            "3",
            "--esmfold2_stagea_soft_steps",
            "42",
            "--esmfold2_track_proposal_confidence",
            "false",
            "--esmfold2_proposal_confidence_tail_steps",
            "5",
            "--esmfold2_best_of_n_verify_candidates_per_outer_step",
            "8",
            "--esmfold2_best_of_n_continuation_policy",
            "verifier_best",
        ]
    )

    assert resolve_model_backend(args, parser) == "esmfold2"
    assert args.esmfold2_best_of_n == 3
    assert args.esmfold2_best_of_n_verify_candidates_per_outer_step == 8
    assert args.esmfold2_best_of_n_continuation_policy == "verifier_best"
    assert args.esmfold2_stagea_soft_steps == 42
    assert args.esmfold2_track_proposal_confidence is False
    assert args.esmfold2_proposal_confidence_tail_steps == 5


def test_esmfold2_legacy_soft_steps_override_sets_both_stages():
    args, parser = _parse(
        [
            "--model_backend",
            "esmfold2",
            "--mode",
            "mcts",
            "--esmfold2_soft_steps",
            "2",
        ]
    )

    assert resolve_model_backend(args, parser) == "esmfold2"
    assert args.esmfold2_stagea_soft_steps == 2
    assert args.esmfold2_stageb_soft_steps == 2


def test_esmfold2_backend_rejects_non_mcts_modes_until_pipeline_exists():
    args, parser = _parse(["--model_backend", "esmfold2", "--mode", "hybrid"])

    with pytest.raises(SystemExit):
        resolve_model_backend(args, parser)


def test_esmfold2_backend_exposes_separate_model_and_proposal_options():
    args, parser = _parse(
        [
            "--model_backend",
            "esmfold2",
            "--mode",
            "mcts",
            "--esmfold2_verifier_model",
            "biohub/ESMFold2-Fast",
            "--esmfold2_experimental_model",
            "biohub/ESMFold2-Experimental-Fast",
            "--esmfold2_soft_steps",
            "2",
            "--esmfold2_stagea_soft_steps",
            "30",
            "--esmfold2_stageb_soft_steps",
            "5",
            "--esmfold2_lambda_esmc",
            "0.2",
            "--esmfold2_task_type",
            "tcr",
            "--esmfold2_logit_init_mode",
            "legacy_anchor",
            "--esmfold2_mutable_mask",
            "0,1,1",
            "--esmfold2_verifier_num_loops",
            "2",
            "--esmfold2_verifier_num_sampling_steps",
            "12",
            "--esmfold2_proposal_num_loops",
            "1",
            "--esmfold2_proposal_num_sampling_steps",
            "3",
            "--esmfold2_proposal_lm_dropout",
            "0.5",
            "--esmfold2_force_proposal_lm_dropout",
            "true",
            "--esmfold2_track_proposal_confidence",
            "true",
            "--esmfold2_proposal_confidence_temperature",
            "0.07",
            "--esmfold2_proposal_confidence_interval",
            "2",
            "--esmfold2_proposal_confidence_num_loops",
            "1",
            "--esmfold2_proposal_confidence_num_sampling_steps",
            "50",
            "--esmfold2_proposal_confidence_max_candidates",
            "6",
            "--esmfold2_proposal_confidence_tail_steps",
            "20",
            "--esmfold2_hotspot_verify_weight",
            "0.2",
            "--esmfold2_hotspot_verify_near_cutoff",
            "5.0",
            "--esmfold2_hotspot_verify_far_cutoff",
            "20.0",
            "--esmfold2_stageb_hotspot_distance_weight",
            "0.1",
            "--esmfold2_stageb_hotspot_miss_penalty",
            "0.25",
            "--esmfold2_stageb_min_pair_chain_iptm",
            "0.25",
            "--stagea_max_mutations_per_seq",
            "9",
            "--stageb_max_mutations_per_seq",
            "1",
        ]
    )

    assert resolve_model_backend(args, parser) == "esmfold2"
    assert args.esmfold2_verifier_model == "biohub/ESMFold2-Fast"
    assert args.esmfold2_experimental_model == "biohub/ESMFold2-Experimental-Fast"
    assert args.esmfold2_soft_steps == 2
    assert args.esmfold2_stagea_soft_steps == 30
    assert args.esmfold2_stageb_soft_steps == 5
    assert args.mcts_expand_actions_per_step == 2
    assert args.esmfold2_lambda_esmc == "0.2"
    assert args.esmfold2_task_type == "tcr"
    assert args.esmfold2_logit_init_mode == "legacy_anchor"
    assert args.esmfold2_mutable_mask == "0,1,1"
    assert args.esmfold2_verifier_num_loops == 2
    assert args.esmfold2_verifier_num_sampling_steps == 12
    assert args.esmfold2_proposal_num_loops == 1
    assert args.esmfold2_proposal_num_sampling_steps == 3
    assert args.esmfold2_proposal_lm_dropout == pytest.approx(0.5)
    assert args.esmfold2_force_proposal_lm_dropout is True
    assert args.esmfold2_track_proposal_confidence is True
    assert args.esmfold2_proposal_confidence_temperature == pytest.approx(0.07)
    assert args.esmfold2_proposal_confidence_interval == 2
    assert args.esmfold2_proposal_confidence_num_loops == 1
    assert args.esmfold2_proposal_confidence_num_sampling_steps == 50
    assert args.esmfold2_proposal_confidence_max_candidates == 6
    assert args.esmfold2_proposal_confidence_tail_steps == 20
    assert args.esmfold2_hotspot_verify_weight == pytest.approx(0.2)
    assert args.esmfold2_hotspot_verify_near_cutoff == pytest.approx(5.0)
    assert args.esmfold2_hotspot_verify_far_cutoff == pytest.approx(20.0)
    assert args.stagea_max_mutations_per_seq == 9
    assert args.stageb_max_mutations_per_seq == 1


def test_main_dispatches_esmfold2_without_loading_protenix(monkeypatch, tmp_path):
    captured = {}

    class FakeBackend:
        def verify(self, *, binder_seq, target_seq, seed=None):
            score = 0.9 if binder_seq == "AAY" else 0.1
            return build_esm_verifier_result(
                binder_seq=binder_seq,
                target_seq=target_seq,
                summary_confidence={"pair_chain_iptm": score},
            )

        def propose(self, **kwargs):
            return ["AAY"]

    def fail_if_protenix_loaded(*args, **kwargs):
        raise AssertionError("Protenix loader should not run for ESMFold2 backend")

    monkeypatch.setattr(
        run_hallucination_script,
        "load_protenix_model",
        fail_if_protenix_loaded,
    )
    def fake_load_esmfold2_backend(args, **kwargs):
        captured["task_type_arg"] = args.esmfold2_task_type
        return FakeBackend()

    monkeypatch.setattr(run_hallucination_script, "load_esmfold2_backend", fake_load_esmfold2_backend)

    def fake_esm_pipeline(**kwargs):
        captured.update(kwargs)
        return "AAY", "AAY", [0.1]

    monkeypatch.setattr(
        "bindcraft.hallucination.esmfold2_pipeline.run_esmfold2_mcts_hallucination",
        fake_esm_pipeline,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_hallucination.py",
            "--target",
            "TTTT",
            "--mode",
            "mcts",
            "--model_backend",
            "esmfold2",
            "--binder_length",
            "3",
            "--n_steps",
            "1",
            "--mcts_num_beams",
            "1",
            "--b_candidates",
            "1",
            "--stagea_max_mutations_per_seq",
            "7",
            "--stageb_max_mutations_per_seq",
            "1",
            "--init_binder_seq",
            "AAA",
            "--esmfold2_task_type",
            "tcr",
            "--esmfold2_mutable_mask",
            "0,1,1",
            "--target_hotspots",
            "2",
            "--esmfold2_hotspot_weight",
            "0.4",
            "--esmfold2_hotspot_cutoff",
            "8.0",
            "--esmfold2_hotspot_verify_weight",
            "0.2",
            "--esmfold2_hotspot_verify_near_cutoff",
            "5.0",
            "--esmfold2_hotspot_verify_far_cutoff",
            "20.0",
            "--esmfold2_stageb_hotspot_distance_weight",
            "0.1",
            "--esmfold2_stageb_hotspot_miss_penalty",
            "0.25",
            "--esmfold2_stageb_min_pair_chain_iptm",
            "0.25",
            "--target_template_path",
            str(tmp_path / "target.cif"),
            "--esmfold2_initial_logits_checkpoint",
            str(tmp_path / "initial_logits.pt"),
            "--esmfold2_save_visualization_trace",
            "true",
            "--esmfold2_stagea_root_selection_mode",
            "hierarchical_ucb",
            "--esmfold2_stagea_min_root_expansions",
            "2",
            "--output_dir",
            str(tmp_path),
            "--log_file",
            str(tmp_path / "run.log"),
        ],
    )

    run_hallucination_script.main()

    result_text = (tmp_path / "hallucination_result.txt").read_text()
    log_text = (tmp_path / "run.log").read_text()
    assert "Model Backend: esmfold2" in result_text
    assert "Best Sequence: AAY" in result_text
    assert "ESMFold2 hotspot contact configured" in log_text
    assert "ESMFold2 Hotspot Weight: 0.4" in result_text
    assert "ESMFold2 Hotspot Interface Mode: localized" in result_text
    assert "ESMFold2 Hotspot Cutoff: 8.0 A" in result_text
    assert "ESMFold2 Hotspot Verify Weight: 0.2" in result_text
    assert "ESMFold2 Hotspot Verify Near Cutoff: 5.0 A" in result_text
    assert "ESMFold2 Hotspot Verify Far Cutoff: 20.0 A" in result_text
    assert "ESMFold2 hotspot verifier configured" in log_text
    assert "ESMFold2 backend received target_template_path but does not use templates yet" in log_text
    assert captured["mcts_strategy"] == "two_stage"
    assert captured["mcts_bootstrap_steps"] == 70
    assert captured["mcts_expand_actions_per_step"] == 2
    assert captured["mcts_handoff_top_k"] == 2
    assert captured["mcts_prior_mode"] == "no_group"
    assert captured["stagea_max_mutations_per_seq"] == 7
    assert captured["stageb_max_mutations_per_seq"] == 1
    assert captured["mutable_mask"] == [False, True, True]
    assert captured["hotspot_params"] == {
        "target_hotspot_residues_1based": [2],
        "target_hotspot_concat_indices": [1],
    }
    assert captured["esmfold2_logit_init_mode"] == "official_exact"
    assert captured["esmfold2_initial_logits_checkpoint"] == str(
        tmp_path / "initial_logits.pt"
    )
    assert captured["save_verified_candidate_cif"] is False
    assert captured["save_visualization_trace"] is True
    assert captured["esmfold2_stagea_soft_steps"] == 30
    assert captured["esmfold2_stagea_tree_children"] == 2
    assert captured["esmfold2_stagea_root_selection_mode"] == "hierarchical_ucb"
    assert captured["esmfold2_stagea_min_root_expansions"] == 2
    assert captured["esmfold2_stageb_soft_steps"] == 20
    assert captured["esmfold2_stageb_discrete_rerank"] is True
    assert captured["esmfold2_stageb_rerank_pool_size"] == 20
    assert captured["esmfold2_stageb_exploration_actions"] == 1
    assert captured["esmfold2_stageb_verifier_candidates"] == 8
    assert captured["esmfold2_stageb_hotspot_coverage_weight"] == pytest.approx(0.25)
    assert captured["esmfold2_stageb_hotspot_distance_weight"] == pytest.approx(0.1)
    assert captured["esmfold2_stageb_hotspot_miss_penalty"] == pytest.approx(0.25)
    assert captured["esmfold2_stageb_min_pair_chain_iptm"] == pytest.approx(0.25)
    assert captured["esmfold2_stageb_handoff_quality_tolerance"] == pytest.approx(0.05)
    assert captured["esmfold2_stageb_paired_verifier_seed"] is True
    assert captured["esmfold2_stageb_verifier_seed"] is None
    assert captured["esmfold2_stagea_handoff_checkpoint_interval"] == 10
    assert captured["esmfold2_resume_handoff_checkpoint"] is None
    assert captured["esmfold2_resume_reverify_roots"] is False
    assert captured["task_type_arg"] == "tcr"


def test_main_dispatches_best_of_n_without_entering_mcts(monkeypatch, tmp_path):
    captured = {}

    monkeypatch.setattr(
        run_hallucination_script,
        "load_protenix_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Protenix loader should not run")
        ),
    )
    monkeypatch.setattr(
        run_hallucination_script,
        "load_esmfold2_backend",
        lambda *args, **kwargs: object(),
    )

    def fake_best_of_n(**kwargs):
        captured.update(kwargs)
        return "AAY", "AAY", [0.3, 0.2, 0.1]

    monkeypatch.setattr(
        "bindcraft.hallucination.esmfold2_pipeline.run_esmfold2_best_of_n",
        fake_best_of_n,
    )
    monkeypatch.setattr(
        "bindcraft.hallucination.esmfold2_pipeline.run_esmfold2_mcts_hallucination",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("Best-of-N must not enter the MCTS pipeline")
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_hallucination.py",
            "--target",
            "TTTT",
            "--mode",
            "mcts",
            "--model_backend",
            "esmfold2",
            "--mcts_strategy",
            "best_of_n",
            "--esmfold2_best_of_n",
            "3",
            "--esmfold2_stagea_soft_steps",
            "4",
            "--binder_length",
            "3",
            "--output_dir",
            str(tmp_path),
        ],
    )

    run_hallucination_script.main()

    assert captured["best_of_n"] == 3
    assert captured["esmfold2_soft_steps"] == 4
    assert captured["track_confidence"] is True
    assert captured["mutable_mask"] is None
    assert "MCTS Strategy: best_of_n" in (
        tmp_path / "hallucination_result.txt"
    ).read_text()


def test_multichain_hotspot_parser_uses_residue_count_without_separator():
    parser = build_arg_parser()
    args = parser.parse_args(
        ["--target", "AA|BBB", "--target_hotspots", "5"]
    )

    assert run_hallucination_script._build_hotspot_params(args, parser) == {
        "target_hotspot_residues_1based": [5],
        "target_hotspot_concat_indices": [4],
    }

    out_of_range = parser.parse_args(
        ["--target", "AA|BBB", "--target_hotspots", "6"]
    )
    with pytest.raises(SystemExit):
        run_hallucination_script._build_hotspot_params(out_of_range, parser)


def test_main_rejects_positive_esmfold2_hotspot_weight_without_hotspots(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_hallucination.py",
            "--target",
            "TTTT",
            "--model_backend",
            "esmfold2",
            "--mode",
            "mcts",
            "--esmfold2_hotspot_weight",
            "0.4",
        ],
    )

    with pytest.raises(SystemExit):
        run_hallucination_script.main()


def test_main_rejects_positive_esmfold2_hotspot_verify_weight_without_hotspots(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_hallucination.py",
            "--target",
            "TTTT",
            "--model_backend",
            "esmfold2",
            "--mode",
            "mcts",
            "--esmfold2_hotspot_verify_weight",
            "0.2",
        ],
    )

    with pytest.raises(SystemExit):
        run_hallucination_script.main()


def test_esmfold2_mutable_mask_parser_accepts_compact_and_delimited_masks():
    parser = build_arg_parser()

    assert run_hallucination_script.parse_esmfold2_mutable_mask("011", 3, parser) == [
        False,
        True,
        True,
    ]
    assert run_hallucination_script.parse_esmfold2_mutable_mask("0,true,1", 3, parser) == [
        False,
        True,
        True,
    ]

    with pytest.raises(SystemExit):
        run_hallucination_script.parse_esmfold2_mutable_mask("01", 3, parser)


def test_esmfold2_target_chain_roles_parser_is_explicit_and_validated():
    parser = build_arg_parser()

    assert run_hallucination_script.parse_esmfold2_target_chain_roles(
        "target,scaffold,scaffold",
        n_target_chains=3,
        parser=parser,
    ) == ("target", "scaffold", "scaffold")
    assert (
        run_hallucination_script.parse_esmfold2_target_chain_roles(
            None,
            n_target_chains=2,
            parser=parser,
        )
        is None
    )

    with pytest.raises(SystemExit):
        run_hallucination_script.parse_esmfold2_target_chain_roles(
            "target,scaffold",
            n_target_chains=3,
            parser=parser,
        )


def test_esmfold2_helper_import_does_not_require_protenix_package():
    code = """
import importlib.abc
import sys

class BlockProtenix(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'protenix' or fullname.startswith('protenix.'):
            raise ImportError('blocked protenix import')
        return None

sys.meta_path.insert(0, BlockProtenix())
import bindcraft.hallucination.esmfold2_backend
print('esm-only import ok')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(run_hallucination_script.PROJECT_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "esm-only import ok" in completed.stdout
