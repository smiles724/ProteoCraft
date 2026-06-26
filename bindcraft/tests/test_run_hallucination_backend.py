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


def test_esmfold2_backend_is_explicit_and_mcts_only_for_first_version():
    args, parser = _parse(["--model_backend", "esmfold2", "--mode", "mcts"])

    assert args.model_backend == "esmfold2"
    assert resolve_model_backend(args, parser) == "esmfold2"


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
            "--esmfold2_lambda_esmc",
            "0.2",
            "--esmfold2_task_type",
            "tcr",
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
    assert args.esmfold2_lambda_esmc == "0.2"
    assert args.esmfold2_task_type == "tcr"
    assert args.esmfold2_mutable_mask == "0,1,1"
    assert args.esmfold2_verifier_num_loops == 2
    assert args.esmfold2_verifier_num_sampling_steps == 12
    assert args.esmfold2_proposal_num_loops == 1
    assert args.esmfold2_proposal_num_sampling_steps == 3
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
            "--target_template_path",
            str(tmp_path / "target.cif"),
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
    assert "ESMFold2 backend received target_hotspots but does not use hotspot-aware routing yet" in log_text
    assert "ESMFold2 backend received target_template_path but does not use templates yet" in log_text
    assert captured["mcts_strategy"] == "two_stage"
    assert captured["mcts_bootstrap_steps"] == 50
    assert captured["mcts_expand_actions_per_step"] == 2
    assert captured["mcts_prior_mode"] == "full"
    assert captured["stagea_max_mutations_per_seq"] == 7
    assert captured["stageb_max_mutations_per_seq"] == 1
    assert captured["mutable_mask"] == [False, True, True]
    assert captured["task_type_arg"] == "tcr"


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
