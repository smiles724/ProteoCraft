import json

import pytest

from bindcraft.affibody.profiles import get_affibody_profile
from bindcraft.affibody.targets import build_pmhc_target
from scripts.run_affibody_design import (
    build_esmfold2_affibody_command,
    validate_completed_esmfold2_run,
)
from scripts.score_affibody_complexes import (
    build_parser as build_blind_score_parser,
    main as score_affibody_complexes,
)


def _command_options(command):
    options = {}
    for index, token in enumerate(command):
        if token.startswith("--") and index + 1 < len(command):
            options[token] = command[index + 1]
    return options


def test_affibody_command_is_formal_esmfold2_scaffold_redesign(tmp_path):
    profile = get_affibody_profile("libb")
    target = build_pmhc_target(
        target_id="T001",
        peptide="SLLMWITQV",
        hla_sequence="AAAA",
        b2m_sequence="GGG",
    )

    command = build_esmfold2_affibody_command(
        python_executable="python",
        target=target,
        profile=profile,
        initial_sequence=profile.render(),
        run_dir=tmp_path / "run",
        device="cuda:0",
        seed=43,
    )
    options = _command_options(command)

    assert options["--profile"] == "esmfold2_d"
    assert options["--model_backend"] == "esmfold2"
    assert options["--mode"] == "mcts"
    assert options["--mcts_strategy"] == "two_stage"
    assert options["--esmfold2_task_type"] == "affibody_redesign"
    assert options["--esmfold2_logit_init_mode"] == "official_exact"
    assert options["--init_binder_seq"] == profile.render()
    assert options["--esmfold2_mutable_mask"] == profile.mutable_mask_argument
    assert options["--esmfold2_allowed_binder_aas"] == profile.allowed_aas_argument
    assert options["--esmfold2_target_chain_roles"] == "scaffold,scaffold,target"
    assert options["--esmfold2_verifier_loss_mode"] == "aligned_common"
    assert options["--target_hotspots"] == "11,12"


def test_completed_run_contract_rejects_framework_or_alphabet_edits(tmp_path):
    profile = get_affibody_profile("libb")
    invalid = list(profile.render())
    invalid[0] = "A"
    result = tmp_path / "hallucination_result.txt"
    result.write_text(
        "\n".join(
            [
                "Model Backend: esmfold2",
                "Launch Profile: esmfold2_d",
                "Mode: mcts",
                "MCTS Strategy: two_stage",
                "ESMFold2 Task Type: affibody_redesign",
                "ESMFold2 Logit Init Mode: official_exact",
                f"ESMFold2 Mutable Mask: {profile.mutable_mask_argument}",
                f"ESMFold2 Allowed Binder AAs: {profile.allowed_aas_argument}",
                "ESMFold2 Target Chain Roles: scaffold,scaffold,target",
                "ESMFold2 Verifier Loss Mode: aligned_common",
                f"Best Sequence: {''.join(invalid)}",
                f"Final Sequence: {profile.render()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="framework position 1"):
        validate_completed_esmfold2_run(run_dir=tmp_path, profile=profile)


def test_completed_run_contract_writes_validated_summary(tmp_path):
    profile = get_affibody_profile("liba")
    result = tmp_path / "hallucination_result.txt"
    result.write_text(
        "\n".join(
            [
                "Model Backend: esmfold2",
                "Launch Profile: esmfold2_d",
                "Mode: mcts",
                "MCTS Strategy: two_stage",
                "ESMFold2 Task Type: affibody_redesign",
                "ESMFold2 Logit Init Mode: official_exact",
                f"ESMFold2 Mutable Mask: {profile.mutable_mask_argument}",
                f"ESMFold2 Allowed Binder AAs: {profile.allowed_aas_argument}",
                "ESMFold2 Target Chain Roles: scaffold,scaffold,target",
                "ESMFold2 Verifier Loss Mode: aligned_common",
                f"Best Sequence: {profile.render()}",
                f"Final Sequence: {profile.render()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = validate_completed_esmfold2_run(run_dir=tmp_path, profile=profile)

    assert summary["model_backend"] == "esmfold2"
    assert summary["target_chain_roles"] == ["scaffold", "scaffold", "target"]
    assert summary["verifier_loss_mode"] == "aligned_common"
    assert summary["best_tuple"] == profile.default_tuple
    assert summary["framework_unchanged"] is True
    assert (tmp_path / "affibody_result.json").is_file()


def test_completed_run_contract_rejects_old_all_target_roles(tmp_path):
    profile = get_affibody_profile("liba")
    result = tmp_path / "hallucination_result.txt"
    result.write_text(
        "\n".join(
            [
                "Model Backend: esmfold2",
                "Launch Profile: esmfold2_d",
                "Mode: mcts",
                "MCTS Strategy: two_stage",
                "ESMFold2 Task Type: affibody_redesign",
                "ESMFold2 Logit Init Mode: official_exact",
                f"ESMFold2 Mutable Mask: {profile.mutable_mask_argument}",
                f"ESMFold2 Allowed Binder AAs: {profile.allowed_aas_argument}",
                "ESMFold2 Target Chain Roles: target,target,target",
                "ESMFold2 Verifier Loss Mode: aligned_common",
                f"Best Sequence: {profile.render()}",
                f"Final Sequence: {profile.render()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Target Chain Roles"):
        validate_completed_esmfold2_run(run_dir=tmp_path, profile=profile)


def test_blind_score_plan_uses_design_loss_and_peptide_only_target(tmp_path):
    profile = get_affibody_profile("libb")
    complexes = tmp_path / "complexes.csv"
    complexes.write_text(
        "complex_id,library,peptide,register,affibody_sequence,affibody_tuple\n"
        f"C001,libb,SLLMWITQV,P4/P5,{profile.render()},{profile.default_tuple}\n",
        encoding="utf-8",
    )
    reference = tmp_path / "reference.pdb"
    reference.write_text("REMARK plan-only test\n", encoding="utf-8")
    output = tmp_path / "scores.jsonl"

    assert build_blind_score_parser().parse_args([]).verifier_loss_mode == "aligned_common"
    assert score_affibody_complexes(
        [
            "--complexes", str(complexes),
            "--reference-pdb", str(reference),
            "--output", str(output),
            "--seeds", "101",
        ]
    ) == 0

    plan = json.loads(
        (tmp_path / "scores.jsonl.plan.json").read_text(encoding="utf-8")
    )
    assert plan["executed"] is False
    assert plan["config"]["loss_mode"] == "aligned_common"
    assert plan["config"]["target_chain_roles"] == [
        "scaffold", "scaffold", "target"
    ]
