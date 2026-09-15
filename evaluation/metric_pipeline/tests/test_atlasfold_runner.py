"""Tests for the pinned AtlasFold-M sequence-only evaluator adapter."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_chain_layout_keeps_targets_first_and_binder_last():
    from eval_pipeline.atlasfold_runner import build_chain_layout

    layout = build_chain_layout(
        "ACD:EFG",
        "HIK",
        target_chain_roles="target,scaffold",
    )
    assert layout.sequences == ("ACD", "EFG", "HIK")
    assert layout.chain_ids == ("A", "B", "C")
    assert layout.target_metric_chain_ids == ("A",)
    assert layout.scaffold_chain_ids == ("B",)
    assert layout.binder_chain_id == "C"
    assert [segment["kind"] for segment in layout.clean_segments] == [
        "target_chain_0",
        "target_chain_1",
        "binder",
    ]
    assert [segment["clean_start"] for segment in layout.clean_segments] == [0, 3, 6]


def test_chain_layout_rejects_nonstandard_residue_instead_of_replacing_x():
    import pytest

    from eval_pipeline.atlasfold_runner import build_chain_layout

    with pytest.raises(ValueError, match="implicit X replacement"):
        build_chain_layout("ACX", "HIK")


def test_stage_inputs_writes_colon_multimer_fasta_and_mapping(tmp_path):
    from eval_pipeline.atlasfold_runner import stage_inputs_for_sub_run

    rows = [{
        "method": "ProteoCraft",
        "case": "case_1",
        "design_id": "candidate_1",
        "target_seq": "ACD:EFG",
        "target_chain_roles": "target,scaffold",
        "binder_seq": "HIK",
    }]
    staged = stage_inputs_for_sub_run(rows=rows, staging_dir=tmp_path)
    fasta = (tmp_path / "atlasfold_m_inputs.fasta").read_text().splitlines()
    assert fasta[0] == f">{staged[0]['job_name']}"
    assert fasta[1] == "ACD:EFG:HIK"
    mapping = json.loads((tmp_path / "chain_mapping.json").read_text())
    assert mapping["jobs"][0]["chain_ids"] == ["A", "B", "C"]
    assert mapping["jobs"][0]["target_chain_roles"] == ["target", "scaffold"]
    assert mapping["jobs"][0]["binder_chain_id"] == "C"


def test_build_command_uses_official_multimer_cli_and_reviewed_defaults(tmp_path):
    from eval_pipeline.atlasfold_runner import build_command

    command = build_command(
        atlasfold_python="/env/bin/python",
        fasta_path=tmp_path / "in.fasta",
        output_dir=tmp_path / "out",
        device="cuda:0",
    )
    assert command[:4] == ["/env/bin/python", "-m", "atlasfold.cli", "multimer"]
    assert command[command.index("--num-recycles") + 1] == "10"
    assert command[command.index("--num-samples") + 1] == "5"
    assert command[command.index("--num-steps") + 1] == "200"
    assert command[command.index("--seed") + 1] == "1"
    assert "--save-confidence-arrays" in command
    assert command[command.index("--format") + 1] == "cif"
    assert "--overwrite" not in command

    overwrite_command = build_command(
        atlasfold_python="/env/bin/python",
        fasta_path=tmp_path / "in.fasta",
        output_dir=tmp_path / "out",
        device="cuda:0",
        overwrite=True,
    )
    assert "--overwrite" in overwrite_command


def test_job_name_changes_when_sequence_changes():
    from eval_pipeline.atlasfold_runner import _safe_job_name

    row = {
        "method": "ProteoCraft",
        "case": "PD1",
        "design_id": "candidate_1",
        "target_seq": "AAA",
        "binder_seq": "CCC",
    }
    changed = {**row, "binder_seq": "CAC"}
    assert _safe_job_name(row) != _safe_job_name(changed)


def test_run_config_refuses_mismatched_resume(tmp_path):
    import pytest

    from eval_pipeline.cli import _write_or_validate_atlasfold_run_config

    _write_or_validate_atlasfold_run_config(
        tmp_path, {"num_steps": 200}, allow_replace=False
    )
    _write_or_validate_atlasfold_run_config(
        tmp_path, {"num_steps": 200}, allow_replace=False
    )
    with pytest.raises(SystemExit, match="different run configuration"):
        _write_or_validate_atlasfold_run_config(
            tmp_path, {"num_steps": 10}, allow_replace=False
        )


def test_collect_atlasfold_env_preserves_venv_python_symlink(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from eval_pipeline import cli

    base_python = tmp_path / "base-python"
    base_python.write_text("")
    venv_dir = tmp_path / "venv" / "bin"
    venv_dir.mkdir(parents=True)
    venv_python = venv_dir / "python"
    venv_python.symlink_to(base_python)
    source_root = tmp_path / "atlasfold"
    source_root.mkdir()
    model_path = tmp_path / "model.pth"
    lm_path = tmp_path / "lm.pth"
    receipt_path = tmp_path / "receipt.json"
    for path in (model_path, lm_path, receipt_path):
        path.write_text("")

    monkeypatch.setattr(
        cli.atlasfold_runner,
        "verify_source_checkout",
        lambda path: {"source_root": str(path)},
    )
    monkeypatch.setattr(
        cli.atlasfold_runner,
        "probe_environment",
        lambda python, source: {"python": python, "source": str(source)},
    )
    monkeypatch.setattr(
        cli.atlasfold_runner,
        "validate_weights_receipt",
        lambda receipt, model_path, lm_path: {"receipt": str(receipt)},
    )
    args = SimpleNamespace(
        atlasfold_python=str(venv_python),
        atlasfold_source_root=str(source_root),
        atlasfold_model_path=str(model_path),
        atlasfold_lm_path=str(lm_path),
        atlasfold_weights_receipt=str(receipt_path),
    )

    metadata = cli._collect_atlasfold_env(args)

    assert metadata["atlasfold_python"] == str(venv_python.absolute())
    assert metadata["atlasfold_python"] != str(base_python.resolve())


def test_weights_receipt_requires_immutable_matching_snapshots(tmp_path):
    import pytest

    from eval_pipeline.atlasfold_runner import validate_weights_receipt

    model_path = tmp_path / "atlasfold-m.pth"
    lm_path = tmp_path / "atlaslm.pth"
    model_path.write_bytes(b"model")
    lm_path.write_bytes(b"lm")
    receipt_path = tmp_path / "receipt.json"
    receipt = {
        "schema_version": 1,
        "downloads": {
            "atlasfold_m": {
                "repo_id": "SeonghwanSeo/atlasfold-m-260725",
                "resolved_revision": "a9f07668b7ef544d4470795ef03aa492f243a461",
                "weight_path": str(model_path),
                "weight_size_bytes": model_path.stat().st_size,
            },
            "atlaslm_3b": {
                "repo_id": "SeonghwanSeo/atlaslm-3b-base",
                "resolved_revision": "b4e6d8abb92d3432e3a65d37f1cfafc18575f78e",
                "weight_path": str(lm_path),
                "weight_size_bytes": lm_path.stat().st_size,
            },
        },
    }
    receipt_path.write_text(json.dumps(receipt))
    assert validate_weights_receipt(
        receipt_path, model_path=model_path, lm_path=lm_path
    )["receipt_path"] == str(receipt_path.resolve())

    receipt["downloads"]["atlasfold_m"]["resolved_revision"] = "main"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="non-immutable revision"):
        validate_weights_receipt(receipt_path, model_path=model_path, lm_path=lm_path)


def test_atlasfold_rerun_replaces_stale_csv_identity():
    from eval_pipeline.cli import _drop_existing_rows_for_rerun

    stale = {"method": "ProteoCraft", "case": "PD1", "design_id": "d1"}
    untouched = {"method": "ProteoCraft", "case": "PD1", "design_id": "d2"}
    assert _drop_existing_rows_for_rerun([stale, untouched], [stale]) == [untouched]


def test_mock_official_outputs_parse_target_and_scaffold_interfaces(tmp_path, monkeypatch):
    from eval_pipeline import atlasfold_runner

    row = {
        "method": "ProteoCraft",
        "case": "cdr_case",
        "design_id": "candidate_1",
        "target_seq": "ACD:EFG",
        "target_chain_roles": "target,scaffold",
        "binder_seq": "HIK",
    }
    staging = tmp_path / "staging"
    output = tmp_path / "raw"
    staged = atlasfold_runner.stage_inputs_for_sub_run(rows=[row], staging_dir=staging)
    monkeypatch.setenv("ATLASFOLD_MOCK", "1")
    result = atlasfold_runner.run_atlasfold_batched(
        staging_dir=staging,
        output_dir=output,
        atlasfold_python="/fake/python",
        source_root=tmp_path / "fake-source",
        device="cuda:0",
        timeout_s=30,
    )
    assert result["exit_code"] == 0
    parsed = atlasfold_runner.parse_one_atlasfold_design(
        atlasfold_output_dir=output,
        staged_design=staged[0],
        amortized_runtime_s=1.25,
        out_dir=tmp_path / "parsed",
        run_config={"test": True},
    )
    assert parsed.plddt.shape == (9,)
    assert parsed.pae.shape == (9, 9)
    assert parsed.native["pair_chain_iptm"] == 0.75
    assert parsed.native["scaffold_pair_chain_iptm"] == 0.75
    assert parsed.native["target_plddt"] == 90.0
    assert parsed.native["binder_plddt"] == 90.0
    assert Path(parsed.mmcif_path).exists()
    assert Path(parsed.pdb_path).exists()


def _write_reference_pdb(path: Path) -> None:
    one_to_three = {"A": "ALA", "C": "CYS"}
    serial = 1
    lines: list[str] = []
    for chain_index, (chain_id, sequence) in enumerate((("A", "AAA"), ("B", "CCC"))):
        for residue_index, aa in enumerate(sequence, start=1):
            x = (residue_index - 1) * 3.8 + 1.3
            y = chain_index * 4.0
            lines.append(
                f"ATOM  {serial:5d}  CA  {one_to_three[aa]} {chain_id}{residue_index:4d}    "
                f"{x:8.3f}{y:8.3f}{0.0:8.3f}  1.00 90.00           C"
            )
            serial += 1
        lines.append("TER")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def test_atlasfold_mock_cli_e2e_does_not_route_through_other_evaluators(tmp_path):
    reference = tmp_path / "reference.pdb"
    _write_reference_pdb(reference)
    manifest = tmp_path / "manifest.csv"
    first_row = {
        "method": "ProteoCraft",
        "case": "case_1",
        "paper_case": "case_1",
        "design_id": "candidate_1",
        "target_pdb_path": str(reference),
        "target_chain": "A",
        "target_seq": "AAA",
        "binder_seq": "CCC",
        "designed_complex_pdb_path": str(reference),
        "target_chain_in_complex": "A",
        "binder_chain_in_complex": "B",
        "n_generated_trajectories": 1,
        "n_mpnn_attempted": -1,
        "n_accepted_by_method": 1,
        "candidate_rank": 0,
        "source_filter_status": "accepted",
        "verifier_calls": 1,
        "design_gpu_seconds": 1.0,
        "target_chain_roles": "target",
        "target_hotspot_concat_indices": "",
    }
    second_row = {
        **first_row,
        "design_id": "candidate_2",
        "designed_complex_pdb_path": "",
    }
    pd.DataFrame([first_row, second_row]).to_csv(manifest, index=False)

    env = {**os.environ, "ATLASFOLD_MOCK": "1"}
    command = [
        sys.executable,
        "-m",
        "eval_pipeline.cli",
        "--manifest",
        str(manifest),
        "--out-dir",
        str(tmp_path / "work"),
        "--evaluator",
        "atlasfold_m",
        "--paper-case",
        "case_1",
        "--sub-run",
        "case_1",
        "--atlasfold-python",
        "/fake/python",
        "--atlasfold-source-root",
        "/fake/source",
        "--no-skip-existing",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"stdout={completed.stdout}\nstderr={completed.stderr}"
    )
    output_csv = tmp_path / "work" / "eval" / "case_1__case_1__atlasfold_m.csv"
    result = pd.read_csv(output_csv)
    assert len(result) == 2
    assert result.loc[0, "evaluator"] == "atlasfold_m"
    assert result.loc[0, "atlasfold_model"] == "atlasfold-m-260725"
    assert result.loc[0, "pair_chain_iptm"] == 0.75
    assert int(result.loc[0, "eval_n_successes"]) == 1
    assert int(result.loc[1, "eval_n_successes"]) == 1
    assert result.loc[0, "atlasfold_reference_metric_status"] == "computed"
    assert result.loc[1, "atlasfold_reference_metric_status"] == "missing"
    assert pd.isna(result.loc[1, "binder_scRMSD"])
    assert not bool(result.loc[0, "tier1_pass"])
    assert not bool(result.loc[0, "tier2_pass"])

    report_command = [
        sys.executable,
        "-m",
        "eval_pipeline.report",
        "--out-dir",
        str(tmp_path / "work"),
        "--auto-discover",
        "--evaluators",
        "atlasfold_m",
        "--run-id",
        "atlasfold_mock",
    ]
    report = subprocess.run(
        report_command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert report.returncode == 0, f"stdout={report.stdout}\nstderr={report.stderr}"
    report_text = (tmp_path / "work" / "report_atlasfold_m.md").read_text()
    assert "AtlasFold-M native metrics" in report_text
    assert "not calibrated for AtlasFold-M" in report_text
    assert (tmp_path / "work" / "bar_chart_atlasfold_m.png").is_file()
    aggregate = pd.read_csv(
        tmp_path / "work" / "eval_per_paper_case_aggregate.csv"
    )
    assert pd.isna(aggregate.loc[0, "tier1_rate"])
    assert pd.isna(aggregate.loc[0, "tier2_rate"])
