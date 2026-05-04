"""Tests for cli.py evaluator branching."""
import argparse
import sys

import numpy as np
import pandas as pd
import pytest


def test_argparse_accepts_per_sub_run_flags():
    from eval_pipeline.cli import _build_argparser

    ap = _build_argparser()
    args = ap.parse_args([
        "--manifest", "m.csv", "--out-dir", "/tmp/out",
        "--evaluator", "af3",
        "--paper-case", "BetV1", "--sub-run", "BetV1_h23",
        "--af3-python", "/p", "--af3-run-script", "/r",
        "--af3-model-dir", "/m", "--af3-db-dir", "/d",
    ])
    assert args.evaluator == "af3"
    assert args.paper_case == "BetV1"
    assert args.sub_run == "BetV1_h23"
    assert args.skip_existing is True


def test_argparse_no_skip_existing():
    from eval_pipeline.cli import _build_argparser

    ap = _build_argparser()
    args = ap.parse_args([
        "--manifest", "m.csv", "--out-dir", "/tmp/out",
        "--no-skip-existing",
    ])
    assert args.skip_existing is False


def test_design_dir_derived_from_out_root(tmp_path):
    from eval_pipeline.cli import _design_dir_for_row

    row = {"method": "BCv1", "case": "PD1", "design_id": "PD1_d_001"}
    out_root = tmp_path / "work"
    assert _design_dir_for_row(out_root, row) == (
        out_root / "per_design" / "BCv1" / "PD1" / "PD1_d_001"
    )


def test_resolve_output_csv_naming():
    from eval_pipeline.cli import _resolve_output_csv

    args = argparse.Namespace(
        out_dir="/work", paper_case="BetV1", sub_run="BetV1_h23", evaluator="af3",
    )
    p = _resolve_output_csv(args, sub_run="BetV1_h23")
    assert str(p).endswith("/work/eval/BetV1__BetV1_h23__af3.csv")


def test_resolve_output_csv_naming_no_sub_run():
    from eval_pipeline.cli import _resolve_output_csv

    args = argparse.Namespace(
        out_dir="/work", paper_case="PD1", sub_run=None, evaluator="esmfold",
    )
    p = _resolve_output_csv(args, sub_run=None)
    assert str(p).endswith("/work/eval/PD1__PD1__esmfold.csv")


def test_main_with_zero_row_manifest_writes_header_only_csv(tmp_path, monkeypatch):
    from eval_pipeline import cli
    from eval_pipeline.manifest import DENOMINATOR_COLS, OPTIONAL_COLS, REQUIRED_COLS

    manifest_path = tmp_path / "empty.manifest.csv"
    cols = REQUIRED_COLS + DENOMINATOR_COLS + OPTIONAL_COLS
    pd.DataFrame(columns=cols).to_csv(manifest_path, index=False)

    out_dir = tmp_path / "work"
    out_dir.mkdir()
    monkeypatch.setattr(sys, "argv", [
        "cli.py", "--manifest", str(manifest_path), "--out-dir", str(out_dir),
        "--evaluator", "esmfold",
        "--paper-case", "PD1", "--sub-run", "PD1",
    ])
    cli.main()
    expected_csv = out_dir / "eval" / "PD1__PD1__esmfold.csv"
    assert expected_csv.exists()
    df = pd.read_csv(expected_csv)
    assert len(df) == 0


def _manifest_row(case: str, design_id: str, paper_case: str) -> dict:
    return {
        "method": "BCv1",
        "case": case,
        "paper_case": paper_case,
        "design_id": design_id,
        "target_pdb_path": "/tmp/target.pdb",
        "target_chain": "A",
        "target_seq": "AAAA",
        "binder_seq": "BBBB",
        "designed_complex_pdb_path": "/tmp/ref.pdb",
        "target_chain_in_complex": "A",
        "binder_chain_in_complex": "B",
        "n_generated_trajectories": -1,
        "n_mpnn_attempted": -1,
        "n_accepted_by_method": 1,
        "candidate_rank": 0,
        "source_filter_status": "accepted",
        "verifier_calls": -1,
        "design_gpu_seconds": float("nan"),
        "binder_length": 4,
        "notes": "",
    }


def test_filter_by_paper_case_and_sub_run(tmp_path, monkeypatch):
    # Filter is evaluator-agnostic; the esmfold lane keeps this test lightweight.
    from eval_pipeline import cli

    manifest_path = tmp_path / "selected.manifest.csv"
    pd.DataFrame([
        _manifest_row("BetV1_h23", "keep", "BetV1"),
        _manifest_row("BetV1_h24", "drop_sub_run", "BetV1"),
        _manifest_row("PD1", "drop_paper_case", "PD1"),
    ]).to_csv(manifest_path, index=False)

    def fake_evaluate_one(row, out_root, tokenizer, model, device, *, evaluator):
        out = cli._empty_row()
        out.update({
            "method": row["method"],
            "case": row["case"],
            "paper_case": row["paper_case"],
            "design_id": row["design_id"],
            "target_chain": row["target_chain"],
            "binder_length": len(row["binder_seq"]),
            "designed_complex_pdb_path": row["designed_complex_pdb_path"],
            "evaluator": evaluator,
            "eval_n_calls": 1,
            "eval_n_successes": 1,
            "error_msg": "",
        })
        return out

    out_dir = tmp_path / "work"
    monkeypatch.setattr(cli, "load_esmfold", lambda device: (None, None))
    monkeypatch.setattr(cli, "evaluate_one", fake_evaluate_one)
    monkeypatch.setattr(sys, "argv", [
        "cli.py", "--manifest", str(manifest_path), "--out-dir", str(out_dir),
        "--evaluator", "esmfold",
        "--paper-case", "BetV1", "--sub-run", "BetV1_h23",
    ])

    cli.main()

    output_csv = out_dir / "eval" / "BetV1__BetV1_h23__esmfold.csv"
    df = pd.read_csv(output_csv)
    assert df["design_id"].tolist() == ["keep"]
    assert df["paper_case"].tolist() == ["BetV1"]
    assert df["case"].tolist() == ["BetV1_h23"]


def _write_ca_pdb(path, chain_id: str, n_res: int):
    lines = []
    for i in range(n_res):
        lines.append(
            f"ATOM  {i + 1:5d}  CA  ALA {chain_id}{i + 1:4d}    "
            f"{i * 3.8:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 80.00           C"
        )
    path.write_text("\n".join(lines) + "\nTER\nEND\n")


def test_af3_preflight_raises_on_missing_msa(tmp_path, monkeypatch):
    from eval_pipeline import cli

    ref_pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(ref_pdb, "A", 4)
    manifest_path = tmp_path / "m.csv"
    row = _manifest_row("PD1", "d1", "PD1")
    row.update({
        "target_seq": "AAAA",
        "designed_complex_pdb_path": str(ref_pdb),
    })
    pd.DataFrame([row]).to_csv(manifest_path, index=False)

    monkeypatch.setattr(sys, "argv", [
        "cli.py", "--manifest", str(manifest_path), "--out-dir", str(tmp_path),
        "--evaluator", "af3", "--paper-case", "PD1", "--sub-run", "PD1",
        "--af3-python", "/usr/bin/python", "--af3-run-script", str(manifest_path),
        "--af3-model-dir", str(tmp_path), "--af3-db-dir", str(tmp_path),
    ])

    with pytest.raises(SystemExit, match="missing MSA"):
        cli.main()


def test_af3_preflight_raises_on_ca_count_mismatch(tmp_path, monkeypatch):
    from eval_pipeline import cli

    ref_pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(ref_pdb, "A", 3)
    manifest_path = tmp_path / "m.csv"
    row = _manifest_row("TCR_1QRN", "d1", "TCR_1QRN")
    row.update({
        "target_seq": "AAAA:BBB",
        "designed_complex_pdb_path": str(ref_pdb),
        "target_chain_in_complex": "A",
    })
    pd.DataFrame([row]).to_csv(manifest_path, index=False)
    msa_dir = tmp_path / "per_paper_case_msa" / "TCR_1QRN"
    msa_dir.mkdir(parents=True)
    (msa_dir / "target_chain_B.a3m").write_text(">x\nAAAA\n")
    (msa_dir / "target_chain_C.a3m").write_text(">x\nBBB\n")

    monkeypatch.setattr(sys, "argv", [
        "cli.py", "--manifest", str(manifest_path), "--out-dir", str(tmp_path),
        "--evaluator", "af3", "--paper-case", "TCR_1QRN", "--sub-run", "TCR_1QRN",
        "--af3-python", "/usr/bin/python", "--af3-run-script", str(ref_pdb),
        "--af3-model-dir", str(tmp_path), "--af3-db-dir", str(tmp_path),
    ])

    with pytest.raises(SystemExit, match="CA count mismatch"):
        cli.main()


def test_af3_preflight_single_chain_mismatch_raises(tmp_path, monkeypatch):
    from eval_pipeline import cli

    ref_pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(ref_pdb, "A", 3)
    manifest_path = tmp_path / "m.csv"
    row = _manifest_row("PD1", "d1", "PD1")
    row.update({
        "target_seq": "AAAA",
        "designed_complex_pdb_path": str(ref_pdb),
        "target_chain_in_complex": "A",
    })
    pd.DataFrame([row]).to_csv(manifest_path, index=False)
    msa_dir = tmp_path / "per_paper_case_msa" / "PD1"
    msa_dir.mkdir(parents=True)
    (msa_dir / "target_chain_B.a3m").write_text(">x\nAAAA\n")

    monkeypatch.setattr(sys, "argv", [
        "cli.py", "--manifest", str(manifest_path), "--out-dir", str(tmp_path),
        "--evaluator", "af3", "--paper-case", "PD1", "--sub-run", "PD1",
        "--af3-python", "/usr/bin/python", "--af3-run-script", str(ref_pdb),
        "--af3-model-dir", str(tmp_path), "--af3-db-dir", str(tmp_path),
    ])

    with pytest.raises(SystemExit, match="CA count mismatch"):
        cli.main()


def test_af3_preflight_skipped_under_mock(tmp_path, monkeypatch):
    from eval_pipeline import cli

    manifest_path = tmp_path / "m.csv"
    row = _manifest_row("PD1", "d1", "PD1")
    row["designed_complex_pdb_path"] = str(tmp_path / "missing_ref.pdb")
    pd.DataFrame([row]).to_csv(manifest_path, index=False)

    monkeypatch.setenv("AF3_MOCK", "1")
    monkeypatch.setattr(cli.af3_runner, "cleanup_for_rerun", lambda *a, **kw: None)
    monkeypatch.setattr(cli.af3_runner, "stage_inputs_for_sub_run", lambda **kw: [])
    monkeypatch.setattr(cli.af3_runner, "run_af3_batched", lambda **kw: {
        "exit_code": 0,
        "wall_clock_s": 0.0,
        "stdout_tail": "",
        "stderr_tail": "",
        "timed_out": False,
    })
    monkeypatch.setattr(sys, "argv", [
        "cli.py", "--manifest", str(manifest_path), "--out-dir", str(tmp_path),
        "--evaluator", "af3", "--paper-case", "PD1", "--sub-run", "PD1",
        "--af3-python", "/not/real/python", "--af3-run-script", "/not/real/run.py",
        "--af3-model-dir", "/not/real/model", "--af3-db-dir", "/not/real/db",
        "--no-skip-existing",
    ])

    cli.main()


def test_run_af3_for_one_sub_run_empty_staged_writes_header_only_csv(tmp_path, monkeypatch):
    """SF2-2: zero staged AF3 designs should still leave a readable eval CSV."""
    from eval_pipeline import cli

    manifest_path = tmp_path / "m.csv"
    row = _manifest_row("PD1", "d1", "PD1")
    pd.DataFrame([row]).to_csv(manifest_path, index=False)

    monkeypatch.setenv("AF3_MOCK", "1")
    monkeypatch.setattr(cli.af3_runner, "cleanup_for_rerun", lambda *a, **kw: None)
    monkeypatch.setattr(cli.af3_runner, "stage_inputs_for_sub_run", lambda **kw: [])

    def fail_if_called(**kwargs):
        raise AssertionError("run_af3_batched should not run with zero staged designs")

    monkeypatch.setattr(cli.af3_runner, "run_af3_batched", fail_if_called)
    monkeypatch.setattr(sys, "argv", [
        "cli.py", "--manifest", str(manifest_path), "--out-dir", str(tmp_path),
        "--evaluator", "af3", "--paper-case", "PD1", "--sub-run", "PD1",
        "--af3-python", "/not/real/python", "--af3-run-script", "/not/real/run.py",
        "--af3-model-dir", "/not/real/model", "--af3-db-dir", "/not/real/db",
        "--no-skip-existing",
    ])

    cli.main()

    output_csv = tmp_path / "eval" / "PD1__PD1__af3.csv"
    out = pd.read_csv(output_csv)
    assert list(out.columns) == cli.CSV_COLS
    assert len(out) == 0


def test_build_failed_row_with_malformed_input_does_not_raise():
    """SF-19: _build_failed_row must produce a row for partially malformed input."""
    from eval_pipeline.cli import _build_failed_row

    row = _build_failed_row({"row": {"design_id": "d1"}}, "PD1", "PD1", "test")

    assert row["evaluator"] == "af3"
    assert row["case"] == "PD1"
    assert row["method"] == "unknown"
    assert row["binder_length"] == -1
    assert row["error_msg"] == "test"


def test_run_af3_for_one_sub_run_staging_failure_emits_csv(tmp_path, monkeypatch):
    """SF-18: staging failure must produce per-design failure rows, not abort."""
    from eval_pipeline import cli

    manifest_path = tmp_path / "m.csv"
    row = _manifest_row("PD1", "d1", "PD1")
    pd.DataFrame([row]).to_csv(manifest_path, index=False)

    monkeypatch.setenv("AF3_MOCK", "1")
    monkeypatch.setattr(cli.af3_runner, "cleanup_for_rerun", lambda *a, **kw: None)
    monkeypatch.setattr(
        cli.af3_runner,
        "stage_inputs_for_sub_run",
        lambda **kw: (_ for _ in ()).throw(FileNotFoundError("missing msa")),
    )
    monkeypatch.setattr(sys, "argv", [
        "cli.py", "--manifest", str(manifest_path), "--out-dir", str(tmp_path),
        "--evaluator", "af3", "--paper-case", "PD1", "--sub-run", "PD1",
        "--af3-python", "/not/real/python", "--af3-run-script", "/not/real/run.py",
        "--af3-model-dir", "/not/real/model", "--af3-db-dir", "/not/real/db",
        "--no-skip-existing",
    ])

    cli.main()

    output_csv = tmp_path / "eval" / "PD1__PD1__af3.csv"
    out = pd.read_csv(output_csv)
    assert len(out) == 1
    assert out["eval_n_successes"].iloc[0] == 0
    assert "staging_failed: FileNotFoundError: missing msa" in out["error_msg"].iloc[0]


def test_esmfold_loop_skip_existing_preserves_existing_csv(tmp_path, monkeypatch):
    """SF-20: ESMFold --skip-existing should preserve existing eval csv."""
    from eval_pipeline import cli

    manifest_path = tmp_path / "m.csv"
    row = _manifest_row("PD1", "d1", "PD1")
    pd.DataFrame([row]).to_csv(manifest_path, index=False)

    design_dir = tmp_path / "per_design" / "BCv1" / "PD1" / "d1"
    design_dir.mkdir(parents=True)
    (design_dir / "esmfold.pdb").write_text("END\n")
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    existing = eval_dir / "PD1__PD1__esmfold.csv"
    existing.write_text("evaluator,case,design_id,binder_pLDDT\nesmfold,PD1,d1,90.0\n")

    monkeypatch.setattr(
        cli,
        "load_esmfold",
        lambda device: (_ for _ in ()).throw(AssertionError("load_esmfold called")),
    )
    monkeypatch.setattr(sys, "argv", [
        "cli.py", "--manifest", str(manifest_path), "--out-dir", str(tmp_path),
        "--evaluator", "esmfold", "--paper-case", "PD1", "--sub-run", "PD1",
    ])

    cli.main()

    assert existing.read_text() == "evaluator,case,design_id,binder_pLDDT\nesmfold,PD1,d1,90.0\n"


def test_esmfold_loop_skip_existing_legacy_schema_csv_continues(tmp_path, monkeypatch):
    """SF2-1: partial reruns must tolerate older eval CSV schemas."""
    from eval_pipeline import cli

    manifest_path = tmp_path / "m.csv"
    pd.DataFrame([
        _manifest_row("PD1", "d1", "PD1"),
        _manifest_row("PD1", "d2", "PD1"),
    ]).to_csv(manifest_path, index=False)

    design_dir = tmp_path / "per_design" / "BCv1" / "PD1" / "d1"
    design_dir.mkdir(parents=True)
    (design_dir / "esmfold.pdb").write_text("END\n")
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    existing = eval_dir / "PD1__PD1__esmfold.csv"
    existing.write_text(
        "evaluator,case,paper_case,design_id,binder_pLDDT,error_msg\n"
        "esmfold,PD1,PD1,d1,90.0,\n"
    )

    def fake_evaluate_one(row, out_root, tokenizer, model, device, *, evaluator):
        out = cli._empty_row()
        out.update({
            "evaluator": evaluator,
            "method": row["method"],
            "case": row["case"],
            "paper_case": row["paper_case"],
            "design_id": row["design_id"],
            "target_chain": row["target_chain"],
            "binder_length": len(row["binder_seq"]),
            "designed_complex_pdb_path": row["designed_complex_pdb_path"],
            "binder_pLDDT": 91.0,
            "ipAE": 4.0,
            "binder_scRMSD": 1.0,
            "eval_n_calls": 1,
            "eval_n_successes": 1,
            "error_msg": "",
        })
        return out

    monkeypatch.setattr(cli, "load_esmfold", lambda device: (None, None))
    monkeypatch.setattr(cli, "evaluate_one", fake_evaluate_one)
    monkeypatch.setattr(sys, "argv", [
        "cli.py", "--manifest", str(manifest_path), "--out-dir", str(tmp_path),
        "--evaluator", "esmfold", "--paper-case", "PD1", "--sub-run", "PD1",
    ])

    cli.main()

    out = pd.read_csv(existing, keep_default_na=False)
    assert list(out.columns) == cli.CSV_COLS
    assert out["design_id"].tolist() == ["d1", "d2"]
    assert out["error_msg"].tolist() == ["", ""]


def test_align_eval_csv_rows_pads_legacy_rows_and_preserves_blank_errors():
    """SF2-1: schema alignment should make legacy rows safe to write."""
    from eval_pipeline.cli import CSV_COLS, _align_eval_csv_rows

    df = _align_eval_csv_rows([
        {
            "evaluator": "esmfold",
            "case": "PD1",
            "paper_case": "PD1",
            "design_id": "d1",
            "binder_pLDDT": 90.0,
            "error_msg": float("nan"),
        }
    ])

    assert list(df.columns) == CSV_COLS
    assert df["error_msg"].iloc[0] == ""
    assert pd.isna(df["af3_ranking_score"].iloc[0])


def test_evaluate_one_nan_paper_case_uses_case_fallback(tmp_path, monkeypatch):
    """SF-21: NaN paper_case should not propagate into success/failure rows."""
    from eval_pipeline import cli

    row = pd.Series(_manifest_row("PD1", "d1", "PD1"))
    row["paper_case"] = float("nan")

    def broken(*args, **kwargs):
        raise RuntimeError("forced")

    monkeypatch.setattr(cli, "run_esmfold", broken)
    out = cli.evaluate_one(row, tmp_path, None, None, "cpu", evaluator="esmfold")

    assert out["paper_case"] == "PD1"
    assert "forced" in out["error_msg"]


def test_evaluate_one_af3_result_path_propagates_native_columns(tmp_path):
    """SF-23: AF3Result path must populate all native AF3 columns."""
    from eval_pipeline.af3_runner import AF3Result
    from eval_pipeline.cli import evaluate_one

    pdb = tmp_path / "af3_pred.pdb"
    pdb.write_text(
        "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00 85.00           N\n"
        "ATOM      2  CA  MET A   1       1.000   0.000   0.000  1.00 85.00           C\n"
        "ATOM      3  N   MET A   2       3.800   0.000   0.000  1.00 85.00           N\n"
        "ATOM      4  CA  MET A   2       4.800   0.000   0.000  1.00 85.00           C\n"
        "ATOM      5  N   ALA B   1      20.000   0.000   0.000  1.00 92.00           N\n"
        "ATOM      6  CA  ALA B   1      21.000   0.000   0.000  1.00 92.00           C\n"
        "ATOM      7  N   ALA B   2      23.800   0.000   0.000  1.00 92.00           N\n"
        "ATOM      8  CA  ALA B   2      24.800   0.000   0.000  1.00 92.00           C\n"
        "TER\nEND\n"
    )
    af3_result = AF3Result(
        pdb_path=str(pdb),
        plddt=np.array([85.0, 85.0, 92.0, 92.0], dtype=np.float32),
        pae=np.array([
            [0.0, 1.0, 5.0, 5.0],
            [1.0, 0.0, 5.0, 5.0],
            [5.0, 5.0, 0.0, 1.0],
            [5.0, 5.0, 1.0, 0.0],
        ], dtype=np.float32),
        clean_segments=[
            {"kind": "binder", "length": 2, "dirty_start": 0, "dirty_end": 2,
             "clean_start": 0, "clean_end": 2, "pdb_chain_id": "A"},
            {"kind": "target_chain_0", "length": 2, "dirty_start": 2, "dirty_end": 4,
             "clean_start": 2, "clean_end": 4, "pdb_chain_id": "B"},
        ],
        native={
            "ranking_score": 0.75,
            "ptm": 0.82,
            "iptm": 0.65,
            "has_clash": 0.0,
            "fraction_disordered": 0.1,
        },
        runtime_s=12.5,
        raw_output_subdir=str(tmp_path),
    )
    row = _manifest_row("PD1", "d1", "PD1")
    row.update({
        "target_seq": "AA",
        "binder_seq": "MM",
        "designed_complex_pdb_path": str(pdb),
        "target_chain_in_complex": "B",
        "binder_chain_in_complex": "A",
    })

    out = evaluate_one(row, tmp_path, af3_result=af3_result, evaluator="af3")

    assert out["evaluator"] == "af3"
    assert out["esmfold_pdb_path"] == str(pdb)
    assert out["eval_runtime_s"] == 12.5
    assert out["af3_ranking_score"] == 0.75
    assert out["af3_iptm"] == 0.65
    assert out["af3_ptm_native"] == 0.82
    assert out["af3_has_clash"] == 0.0
    assert out["af3_fraction_disordered"] == 0.1
    assert out["binder_pLDDT"] == pytest.approx(85.0)
    assert out["error_msg"] == ""
    assert out["eval_n_successes"] == 1
