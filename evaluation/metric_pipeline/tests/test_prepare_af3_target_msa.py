"""Tests for scripts/prepare_af3_target_msa.py."""
from __future__ import annotations

import sys

import pandas as pd
import pytest


def test_extract_msa_distinguishes_null_from_empty_string(tmp_path):
    """SF-1: unpairedMsa=null and empty string should produce distinct errors."""
    import json
    from scripts.prepare_af3_target_msa import _extract_msas_from_data_json

    null_json = tmp_path / "null_data.json"
    null_json.write_text(json.dumps({
        "sequences": [{"protein": {"id": "B", "unpairedMsa": None}}],
    }))
    with pytest.raises(RuntimeError, match="unpairedMsa=null"):
        _extract_msas_from_data_json(null_json, tmp_path / "out-null")

    empty_json = tmp_path / "empty_data.json"
    empty_json.write_text(json.dumps({
        "sequences": [{"protein": {"id": "B", "unpairedMsa": "   "}}],
    }))
    with pytest.raises(RuntimeError, match="empty MSA string"):
        _extract_msas_from_data_json(empty_json, tmp_path / "out-empty")


def test_prepare_msa_target_seq_nan_raises(tmp_path, monkeypatch):
    """SF-2: NaN target_seq must raise, not silently fold a single 'n' residue."""
    from scripts import prepare_af3_target_msa

    manifest = tmp_path / "m.csv"
    pd.DataFrame([{
        "method": "BCv1", "case": "PD1", "design_id": "d1",
        "target_seq": float("nan"), "binder_seq": "MSKG", "paper_case": "PD1",
    }]).to_csv(manifest, index=False)

    monkeypatch.setattr(sys, "argv", [
        "prepare_af3_target_msa.py",
        "--manifest", str(manifest),
        "--work-root", str(tmp_path),
        "--af3-python", "/usr/bin/python",
        "--af3-run-script", "/tmp/x",
        "--af3-model-dir", "/tmp",
        "--af3-db-dir", "/tmp",
    ])
    with pytest.raises(SystemExit, match="empty/NaN target_seq"):
        prepare_af3_target_msa.main()


def test_prepare_msa_inconsistent_target_seq_raises(tmp_path, monkeypatch):
    """SF-2: rows for the same paper_case must agree on target_seq."""
    from scripts import prepare_af3_target_msa

    manifest = tmp_path / "m.csv"
    pd.DataFrame([
        {"case": "PD1", "paper_case": "PD1", "design_id": "d1", "target_seq": "AAA"},
        {"case": "PD1", "paper_case": "PD1", "design_id": "d2", "target_seq": "BBB"},
    ]).to_csv(manifest, index=False)

    monkeypatch.setattr(sys, "argv", [
        "prepare_af3_target_msa.py",
        "--manifest", str(manifest),
        "--work-root", str(tmp_path),
        "--af3-python", "/usr/bin/python",
        "--af3-run-script", "/tmp/x",
        "--af3-model-dir", "/tmp",
        "--af3-db-dir", "/tmp",
    ])
    with pytest.raises(SystemExit, match="inconsistent target_seq"):
        prepare_af3_target_msa.main()


def test_prepare_msa_partial_cache_raises(tmp_path):
    """SF-3: partial per-paper_case MSA cache must raise before clobbering."""
    from scripts.prepare_af3_target_msa import run_prep_for_paper_case

    cache = tmp_path / "per_paper_case_msa" / "PD1"
    cache.mkdir(parents=True)
    (cache / "target_chain_B.a3m").write_text(">B\nAAAA\n")

    with pytest.raises(RuntimeError, match="partial MSA cache"):
        run_prep_for_paper_case(
            paper_case="PD1",
            target_seqs=["AAAA", "BBBB"],
            work_root=tmp_path,
            af3_python="/usr/bin/python",
            af3_run_script="/tmp/x",
            af3_model_dir="/tmp",
            af3_db_dir="/tmp",
        )


def test_prepare_msa_paper_case_typo_raises(tmp_path, monkeypatch):
    """SF-5: --paper-case typo should fail loud instead of doing zero work."""
    from scripts import prepare_af3_target_msa

    manifest = tmp_path / "m.csv"
    pd.DataFrame([{
        "case": "PD1", "paper_case": "PD1", "design_id": "d1", "target_seq": "AAA",
    }]).to_csv(manifest, index=False)

    monkeypatch.setattr(sys, "argv", [
        "prepare_af3_target_msa.py",
        "--manifest", str(manifest),
        "--work-root", str(tmp_path),
        "--af3-python", "/usr/bin/python",
        "--af3-run-script", "/tmp/x",
        "--af3-model-dir", "/tmp",
        "--af3-db-dir", "/tmp",
        "--paper-case", "PD2",
    ])
    with pytest.raises(SystemExit, match="not found in manifest"):
        prepare_af3_target_msa.main()
