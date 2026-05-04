"""Pytest-tracked AF3_MOCK end-to-end and report OR-merge checks."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def _build_mock_manifest(work: Path) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    (work / "manifests").mkdir(parents=True, exist_ok=True)
    msa_dir = work / "per_paper_case_msa" / "PD1"
    msa_dir.mkdir(parents=True, exist_ok=True)
    (msa_dir / "target_chain_B.a3m").write_text(">x\nAAA\n")

    ref = work / "ref.pdb"
    ref.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 80.00           C\n"
        "ATOM      2  CA  ALA A   2       3.800   0.000   0.000  1.00 80.00           C\n"
        "ATOM      3  CA  ALA A   3       7.600   0.000   0.000  1.00 80.00           C\n"
        "ATOM      4  CA  ALA B   1      20.000   0.000   0.000  1.00 80.00           C\n"
        "ATOM      5  CA  ALA B   2      23.800   0.000   0.000  1.00 80.00           C\n"
        "ATOM      6  CA  ALA B   3      27.600   0.000   0.000  1.00 80.00           C\n"
        "ATOM      7  CA  ALA B   4      31.400   0.000   0.000  1.00 80.00           C\n"
        "TER\nEND\n"
    )
    manifest = work / "manifests" / "PD1.selected.manifest.csv"
    pd.DataFrame([{
        "method": "BCv1",
        "case": "PD1",
        "paper_case": "PD1",
        "design_id": "PD1_d_001",
        "target_pdb_path": str(ref),
        "target_chain": "A",
        "target_seq": "AAA",
        "binder_seq": "AAAA",
        "designed_complex_pdb_path": str(ref),
        "target_chain_in_complex": "A",
        "binder_chain_in_complex": "B",
        "n_generated_trajectories": -1,
        "n_mpnn_attempted": -1,
        "n_accepted_by_method": 1,
        "candidate_rank": 0,
        "source_filter_status": "accepted",
        "verifier_calls": -1,
        "design_gpu_seconds": float("nan"),
        "notes": "",
    }]).to_csv(manifest, index=False)
    return manifest


def test_af3_mock_e2e_produces_one_real_row(tmp_path):
    """SF-27: AF3_MOCK=1 CLI must produce one successful per-sub-run row."""
    work = tmp_path / "work"
    manifest = _build_mock_manifest(work)
    env = {**os.environ, "AF3_MOCK": "1"}
    cmd = [
        PYTHON, "-m", "eval_pipeline.cli",
        "--manifest", str(manifest),
        "--out-dir", str(work),
        "--evaluator", "af3",
        "--paper-case", "PD1",
        "--sub-run", "PD1",
        "--af3-python", "/usr/bin/python",
        "--af3-run-script", "/tmp/fake.py",
        "--af3-model-dir", "/tmp",
        "--af3-db-dir", "/tmp",
        "--no-skip-existing",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, f"cli failed:\nstdout={result.stdout}\nstderr={result.stderr}"

    eval_csv = work / "eval" / "PD1__PD1__af3.csv"
    assert eval_csv.exists()
    df = pd.read_csv(eval_csv)
    assert len(df) == 1
    assert df["evaluator"].iloc[0] == "af3"
    assert int(df["eval_n_successes"].iloc[0]) == 1


def test_or_merge_integration_derf21_aggregates_to_one_row(tmp_path):
    """SF-27: DerF21 sub-run eval CSVs aggregate to one paper_case row with n=100."""
    work = tmp_path / "work"
    eval_dir = work / "eval"
    eval_dir.mkdir(parents=True)
    for case in ["DerF21_h17", "DerF21_h35"]:
        rows = [{
            "evaluator": "af3",
            "method": "BCv1",
            "case": case,
            "paper_case": "DerF21",
            "design_id": f"{case}_d_{i:03d}",
            "binder_pLDDT": 88.0,
            "ipAE": 5.0,
            "binder_scRMSD": 1.2,
            "tier1_pass": True,
            "tier2_pass": True,
            "eval_n_calls": 1,
            "eval_n_successes": 1,
            "eval_runtime_s": 5.0,
            "error_msg": "",
        } for i in range(50)]
        pd.DataFrame(rows).to_csv(eval_dir / f"DerF21__{case}__af3.csv", index=False)

    cmd = [
        PYTHON, "-m", "eval_pipeline.report",
        "--auto-discover",
        "--case-roster", str(REPO_ROOT / "configs" / "bcv2_case_roster.json"),
        "--evaluators", "af3",
        "--out-dir", str(work),
        "--run-id", "or_merge_smoke",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, f"report failed:\nstdout={result.stdout}\nstderr={result.stderr}"

    agg = pd.read_csv(work / "eval_per_paper_case_aggregate.csv")
    der = agg[(agg["evaluator"] == "af3") & (agg["paper_case"] == "DerF21")]
    assert len(der) == 1
    assert int(der["n_designs_evaluated"].iloc[0]) == 100
