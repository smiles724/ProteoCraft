import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "benchmark/script/run_esm_optimization.py"
spec = importlib.util.spec_from_file_location("esm_optimization_experiments", SCRIPT)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_candidate_selection_is_unique_and_deterministic(tmp_path):
    directory = tmp_path / "esmfold2_artifacts"
    directory.mkdir()
    rows = [
        {"sequence": "AAAA", "total_loss": 3},
        {"sequence": "CCCC", "total_loss": 2},
        {"sequence": "AAAA", "total_loss": 1},
        {"sequence": "DDDD", "total_loss": 2},
    ]
    (directory / "candidates.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    selected, unique, calls = experiment.selected_candidates(tmp_path, 2)
    assert [row["sequence"] for row in selected] == ["AAAA", "CCCC"]
    assert unique == 3 and calls == 4


def test_cluster_bootstrap_does_not_overweight_hotspot_variants():
    result = experiment.clustered_interval({"target1": [1] * 100, "target2": [3]})
    assert result["mean"] == 2
    assert result["source_targets"] == 2
    assert result["interpretation"] == "exploratory"
    assert result == experiment.clustered_interval({"target1": [1] * 100, "target2": [3]})


def test_prepare_uses_disjoint_outputs_and_common_seeds(tmp_path):
    args = experiment.parser().parse_args([
        "prepare", "--output", str(tmp_path), "--kit-root", str(tmp_path / "kit"),
        "--cases", "denovo_PD1", "--arms", "baseline", "exact", "--seeds", "43", "44",
    ])
    experiment.prepare(args)
    manifest = json.loads((tmp_path / "campaign.json").read_text())
    rows = manifest["records"]
    assert len(rows) == 4
    assert len({row["directory"] for row in rows}) == 4
    for row in rows:
        command = row["command"]
        assert command[command.index("--seed") + 1] == str(row["seed"])
        assert command[command.index("--esmfold2_save_verified_candidate_cif") + 1] == "true"
        assert "--esmfold2_optimization_timing" in command
    experiment.prepare(args)
    assert json.loads((tmp_path / "campaign.json").read_text()) == manifest


def test_prepare_refuses_override_of_arm_factor(tmp_path):
    args = experiment.parser().parse_args([
        "prepare", "--output", str(tmp_path), "--kit-root", str(tmp_path),
        "--cases", "denovo_PD1", "--set", "esmfold2_esmc_optimization=exact",
    ])
    with pytest.raises(ValueError, match="experimental factors"):
        experiment.prepare(args)


def test_report_preserves_missing_quality_and_failures(tmp_path):
    records = []
    for arm, code in (("baseline", 0), ("exact", 1)):
        directory = tmp_path / arm
        directory.mkdir()
        (directory / "experiment_run.json").write_text(json.dumps({"returncode": code, "wall_s": 10}))
        records.append({"arm": arm, "seed": 43, "case": {"case_id": "x", "target_name": "source"}, "directory": str(directory)})
    manifest = tmp_path / "campaign.json"
    manifest.write_text(json.dumps({"records": records}))
    output = tmp_path / "report.csv"
    experiment.report(SimpleNamespace(manifest=str(manifest), scores=str(tmp_path / "absent"), output=str(output)))
    report = json.loads(output.with_suffix(".summary.json").read_text())
    assert report["exact"]["failed"] == 1
    assert report["exact"]["paired_quality_delta"] is None
    assert report["exact"]["paired_speedup"] is None
