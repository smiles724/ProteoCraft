"""Tests for sub_run_aggregator."""
from __future__ import annotations

import sys

import pandas as pd
import pytest


def _make_manifest_rows(case: str, n: int, model_suffix: bool = True) -> pd.DataFrame:
    rows = []
    for i in range(n):
        design_id = f"{case}_l50_s{1000 + i}_mpnn1"
        if model_suffix:
            design_id += "_model1"
        rows.append({
            "method": "BCv1",
            "case": case,
            "design_id": design_id,
            "target_pdb_path": "/x/template.pdb",
            "target_chain": "A",
            "target_seq": "PEPTIDE",
            "binder_seq": "MSKG" * (i + 1),
            "designed_complex_pdb_path": f"/x/{design_id}.pdb",
            "target_chain_in_complex": "A",
            "binder_chain_in_complex": "B",
            "n_accepted_by_method": n,
            "n_generated_trajectories": -1,
            "n_mpnn_attempted": -1,
            "candidate_rank": i,
            "source_filter_status": "accepted",
            "verifier_calls": -1,
            "design_gpu_seconds": float("nan"),
        })
    return pd.DataFrame(rows)


def _make_stats(case: str, n: int, iptm_values: list[float]) -> pd.DataFrame:
    rows = []
    for i, iptm in enumerate(iptm_values):
        rows.append({
            "Design": f"{case}_l50_s{1000 + i}_mpnn1",
            "Average_i_pTM": iptm,
        })
    return pd.DataFrame(rows)


def test_betv1_yield_higher_pick(tmp_path):
    from adapters.sub_run_aggregator import aggregate_sub_runs

    h23_manifest = _make_manifest_rows("BetV1_h23", n=10)
    h23_manifest["n_accepted_by_method"] = 10
    h23_stats = _make_stats("BetV1_h23", n=10, iptm_values=[0.5] * 10)

    h24_manifest = _make_manifest_rows("BetV1_h24", n=80)
    h24_manifest["n_accepted_by_method"] = 80
    h24_stats = _make_stats("BetV1_h24", n=80, iptm_values=[0.6] * 80)

    out = aggregate_sub_runs(
        sub_run_manifests={"BetV1_h23": h23_manifest, "BetV1_h24": h24_manifest},
        bindcraft_stats={"BetV1_h23": h23_stats, "BetV1_h24": h24_stats},
        out_dir=tmp_path,
    )
    selected = pd.read_csv(out["BetV1"])
    assert len(selected) == 80
    assert (selected["case"] == "BetV1_h24").all()
    assert (selected["paper_case"] == "BetV1").all()


def test_betv1_yield_tie_deterministic_alpha(tmp_path, capsys):
    from adapters.sub_run_aggregator import aggregate_sub_runs

    h23_manifest = _make_manifest_rows("BetV1_h23", n=50)
    h24_manifest = _make_manifest_rows("BetV1_h24", n=50)
    h23_manifest["n_accepted_by_method"] = 50
    h24_manifest["n_accepted_by_method"] = 50
    h23_stats = _make_stats("BetV1_h23", n=50, iptm_values=[0.5] * 50)
    h24_stats = _make_stats("BetV1_h24", n=50, iptm_values=[0.5] * 50)

    out1 = aggregate_sub_runs(
        {"BetV1_h23": h23_manifest, "BetV1_h24": h24_manifest},
        {"BetV1_h23": h23_stats, "BetV1_h24": h24_stats},
        tmp_path / "order1",
    )
    out2 = aggregate_sub_runs(
        {"BetV1_h24": h24_manifest, "BetV1_h23": h23_manifest},
        {"BetV1_h24": h24_stats, "BetV1_h23": h23_stats},
        tmp_path / "order2",
    )

    selected1 = pd.read_csv(out1["BetV1"])
    selected2 = pd.read_csv(out2["BetV1"])
    assert (selected1["case"] == "BetV1_h23").all()
    assert (selected2["case"] == "BetV1_h23").all()
    assert len(selected1) == 50
    assert len(selected2) == 50
    assert "BetV1 yield tie" in capsys.readouterr().out


def test_derf21_or_merge_top_100(tmp_path):
    from adapters.sub_run_aggregator import aggregate_sub_runs

    sub_runs = ["DerF21_h17", "DerF21_h35", "DerF21_h70", "DerF21_h88", "DerF21_h99"]
    manifests = {sr: _make_manifest_rows(sr, n=50) for sr in sub_runs}
    stats = {
        "DerF21_h17": _make_stats("DerF21_h17", n=50, iptm_values=[0.9 - i * 0.001 for i in range(50)]),
        "DerF21_h35": _make_stats("DerF21_h35", n=50, iptm_values=[0.5 - i * 0.001 for i in range(50)]),
        "DerF21_h70": _make_stats("DerF21_h70", n=50, iptm_values=[0.4] * 50),
        "DerF21_h88": _make_stats("DerF21_h88", n=50, iptm_values=[0.3] * 50),
        "DerF21_h99": _make_stats("DerF21_h99", n=50, iptm_values=[0.2] * 50),
    }
    out = aggregate_sub_runs(manifests, stats, tmp_path)
    selected = pd.read_csv(out["DerF21"])
    assert len(selected) == 100
    assert (selected["paper_case"] == "DerF21").all()
    assert (selected["case"] == "DerF21_h17").sum() == 50


def _empty_manifest() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "method", "case", "design_id", "target_pdb_path", "target_chain",
        "target_seq", "binder_seq", "designed_complex_pdb_path",
        "target_chain_in_complex", "binder_chain_in_complex",
        "n_accepted_by_method", "n_generated_trajectories", "n_mpnn_attempted",
        "candidate_rank", "source_filter_status", "verifier_calls",
        "design_gpu_seconds",
    ])


def test_betv1_empty_manifests_does_not_crash(tmp_path):
    from adapters.sub_run_aggregator import aggregate_sub_runs

    empty_stats = pd.DataFrame(columns=["Design", "Average_i_pTM"])
    out = aggregate_sub_runs(
        {"BetV1_h23": _empty_manifest(), "BetV1_h24": _empty_manifest()},
        {"BetV1_h23": empty_stats, "BetV1_h24": empty_stats},
        tmp_path,
    )

    selected = pd.read_csv(out["BetV1"])
    assert len(selected) == 0
    assert "paper_case" in selected.columns


def test_or_merge_empty_manifests_does_not_crash(tmp_path):
    from adapters.sub_run_aggregator import aggregate_sub_runs

    empty_stats = pd.DataFrame(columns=["Design", "Average_i_pTM"])
    out = aggregate_sub_runs(
        {"DerF7_h132": _empty_manifest(), "DerF7_h178": _empty_manifest()},
        {"DerF7_h132": empty_stats, "DerF7_h178": empty_stats},
        tmp_path,
    )

    selected = pd.read_csv(out["DerF7"])
    assert len(selected) == 0
    assert "paper_case" in selected.columns


def test_cli_header_only_manifest_writes_selected_manifest(tmp_path, monkeypatch):
    from adapters import sub_run_aggregator

    manifest = tmp_path / "BetV1_h23.raw.manifest.csv"
    _empty_manifest().assign(case=pd.Series(dtype=str)).to_csv(manifest, index=False)
    stats_dir = tmp_path / "bindcraft" / "BetV1_h23"
    stats_dir.mkdir(parents=True)
    pd.DataFrame(columns=["Design", "Average_i_pTM"]).to_csv(
        stats_dir / "final_design_stats.csv", index=False
    )

    out_dir = tmp_path / "work"
    monkeypatch.setattr(sys, "argv", [
        "sub_run_aggregator.py",
        "--sub-run-manifests", str(manifest),
        "--bindcraft-outputs", str(tmp_path / "bindcraft"),
        "--out-dir", str(out_dir),
    ])
    sub_run_aggregator.main()

    selected_path = out_dir / "manifests" / "BetV1.selected.manifest.csv"
    assert selected_path.exists()
    selected = pd.read_csv(selected_path)
    assert len(selected) == 0
    assert "paper_case" in selected.columns


def test_cli_missing_bindcraft_stats_has_layout_hint(tmp_path, monkeypatch):
    """SF-7: missing BindCraft stats error should explain expected directory layout."""
    from adapters import sub_run_aggregator

    manifest = tmp_path / "PD1.raw.manifest.csv"
    _make_manifest_rows("PD1", n=1).to_csv(manifest, index=False)

    monkeypatch.setattr(sys, "argv", [
        "sub_run_aggregator.py",
        "--sub-run-manifests", str(manifest),
        "--bindcraft-outputs", str(tmp_path / "bindcraft"),
        "--out-dir", str(tmp_path / "work"),
    ])

    with pytest.raises(FileNotFoundError, match="expected layout"):
        sub_run_aggregator.main()


@pytest.mark.parametrize("case", [
    "PD1", "PDL1", "IFNAR2", "SpCas9",
    "TCR_1QRN", "TCR_3QDJ", "TCR_5BS0", "TCR_6BJ8", "TCR_9C3E",
])
def test_passthrough_paper_case(tmp_path, case):
    """SF-25: all 9 passthrough paper_cases must round-trip rows unchanged."""
    from adapters.sub_run_aggregator import aggregate_sub_runs

    manifest = _make_manifest_rows(case, n=15)
    stats = _make_stats(case, n=15, iptm_values=[0.5] * 15)
    out = aggregate_sub_runs({case: manifest}, {case: stats}, tmp_path)
    selected = pd.read_csv(out[case])
    assert len(selected) == 15
    assert (selected["paper_case"] == case).all()
    assert (selected["case"] == case).all()


def test_unknown_paper_case_raises(tmp_path):
    from adapters.sub_run_aggregator import aggregate_sub_runs

    typo_manifest = _make_manifest_rows("PDL_1", n=10)
    typo_stats = _make_stats("PDL_1", n=10, iptm_values=[0.5] * 10)

    with pytest.raises(KeyError, match="Unknown sub_run/paper_case"):
        aggregate_sub_runs({"PDL_1": typo_manifest}, {"PDL_1": typo_stats}, tmp_path)


def test_known_passthrough_cases_accepted(tmp_path):
    from adapters.sub_run_aggregator import aggregate_sub_runs

    for case in [
        "PD1", "PDL1", "IFNAR2", "SpCas9",
        "TCR_1QRN", "TCR_3QDJ", "TCR_5BS0", "TCR_6BJ8", "TCR_9C3E",
    ]:
        manifest = _make_manifest_rows(case, n=3)
        stats = _make_stats(case, n=3, iptm_values=[0.5] * 3)
        out = aggregate_sub_runs({case: manifest}, {case: stats}, tmp_path / case)
        assert case in out
        selected = pd.read_csv(out[case])
        assert (selected["paper_case"] == case).all()


def test_modelN_suffix_join_correctly(tmp_path):
    from adapters.sub_run_aggregator import aggregate_sub_runs

    manifest = _make_manifest_rows("DerF7_h132", n=10, model_suffix=True)
    stats = _make_stats("DerF7_h132", n=10, iptm_values=[0.5] * 10)
    out = aggregate_sub_runs({"DerF7_h132": manifest}, {"DerF7_h132": stats}, tmp_path)
    selected = pd.read_csv(out["DerF7"])
    assert selected["design_avg_iptm"].notna().all()


def test_high_nan_rate_raises(tmp_path):
    from adapters.sub_run_aggregator import aggregate_sub_runs

    manifest = _make_manifest_rows("DerF7_h132", n=10, model_suffix=True)
    stats = pd.DataFrame({
        "Design": ["WRONG_KEY_1", "WRONG_KEY_2"],
        "Average_i_pTM": [0.5, 0.5],
    })
    with pytest.raises(RuntimeError, match="NaN rate"):
        aggregate_sub_runs({"DerF7_h132": manifest}, {"DerF7_h132": stats}, tmp_path)
