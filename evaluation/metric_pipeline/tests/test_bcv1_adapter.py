import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.bcv1_adapter import (  # noqa: E402
    BCV1_CHAIN_MAP,
    BCV1_SUB_RUN_TO_PAPER_CASE,
    TARGET_TEMPLATE_MAP,
    _design_id_to_design_key,
    build_bcv1_manifest_for_case,
    is_sub_run,
    parse_design_time_to_seconds,
    resolve_paper_case,
)

EXPECTED_CASES = {
    "BetV1", "DerF21", "DerF7", "HER2", "IFNAR2", "PD1", "PDL1", "SpCas9",
    "TCR_1QRN", "TCR_3QDJ", "TCR_5BS0", "TCR_6BJ8", "TCR_9C3E",
}

# Spec §3.4 — 11 sub-runs split from 4 paper cases.
EXPECTED_SUB_RUNS = {
    "BetV1_h23", "BetV1_h24",
    "DerF7_h132", "DerF7_h178",
    "DerF21_h17", "DerF21_h35", "DerF21_h70", "DerF21_h88", "DerF21_h99",
    "HER2_h148", "HER2_h143_148",
}


def test_parse_design_time():
    assert parse_design_time_to_seconds("0 hours, 1 minutes, 26 seconds") == 86.0
    assert parse_design_time_to_seconds("2 hours, 0 minutes, 0 seconds") == 7200.0
    assert pd.isna(parse_design_time_to_seconds(""))
    assert pd.isna(parse_design_time_to_seconds(None))


def test_chain_map_completeness():
    assert set(BCV1_CHAIN_MAP.keys()) == EXPECTED_CASES


def test_target_template_paths_all_exist():
    """v1.1 reviewer P0 #2: every TARGET_TEMPLATE_MAP entry must point to a real file."""
    assert set(TARGET_TEMPLATE_MAP.keys()) == EXPECTED_CASES
    missing = [(c, p) for c, p in TARGET_TEMPLATE_MAP.items() if not p.exists()]
    assert not missing, f"missing template files: {missing}"


def test_target_template_chain_map_consistent():
    """For each case, BCV1_CHAIN_MAP[case]['target'] chains must exist in template PDB."""
    from vendor.proteina_complexa.pdb_chain_utils import list_chain_ids
    for case, spec in BCV1_CHAIN_MAP.items():
        template = TARGET_TEMPLATE_MAP[case]
        if not template.exists():
            pytest.skip(f"{case} template missing")
        chains = list_chain_ids(str(template))
        for tc in spec["target"]:  # type: ignore
            assert tc in chains, f"{case}: target chain {tc} not in template chains {chains}"


def test_design_id_to_design_key_strips_modelN():
    """v1.1 reviewer P1 #4: PDB stem has _modelN, CSV 'Design' col does not."""
    assert _design_id_to_design_key("PDL1_l155_s752946_mpnn5_model1") == "PDL1_l155_s752946_mpnn5"
    assert _design_id_to_design_key("PDL1_l155_s752946_mpnn5_model12") == "PDL1_l155_s752946_mpnn5"
    assert _design_id_to_design_key("foo_bar") == "foo_bar"


def test_empty_accepted_writes_header_only(tmp_path):
    """v1.1 reviewer P0 #3: empty Accepted/ writes header-only CSV (no error)."""
    fake_batch = tmp_path / "batch"
    (fake_batch / "PDL1").mkdir(parents=True)
    out = tmp_path / "manifest.csv"
    n = build_bcv1_manifest_for_case(
        batch_dir=fake_batch,
        case="PDL1",
        target_template=None,
        target_only_dir=tmp_path / "to",
        out_csv=out,
    )
    assert n == 0
    assert out.exists()
    df = pd.read_csv(out)
    assert len(df) == 0
    assert "method" in df.columns and "target_chain_in_complex" in df.columns


def test_build_pdl1_manifest_real_data(tmp_path):
    """End-to-end on real PDL1 batch_v1 output. Skips if BCv1 outputs unavailable."""
    batch = Path("/mnt/shared-storage-user/liyanchao/af3/BindCraft/outputs/batch_v1")
    if not (batch / "PDL1" / "Accepted").exists():
        pytest.skip("BCv1 PDL1 outputs not available")
    out = tmp_path / "manifest.csv"
    n = build_bcv1_manifest_for_case(
        batch_dir=batch,
        case="PDL1",
        target_template=None,
        target_only_dir=tmp_path / "target_only",
        out_csv=out,
    )
    if n == 0:
        pytest.skip("PDL1 Accepted/ empty (jobs still running?)")
    df = pd.read_csv(out)
    assert len(df) == n
    assert (df["method"] == "BCv1").all()
    assert (df["case"] == "PDL1").all()
    assert (df["target_chain_in_complex"] == "A").all()
    n_acc_disk = len(list((batch / "PDL1" / "Accepted").glob("*.pdb")))
    assert (df["n_accepted_by_method"] == n_acc_disk).all()


# ─── Sub-run support (spec §3.4) ─────────────────────────────────────


def test_sub_run_map_completeness():
    """11 sub-runs split from 4 paper cases per spec §3.4."""
    assert set(BCV1_SUB_RUN_TO_PAPER_CASE.keys()) == EXPECTED_SUB_RUNS


def test_sub_run_map_targets_are_valid_paper_cases():
    """Every sub-run's parent paper case must be in BCV1_CHAIN_MAP and TARGET_TEMPLATE_MAP."""
    for sub_run, paper_case in BCV1_SUB_RUN_TO_PAPER_CASE.items():
        assert paper_case in BCV1_CHAIN_MAP, f"{sub_run} → {paper_case} not in BCV1_CHAIN_MAP"
        assert paper_case in TARGET_TEMPLATE_MAP, f"{sub_run} → {paper_case} not in TARGET_TEMPLATE_MAP"


def test_resolve_paper_case_identity_for_paper_cases():
    """Paper case names map to themselves."""
    for case in EXPECTED_CASES:
        assert resolve_paper_case(case) == case


def test_resolve_paper_case_for_sub_runs():
    """Sub-runs map to their parent paper case per spec §3.4 split rules."""
    assert resolve_paper_case("BetV1_h23") == "BetV1"
    assert resolve_paper_case("BetV1_h24") == "BetV1"
    assert resolve_paper_case("DerF7_h132") == "DerF7"
    assert resolve_paper_case("DerF7_h178") == "DerF7"
    for h in ("17", "35", "70", "88", "99"):
        assert resolve_paper_case(f"DerF21_h{h}") == "DerF21"
    assert resolve_paper_case("HER2_h148") == "HER2"
    assert resolve_paper_case("HER2_h143_148") == "HER2"


def test_resolve_paper_case_unknown_returns_input():
    """Unknown / typo case names return as-is (KeyError surfaces later in build_bcv1_manifest_for_case)."""
    assert resolve_paper_case("Garbage") == "Garbage"
    assert resolve_paper_case("BetV1_h99") == "BetV1_h99"  # not a real split for BetV1


def test_is_sub_run():
    """Predicate: True iff in BCV1_SUB_RUN_TO_PAPER_CASE."""
    for sub in EXPECTED_SUB_RUNS:
        assert is_sub_run(sub), f"{sub} should be sub-run"
    for paper in EXPECTED_CASES:
        assert not is_sub_run(paper), f"{paper} should not be sub-run"
    assert not is_sub_run("Garbage")


def test_total_20_sub_runs_or_paper_cases():
    """Across 9 in-place edited paper cases + 11 sub-runs = 20 batch_v1 sub-run jobs (spec §4)."""
    # 4 BC in-place: PD1, PDL1, IFNAR2, SpCas9 (BetV1/DerF7/DerF21/HER2 are SPLIT not in-place)
    # 5 TCR in-place
    # 11 split sub-runs
    in_place_paper_cases = {"PD1", "PDL1", "IFNAR2", "SpCas9",
                             "TCR_1QRN", "TCR_3QDJ", "TCR_5BS0", "TCR_6BJ8", "TCR_9C3E"}
    assert in_place_paper_cases.issubset(EXPECTED_CASES)
    n_in_place = len(in_place_paper_cases)            # 9
    n_sub_runs = len(EXPECTED_SUB_RUNS)               # 11
    assert n_in_place + n_sub_runs == 20


def test_sub_run_build_empty_accepted(tmp_path):
    """Sub-run case name should resolve OK even when Accepted/ empty (jobs not run yet)."""
    fake_batch = tmp_path / "batch"
    (fake_batch / "BetV1_h23").mkdir(parents=True)
    out = tmp_path / "manifest.csv"
    n = build_bcv1_manifest_for_case(
        batch_dir=fake_batch,
        case="BetV1_h23",
        target_template=None,
        target_only_dir=tmp_path / "to",
        out_csv=out,
    )
    assert n == 0
    assert out.exists()


def test_unknown_case_raises_keyerror_with_helpful_message(tmp_path):
    """Bad case name fails fast with a message listing both paper cases and sub-runs."""
    fake_batch = tmp_path / "batch"
    (fake_batch / "GarbageCase").mkdir(parents=True)
    (fake_batch / "GarbageCase" / "Accepted").mkdir()  # so it's not the empty-dir branch
    # drop a fake pdb so the loop reaches the chain map lookup
    (fake_batch / "GarbageCase" / "Accepted" / "x.pdb").write_text("")
    out = tmp_path / "manifest.csv"
    with pytest.raises(KeyError) as exc_info:
        build_bcv1_manifest_for_case(
            batch_dir=fake_batch,
            case="GarbageCase",
            target_template=None,
            target_only_dir=tmp_path / "to",
            out_csv=out,
        )
    msg = str(exc_info.value)
    assert "GarbageCase" in msg
    assert "BetV1" in msg or "PD1" in msg  # paper cases listed
    assert "BetV1_h23" in msg or "DerF7_h132" in msg  # sub-runs listed
