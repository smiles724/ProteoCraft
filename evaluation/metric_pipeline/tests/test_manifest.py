import pandas as pd
import pytest

from eval_pipeline.manifest import (
    DENOMINATOR_COLS,
    REQUIRED_COLS,
    ManifestRow,
    validate_manifest,
    write_manifest,
)


def _valid_df():
    df = pd.DataFrame([{c: "x" for c in REQUIRED_COLS}])
    for c in DENOMINATOR_COLS:
        df[c] = -1
    return df


def test_required_cols_present():
    validate_manifest(_valid_df())


def test_missing_required_col_raises():
    df = pd.DataFrame([{c: "x" for c in REQUIRED_COLS if c != "method"}])
    for c in DENOMINATOR_COLS:
        df[c] = -1
    with pytest.raises(ValueError, match="missing required"):
        validate_manifest(df)


def test_denominator_cols_must_exist_even_if_negative():
    df = pd.DataFrame([{c: "x" for c in REQUIRED_COLS}])
    with pytest.raises(ValueError, match="denominator"):
        validate_manifest(df)


def test_manifest_row_dataclass():
    row = ManifestRow(
        method="BCv1",
        case="PDL1",
        design_id="d1",
        target_pdb_path="/x.pdb",
        target_chain="A",
        target_seq="MEELK",
        binder_seq="GVFNYE",
        designed_complex_pdb_path="/y.pdb",
        target_chain_in_complex="A",
        binder_chain_in_complex="B",
        n_generated_trajectories=20,
        n_mpnn_attempted=200,
        n_accepted_by_method=10,
        candidate_rank=5,
        source_filter_status="accepted",
        verifier_calls=100,
        design_gpu_seconds=180.0,
    )
    assert row.binder_length == len("GVFNYE")


def test_empty_manifest_writes_header_only(tmp_path):
    """v1.1 reviewer P0 #3: empty rows → header-only CSV (not 0-byte file)."""
    out = tmp_path / "empty.csv"
    write_manifest([], str(out))
    df = pd.read_csv(out)
    assert len(df) == 0
    for c in REQUIRED_COLS + DENOMINATOR_COLS:
        assert c in df.columns
