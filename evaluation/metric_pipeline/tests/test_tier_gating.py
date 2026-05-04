import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_pipeline.tier_gating import compute_tier_flags  # noqa: E402


def _metrics(**overrides):
    base = dict(
        binder_pLDDT=95.0,
        complex_pLDDT=92.0,
        ipAE=5.0,
        binder_scRMSD=1.0,
        inter_chain_clash_count=0,
    )
    base.update(overrides)
    return base


def test_perfect_design_passes_all():
    f = compute_tier_flags(_metrics())
    assert f["tier1_plddt80_pass"]
    assert f["tier1_plddt90_pass"]
    assert f["tier1_pass"]
    assert f["tier2_complexa_pass"]
    assert f["tier2_strict_physical_pass"]
    assert f["tier2_pass"]


def test_binder_plddt_85_passes_80_not_90():
    f = compute_tier_flags(_metrics(binder_pLDDT=85.0))
    assert f["tier1_plddt80_pass"] and not f["tier1_plddt90_pass"]
    assert not f["tier1_pass"]
    assert not f["tier2_complexa_pass"]


def test_clash_breaks_strict_physical_only():
    f = compute_tier_flags(_metrics(inter_chain_clash_count=1))
    assert f["tier2_complexa_pass"]
    assert not f["tier2_strict_physical_pass"]
    assert f["tier2_pass"]   # main = complexa


def test_high_ipae_breaks_tier2():
    f = compute_tier_flags(_metrics(ipAE=8.0))
    assert f["tier1_pass"]
    assert not f["tier2_complexa_pass"]
    assert not f["tier2_pass"]


def test_high_scrmsd_breaks_tier2():
    f = compute_tier_flags(_metrics(binder_scRMSD=2.0))
    assert not f["tier2_complexa_pass"]


def test_nan_metric_yields_false_pass():
    f = compute_tier_flags(_metrics(binder_pLDDT=float("nan")))
    assert not f["tier1_pass"]
    assert not f["tier2_pass"]


def test_missing_metric_keys_yield_false():
    f = compute_tier_flags({})
    assert not f["tier1_pass"]
    assert not f["tier2_pass"]
    assert not f["tier2_strict_physical_pass"]
