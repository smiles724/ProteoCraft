"""Pure tier gating logic (spec §6.2 §6.3)."""
from __future__ import annotations

import math

THRESHOLDS_BY_EVALUATOR = {
    "esmfold": {
        "binder_plddt_strict": 90.0,    # Tier 1 strict (Proteina 严)
        "binder_plddt_loose": 80.0,     # Tier 1 aux (BCv1 默认)
        "complex_plddt": 90.0,          # Tier 2
        "ipae_max": 7.0,                # Å, raw HF ESMFold
        "scrmsd_max": 1.5,              # Å, target-aligned
        "clash_max": 0,                 # for strict_physical
    },
    "af3": {
        # Placeholder: occurrence-locked equal to ESMFold until Phase 1 calibration.
        "binder_plddt_strict": 90.0,
        "binder_plddt_loose": 80.0,
        "complex_plddt": 90.0,
        "ipae_max": 7.0,
        "scrmsd_max": 1.5,
        "clash_max": 0,
    },
}

THRESHOLDS = THRESHOLDS_BY_EVALUATOR["esmfold"]


def _is_num(x):
    return (x is not None) and (isinstance(x, (int, float))) and (not math.isnan(float(x)))


def _gt(x, thr):
    return _is_num(x) and float(x) > thr


def _lt(x, thr):
    return _is_num(x) and float(x) < thr


def _le(x, thr):
    return _is_num(x) and float(x) <= thr


def compute_tier_flags(metrics: dict, evaluator: str = "esmfold") -> dict:
    thr = THRESHOLDS_BY_EVALUATOR[evaluator]
    bpl = metrics.get("binder_pLDDT")
    cpl = metrics.get("complex_pLDDT")
    ipae = metrics.get("ipAE")
    scr = metrics.get("binder_scRMSD")
    clash = metrics.get("inter_chain_clash_count")

    t1_80 = _gt(bpl, thr["binder_plddt_loose"])
    t1_90 = _gt(bpl, thr["binder_plddt_strict"])
    tier1_pass = t1_90

    tier2_complexa = (
        t1_90
        and _gt(cpl, thr["complex_plddt"])
        and _lt(ipae, thr["ipae_max"])
        and _lt(scr, thr["scrmsd_max"])
    )
    tier2_strict_physical = tier2_complexa and _le(clash, thr["clash_max"])

    return {
        "tier1_plddt80_pass": t1_80,
        "tier1_plddt90_pass": t1_90,
        "tier1_pass": tier1_pass,
        "tier2_complexa_pass": tier2_complexa,
        "tier2_strict_physical_pass": tier2_strict_physical,
        "tier2_pass": tier2_complexa,
    }
