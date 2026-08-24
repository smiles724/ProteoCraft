"""Natural structure prior derived from real human class-I TCR-pMHC structures.

Source: ``bindcraft/data/tcr_contact_stats/`` -- per-IMGT-position contact
proportions aggregated over 216 solved human class-I TCR-pMHC complexes.
Each position carries the empirical probability of contacting the peptide, the
MHC, both ("bridge"), the partner TCR chain, or nothing.

This turns "how should a TCR touch a pMHC" from a hand-picked preference into a
measured quantity.  The immediate consequence is that ``r_pep`` has a **target**
rather than a direction: real TCRs put ~30% of their pMHC contacts into
CDR3-peptide, and pushing far past that is as unnatural as falling short.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping, Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "tcr_contact_stats"
PROPORTIONS_CSV = DATA_DIR / "human_classI_00_position_contact_proportions.csv"

PEP_COLS = ("prop_TCR-Peptide", "prop_Bridge(Pep+MHC)")
PMHC_COLS = ("prop_TCR-Peptide", "prop_TCR-MHC", "prop_Bridge(Pep+MHC)")


@dataclass(frozen=True)
class NaturalContactStats:
    """Empirical contact statistics over the reference structure set."""

    n_structures: int
    # fraction of all TCR-pMHC contact mass that is CDR3-peptide, per chain
    r_pep_alpha: float
    r_pep_beta: float
    r_pep_combined: float
    # fraction of total contact mass contributed by each (segment, region)
    region_share: Mapping[tuple, float]
    # per-region mean probability of touching peptide / pMHC
    region_pep_prob: Mapping[tuple, float]
    region_pmhc_prob: Mapping[tuple, float]

    def r_pep_target(self, chain: str) -> float:
        chain = str(chain).strip().lower()
        if chain in ("alpha", "a", "tcr_alpha"):
            return self.r_pep_alpha
        if chain in ("beta", "b", "tcr_beta"):
            return self.r_pep_beta
        raise ValueError(f"unknown chain {chain!r}")


def _num(row: Mapping[str, str], cols) -> float:
    return sum(float(row[c]) for c in cols)


@lru_cache(maxsize=1)
def load_natural_stats(path: Optional[str] = None) -> NaturalContactStats:
    csv_path = Path(path) if path else PROPORTIONS_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"natural contact statistics not found at {csv_path}; the TCR "
            "structure prior cannot be built without it"
        )
    rows = list(csv.DictReader(csv_path.open()))
    tcr = [r for r in rows if r["segment"].startswith("TCR")]
    if not tcr:
        raise ValueError(f"{csv_path} contains no TCR rows")
    n_structures = max(int(r["n_total"]) for r in rows)

    def pep_w(r):
        return _num(r, PEP_COLS) * float(r["n_present"])

    def pmhc_w(r):
        return _num(r, PMHC_COLS) * float(r["n_present"])

    total = sum(pmhc_w(r) for r in tcr)
    if total <= 0:
        raise ValueError("degenerate contact statistics: zero total contact mass")

    def r_pep_for(seg: Optional[str]) -> float:
        sel = [r for r in tcr if seg is None or r["segment"] == seg]
        num = sum(pep_w(r) for r in sel if r["region"] == "CDR3")
        den = sum(pmhc_w(r) for r in sel)
        return num / den if den > 0 else 0.0

    share: Dict[tuple, float] = {}
    pep_prob: Dict[tuple, float] = {}
    pmhc_prob: Dict[tuple, float] = {}
    for seg in ("TCR_alpha", "TCR_beta"):
        for reg in ("FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"):
            sel = [r for r in tcr if r["segment"] == seg and r["region"] == reg]
            if not sel:
                continue
            n = sum(float(r["n_present"]) for r in sel)
            key = (seg[4:], reg)
            share[key] = sum(pmhc_w(r) for r in sel) / total
            pep_prob[key] = (sum(pep_w(r) for r in sel) / n) if n else 0.0
            pmhc_prob[key] = (sum(pmhc_w(r) for r in sel) / n) if n else 0.0

    return NaturalContactStats(
        n_structures=n_structures,
        r_pep_alpha=r_pep_for("TCR_alpha"),
        r_pep_beta=r_pep_for("TCR_beta"),
        r_pep_combined=r_pep_for(None),
        region_share=share,
        region_pep_prob=pep_prob,
        region_pmhc_prob=pmhc_prob,
    )


def natural_r_pep(chain: str, path: Optional[str] = None) -> float:
    """Empirical r_pep target for the designed chain."""

    return load_natural_stats(path).r_pep_target(chain)


def summary_table(path: Optional[str] = None) -> str:
    s = load_natural_stats(path)
    lines = [
        f"natural contact statistics over {s.n_structures} human class-I "
        f"TCR-pMHC structures",
        f"  r_pep target   alpha={s.r_pep_alpha:.4f}  beta={s.r_pep_beta:.4f}  "
        f"combined={s.r_pep_combined:.4f}",
        "  contact share by region (fraction of all TCR-pMHC contact mass):",
    ]
    for key in sorted(s.region_share, key=lambda k: -s.region_share[k]):
        if s.region_share[key] < 0.002:
            continue
        chain, reg = key
        lines.append(
            f"    {chain:5s} {reg:5s} share={s.region_share[key]:.4f}  "
            f"P(touch peptide)={s.region_pep_prob[key]:.3f}  "
            f"P(touch pMHC)={s.region_pmhc_prob[key]:.3f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary_table())
