#!/usr/bin/env python
"""
Real-backend validation for TCR sequence priors (ESM-C / TCRLM / LigandMPNN).

This goes beyond interface smoke tests: for each backend it checks whether the
*real* model is actually loaded, whether it produces *real* per-position
log-probs (not a placeholder/no-op), whether the soft prior loss yields non-zero
gradients on the sequence logits, and whether it would fall back to a dummy/no-op.

It also exercises STRICT mode (``require_real_tcr_prior=True``) to confirm that
missing weights / deps / structure raise an error instead of silently degrading.

Usage:
    python scripts/validate_tcr_prior_backends.py \
        [--esmc-model esmc_300m] \
        [--tcrlm-repo /path/to/tcrLM --tcrlm-ckpt /path/to/ckpt.pt] \
        [--ligandmpnn-repo /path --ligandmpnn-ckpt /path/*.pt --pdb /path/example.pdb]

The full ``bindcraft`` import chain (protenix/scipy) is not required; this script
stubs ``bindcraft.utils`` so that ``bindcraft.hallucination.tcr_prior`` can be
imported in isolation.
"""
from __future__ import annotations

import argparse
import os
import sys
import types

# --- Make ``import bindcraft.hallucination.tcr_prior`` work in isolation. -----
# tcr_prior only needs three constants from bindcraft.utils; the real module
# pulls in protenix/scipy which may be broken in some envs. Provide a light stub
# ONLY if the real import fails.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _install_bindcraft_stub() -> None:
    PROTEIN_RESIDUE_NAMES = [
        "ALA", "ARG", "ASN", "ASP", "CYS",
        "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO",
        "SER", "THR", "TRP", "TYR", "VAL",
    ]
    AA_1LETTER = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    bindcraft = types.ModuleType("bindcraft")
    bindcraft.__path__ = [os.path.join(REPO_ROOT, "bindcraft")]
    utils = types.ModuleType("bindcraft.utils")
    utils.PROTEIN_RESIDUE_NAMES = PROTEIN_RESIDUE_NAMES
    utils.AA_1LETTER = AA_1LETTER
    utils.NUM_AMINO_ACIDS = 20
    from typing import List, Union  # noqa
    utils.TargetSeqInput = "Union[str, List[str]]"
    hall = types.ModuleType("bindcraft.hallucination")
    hall.__path__ = [os.path.join(REPO_ROOT, "bindcraft", "hallucination")]
    sys.modules.setdefault("bindcraft", bindcraft)
    sys.modules["bindcraft.utils"] = utils
    sys.modules.setdefault("bindcraft.hallucination", hall)


try:
    import bindcraft.hallucination.tcr_prior as tp  # noqa: F401
except Exception:
    _install_bindcraft_stub()
    import importlib
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bindcraft.hallucination.tcr_prior",
        os.path.join(REPO_ROOT, "bindcraft", "hallucination", "tcr_prior.py"),
    )
    tp = importlib.util.module_from_spec(spec)
    sys.modules["bindcraft.hallucination.tcr_prior"] = tp
    spec.loader.exec_module(tp)

import torch  # noqa: E402


# --- A canonical human TRB sequence (variable region-ish) for testing. --------
TEST_TCR_SEQ = (
    "NAGVTQTPKFQVLKTGQSMTLQCAQDMNHEYMSWYRQDPGMGLRLIHYSVGAGITDQGEVPNGYNVSRSTTEDFPLRLLSAAPSQTSVYFCASSYSTGGADTQYFGPGTRLTVL"
)
# A CDR3-ish short peptide for TCRLM (CDR3 model).
TEST_CDR3 = "CASSYSTGGADTQYF"


def _soft_gradient_ok(scorer, seq: str, metadata=None) -> bool:
    """Return True iff soft_prior_loss produces a finite non-zero gradient."""
    L = len(seq)
    logits = torch.randn(L, 20, requires_grad=True)
    meta = dict(metadata or {})
    meta.setdefault("current_seq", seq)
    loss = scorer.soft_prior_loss(logits, region_mask=None, metadata=meta)
    if not torch.is_tensor(loss) or not loss.requires_grad:
        return False
    loss.backward()
    if logits.grad is None:
        return False
    g = logits.grad.abs().sum().item()
    return bool(g > 0.0 and g == g)  # non-zero and not NaN


def _hard_score_real(scorer, seq: str, metadata=None) -> "tuple[bool, float]":
    """Return (is_real, score). is_real True iff score is non-zero/meaningful."""
    val = scorer.score_hard_sequence(seq, region_mask=None, metadata=metadata)
    return (val != 0.0, float(val))


def validate_backend(name: str, config: dict, seq: str, metadata=None) -> dict:
    """Validate one backend; return a status row dict."""
    row = {
        "backend": name,
        "loaded": False,
        "real_weights": False,
        "real_output": False,
        "real_score": False,
        "soft_grad": False,
        "fallback": True,
        "missing": [],
        "strict_error": None,
    }

    # 1) Non-strict build: inspect backend status directly.
    cfg = dict(config)
    cfg["require_real_tcr_prior"] = False
    cfg.setdefault("use_tcr_prior_gradient", True)
    cfg.setdefault("use_tcr_prior_verifier", True)
    try:
        scorer = tp.build_tcr_prior_scorer(cfg, device=torch.device("cpu"))
    except Exception as exc:  # pragma: no cover
        row["missing"].append(f"build failed: {exc}")
        return row

    backend = getattr(scorer, "backend", None)
    if backend is not None:
        st = backend.status(metadata)
        row["loaded"] = st["loaded"]
        row["real_weights"] = st["real_weights"]
        row["real_output"] = st["real_output"]
        row["missing"] = st["missing_items"]
        row["fallback"] = not (st["loaded"] and st["real_weights"] and st["real_output"] is True)
    else:
        # Fell back to HF/placeholder scorer -> definitely not a real backend.
        row["missing"].append("fell back to placeholder scorer (no real backend)")
        row["fallback"] = True

    # For structure-conditioned backends the model determines the length from
    # the PDB, so align the test sequence to the native length to exercise the
    # real hard-score / soft-gradient paths on matching tensors.
    test_seq = seq
    if backend is not None and getattr(backend, "requires_structure", False) and metadata:
        try:
            probe = backend.per_position_logprobs(seq, list(range(10000)), metadata)
            if probe is not None:
                test_seq = "A" * probe.shape[0]
        except Exception:
            pass

    # 2) Real score + soft gradient (best-effort).
    try:
        is_real, score = _hard_score_real(scorer, test_seq, metadata)
        row["real_score"] = is_real and not row["fallback"]
        row["hard_score_value"] = score
    except Exception as exc:
        row["missing"].append(f"hard score error: {exc}")
    try:
        row["soft_grad"] = _soft_gradient_ok(scorer, test_seq, metadata) and not row["fallback"]
    except Exception as exc:
        row["missing"].append(f"soft grad error: {exc}")

    # 3) Strict-mode probe: confirm it raises when not fully functional.
    strict_cfg = dict(cfg)
    strict_cfg["require_real_tcr_prior"] = True
    if metadata:
        strict_cfg["tcr_prior_probe_metadata"] = metadata
    try:
        strict_scorer = tp.build_tcr_prior_scorer(strict_cfg, device=torch.device("cpu"))
        # For structure backends, also confirm the runtime strict path scores.
        if metadata:
            strict_scorer.score_hard_sequence(test_seq, region_mask=None, metadata=metadata)
        row["strict_error"] = None  # strict build+score succeeded -> backend fully real
    except tp.TCRPriorStrictError as exc:
        row["strict_error"] = str(exc)
    except Exception as exc:
        row["strict_error"] = f"(non-strict-error) {exc}"

    return row


def print_table(rows: list) -> None:
    def yn(b):
        return "yes" if b else "no"

    header = (
        "| backend | real model loaded? | real score? | soft gradient works? "
        "| fallback/no-op? | missing items |"
    )
    sep = "| ------- | ------------------ | ----------- | -------------------- | --------------- | ------------- |"
    print("\n" + header)
    print(sep)
    for r in rows:
        missing = "; ".join(r["missing"]) if r["missing"] else "-"
        if len(missing) > 80:
            missing = missing[:77] + "..."
        print(
            f"| {r['backend']} | {yn(r['loaded'] and r['real_weights'])} "
            f"| {yn(r['real_score'])} | {yn(r['soft_grad'])} "
            f"| {yn(r['fallback'])} | {missing} |"
        )
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--esmc-model", default="esmc_300m")
    ap.add_argument("--tcrlm-repo", default=os.environ.get("TCRLM_HOME"))
    ap.add_argument("--tcrlm-ckpt", default=os.environ.get("TCRLM_CKPT"))
    ap.add_argument("--ligandmpnn-repo", default=os.environ.get("LIGANDMPNN_HOME"))
    ap.add_argument("--ligandmpnn-ckpt", default=os.environ.get("LIGANDMPNN_CKPT"))
    ap.add_argument("--pdb", default=os.environ.get("TCR_PRIOR_PDB"))
    ap.add_argument("--only", default=None, help="comma list: esmc,tcrlm,ligandmpnn")
    args = ap.parse_args()

    only = set((args.only or "esmc,tcrlm,ligandmpnn").split(","))
    rows = []

    if "esmc" in only:
        print("\n=== Validating ESM-C ===")
        cfg = {"tcr_prior_model_name": f"esmc:{args.esmc_model}"}
        rows.append(validate_backend("esmc", cfg, TEST_TCR_SEQ))

    if "tcrlm" in only:
        print("\n=== Validating TCRLM ===")
        name = "tcrlm"
        if args.tcrlm_repo:
            name = f"tcrlm:{args.tcrlm_repo}"
        cfg = {"tcr_prior_model_name": name, "tcr_prior_checkpoint": args.tcrlm_ckpt}
        rows.append(validate_backend("tcrlm", cfg, TEST_CDR3))

    if "ligandmpnn" in only:
        print("\n=== Validating LigandMPNN ===")
        name = "ligandmpnn"
        if args.ligandmpnn_repo:
            name = f"ligandmpnn:{args.ligandmpnn_repo}"
        cfg = {
            "tcr_prior_model_name": name,
            "tcr_prior_checkpoint": args.ligandmpnn_ckpt,
        }
        meta = {"pdb_path": args.pdb} if args.pdb else None
        rows.append(validate_backend("ligandmpnn", cfg, TEST_TCR_SEQ, metadata=meta))

    # Detailed per-backend report.
    print("\n=== Detailed report ===")
    for r in rows:
        print(f"\n[{r['backend']}]")
        for k, v in r.items():
            if k == "backend":
                continue
            print(f"  {k}: {v}")

    print_table(rows)


if __name__ == "__main__":
    main()
