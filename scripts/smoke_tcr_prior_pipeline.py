#!/usr/bin/env python
"""Minimal end-to-end smoke for TCR prior proposal/verifier/MCTS wiring.

This does not run a full AF3/ProteoCraft hallucination job.  It uses the same
TCR prior scorer, verifier hook, and MCTSTree reward path that the pipeline
calls, but with tiny synthetic ZerothOrderResult-like candidates.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bindcraft.hallucination.mcts_controller import MCTSTree
from bindcraft.hallucination.tcr_prior import (
    AA_1_TO_IDX,
    build_tcr_prior_scorer,
    resolve_region_mask,
)


@dataclass
class ToyVerifierResult:
    binder_seq: str
    total_loss: float
    loss_dict: Dict[str, float]
    coordinate: torch.Tensor
    summary_confidence: Dict[str, Any]


SEQUENCES = [
    ("cdr3_like", "CASSLAPGATNEKLFF"),
    ("repetitive", "AAAAAAAAAAAAAAAA"),
    ("scrambled", "MFWPGLTYHNDQREKV"),
]


def _candidate(seq: str, total_loss: float = 1.0) -> ToyVerifierResult:
    return ToyVerifierResult(
        binder_seq=seq,
        total_loss=float(total_loss),
        loss_dict={"total": float(total_loss), "i_pae": 0.5},
        coordinate=torch.zeros(1, len(seq), 3),
        summary_confidence={},
    )


def _tree_value(result: ToyVerifierResult) -> float:
    tree = MCTSTree(
        C=1.0,
        max_nodes=8,
        max_children_per_node=4,
        use_reward_normalization=False,
        use_loss_grouping=False,
        use_pareto_pruning=False,
        use_elite_archive=False,
        use_structural_grouping=False,
        use_diversity_bonus=False,
    )
    root = tree.add_root_node(
        "AAAAAAAAAA",
        torch.zeros(10, 20),
        L_real=1.0,
        loss_vector={"total": 1.0, "i_pae": 0.5},
    )
    children = tree.expand(root, [result])
    if not children:
        raise RuntimeError("MCTS expand produced no child for toy candidate")
    return float(children[0].tree_reward)


def _one_hot_like_logits(seq: str, on_value: float = 6.0, off_value: float = -6.0) -> torch.Tensor:
    logits = torch.full((len(seq), 20), float(off_value))
    for pos, aa in enumerate(seq):
        logits[pos, AA_1_TO_IDX[aa]] = float(on_value)
    return logits.requires_grad_(True)


def _region_indices(mask: List[bool]) -> List[int]:
    return [idx for idx, keep in enumerate(mask) if keep]


def _hard_distribution_from_backend(
    scorer: Any,
    seq: str,
    mask: List[bool],
    metadata: Dict[str, Any],
) -> tuple[float, torch.Tensor, List[int]]:
    """Compute positive prior NLL = -mean log p from one backend log-prob pass."""
    backend = getattr(scorer, "backend", None)
    if backend is None:
        weighted = scorer.score_hard_sequence(seq, region_mask=mask, metadata=metadata)
        weight = float(getattr(scorer, "verifier_weight", 1.0) or 1.0)
        indices = _region_indices(mask)
        logp = torch.full((len(indices), 20), -float("inf"))
        return -float(weighted) / weight, logp, indices
    indices = _region_indices(mask)
    logp = backend.per_position_logprobs(seq, indices, metadata)
    if logp is None:
        raise RuntimeError("backend produced no hard-score distribution")
    aa_idx = torch.tensor([AA_1_TO_IDX[seq[i]] for i in indices], dtype=torch.long)
    nll = -logp.gather(1, aa_idx[:, None]).squeeze(1).mean()
    return float(nll), logp.detach(), indices


def _soft_gradient_from_distribution(
    seq: str,
    logp: torch.Tensor,
    indices: List[int],
    gradient_weight: float,
) -> tuple[float, float]:
    logits = _one_hot_like_logits(seq)
    idx = torch.tensor(indices, dtype=torch.long)
    q_region = torch.log_softmax(logits.index_select(0, idx), dim=-1)
    p_prior = logp.to(logits.device).exp().detach()
    soft = -(p_prior * q_region).sum(dim=-1).mean() * float(gradient_weight)
    soft.backward()
    grad = logits.grad
    grad_norm = float(grad.norm().detach().cpu()) if grad is not None else 0.0
    return float(soft.detach().cpu()), grad_norm


def _scorer_soft_path_works(scorer: Any, seq: str, mask: List[bool], metadata: Dict[str, Any]) -> bool:
    logits = _one_hot_like_logits(seq)
    soft = scorer.soft_prior_loss(logits, region_mask=mask, metadata={**metadata, "current_seq": seq})
    soft.backward()
    grad = logits.grad
    return bool(float(soft.detach().cpu()) != 0.0 and grad is not None and grad.norm().item() > 0.0)


def _inject_weighted_prior_loss(result: ToyVerifierResult, weighted_prior_loss: float) -> ToyVerifierResult:
    out = copy.deepcopy(result)
    loss = float(weighted_prior_loss)
    out.total_loss = float(out.total_loss) + loss
    out.loss_dict["tcr_prior_hard"] = loss
    return out


def _run_sequence(
    backend: str,
    scorer: Any,
    seq_type: str,
    seq: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "backend": backend,
        "sequence_type": seq_type,
        "prior_nll": None,
        "weighted_prior_loss": None,
        "tree_value_changed": False,
        "soft_grad_norm": None,
        "notes": "-",
    }
    try:
        mask = resolve_region_mask("all", None, len(seq))
        metadata = {**metadata, "sequence_type": seq_type, "seq": seq}

        prior_nll, logp, indices = _hard_distribution_from_backend(scorer, seq, mask, metadata)
        verifier_weight = float(getattr(scorer, "verifier_weight", 1.0) or 1.0)
        gradient_weight = float(getattr(scorer, "gradient_weight", 1.0) or 1.0)
        weighted_prior_loss = prior_nll * verifier_weight

        off = _candidate(seq)
        on = _inject_weighted_prior_loss(off, weighted_prior_loss)
        off_value = _tree_value(off)
        on_value = _tree_value(on)

        soft_loss, grad_norm = _soft_gradient_from_distribution(seq, logp, indices, gradient_weight)

        has_prior_loss = "tcr_prior_hard" in on.loss_dict
        row.update(
            prior_nll=prior_nll,
            weighted_prior_loss=float(weighted_prior_loss),
            tree_value_changed=has_prior_loss and (on.total_loss != off.total_loss) and (on_value != off_value),
            soft_grad_norm=grad_norm,
            candidate_total_loss=float(on.total_loss),
            tree_value_off=off_value,
            tree_value_on=on_value,
            soft_loss=soft_loss,
        )
        if not has_prior_loss:
            row["notes"] = "missing tcr_prior_hard"
        elif grad_norm <= 0.0 or soft_loss == 0.0:
            row["notes"] = "soft path produced zero loss/gradient"
    except Exception as exc:
        row["notes"] = f"{type(exc).__name__}: {exc}"
    return row


def _run_backend(name: str, cfg: Dict[str, Any], pdb_path: Optional[str]) -> List[Dict[str, Any]]:
    print(f"[smoke] loading backend={name}", file=sys.stderr, flush=True)
    try:
        scorer = build_tcr_prior_scorer(cfg, device=torch.device("cpu"))
    except Exception as exc:
        return [
            {
                "backend": name,
                "sequence_type": seq_type,
                "prior_nll": None,
                "weighted_prior_loss": None,
                "tree_value_changed": False,
                "soft_grad_norm": None,
                "notes": f"build failed: {type(exc).__name__}: {exc}",
            }
            for seq_type, _seq in SEQUENCES
        ]

    metadata: Dict[str, Any] = {"stage": "candidate"}
    if pdb_path:
        metadata["pdb_path"] = pdb_path
    rows = []
    for seq_type, seq in SEQUENCES:
        print(f"[smoke] scoring backend={name} sequence_type={seq_type}", file=sys.stderr, flush=True)
        rows.append(_run_sequence(name, scorer, seq_type, seq, metadata))
    seq_type, seq = SEQUENCES[0]
    mask = resolve_region_mask("all", None, len(seq))
    try:
        soft_path_ok = _scorer_soft_path_works(
            scorer,
            seq,
            mask,
            {**metadata, "sequence_type": seq_type, "seq": seq},
        )
    except Exception as exc:
        soft_path_ok = False
        rows[0]["notes"] = f"scorer.soft_prior_loss failed: {type(exc).__name__}: {exc}"
    if soft_path_ok and rows[0]["notes"] == "-":
        rows[0]["notes"] = "scorer.soft_prior_loss path ok"
    for row in rows:
        if name == "esmc" and row["notes"] == "-":
            row["notes"] = "ESM-C unmasked baseline; self-conditioning/token leakage possible"
        elif name == "esmc" and "scorer.soft_prior_loss path ok" in row["notes"]:
            row["notes"] += "; unmasked baseline"
        elif name == "esmc_masked" and row["notes"] == "-":
            row["notes"] = "ESM-C masked PLL"
        elif name == "esmc_masked" and "scorer.soft_prior_loss path ok" in row["notes"]:
            row["notes"] += "; masked PLL"
    return rows


def _ligandmpnn_missing_pdb_note(cfg: Dict[str, Any]) -> str:
    try:
        scorer = build_tcr_prior_scorer(cfg, device=torch.device("cpu"))
        seq = SEQUENCES[0][1]
        mask = resolve_region_mask("all", None, len(seq))
        scorer.score_hard_sequence(seq, region_mask=mask, metadata={"stage": "candidate", "seq": seq})
    except Exception as exc:
        return f"without metadata['pdb_path']: {type(exc).__name__}: {exc}"
    return "without metadata['pdb_path']: unexpectedly scored"


def _fmt_float(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.6f}"


def _fmt_bool(value: bool) -> str:
    return "yes" if value else "no"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcrlm-repo", default=os.environ.get("TCRLM_HOME"))
    ap.add_argument("--tcrlm-ckpt", default=os.environ.get("TCRLM_CKPT"))
    ap.add_argument("--esmc-model", default=os.environ.get("ESMC_MODEL", "esmc_300m"))
    ap.add_argument("--ligandmpnn-repo", default=os.environ.get("LIGANDMPNN_HOME"))
    ap.add_argument("--ligandmpnn-ckpt", default=os.environ.get("LIGANDMPNN_CKPT"))
    ap.add_argument("--pdb", default=os.environ.get("TCR_PRIOR_PDB"))
    ap.add_argument(
        "--backends",
        default="tcrlm,esmc,ligandmpnn",
        help="Comma-separated subset: tcrlm,esmc,esmc_masked,ligandmpnn",
    )
    args = ap.parse_args()
    selected_backends = {item.strip().lower() for item in args.backends.split(",") if item.strip()}

    base = {
        "use_tcr_prior_gradient": True,
        "tcr_prior_gradient_weight": 0.1,
        "use_tcr_prior_verifier": True,
        "tcr_prior_verifier_weight": 0.1,
        "require_real_tcr_prior": True,
        "tcr_prior_region": "all",
    }
    configs: List[tuple[str, Dict[str, Any], Optional[str]]] = [
        (
            "tcrlm",
            {
                **base,
                "tcr_prior_model_name": f"tcrlm:{args.tcrlm_repo}" if args.tcrlm_repo else "tcrlm",
                "tcr_prior_checkpoint": args.tcrlm_ckpt,
            },
            None,
        ),
        (
            "esmc",
            {**base, "tcr_prior_model_name": f"esmc:{args.esmc_model}"},
            None,
        ),
        (
            "esmc_masked",
            {**base, "tcr_prior_model_name": f"esmc:{args.esmc_model}", "tcr_prior_masked": True},
            None,
        ),
        (
            "ligandmpnn",
            {
                **base,
                "tcr_prior_model_name": (
                    f"ligandmpnn:{args.ligandmpnn_repo}" if args.ligandmpnn_repo else "ligandmpnn"
                ),
                "tcr_prior_checkpoint": args.ligandmpnn_ckpt,
                "tcr_prior_probe_metadata": {"pdb_path": args.pdb} if args.pdb else None,
            },
            args.pdb,
        ),
    ]

    rows: List[Dict[str, Any]] = []
    ligand_cfg = None
    for name, cfg, pdb_path in configs:
        if name not in selected_backends:
            continue
        if name == "ligandmpnn":
            ligand_cfg = cfg
        rows.extend(_run_backend(name, cfg, pdb_path))

    if ligand_cfg is not None:
        missing_pdb_note = _ligandmpnn_missing_pdb_note(ligand_cfg)
        for row in rows:
            if row["backend"] == "ligandmpnn":
                row["notes"] = (
                    "explicit PDB works; actual pipeline needs metadata['pdb_path']"
                    if row["notes"] == "-"
                    else row["notes"]
                )
    else:
        missing_pdb_note = "LigandMPNN config not present"

    print("| backend | sequence type | prior_nll | weighted_prior_loss | tree value changed? | soft grad norm | notes |")
    print("| ------- | ------------- | --------- | ------------------- | ------------------- | -------------- | ----- |")
    for row in rows:
        print(
            "| {backend} | {seq_type} | {nll} | {weighted} | {changed} | {grad} | {notes} |".format(
                backend=row["backend"],
                seq_type=row["sequence_type"],
                nll=_fmt_float(row["prior_nll"]),
                weighted=_fmt_float(row["weighted_prior_loss"]),
                changed=_fmt_bool(bool(row["tree_value_changed"])),
                grad=_fmt_float(row["soft_grad_norm"]),
                notes=row["notes"],
            )
        )

    print("\nDetails:")
    for row in rows:
        print(
            "{backend}/{seq_type}: total_loss={total_loss} tree={tree_off}->{tree_on} soft_loss={soft_loss}".format(
                backend=row["backend"],
                seq_type=row["sequence_type"],
                total_loss=_fmt_float(row.get("candidate_total_loss")),
                tree_off=_fmt_float(row.get("tree_value_off")),
                tree_on=_fmt_float(row.get("tree_value_on")),
                soft_loss=_fmt_float(row.get("soft_loss")),
            )
        )
    print(f"\nLigandMPNN structure metadata check: {missing_pdb_note}")

    ok = all(
        bool(r["tree_value_changed"]) and (r["soft_grad_norm"] is not None) and float(r["soft_grad_norm"]) > 0.0
        for r in rows
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
