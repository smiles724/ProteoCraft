#!/usr/bin/env python3
"""Blindly score fixed Affibody/pMHC complexes with the ESMFold2 verifier."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bindcraft.affibody.data import read_csv_rows, sha256_file, write_json  # noqa: E402
from bindcraft.affibody.profiles import get_affibody_profile  # noqa: E402
from bindcraft.affibody.targets import (  # noqa: E402
    PMHC_TARGET_CHAIN_ROLES,
    build_pmhc_target,
    chains_from_reference_pdb,
)


FORBIDDEN_BLIND_COLUMNS = {
    "retention_percent",
    "cleavage_binder",
    "binder_threshold_percent",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "固定序列盲评分器。默认只检查输入并写计划；"
            "显式传 --execute 才加载 ESMFold2/GPU"
        )
    )
    parser.add_argument(
        "--complexes",
        type=Path,
        default=PROJECT_ROOT / "data_prepared" / "affibody" / "known_complexes_blind.csv",
    )
    parser.add_argument(
        "--reference-pdb",
        type=Path,
        default=PROJECT_ROOT / "datafolder" / "nyeso_xx133_complex.pdb",
    )
    parser.add_argument("--hla-chain", default="A")
    parser.add_argument("--b2m-chain", default="B")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "validation" / "known_pair_scores_blind.jsonl",
    )
    parser.add_argument("--complex-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seeds", default="101,102,103")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verifier-model", default="biohub/ESMFold2")
    parser.add_argument("--verifier-num-loops", type=int, default=20)
    parser.add_argument("--verifier-num-sampling-steps", type=int, default=200)
    parser.add_argument(
        "--verifier-loss-mode",
        choices=["pair_chain_iptm", "aligned_common"],
        default="aligned_common",
        help=(
            "full-verifier loss；默认 aligned_common，与正式 Affibody 设计一致。"
            "pair_chain_iptm 仅用于单独诊断，不能与正式盲评混用"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return _json_value(float(value.detach().cpu().item()))
            return None
    except ImportError:
        pass
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if math.isfinite(scalar) else None


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(token.strip()) for token in str(raw).split(",") if token.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    complexes_path = args.complexes.resolve()
    reference_pdb = args.reference_pdb.resolve()
    rows = read_csv_rows(complexes_path)
    if not rows:
        raise ValueError("complex manifest is empty")
    leaked = FORBIDDEN_BLIND_COLUMNS.intersection(rows[0])
    if leaked:
        raise ValueError(
            "blind complex input contains experimental labels: " + ", ".join(sorted(leaked))
        )
    required = {
        "complex_id", "library", "peptide", "register",
        "affibody_sequence", "affibody_tuple",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"complex manifest is missing columns: {sorted(missing)}")
    if args.complex_id:
        wanted = {value.strip() for item in args.complex_id for value in item.split(",")}
        rows = [row for row in rows if row["complex_id"] in wanted]
        found = {row["complex_id"] for row in rows}
        if wanted - found:
            raise ValueError(f"unknown complex ID(s): {sorted(wanted - found)}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    seeds = _parse_seeds(args.seeds)
    for row in rows:
        profile = get_affibody_profile(row["library"])
        sequence = profile.validate_sequence(row["affibody_sequence"])
        if profile.extract_tuple(sequence) != row["affibody_tuple"].upper():
            raise ValueError(f"{row['complex_id']} tuple does not match its full sequence")
    output_path = args.output.resolve()
    plan = {
        "schema_version": 1,
        "blind_input": str(complexes_path),
        "blind_input_sha256": sha256_file(complexes_path),
        "reference_pdb": str(reference_pdb),
        "reference_pdb_sha256": sha256_file(reference_pdb),
        "complex_count": len(rows),
        "seeds": seeds,
        "planned_verifier_calls": len(rows) * len(seeds),
        "output": str(output_path),
        "executed": bool(args.execute),
        "config": {
            "model_backend": "esmfold2",
            "task_type": "affibody_redesign",
            "model": args.verifier_model,
            "num_loops": args.verifier_num_loops,
            "num_sampling_steps": args.verifier_num_sampling_steps,
            "loss_mode": args.verifier_loss_mode,
            "target_chain_roles": list(PMHC_TARGET_CHAIN_ROLES),
            "device": args.device,
        },
    }
    plan_path = output_path.with_suffix(output_path.suffix + ".plan.json")
    write_json(plan_path, plan)
    print(
        f"Validated blind input: {len(rows)} complexes × {len(seeds)} seeds "
        f"= {len(rows) * len(seeds)} verifier calls"
    )
    if not args.execute:
        print(f"Plan: {plan_path}")
        print("Preparation only: ESMFold2 was not loaded and no score was computed.")
        return 0

    from bindcraft.hallucination.esmfold2_backend import ESMFold2ProposalConfig
    from bindcraft.hallucination.esmfold2_runtime import ESMFold2RuntimeBackend

    hla_sequence, b2m_sequence = chains_from_reference_pdb(
        reference_pdb, hla_chain=args.hla_chain, b2m_chain=args.b2m_chain
    )
    config = ESMFold2ProposalConfig(
        verifier_model_name=args.verifier_model,
        verifier_num_loops=args.verifier_num_loops,
        verifier_num_sampling_steps=args.verifier_num_sampling_steps,
        verifier_loss_mode=args.verifier_loss_mode,
        target_chain_roles=PMHC_TARGET_CHAIN_ROLES,
    )
    backend = ESMFold2RuntimeBackend(
        config=config, device=args.device, task_type="affibody_redesign"
    )
    completed: set[tuple[str, int]] = set()
    if args.resume and output_path.is_file():
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    completed.add((str(record["complex_id"]), int(record["seed"])))
    elif output_path.exists():
        raise FileExistsError(
            f"output already exists: {output_path}; use --resume or choose a new path"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for row_index, row in enumerate(rows, start=1):
            target = build_pmhc_target(
                target_id=row["complex_id"],
                peptide=row["peptide"],
                hla_sequence=hla_sequence,
                b2m_sequence=b2m_sequence,
                register=row["register"],
            )
            hotspot_params = {
                "target_hotspot_residues_1based": list(target.hotspot_positions_1based),
                "target_hotspot_concat_indices": [
                    value - 1 for value in target.hotspot_positions_1based
                ],
            }
            for seed in seeds:
                if (row["complex_id"], seed) in completed:
                    continue
                print(
                    f"[{row_index}/{len(rows)}] {row['complex_id']} seed={seed}",
                    flush=True,
                )
                result = backend.verify(
                    binder_seq=row["affibody_sequence"],
                    target_seq=[hla_sequence, b2m_sequence, row["peptide"]],
                    seed=seed,
                    hotspot_params=hotspot_params,
                )
                summary = {
                    str(key): converted
                    for key, value in dict(result.summary_confidence).items()
                    if key != "mmcif" and (converted := _json_value(value)) is not None
                }
                loss_dict = {
                    str(key): converted
                    for key, value in dict(result.loss_dict).items()
                    if (converted := _json_value(value)) is not None
                }
                payload = {
                    "complex_id": row["complex_id"],
                    "library": row["library"],
                    "peptide_pair": row.get("peptide_pair", target.up_facing_pair),
                    "peptide": row["peptide"],
                    "affibody_tuple": row["affibody_tuple"],
                    "seed": seed,
                    "target_chain_roles": list(PMHC_TARGET_CHAIN_ROLES),
                    "verifier_loss_mode": args.verifier_loss_mode,
                    "total_loss": float(result.total_loss),
                    "loss_dict": loss_dict,
                    "summary_confidence": summary,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                handle.flush()
    print(f"Blind scores: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
