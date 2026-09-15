#!/usr/bin/env python3
"""Run one real scaffold-consistency verifier smoke test for a CDR case.

The script intentionally does not invoke MCTS.  It verifies the template's
native binder sequence once with the new constrained verifier and once with
the existing sequence-only verifier in the same runtime.  The latter confirms
that temporary sampler replacement has been restored before a normal call.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from bindcraft.hallucination.esmfold2_backend import ESMFold2ProposalConfig  # noqa: E402
from bindcraft.hallucination.esmfold2_cdr import prepare_esmfold2_cdr_case  # noqa: E402
from bindcraft.hallucination.esmfold2_runtime import ESMFold2RuntimeBackend  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="CDR case YAML containing input_dir and cases with template paths",
    )
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def write_mmcif(summary: dict[str, Any], destination: Path) -> None:
    mmcif = summary.pop("mmcif", None)
    if not isinstance(mmcif, str) or not mmcif.strip():
        raise RuntimeError("ESMFold2 verifier did not return a mmCIF structure")
    destination.write_text(mmcif, encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    case_config = next(
        (item for item in raw_config.get("cases", []) if item.get("id") == args.case),
        None,
    )
    if case_config is None:
        raise ValueError(f"Case not found in config: {args.case}")
    input_dir = resolve_path(str(raw_config["input_dir"]))
    case = prepare_esmfold2_cdr_case(case_config, input_dir=input_dir)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    proposal_config = ESMFold2ProposalConfig(
        verifier_num_loops=20,
        verifier_num_sampling_steps=200,
        verifier_structure_mode="scaffold_consistency",
        target_chain_roles=case.target_chain_roles,
        binder_mutable_mask=case.mutable_mask,
        template_structure_path=case.template_path,
        template_chain_ids=case.template_chain_ids,
        template_chain_sequences=(*case.partner_sequences, case.binder_sequence),
        binder_structure_regeneration_mask=case.mutable_mask,
    )
    backend = ESMFold2RuntimeBackend(
        config=proposal_config,
        device=args.device,
        task_type="cdr_redesign",
    )

    constrained = backend.verify_scaffold_consistency(
        binder_seq=case.binder_sequence,
        target_seq=case.partner_sequences,
        seed=args.seed,
    )
    constrained_summary = dict(constrained.summary_confidence)
    write_mmcif(constrained_summary, output_dir / "scaffold_consistency.cif")

    # The regular verifier must still use its original sampler after the call above.
    sequence_only = backend.verify(
        binder_seq=case.binder_sequence,
        target_seq=case.partner_sequences,
        seed=args.seed,
    )
    sequence_only_summary = dict(sequence_only.summary_confidence)
    write_mmcif(sequence_only_summary, output_dir / "sequence_only_after_restore.cif")

    report = {
        "status": "ok",
        "case": case.to_dict(),
        "config": {
            "verifier_num_loops": 20,
            "verifier_num_sampling_steps": 200,
            "verifier_structure_mode": "scaffold_consistency",
            "seed": args.seed,
        },
        "scaffold_consistency": {
            "total_loss": constrained.total_loss,
            "loss_dict": constrained.loss_dict,
            "summary_confidence": constrained_summary,
        },
        "sequence_only_after_restore": {
            "total_loss": sequence_only.total_loss,
            "loss_dict": sequence_only.loss_dict,
            "summary_confidence": sequence_only_summary,
        },
    }
    (output_dir / "result.json").write_text(
        json.dumps(json_safe(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
