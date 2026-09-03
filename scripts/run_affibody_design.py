#!/usr/bin/env python3
"""Generate or run constrained Affibody redesign with the ESMFold2 workflow."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bindcraft.affibody.data import read_csv_rows, sha256_file, write_json  # noqa: E402
from bindcraft.affibody.profiles import (  # noqa: E402
    AffibodyProfile,
    get_affibody_profile,
)
from bindcraft.affibody.targets import (  # noqa: E402
    PMHC_TARGET_CHAIN_ROLES,
    PMHC_TARGET_CHAIN_ROLES_ARGUMENT,
    PMHCTarget,
    build_pmhc_target,
    chains_from_reference_pdb,
)


ESMFOLD2_LAUNCH_PROFILE = "esmfold2_d"
ESMFOLD2_TASK_TYPE = "affibody_redesign"
ESMFOLD2_SEARCH_MODE = "mcts"
ESMFOLD2_MCTS_STRATEGY = "two_stage"
ESMFOLD2_LOGIT_INIT_MODE = "official_exact"
ESMFOLD2_VERIFIER_LOSS_MODE = "aligned_common"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "生成 Affibody/pMHC ProteoCraft 命令。默认只写命令清单；"
            "只有显式传入 --execute 才启动设计"
        )
    )
    parser.add_argument(
        "--target-manifest",
        type=Path,
        default=PROJECT_ROOT / "data_prepared" / "affibody" / "targets.csv",
    )
    parser.add_argument(
        "--target-id",
        action="append",
        default=[],
        help="目标 ID，可重复或用逗号分隔；默认 T001；传 all 选择全部",
    )
    parser.add_argument("--library", choices=["liba", "libb"], default="libb")
    parser.add_argument("--init-tuple", default=None)
    parser.add_argument(
        "--reference-pdb",
        type=Path,
        default=PROJECT_ROOT / "datafolder" / "nyeso_xx133_complex.pdb",
    )
    parser.add_argument("--hla-chain", default="A")
    parser.add_argument("--b2m-chain", default="B")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "affibody_design",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="每个 target 的独立 seed 数；开发完成后建议 8，默认 1 用于安全试运行",
    )
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真的逐条运行命令；不传时只生成 commands.sh 和 design_runs.jsonl",
    )
    return parser


def _requested_ids(values: list[str]) -> list[str]:
    tokens = [
        token.strip()
        for value in values
        for token in str(value).split(",")
        if token.strip()
    ]
    return tokens or ["T001"]


def build_esmfold2_affibody_command(
    *,
    python_executable: str,
    target: PMHCTarget,
    profile: AffibodyProfile,
    initial_sequence: str,
    run_dir: Path,
    device: str,
    seed: int,
    n_steps: int | None = None,
) -> list[str]:
    """Build one command while keeping the formal ESMFold2 workflow intact.

    Affibody-specific behavior is expressed only through task inputs: the full
    starting sequence, mutable mask, experimental alphabet, pMHC chain roles,
    and peptide hotspots. Search and verification stay on the shared
    ``esmfold2_d`` two-stage MCTS implementation.
    """

    initial_sequence = profile.validate_sequence(initial_sequence)
    artifact_dir = run_dir / "esmfold2_artifacts"
    command = [
        str(python_executable),
        str(PROJECT_ROOT / "scripts" / "run_hallucination.py"),
        "--profile", ESMFOLD2_LAUNCH_PROFILE,
        "--model_backend", "esmfold2",
        "--target", target.target_argument,
        "--mode", ESMFOLD2_SEARCH_MODE,
        "--mcts_strategy", ESMFOLD2_MCTS_STRATEGY,
        "--esmfold2_task_type", ESMFOLD2_TASK_TYPE,
        "--esmfold2_logit_init_mode", ESMFOLD2_LOGIT_INIT_MODE,
        "--binder_length", str(profile.length),
        "--init_binder_seq", initial_sequence,
        "--esmfold2_mutable_mask", profile.mutable_mask_argument,
        "--esmfold2_allowed_binder_aas", profile.allowed_aas_argument,
        "--esmfold2_target_chain_roles", target.target_chain_roles_argument,
        "--esmfold2_verifier_loss_mode", ESMFOLD2_VERIFIER_LOSS_MODE,
        "--target_hotspots", target.hotspot_argument,
        "--output_dir", str(run_dir),
        "--esmfold2_artifact_dir", str(artifact_dir),
        "--save_checkpoint_dir", str(run_dir / "checkpoints"),
        "--log_file", str(run_dir / "run.log"),
        "--device", str(device),
        "--seed", str(seed),
    ]
    if n_steps is not None:
        command.extend(["--n_steps", str(n_steps)])
    return command


def parse_hallucination_result(path: Path) -> dict[str, str]:
    """Parse the key/value portion of ``hallucination_result.txt``."""

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            values[key.strip()] = value.strip()
    return values


def validate_completed_esmfold2_run(
    *,
    run_dir: Path,
    profile: AffibodyProfile,
) -> dict[str, object]:
    """Fail closed if a completed run violated the Affibody workflow contract."""

    result_path = run_dir / "hallucination_result.txt"
    if not result_path.is_file():
        raise FileNotFoundError(f"missing ESMFold2 result: {result_path}")
    result = parse_hallucination_result(result_path)
    expected = {
        "Model Backend": "esmfold2",
        "Launch Profile": ESMFOLD2_LAUNCH_PROFILE,
        "Mode": ESMFOLD2_SEARCH_MODE,
        "MCTS Strategy": ESMFOLD2_MCTS_STRATEGY,
        "ESMFold2 Task Type": ESMFOLD2_TASK_TYPE,
        "ESMFold2 Logit Init Mode": ESMFOLD2_LOGIT_INIT_MODE,
        "ESMFold2 Mutable Mask": profile.mutable_mask_argument,
        "ESMFold2 Allowed Binder AAs": profile.allowed_aas_argument,
        "ESMFold2 Target Chain Roles": PMHC_TARGET_CHAIN_ROLES_ARGUMENT,
        "ESMFold2 Verifier Loss Mode": ESMFOLD2_VERIFIER_LOSS_MODE,
    }
    mismatches = {
        key: {"expected": value, "observed": result.get(key)}
        for key, value in expected.items()
        if result.get(key) != value
    }
    if mismatches:
        raise ValueError(f"ESMFold2 workflow contract mismatch: {mismatches}")
    best_sequence = profile.validate_sequence(result.get("Best Sequence", ""))
    final_sequence = profile.validate_sequence(result.get("Final Sequence", ""))
    summary: dict[str, object] = {
        "status": "ok",
        "model_backend": "esmfold2",
        "task_type": ESMFOLD2_TASK_TYPE,
        "launch_profile": ESMFOLD2_LAUNCH_PROFILE,
        "search_mode": ESMFOLD2_SEARCH_MODE,
        "mcts_strategy": ESMFOLD2_MCTS_STRATEGY,
        "target_chain_roles": list(PMHC_TARGET_CHAIN_ROLES),
        "verifier_loss_mode": ESMFOLD2_VERIFIER_LOSS_MODE,
        "best_sequence": best_sequence,
        "best_tuple": profile.extract_tuple(best_sequence),
        "final_sequence": final_sequence,
        "final_tuple": profile.extract_tuple(final_sequence),
        "framework_unchanged": True,
        "experimental_alphabet_satisfied": True,
        "source_result": str(result_path),
    }
    write_json(run_dir / "affibody_result.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.target_manifest.resolve()
    reference_pdb = args.reference_pdb.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"target manifest not found: {manifest_path}; run prepare_affibody_validation.py first"
        )
    if not reference_pdb.is_file():
        raise FileNotFoundError(f"reference PDB not found: {reference_pdb}")
    targets = read_csv_rows(manifest_path)
    by_id = {row["target_id"]: row for row in targets}
    requested = _requested_ids(args.target_id)
    selected = targets if any(value.lower() == "all" for value in requested) else []
    if not selected:
        unknown = [target_id for target_id in requested if target_id not in by_id]
        if unknown:
            raise ValueError(f"unknown target ID(s): {unknown}")
        selected = [by_id[target_id] for target_id in requested]

    profile = get_affibody_profile(args.library)
    if args.replicates < 1:
        raise ValueError("--replicates must be positive")
    initial_sequence = profile.render(args.init_tuple)
    hla_sequence, b2m_sequence = chains_from_reference_pdb(
        reference_pdb,
        hla_chain=args.hla_chain,
        b2m_chain=args.b2m_chain,
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    commands: list[list[str]] = []
    run_records: list[dict[str, object]] = []
    run_index = 0
    for target_row in selected:
        target = build_pmhc_target(
            target_id=target_row["target_id"],
            peptide=target_row["peptide"],
            hla_sequence=hla_sequence,
            b2m_sequence=b2m_sequence,
            register=target_row["register"],
        )
        for replicate_index in range(args.replicates):
            run_seed = args.seed + run_index
            run_id = f"{target.target_id}_{profile.name}_seed{run_seed}"
            run_dir = output_root / run_id
            artifact_dir = run_dir / "esmfold2_artifacts"
            command = build_esmfold2_affibody_command(
                python_executable=str(args.python),
                target=target,
                profile=profile,
                initial_sequence=initial_sequence,
                run_dir=run_dir,
                device=str(args.device),
                seed=run_seed,
                n_steps=args.n_steps,
            )
            commands.append(command)
            applicable = {
                item.strip().lower()
                for item in str(target_row.get("applicable_libraries", "")).split(";")
                if item.strip()
            }
            run_records.append(
                {
                    "run_id": run_id,
                    "target_id": target.target_id,
                    "library": profile.name,
                    "peptide": target.peptide,
                    "register": target.register,
                    "peptide_pair": target.up_facing_pair,
                    "target_support_level": target_row.get("support_level", ""),
                    "library_has_pair_cleavage": profile.name.lower() in applicable,
                    "init_tuple": profile.extract_tuple(initial_sequence),
                    "init_sequence": initial_sequence,
                    "output_dir": str(run_dir),
                    "artifact_summary": str(artifact_dir / "run_summary.json"),
                    "replicate_index": replicate_index + 1,
                    "seed": run_seed,
                    "model_backend": "esmfold2",
                    "task_type": ESMFOLD2_TASK_TYPE,
                    "launch_profile": ESMFOLD2_LAUNCH_PROFILE,
                    "mcts_strategy": ESMFOLD2_MCTS_STRATEGY,
                    "target_chain_roles": list(PMHC_TARGET_CHAIN_ROLES),
                    "verifier_loss_mode": ESMFOLD2_VERIFIER_LOSS_MODE,
                    "mutable_mask": profile.mutable_mask_argument,
                    "allowed_aas": profile.allowed_aas_argument,
                    "command": shlex.join(command),
                }
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                run_dir / "affibody_case.json",
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "model_backend": "esmfold2",
                    "task_type": ESMFOLD2_TASK_TYPE,
                    "launch_profile": ESMFOLD2_LAUNCH_PROFILE,
                    "search_mode": ESMFOLD2_SEARCH_MODE,
                    "mcts_strategy": ESMFOLD2_MCTS_STRATEGY,
                    "logit_init_mode": ESMFOLD2_LOGIT_INIT_MODE,
                    "structure_condition": "sequence_only",
                    "reference_pdb_usage": "HLA_and_beta2m_sequence_extraction_only",
                    "target": target.as_dict(),
                    "affibody_profile": profile.as_dict(),
                    "initial_sequence": initial_sequence,
                    "command": shlex.join(command),
                },
            )
            run_index += 1

    commands_path = output_root / "commands.sh"
    with commands_path.open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for command in commands:
            handle.write(shlex.join(command) + "\n")
    runs_path = output_root / "design_runs.jsonl"
    with runs_path.open("w", encoding="utf-8") as handle:
        for record in run_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(
        output_root / "design_manifest.json",
        {
            "schema_version": 1,
            "executed": bool(args.execute),
            "target_manifest_sha256": sha256_file(manifest_path),
            "reference_pdb_sha256": sha256_file(reference_pdb),
            "profile": profile.as_dict(),
            "workflow_contract": {
                "model_backend": "esmfold2",
                "task_type": ESMFOLD2_TASK_TYPE,
                "launch_profile": ESMFOLD2_LAUNCH_PROFILE,
                "search_mode": ESMFOLD2_SEARCH_MODE,
                "mcts_strategy": ESMFOLD2_MCTS_STRATEGY,
                "logit_init_mode": ESMFOLD2_LOGIT_INIT_MODE,
                "target_chain_roles": list(PMHC_TARGET_CHAIN_ROLES),
                "verifier_loss_mode": ESMFOLD2_VERIFIER_LOSS_MODE,
                "sequence_only": True,
            },
            "run_count": len(run_records),
            "commands_file": str(commands_path),
            "runs_file": str(runs_path),
        },
    )
    print(f"Generated {len(commands)} command(s): {commands_path}")
    if not args.execute:
        print("Preparation only: no model or GPU design was run.")
        return 0
    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] executing {run_records[index - 1]['run_id']}")
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        validate_completed_esmfold2_run(
            run_dir=Path(str(run_records[index - 1]["output_dir"])),
            profile=profile,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
