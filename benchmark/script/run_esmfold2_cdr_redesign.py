#!/usr/bin/env python3
"""Run sequence-only ESMFold2 CDR redesign from the four-case YAML format."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from bindcraft.hallucination.esmfold2_cdr import (  # noqa: E402
    ESMFold2CDRCase,
    framework_unchanged,
    prepare_esmfold2_cdr_case,
)


DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "benchmark" / "yaml" / "cdr_tcr" / "esmfold2_cdr_redesign.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--case", default=None)
    parser.add_argument("--case-index", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def resolve_path(raw_path: str, *, base_dir: Path = PROJECT_ROOT) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else base_dir / path


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = resolve_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"YAML config must be a mapping: {config_path}")
    config["config_path"] = str(config_path)
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.device is not None:
        config["device"] = args.device
    config["prepare_only"] = bool(args.prepare_only)
    config["case"] = args.case
    config["case_index"] = args.case_index
    return config


def select_cases(config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    cases = list(config.get("cases") or [])
    if not cases:
        raise ValueError("Config must define a non-empty cases list")
    if config.get("case") is not None:
        cases = [case for case in cases if str(case.get("id")) == str(config["case"])]
        if not cases:
            raise ValueError(f"Case id not found: {config['case']}")
    if config.get("case_index") is not None:
        index = int(config["case_index"])
        if index < 0 or index >= len(cases):
            raise ValueError(f"case_index out of range: {index}")
        cases = [cases[index]]
    return cases


def _append_cli_argument(command: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        value = str(value).lower()
    command.extend((f"--{name}", str(value)))


def build_hallucination_command(
    case: ESMFold2CDRCase,
    config: Mapping[str, Any],
    *,
    case_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_hallucination.py"),
        "--model_backend",
        "esmfold2",
        "--mode",
        "mcts",
        "--esmfold2_task_type",
        "cdr_redesign",
        "--target",
        case.target_sequence_argument,
        "--esmfold2_target_chain_roles",
        case.target_chain_roles_argument,
        "--binder_length",
        str(len(case.binder_sequence)),
        "--init_binder_seq",
        case.binder_sequence,
        "--esmfold2_mutable_mask",
        case.mutable_mask_argument,
        "--output_dir",
        str(case_dir),
        "--esmfold2_artifact_dir",
        str(case_dir / "esmfold2_artifacts"),
        "--log_file",
        str(case_dir / "run.log"),
    ]
    option_map = {
        "device": "device",
        "n_steps": "n_steps",
        "seed": "seed",
        "k_top": "k_top",
        "b_candidates": "b_candidates",
        "t_sample": "t_sample",
        "max_mutations_per_seq": "max_mutations_per_seq",
        "stagea_max_mutations_per_seq": "stagea_max_mutations_per_seq",
        "stageb_max_mutations_per_seq": "stageb_max_mutations_per_seq",
        "mcts_num_beams": "mcts_num_beams",
        "mcts_strategy": "mcts_strategy",
        "mcts_bootstrap_steps": "mcts_bootstrap_steps",
        "mcts_bootstrap_max_steps": "mcts_bootstrap_max_steps",
        "mcts_expand_actions_per_step": "mcts_expand_actions_per_step",
        "esmfold2_stagea_soft_steps": "esmfold2_stagea_soft_steps",
        "esmfold2_stageb_soft_steps": "esmfold2_stageb_soft_steps",
        "esmfold2_verifier_num_loops": "esmfold2_verifier_num_loops",
        "esmfold2_verifier_num_sampling_steps": "esmfold2_verifier_num_sampling_steps",
        "esmfold2_proposal_num_loops": "esmfold2_proposal_num_loops",
        "esmfold2_proposal_num_sampling_steps": "esmfold2_proposal_num_sampling_steps",
        "esmfold2_proposal_load_internal_esmc": "esmfold2_proposal_load_internal_esmc",
        "esmfold2_proposal_lm_dropout": "esmfold2_proposal_lm_dropout",
        "esmfold2_force_proposal_lm_dropout": "esmfold2_force_proposal_lm_dropout",
        "esmfold2_lambda_esmc": "esmfold2_lambda_esmc",
        "esmfold2_logit_init_mode": "esmfold2_logit_init_mode",
        "esmfold2_stagea_handoff_checkpoint_interval": "esmfold2_stagea_handoff_checkpoint_interval",
    }
    for config_key, cli_name in option_map.items():
        _append_cli_argument(command, cli_name, config.get(config_key))
    return command


def parse_hallucination_result(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            values[key.strip()] = value.strip()
    return values


def write_case_record(
    case: ESMFold2CDRCase,
    config: Mapping[str, Any],
    *,
    case_dir: Path,
    command: Iterable[str],
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case_spec.json").write_text(
        json.dumps(case.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (case_dir / "command.txt").write_text(
        shlex.join(list(command)) + "\n",
        encoding="utf-8",
    )
    provenance = {
        "model_backend": "esmfold2",
        "task_type": "cdr_redesign",
        "structure_condition": "sequence_only",
        "template_path": case.template_path,
        "template_usage": "recorded_only_not_consumed_by_esmfold2",
        "config_path": config["config_path"],
        "target_chain_roles": list(case.target_chain_roles),
        "mutable_residue_count": sum(case.mutable_mask),
    }
    (case_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_case(
    case: ESMFold2CDRCase,
    config: Mapping[str, Any],
    *,
    output_root: Path,
) -> int:
    case_dir = output_root / case.case_id
    command = build_hallucination_command(case, config, case_dir=case_dir)
    write_case_record(case, config, case_dir=case_dir, command=command)
    if bool(config.get("prepare_only")):
        return 0

    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    result_path = case_dir / "hallucination_result.txt"
    summary: Dict[str, Any] = {
        "case_id": case.case_id,
        "returncode": int(completed.returncode),
        "status": "failed",
    }
    if completed.returncode == 0 and result_path.is_file():
        result = parse_hallucination_result(result_path)
        best_sequence = result.get("Best Sequence", "")
        final_sequence = result.get("Final Sequence", "")
        summary.update(
            {
                "status": "ok",
                "best_sequence": best_sequence,
                "final_sequence": final_sequence,
                "best_framework_unchanged": framework_unchanged(
                    case.binder_sequence,
                    best_sequence,
                    case.mutable_mask,
                ),
                "final_framework_unchanged": framework_unchanged(
                    case.binder_sequence,
                    final_sequence,
                    case.mutable_mask,
                ),
            }
        )
        if not summary["best_framework_unchanged"] or not summary[
            "final_framework_unchanged"
        ]:
            summary["status"] = "framework_changed"
            completed = subprocess.CompletedProcess(
                completed.args,
                returncode=2,
            )
    (case_dir / "cdr_result.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return int(completed.returncode)


def main() -> int:
    args = parse_args()
    config = load_config(args)
    input_dir = resolve_path(str(config["input_dir"]))
    output_root = resolve_path(str(config["output_dir"]))
    output_root.mkdir(parents=True, exist_ok=True)
    cases = [
        prepare_esmfold2_cdr_case(case, input_dir=input_dir)
        for case in select_cases(config)
    ]
    return max(
        (run_case(case, config, output_root=output_root) for case in cases),
        default=0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
