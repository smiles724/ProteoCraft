#!/usr/bin/env python3
"""Run CDR3-only TCR domain redesign jobs from FASTA/CIF inputs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from bindcraft.hallucination.pipeline_cdr import (  # noqa: E402
    run_hybrid_hallucination_cdr,
    run_mcts_hallucination_cdr,
)
from bindcraft.utils import configure_protenix_common_paths  # noqa: E402
from scripts.run_hallucination import load_protenix_model  # noqa: E402


DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "benchmark" / "yaml" / "cdr_tcr" / "cdr_tcr_domain_redesign.yaml"
)

LOSS_WEIGHT_KEYS = (
    "plddt",
    "pae",
    "i_pae",
    "con",
    "i_con",
    "iptm",
    "rg",
    "helix",
    "clash",
    "i_clash",
    "pde",
    "i_pde",
)


def setup_logging(log_file: Optional[Path]) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def add_bool_argument(parser: argparse.ArgumentParser, name: str) -> None:
    dest = name.replace("-", "_")
    parser.add_argument(f"--{name}", dest=dest, action="store_true", default=argparse.SUPPRESS)
    parser.add_argument(f"--no-{name}", dest=dest, action="store_false", default=argparse.SUPPRESS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--case", default=None, help="Run only one case id from cases[].id")
    parser.add_argument("--case-index", type=int, default=None, help="Run one 0-based case index")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--device", default=argparse.SUPPRESS)
    parser.add_argument("--n-steps", dest="n_steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--output-dir", dest="output_dir", default=argparse.SUPPRESS)
    parser.add_argument("--log-file", dest="log_file", default=argparse.SUPPRESS)
    add_bool_argument(parser, "save-checkpoints")
    return parser.parse_args()


def resolve_path(raw: Optional[str], *, base_dir: Path = PROJECT_ROOT) -> Optional[Path]:
    if raw is None:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = resolve_path(args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"YAML config must be a mapping: {config_path}")

    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        if value is not argparse.SUPPRESS:
            config[key] = value
    config["config"] = str(config_path)
    return config


def read_fasta_records(fasta_path: Path) -> Dict[str, Dict[str, str]]:
    if not fasta_path.exists():
        raise FileNotFoundError(fasta_path)

    records: Dict[str, Dict[str, str]] = {}
    header: Optional[str] = None
    seq_parts: List[str] = []

    def flush() -> None:
        if header is None:
            return
        parts = header.split("|")
        chain_id = None
        for part in parts:
            part = part.strip()
            if part.startswith("Chain "):
                chain_id = part.replace("Chain ", "", 1).strip()
                break
        if not chain_id:
            raise ValueError(f"Could not parse chain id from FASTA header: {header}")
        seq = "".join(seq_parts).upper()
        invalid = sorted(set(seq) - set("ACDEFGHIKLMNPQRSTVWY"))
        if invalid:
            raise ValueError(f"{fasta_path} chain {chain_id} has nonstandard amino acids: {invalid}")
        records[chain_id] = {"header": header, "sequence": seq}

    with fasta_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    flush()
    return records


def make_cdr_mask(sequence: str, motif: str) -> Tuple[List[bool], int, int]:
    motif = motif.strip().upper()
    start = sequence.find(motif)
    if start < 0:
        raise ValueError(f"CDR motif not found: {motif}")
    if sequence.find(motif, start + 1) >= 0:
        raise ValueError(f"CDR motif is not unique: {motif}")
    mask = [False] * len(sequence)
    for idx in range(start, start + len(motif)):
        mask[idx] = True
    return mask, start, start + len(motif)


def resolve_loss_weights(config: Dict[str, Any]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for key in LOSS_WEIGHT_KEYS:
        value = config.get(f"w_{key}")
        if value is not None:
            weights[key] = float(value)
    return weights


def prepare_runtime(config: Dict[str, Any]) -> None:
    common_dir = resolve_path(config.get("components_cache_dir"))
    checkpoint_dir = resolve_path(config.get("model_checkpoint_dir"))
    protenix_root = resolve_path(config.get("protenix_root_dir"))
    if common_dir is None or checkpoint_dir is None or protenix_root is None:
        return

    runtime_common = protenix_root / "common"
    runtime_common.mkdir(parents=True, exist_ok=True)
    for name in ("components.cif", "components.cif.rdkit_mol.pkl"):
        src = common_dir / name
        dst = runtime_common / name
        if not src.exists():
            raise FileNotFoundError(src)
        if dst.is_symlink() and Path(os.path.realpath(dst)) == src.resolve():
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
    checkpoint_link = protenix_root / "checkpoint"
    if checkpoint_link.is_symlink() and Path(os.path.realpath(checkpoint_link)) == checkpoint_dir.resolve():
        pass
    else:
        if checkpoint_link.exists() or checkpoint_link.is_symlink():
            checkpoint_link.unlink()
        checkpoint_link.symlink_to(checkpoint_dir)

    os.environ["PROTENIX_ROOT_DIR"] = str(protenix_root)
    os.environ["PROTENIX_CHECKPOINT_DIR"] = str(checkpoint_dir)
    configure_protenix_common_paths(str(runtime_common))


def select_cases(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Config must define a non-empty cases list")
    case_id = config.get("case")
    case_index = config.get("case_index")
    if case_id is not None:
        cases = [case for case in cases if case.get("id") == case_id]
        if not cases:
            raise ValueError(f"Case id not found: {case_id}")
    if case_index is not None:
        idx = int(case_index)
        if idx < 0 or idx >= len(cases):
            raise ValueError(f"case_index out of range: {idx}")
        cases = [cases[idx]]
    return cases


def validate_case(case: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    input_dir = resolve_path(config.get("input_dir"))
    if input_dir is None:
        raise ValueError("input_dir is required")
    fasta_path = resolve_path(case["fasta_path"], base_dir=input_dir)
    template_path = resolve_path(case["template_path"], base_dir=input_dir)
    records = read_fasta_records(fasta_path)

    binder_chain = str(case["binder_chain"])
    target_chains = [str(chain) for chain in case["target_chains"]]
    scaffold_chains = [str(chain) for chain in case.get("scaffold_chains", [])]
    if binder_chain not in records:
        raise ValueError(f"{case['id']}: binder chain {binder_chain} not in {fasta_path}")
    missing_targets = [chain for chain in target_chains if chain not in records]
    if missing_targets:
        raise ValueError(f"{case['id']}: target chains missing from {fasta_path}: {missing_targets}")
    missing_scaffolds = [chain for chain in scaffold_chains if chain not in records]
    if missing_scaffolds:
        raise ValueError(f"{case['id']}: scaffold chains missing from {fasta_path}: {missing_scaffolds}")
    overlap = set(target_chains) & set(scaffold_chains)
    if overlap:
        raise ValueError(f"{case['id']}: target/scaffold chains overlap: {sorted(overlap)}")
    if binder_chain in target_chains or binder_chain in scaffold_chains:
        raise ValueError(f"{case['id']}: binder chain cannot also be target/scaffold")

    binder_seq = records[binder_chain]["sequence"]
    target_sequences = [records[chain]["sequence"] for chain in target_chains]
    scaffold_sequences = [records[chain]["sequence"] for chain in scaffold_chains]
    partner_chains = target_chains + scaffold_chains
    partner_sequences = target_sequences + scaffold_sequences
    target_chain_roles = ["target"] * len(target_sequences) + ["scaffold"] * len(scaffold_sequences)
    cdr_mask, cdr_start, cdr_end = make_cdr_mask(binder_seq, str(case["cdr_motif"]))
    expected_len = len(str(case["cdr_motif"]))
    if sum(cdr_mask) != expected_len:
        raise AssertionError(f"{case['id']}: mask size mismatch")

    return {
        "id": str(case["id"]),
        "records": records,
        "fasta_path": str(fasta_path),
        "template_path": str(template_path),
        "binder_chain": binder_chain,
        "target_chains": target_chains,
        "scaffold_chains": scaffold_chains,
        "binder_sequence": binder_seq,
        "target_sequences": target_sequences,
        "scaffold_sequences": scaffold_sequences,
        "partner_chains": partner_chains,
        "partner_sequences": partner_sequences,
        "target_chain_roles": target_chain_roles,
        "cdr_motif": str(case["cdr_motif"]).strip().upper(),
        "cdr_mask": cdr_mask,
        "cdr_start_0based": cdr_start,
        "cdr_end_0based_exclusive": cdr_end,
        "cdr_start_1based": cdr_start + 1,
        "cdr_end_1based": cdr_end,
    }


def framework_unchanged(reference: str, designed: str, mask: List[bool]) -> bool:
    if len(reference) != len(designed) or len(reference) != len(mask):
        return False
    return all(is_cdr or reference[idx] == designed[idx] for idx, is_cdr in enumerate(mask))


def run_case(
    case_info: Dict[str, Any],
    config: Dict[str, Any],
    model,
    device: torch.device,
    output_root: Path,
) -> int:
    case_dir = output_root / case_info["id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(case_dir / "run.log")
    result_path = case_dir / "result.json"
    checkpoint_dir = case_dir / str(config.get("checkpoint_subdir_name", "checkpoints"))

    if bool(config.get("save_checkpoints", True)):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_arg: Optional[str] = str(checkpoint_dir)
    else:
        checkpoint_arg = None

    logging.info(
        "Case %s | binder=%s len=%d | partners=%s | CDR=%s %d-%d | template=%s",
        case_info["id"],
        case_info["binder_chain"],
        len(case_info["binder_sequence"]),
        ",".join(case_info["partner_chains"]),
        case_info["cdr_motif"],
        case_info["cdr_start_1based"],
        case_info["cdr_end_1based"],
        case_info["template_path"],
    )

    loss_weights = resolve_loss_weights(config)
    mode = str(config.get("mode", "hybrid"))
    t0 = time.time()
    try:
        if mode == "hybrid":
            best_seq, final_seq, losses = run_hybrid_hallucination_cdr(
                model=model,
                binder_sequence=case_info["binder_sequence"],
                cdr_mask=case_info["cdr_mask"],
                target_seq=case_info["partner_sequences"],
                target_chain_roles=case_info["target_chain_roles"],
                N_steps=int(config.get("n_steps", 50)),
                learning_rate=float(config.get("learning_rate", 0.05)),
                tau_noise=float(config.get("tau_noise", 4.0)),
                device=device,
                loss_weights=loss_weights,
                K_top=int(config.get("k_top", 8)),
                B_candidates=int(config.get("b_candidates", 4)),
                T_sample=float(config.get("t_sample", 1.0)),
                max_mutations_per_seq=int(config.get("max_mutations_per_seq", 3)),
                use_annealing=bool(config.get("use_annealing", True)),
                T_init=float(config.get("t_init", 1.0)),
                T_min=float(config.get("t_min", 0.01)),
                use_tau_scheduling=bool(config.get("use_tau_scheduling", True)),
                tau_jitter=bool(config.get("tau_jitter", True)),
                use_prior=bool(config.get("use_prior", True)),
                prior_strength=float(config.get("prior_strength", 0.1)),
                checkpoint_dir=checkpoint_arg,
                target_template_path=case_info["template_path"],
                chunk_size=config.get("chunk_size"),
            )
        elif mode == "mcts":
            best_seq, final_seq, losses = run_mcts_hallucination_cdr(
                model=model,
                binder_sequence=case_info["binder_sequence"],
                cdr_mask=case_info["cdr_mask"],
                target_seq=case_info["partner_sequences"],
                target_chain_roles=case_info["target_chain_roles"],
                N_steps=int(config.get("n_steps", 50)),
                learning_rate=float(config.get("learning_rate", 0.05)),
                tau_noise=float(config.get("tau_noise", 4.0)),
                device=device,
                loss_weights=loss_weights,
                K_top=int(config.get("k_top", 8)),
                B_candidates=int(config.get("b_candidates", 4)),
                T_sample=float(config.get("t_sample", 1.0)),
                max_mutations_per_seq=int(config.get("max_mutations_per_seq", 3)),
                use_annealing=bool(config.get("use_annealing", True)),
                T_init=float(config.get("t_init", 1.0)),
                T_min=float(config.get("t_min", 0.01)),
                num_beams=int(config.get("mcts_num_beams", 8)),
                use_prior=bool(config.get("use_prior", True)),
                prior_strength=float(config.get("prior_strength", 0.1)),
                checkpoint_dir=checkpoint_arg,
                target_template_path=case_info["template_path"],
                mcts_C=float(config.get("mcts_c", 1.0)),
                mcts_max_nodes=int(config.get("mcts_max_nodes", 5000)),
                mcts_tau_fixed=float(config.get("mcts_tau_fixed", 4.0)),
                chunk_size=config.get("chunk_size"),
            )
        else:
            raise ValueError(f"Unsupported mode for CDR TCR redesign: {mode}")

        elapsed = time.time() - t0
        result = {
            "status": "ok",
            "case_id": case_info["id"],
            "fasta_path": case_info["fasta_path"],
            "template_path": case_info["template_path"],
            "binder_chain": case_info["binder_chain"],
            "target_chains": case_info["target_chains"],
            "scaffold_chains": case_info["scaffold_chains"],
            "binder_sequence": case_info["binder_sequence"],
            "target_sequences": case_info["target_sequences"],
            "scaffold_sequences": case_info["scaffold_sequences"],
            "partner_chains": case_info["partner_chains"],
            "partner_sequences": case_info["partner_sequences"],
            "target_chain_roles": case_info["target_chain_roles"],
            "cdr_motif": case_info["cdr_motif"],
            "cdr_start_1based": case_info["cdr_start_1based"],
            "cdr_end_1based": case_info["cdr_end_1based"],
            "cdr_mask_true_count": int(sum(case_info["cdr_mask"])),
            "best_seq": best_seq,
            "final_seq": final_seq,
            "best_framework_unchanged": framework_unchanged(
                case_info["binder_sequence"], best_seq, case_info["cdr_mask"]
            ),
            "final_framework_unchanged": framework_unchanged(
                case_info["binder_sequence"], final_seq, case_info["cdr_mask"]
            ),
            "loss_history": [float(loss) for loss in losses],
            "best_loss": float(min(losses)) if losses else None,
            "best_step": int(losses.index(min(losses))) if losses else None,
            "final_loss": float(losses[-1]) if losses else None,
            "elapsed_sec": round(elapsed, 2),
            "params": {
                "mode": mode,
                "n_steps": int(config.get("n_steps", 50)),
                "learning_rate": float(config.get("learning_rate", 0.05)),
                "tau_noise": float(config.get("tau_noise", 4.0)),
                "k_top": int(config.get("k_top", 8)),
                "b_candidates": int(config.get("b_candidates", 4)),
                "t_sample": float(config.get("t_sample", 1.0)),
                "max_mutations_per_seq": int(config.get("max_mutations_per_seq", 3)),
                "use_annealing": bool(config.get("use_annealing", True)),
                "use_tau_scheduling": bool(config.get("use_tau_scheduling", True)),
                "tau_jitter": bool(config.get("tau_jitter", True)),
                "use_prior": bool(config.get("use_prior", True)),
                "prior_strength": float(config.get("prior_strength", 0.1)),
                "chunk_size": config.get("chunk_size"),
                "model_name": config.get("model_name"),
                "loss_weights": loss_weights,
            },
            "runtime": {
                "device": str(device),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "checkpoint_dir": checkpoint_arg,
            },
        }
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        if not result["best_framework_unchanged"] or not result["final_framework_unchanged"]:
            logging.error("Framework changed outside CDR mask; result saved to %s", result_path)
            return 2
        logging.info("Saved result: %s", result_path)
        return 0
    except Exception as exc:
        result = {
            "status": "failed",
            "case_id": case_info["id"],
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        logging.exception("Case %s failed", case_info["id"])
        return 1


def main() -> int:
    args = parse_args()
    config = load_config(args)
    output_root = resolve_path(config.get("output_dir"))
    if output_root is None:
        raise ValueError("output_dir is required")
    output_root.mkdir(parents=True, exist_ok=True)

    global_log = resolve_path(config.get("log_file")) if config.get("log_file") else output_root / "launcher.log"
    setup_logging(global_log)
    prepare_runtime(config)

    cases = select_cases(config)
    case_infos = [validate_case(case, config) for case in cases]
    for info in case_infos:
        logging.info(
            "Validated %s | binder=%s len=%d | CDR=%s %d-%d | partners=%s",
            info["id"],
            info["binder_chain"],
            len(info["binder_sequence"]),
            info["cdr_motif"],
            info["cdr_start_1based"],
            info["cdr_end_1based"],
            ",".join(info["partner_chains"]),
        )

    if config.get("prepare_only"):
        logging.info("prepare-only complete")
        return 0

    device = torch.device(str(config.get("device", "cuda:0")))
    if device.type == "cuda":
        torch.cuda.set_device(device)

    model, _ = load_protenix_model(
        model_name=str(config.get("model_name", "protenix_base_20250630_v1.0.0")),
        device=str(device),
        checkpoint_dir=str(resolve_path(config.get("model_checkpoint_dir"))),
    )

    status = 0
    for case_info in case_infos:
        status = max(status, run_case(case_info, config, model, device, output_root))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
