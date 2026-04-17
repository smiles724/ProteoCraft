#!/usr/bin/env python3
"""
从 AF3Score 生成的 PDB 中提取链 A 作为初始 binder，并对固定的 A6 TCR (链 B+C) 批量优化。

支持：
1. benchmark/yaml/default.yaml 作为默认配置
2. CLI 显式参数覆盖 YAML 配置
3. worker_index / num_workers 分片，便于 shell 层多 GPU 并行
4. hybrid 模式下将每个 entry 的优化过程 step checkpoint 保存到 results/<entry_id>/checkpoints/
"""

import os
import sys
import csv
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch
import yaml


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.run_hallucination import load_protenix_model  # noqa: E402
from bindcraft.hallucination.pipeline import (  # noqa: E402
    run_hallucination,
    run_hybrid_hallucination,
)


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

DEFAULT_CONFIG_PATH = Path(PROJECT_ROOT) / "benchmark" / "yaml" / "default.yaml"
DEFAULT_CONFIG: Dict[str, Any] = {
    "source_dir": "/data/huangyanjie/binder_design/AF3Score/input",
    "input_dir": str(Path(PROJECT_ROOT) / "benchmark" / "input" / "A6_TCR"),
    "results_dir": str(Path(PROJECT_ROOT) / "benchmark" / "results"),
    "model_checkpoint_dir": "/data/huangyanjie/Alphaenzyme/LGOS/checkpoints",
    "components_cache_dir": "/data/huangyanjie/Alphaenzyme/LGOS/protenix/release_data/ccd_cache",
    "protenix_root_dir": str(Path(PROJECT_ROOT) / "benchmark" / "protenix_runtime"),
    "model_name": "protenix_base_20250630_v1.0.0",
    "device": "cuda:0",
    "mode": "hybrid",
    "n_steps": 50,
    "learning_rate": 0.05,
    "tau_noise": 4.0,
    "k_top": 8,
    "b_candidates": 4,
    "t_sample": 1.0,
    "max_mutations_per_seq": 3,
    "use_annealing": True,
    "t_init": 1.0,
    "t_min": 0.01,
    "use_tau_scheduling": True,
    "tau_jitter": True,
    "entry_ids": None,
    "skip_existing": False,
    "save_checkpoints": True,
    "checkpoint_subdir_name": "checkpoints",
}
RUNTIME_DEFAULTS: Dict[str, Any] = {
    "prepare_only": False,
    "skip_input_write": False,
    "worker_index": 0,
    "num_workers": 1,
}


def add_bool_argument(parser: argparse.ArgumentParser, name: str, help_text: str) -> None:
    dest = name.replace("-", "_")
    parser.add_argument(
        f"--{name}",
        dest=dest,
        action="store_true",
        default=argparse.SUPPRESS,
        help=help_text,
    )
    parser.add_argument(
        f"--no-{name}",
        dest=dest,
        action="store_false",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )


def setup_logging(log_path: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("benchmark_batch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        # Overwrite log on each run for this worker index.
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"YAML 配置必须是字典: {config_path}")

    unknown_keys = sorted(set(loaded) - set(DEFAULT_CONFIG))
    if unknown_keys:
        raise ValueError(
            f"{config_path} 包含未知配置项: {', '.join(unknown_keys)}"
        )

    config = dict(DEFAULT_CONFIG)
    config.update(loaded)
    return config


def resolve_args(parsed_args: argparse.Namespace) -> argparse.Namespace:
    config_path = Path(parsed_args.config).expanduser()
    config = load_yaml_config(config_path)
    config.update(RUNTIME_DEFAULTS)

    for key, value in vars(parsed_args).items():
        if key == "config":
            continue
        config[key] = value

    config["config"] = str(config_path.resolve())
    return argparse.Namespace(**config)


def extract_chain_sequence(pdb_path: Path, chain_id: str) -> str:
    residues = []
    seen = set()

    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue

            current_chain = line[21].strip() or "_"
            if current_chain != chain_id:
                continue

            resid = (line[22:26].strip(), line[26].strip())
            if resid in seen:
                continue
            seen.add(resid)

            resname = line[17:20].strip()
            residues.append(AA3_TO_1.get(resname, "X"))

    if not residues:
        raise ValueError(f"{pdb_path} 中未找到链 {chain_id} 的蛋白质序列")

    return "".join(residues)


def collect_entries(source_dir: Path) -> tuple[List[Dict[str, Any]], List[str]]:
    pdb_files = sorted(source_dir.rglob("*.pdb"))
    if not pdb_files:
        raise FileNotFoundError(f"{source_dir} 下未找到 PDB 文件")

    entries: List[Dict[str, Any]] = []
    target_b_set = set()
    target_c_set = set()

    for pdb_path in pdb_files:
        binder_seq = extract_chain_sequence(pdb_path, "A")
        target_b = extract_chain_sequence(pdb_path, "B")
        target_c = extract_chain_sequence(pdb_path, "C")

        target_b_set.add(target_b)
        target_c_set.add(target_c)

        entries.append(
            {
                "entry_id": pdb_path.stem,
                "source_pdb": str(pdb_path),
                "binder_chain": "A",
                "binder_length": len(binder_seq),
                "init_binder_seq": binder_seq,
                "target_chain_b_seq": target_b,
                "target_chain_c_seq": target_c,
            }
        )

    if len(target_b_set) != 1 or len(target_c_set) != 1:
        raise ValueError("检测到链 B/C 在不同 PDB 中不一致，无法作为固定 target 批量优化")

    target_sequences = [target_b_set.pop(), target_c_set.pop()]
    return entries, target_sequences


def write_input_files(entries: List[Dict[str, Any]], target_sequences: List[str], input_dir: Path) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = input_dir / "init_binders.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "entry_id",
            "source_pdb",
            "binder_chain",
            "binder_length",
            "init_binder_seq",
            "target_chain_b_seq",
            "target_chain_c_seq",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry)

    target_fasta = input_dir / "target_chains.fasta"
    with target_fasta.open("w", encoding="utf-8") as handle:
        handle.write(">A6_TCR_chain_B\n")
        handle.write(target_sequences[0] + "\n")
        handle.write(">A6_TCR_chain_C\n")
        handle.write(target_sequences[1] + "\n")

    for entry in entries:
        fasta_path = input_dir / f"{entry['entry_id']}.fasta"
        with fasta_path.open("w", encoding="utf-8") as handle:
            handle.write(f">{entry['entry_id']}|chain_A|len={entry['binder_length']}\n")
            handle.write(entry["init_binder_seq"] + "\n")

    return manifest_path


def shard_entries(entries: List[Dict[str, Any]], worker_index: int, num_workers: int) -> List[Dict[str, Any]]:
    if num_workers <= 0:
        raise ValueError(f"num_workers={num_workers} 必须大于 0")
    if worker_index < 0 or worker_index >= num_workers:
        raise ValueError(f"worker_index={worker_index} 必须落在 [0, {num_workers})")
    return entries[worker_index::num_workers]


def ensure_link(src: Path, dst: Path) -> None:
    if dst.is_symlink():
        if Path(os.path.realpath(dst)) == src.resolve():
            return
        dst.unlink()
    elif dst.exists():
        if dst.is_dir():
            raise FileExistsError(f"已有目录占用链接目标，请手动处理: {dst}")
        dst.unlink()

    dst.symlink_to(src)


def prepare_protenix_root(
    protenix_root_dir: Path,
    components_cache_dir: Path,
    model_checkpoint_dir: Path,
) -> Path:
    components_cif = components_cache_dir / "components.cif"
    components_pkl = components_cache_dir / "components.cif.rdkit_mol.pkl"
    if not components_cif.exists():
        raise FileNotFoundError(f"未找到 CCD 组件文件: {components_cif}")
    if not components_pkl.exists():
        raise FileNotFoundError(f"未找到 CCD RDKit 缓存: {components_pkl}")
    if not model_checkpoint_dir.exists():
        raise FileNotFoundError(f"未找到模型 checkpoint 目录: {model_checkpoint_dir}")

    protenix_root_dir.mkdir(parents=True, exist_ok=True)
    common_dir = protenix_root_dir / "common"
    common_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_link = protenix_root_dir / "checkpoint"

    ensure_link(components_cif, common_dir / "components.cif")
    ensure_link(components_pkl, common_dir / "components.cif.rdkit_mol.pkl")
    ensure_link(model_checkpoint_dir, checkpoint_link)
    return protenix_root_dir


def parse_entry_ids(entry_ids: Any) -> Set[str]:
    if not entry_ids:
        return set()
    if isinstance(entry_ids, str):
        return {part.strip() for part in entry_ids.split(",") if part.strip()}
    if isinstance(entry_ids, list):
        return {str(part).strip() for part in entry_ids if str(part).strip()}
    raise TypeError("entry_ids 必须是 null、逗号分隔字符串或字符串列表")


def run_one_entry(
    entry: Dict[str, Any],
    target_sequences: List[str],
    model,
    device: torch.device,
    args: argparse.Namespace,
    logger: logging.Logger,
    results_dir: Path,
) -> None:
    entry_results_dir = results_dir / entry["entry_id"]
    entry_results_dir.mkdir(parents=True, exist_ok=True)
    result_path = entry_results_dir / "result.json"
    if args.skip_existing and result_path.exists():
        logger.info(f"{entry['entry_id']} 已存在，跳过")
        return

    optimization_checkpoint_dir: Optional[Path] = None
    if args.mode == "hybrid" and args.save_checkpoints:
        optimization_checkpoint_dir = entry_results_dir / args.checkpoint_subdir_name
        optimization_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"开始 {entry['entry_id']} | binder_len={entry['binder_length']} | "
        f"target={len(target_sequences[0])}+{len(target_sequences[1])}"
    )
    if optimization_checkpoint_dir is not None:
        logger.info(f"{entry['entry_id']} checkpoint 输出目录: {optimization_checkpoint_dir}")

    t0 = time.time()
    if args.mode == "first_order":
        best_seq, final_seq, losses_history = run_hallucination(
            model=model,
            target_seq=target_sequences,
            L_binder=entry["binder_length"],
            N_steps=args.n_steps,
            learning_rate=args.learning_rate,
            tau_noise=args.tau_noise,
            device=device,
            loss_weights=None,
            init_binder_seq=entry["init_binder_seq"],
        )
    else:
        best_seq, final_seq, losses_history = run_hybrid_hallucination(
            model=model,
            target_seq=target_sequences,
            L_binder=entry["binder_length"],
            N_steps=args.n_steps,
            learning_rate=args.learning_rate,
            tau_noise=args.tau_noise,
            device=device,
            loss_weights=None,
            K_top=args.k_top,
            B_candidates=args.b_candidates,
            T_sample=args.t_sample,
            max_mutations_per_seq=args.max_mutations_per_seq,
            use_annealing=args.use_annealing,
            T_init=args.t_init,
            T_min=args.t_min,
            use_tau_scheduling=args.use_tau_scheduling,
            tau_jitter=args.tau_jitter,
            init_binder_seq=entry["init_binder_seq"],
            checkpoint_dir=str(optimization_checkpoint_dir) if optimization_checkpoint_dir is not None else None,
        )

    elapsed = time.time() - t0
    best_loss = float(min(losses_history)) if losses_history else None
    best_step = int(losses_history.index(best_loss)) if losses_history else None
    final_loss = float(losses_history[-1]) if losses_history else None

    result = {
        "entry_id": entry["entry_id"],
        "source_pdb": entry["source_pdb"],
        "binder_length": entry["binder_length"],
        "init_binder_seq": entry["init_binder_seq"],
        "target_chain_b_seq": target_sequences[0],
        "target_chain_c_seq": target_sequences[1],
        "mode": args.mode,
        "best_seq": best_seq,
        "final_seq": final_seq,
        "loss_history": [float(loss) for loss in losses_history],
        "best_loss": best_loss,
        "best_step": best_step,
        "final_loss": final_loss,
        "elapsed_sec": round(elapsed, 2),
        "params": {
            "n_steps": args.n_steps,
            "learning_rate": args.learning_rate,
            "tau_noise": args.tau_noise,
            "k_top": args.k_top,
            "b_candidates": args.b_candidates,
            "t_sample": args.t_sample,
            "max_mutations_per_seq": args.max_mutations_per_seq,
            "use_annealing": args.use_annealing,
            "t_init": args.t_init,
            "t_min": args.t_min,
            "use_tau_scheduling": args.use_tau_scheduling,
            "tau_jitter": args.tau_jitter,
            "model_name": args.model_name,
            "save_checkpoints": args.save_checkpoints,
            "checkpoint_subdir_name": args.checkpoint_subdir_name,
        },
        "runtime": {
            "config_file": args.config,
            "worker_index": args.worker_index,
            "num_workers": args.num_workers,
            "resolved_device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "model_checkpoint_dir": args.model_checkpoint_dir,
            "optimization_checkpoint_dir": (
                str(optimization_checkpoint_dir) if optimization_checkpoint_dir is not None else None
            ),
            "protenix_root_dir": args.protenix_root_dir,
        },
    }

    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    logger.info(
        f"完成 {entry['entry_id']} | {elapsed:.1f}s | best={best_seq[:20]}... | saved={result_path}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="benchmark 批量 Binder 优化")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--source-dir", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--input-dir", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--results-dir", type=str, default=argparse.SUPPRESS)
    parser.add_argument(
        "--model-checkpoint-dir",
        "--checkpoint-dir",
        dest="model_checkpoint_dir",
        type=str,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--components-cache-dir", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--protenix-root-dir", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--model-name", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--device", type=str, default=argparse.SUPPRESS)
    parser.add_argument(
        "--mode",
        choices=["first_order", "hybrid"],
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--n-steps", dest="n_steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--tau-noise", dest="tau_noise", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--k-top", dest="k_top", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--b-candidates", dest="b_candidates", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--t-sample", dest="t_sample", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--max-mutations-per-seq",
        dest="max_mutations_per_seq",
        type=int,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--t-init", dest="t_init", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--t-min", dest="t_min", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--entry-ids", dest="entry_ids", type=str, default=argparse.SUPPRESS)
    parser.add_argument(
        "--checkpoint-subdir-name",
        dest="checkpoint_subdir_name",
        type=str,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-index", dest="worker_index", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--prepare-only", dest="prepare_only", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument(
        "--skip-input-write",
        dest="skip_input_write",
        action="store_true",
        default=argparse.SUPPRESS,
    )

    add_bool_argument(parser, "use-annealing", "是否启用模拟退火")
    add_bool_argument(parser, "use-tau-scheduling", "是否启用 tau 调度")
    add_bool_argument(parser, "tau-jitter", "是否启用 tau 抖动")
    add_bool_argument(parser, "skip-existing", "结果已存在时跳过")
    add_bool_argument(parser, "save-checkpoints", "hybrid 模式下保存优化过程 step checkpoint")
    return parser


def main() -> None:
    args = resolve_args(build_parser().parse_args())

    input_dir = Path(args.input_dir)
    base_results_dir = Path(args.results_dir)
    base_results_dir.mkdir(parents=True, exist_ok=True)
    task_id = input_dir.name
    results_dir = base_results_dir / task_id
    results_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(PROJECT_ROOT) / "benchmark" / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"worker_{args.worker_index}.log"
    logger = setup_logging(log_path)
    logger.info(f"配置文件: {args.config}")
    logger.info(f"任务: {task_id} | input_dir={input_dir} | results_dir={results_dir}")

    if args.mode == "first_order" and args.save_checkpoints:
        logger.warning("mode=first_order 不支持 step checkpoint，已忽略 save_checkpoints")
        args.save_checkpoints = False

    runtime_root = prepare_protenix_root(
        protenix_root_dir=Path(args.protenix_root_dir),
        components_cache_dir=Path(args.components_cache_dir),
        model_checkpoint_dir=Path(args.model_checkpoint_dir),
    )
    os.environ["PROTENIX_ROOT_DIR"] = str(runtime_root)
    os.environ["PROTENIX_CHECKPOINT_DIR"] = args.model_checkpoint_dir
    logger.info(f"PROTENIX_ROOT_DIR={runtime_root}")

    source_dir = Path(args.source_dir)
    entries, target_sequences = collect_entries(source_dir)
    logger.info(f"扫描到 {len(entries)} 个 source PDB 条目")
    logger.info(
        f"固定 target: chain_B={len(target_sequences[0])} aa, chain_C={len(target_sequences[1])} aa"
    )

    selected_entry_ids = parse_entry_ids(args.entry_ids)
    if selected_entry_ids:
        available_entry_ids = {entry["entry_id"] for entry in entries}
        missing_entry_ids = sorted(selected_entry_ids - available_entry_ids)
        if missing_entry_ids:
            logger.warning(f"以下 entry_id 未找到: {', '.join(missing_entry_ids)}")
        entries = [entry for entry in entries if entry["entry_id"] in selected_entry_ids]
        logger.info(f"按 entry_ids 过滤后剩余 {len(entries)} 个条目")

    if args.skip_input_write:
        logger.info(f"skip-input-write 已启用，跳过写入输入文件目录: {input_dir}")
    else:
        manifest_path = write_input_files(entries, target_sequences, input_dir)
        logger.info(f"已生成输入清单: {manifest_path}")

    if args.prepare_only:
        logger.info("prepare-only 模式，结束")
        return

    worker_entries = shard_entries(entries, args.worker_index, args.num_workers)
    logger.info(
        f"worker {args.worker_index}/{args.num_workers} 负责 {len(worker_entries)} 个条目"
    )
    if not worker_entries:
        logger.info("当前 worker 没有可处理条目")
        return

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    logger.info(f"加载模型 {args.model_name} 到 {device}")
    model, _ = load_protenix_model(
        model_name=args.model_name,
        device=str(device),
        checkpoint_dir=args.model_checkpoint_dir,
    )

    for entry in worker_entries:
        try:
            run_one_entry(
                entry=entry,
                target_sequences=target_sequences,
                model=model,
                device=device,
                args=args,
                logger=logger,
                results_dir=results_dir,
            )
        except Exception as exc:
            logger.exception(f"{entry['entry_id']} 失败: {exc}")

    logger.info("worker 处理完成")


if __name__ == "__main__":
    main()
