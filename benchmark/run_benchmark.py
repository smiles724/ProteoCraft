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
import re
import sys
import csv
import json
import time
import argparse
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import yaml


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.run_hallucination import load_protenix_model  # noqa: E402
from bindcraft.hallucination.pipeline import (  # noqa: E402
    run_hallucination,
    run_hybrid_hallucination,
    run_mcts_hallucination,
)
from bindcraft.utils import LOSS_WEIGHT_PRESETS  # noqa: E402


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
    "target_fasta_dir": None,
    "target_fasta_residue_start": None,
    "random_binder_length": 100,
    "random_binder_lengths": None,
    "init_binder_fasta_dir": None,
    "random_seed": 42,
    "input_dir": str(Path(PROJECT_ROOT) / "benchmark" / "input" / "A6_TCR"),
    "results_dir": str(Path(PROJECT_ROOT) / "benchmark" / "results"),
    "log_dir": None,
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
    "mcts_c": 1.0,
    "mcts_max_nodes": 5000,
    "mcts_tau_fixed": 4.0,
    "mcts_num_beams": 8,
    "mcts_strategy": "two_stage",
    "mcts_bootstrap_steps": 50,
    "mcts_bootstrap_patience": 10,
    "mcts_bootstrap_min_delta": 0.02,
    "mcts_bootstrap_max_steps": 80,
    "mcts_handoff_top_k": None,
    "mcts_expand_actions_per_step": 2,
    "chunk_size": None,
    "use_prior": True,
    "prior_strength": 0.1,
    "target_hotspot_residues": None,
    "loss_preset": "tuned",
    "target_template_path": None,
    "target_template_dir": None,
    "w_plddt": None,
    "w_pae": None,
    "w_i_pae": None,
    "w_con": None,
    "w_i_con": None,
    "w_iptm": None,
    "w_rg": None,
    "w_helix": None,
    "w_clash": None,
    "w_i_clash": None,
    "w_pde": None,
    "w_i_pde": None,
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


HOTSPOT_TOKEN_PATTERN = re.compile(
    r"^(?P<chain>[A-Za-z0-9_])(?P<start>-?\d+[A-Za-z]?)(?:-(?P<end>-?\d+[A-Za-z]?))?$"
)
RESIDUE_REF_PATTERN = re.compile(r"^(?P<number>-?\d+)(?P<icode>[A-Za-z]?)$")


def extract_chain_residues(pdb_path: Path, chain_id: str) -> List[Dict[str, Any]]:
    residues: List[Dict[str, Any]] = []
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

            resid_number_str = line[22:26].strip()
            insertion_code = line[26].strip().upper()
            resid = (resid_number_str, insertion_code)
            if resid in seen:
                continue
            seen.add(resid)

            resname = line[17:20].strip()
            residues.append(
                {
                    "chain_id": chain_id,
                    "residue_number": int(resid_number_str),
                    "insertion_code": insertion_code,
                    "seq_index": len(residues),
                    "aa": AA3_TO_1.get(resname, "X"),
                }
            )

    if not residues:
        raise ValueError(f"{pdb_path} 中未找到链 {chain_id} 的蛋白质序列")

    return residues


def extract_chain_sequence(pdb_path: Path, chain_id: str) -> str:
    return "".join(residue["aa"] for residue in extract_chain_residues(pdb_path, chain_id))


def _parse_residue_ref(label: str) -> Tuple[int, str]:
    match = RESIDUE_REF_PATTERN.fullmatch(label)
    if match is None:
        raise ValueError(f"非法残基编号: {label}")
    return int(match.group("number")), match.group("icode").upper()


def resolve_target_hotspots(
    chain_residue_map: Dict[str, List[Dict[str, Any]]],
    hotspot_spec: Optional[str],
) -> Tuple[List[int], List[str]]:
    if hotspot_spec is None:
        return [], []

    spec = str(hotspot_spec).strip()
    if not spec:
        return [], []

    chain_offsets: Dict[str, int] = {}
    running_offset = 0
    for chain_id in ("B", "C"):
        chain_offsets[chain_id] = running_offset
        running_offset += len(chain_residue_map.get(chain_id, []))

    resolved_indices: List[int] = []
    resolved_labels: List[str] = []
    seen_indices = set()

    for raw_token in spec.split(","):
        token = raw_token.strip()
        if not token:
            continue

        match = HOTSPOT_TOKEN_PATTERN.fullmatch(token)
        if match is None:
            raise ValueError(
                f"非法 hotspot 片段: {token}；当前必须使用带链名前缀的格式，如 B56 或 C12-15"
            )

        chain_id = match.group("chain")
        if chain_id not in chain_residue_map:
            raise ValueError(f"hotspot 指定了不存在的 target 链: {chain_id}")

        residues = chain_residue_map[chain_id]
        start_number, start_icode = _parse_residue_ref(match.group("start"))
        end_label = match.group("end")

        if end_label is None:
            matched = [
                residue
                for residue in residues
                if residue["residue_number"] == start_number
                and residue["insertion_code"] == start_icode
            ]
            if not matched:
                suffix = start_icode or ""
                raise ValueError(f"未在链 {chain_id} 中找到 hotspot 残基 {chain_id}{start_number}{suffix}")
        else:
            end_number, end_icode = _parse_residue_ref(end_label)
            if start_icode or end_icode:
                raise ValueError(f"暂不支持带 insertion code 的区间 hotspot: {token}")
            if end_number < start_number:
                raise ValueError(f"hotspot 区间非法: {token}")
            matched = [
                residue
                for residue in residues
                if start_number <= residue["residue_number"] <= end_number
            ]
            if not matched:
                raise ValueError(f"未在链 {chain_id} 中找到 hotspot 区间 {token}")

        chain_offset = chain_offsets[chain_id]
        for residue in matched:
            concat_index = chain_offset + residue["seq_index"]
            if concat_index in seen_indices:
                continue
            seen_indices.add(concat_index)
            resolved_indices.append(concat_index)
            resolved_labels.append(
                f"{chain_id}{residue['residue_number']}{residue['insertion_code']}"
            )

    if not resolved_indices:
        raise ValueError(f"未解析出任何有效 hotspot: {hotspot_spec}")

    return resolved_indices, resolved_labels



def read_fasta_sequence(fasta_path: Path) -> str:
    sequence_parts: List[str] = []
    with fasta_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            sequence_parts.append(line)
    sequence = "".join(sequence_parts).upper()
    invalid = sorted(set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"))
    if not sequence:
        raise ValueError(f"{fasta_path} 中没有 FASTA 序列")
    if invalid:
        raise ValueError(f"{fasta_path} 包含非标准氨基酸: {','.join(invalid)}")
    return sequence


def make_random_binder_sequence(length: int, rng: random.Random) -> str:
    if length <= 0:
        raise ValueError(f"random_binder_length={length} 必须大于 0")
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    return "".join(rng.choice(alphabet) for _ in range(length))


def make_fasta_target_residues(
    sequence: str,
    chain_id: str = "B",
    residue_start: int = 1,
) -> List[Dict[str, Any]]:
    return [
        {
            "chain_id": chain_id,
            "residue_number": residue_start + seq_index,
            "insertion_code": "",
            "seq_index": seq_index,
            "aa": aa,
        }
        for seq_index, aa in enumerate(sequence)
    ]


def collect_fasta_random_entries(
    target_fasta_dir: Path,
    random_binder_length: int,
    random_seed: int,
    random_binder_lengths: Optional[Dict[str, int]] = None,
    init_binder_fasta_dir: Optional[Path] = None,
    target_hotspot_residues: Optional[str] = None,
    target_fasta_residue_start: Optional[int] = None,
    selected_entry_ids: Optional[Set[str]] = None,
) -> tuple[List[Dict[str, Any]], List[str]]:
    fasta_files = sorted(
        list(target_fasta_dir.glob("*.fa"))
        + list(target_fasta_dir.glob("*.faa"))
        + list(target_fasta_dir.glob("*.fasta"))
    )
    if not fasta_files:
        raise FileNotFoundError(f"{target_fasta_dir} 下未找到 FASTA 文件")

    rng = random.Random(random_seed)
    per_case_lengths = random_binder_lengths or {}
    entries: List[Dict[str, Any]] = []
    for fasta_path in fasta_files:
        entry_id = fasta_path.stem

        target_seq = read_fasta_sequence(fasta_path)
        init_binder_path = init_binder_fasta_dir / f"{entry_id}.fasta" if init_binder_fasta_dir else None
        if init_binder_path is not None and init_binder_path.exists():
            binder_seq = read_fasta_sequence(init_binder_path)
            binder_source = str(init_binder_path)
        else:
            binder_length = int(per_case_lengths.get(entry_id, random_binder_length))
            binder_seq = make_random_binder_sequence(binder_length, rng)
            binder_source = "random"

        if selected_entry_ids and entry_id not in selected_entry_ids:
            continue

        hotspot_concat_indices: List[int] = []
        hotspot_labels: List[str] = []
        if target_hotspot_residues:
            residue_start = int(target_fasta_residue_start or 1)
            target_residues = make_fasta_target_residues(
                target_seq,
                chain_id="B",
                residue_start=residue_start,
            )
            hotspot_concat_indices, hotspot_labels = resolve_target_hotspots(
                {"B": target_residues},
                target_hotspot_residues,
            )

        entries.append(
            {
                "entry_id": entry_id,
                "source_pdb": str(fasta_path),
                "binder_chain": binder_source,
                "binder_length": len(binder_seq),
                "init_binder_seq": binder_seq,
                "target_chain_b_seq": target_seq,
                "target_chain_c_seq": "",
                "target_sequences": [target_seq],
                "target_hotspot_residues": target_hotspot_residues,
                "target_hotspot_concat_indices": hotspot_concat_indices,
                "target_hotspot_labels": hotspot_labels,
            }
        )

    return entries, []

def collect_entries(
    source_dir: Path,
    target_hotspot_residues: Optional[str] = None,
) -> tuple[List[Dict[str, Any]], List[str]]:
    pdb_files = sorted(source_dir.rglob("*.pdb"))
    if not pdb_files:
        raise FileNotFoundError(f"{source_dir} 下未找到 PDB 文件")

    entries: List[Dict[str, Any]] = []
    target_b_set = set()
    target_c_set = set()
    reference_hotspot_indices: Optional[List[int]] = None

    for pdb_path in pdb_files:
        binder_residues = extract_chain_residues(pdb_path, "A")
        target_b_residues = extract_chain_residues(pdb_path, "B")
        target_c_residues = extract_chain_residues(pdb_path, "C")

        binder_seq = "".join(residue["aa"] for residue in binder_residues)
        target_b = "".join(residue["aa"] for residue in target_b_residues)
        target_c = "".join(residue["aa"] for residue in target_c_residues)

        target_b_set.add(target_b)
        target_c_set.add(target_c)

        hotspot_concat_indices, hotspot_labels = resolve_target_hotspots(
            {"B": target_b_residues, "C": target_c_residues},
            target_hotspot_residues,
        )
        if reference_hotspot_indices is None:
            reference_hotspot_indices = hotspot_concat_indices
        elif hotspot_concat_indices != reference_hotspot_indices:
            raise ValueError(
                f"{pdb_path} 的 hotspot 映射与其他 source PDB 不一致，无法作为固定 target 批量优化"
            )

        entries.append(
            {
                "entry_id": pdb_path.stem,
                "source_pdb": str(pdb_path),
                "binder_chain": "A",
                "binder_length": len(binder_seq),
                "init_binder_seq": binder_seq,
                "target_chain_b_seq": target_b,
                "target_chain_c_seq": target_c,
                "target_hotspot_residues": target_hotspot_residues,
                "target_hotspot_concat_indices": hotspot_concat_indices,
                "target_hotspot_labels": hotspot_labels,
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
            "target_hotspot_residues",
            "target_hotspot_concat_indices",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for entry in entries:
            row = dict(entry)
            row["target_hotspot_concat_indices"] = json.dumps(
                entry.get("target_hotspot_concat_indices", []), ensure_ascii=False
            )
            writer.writerow(row)

    target_fasta = input_dir / "target_chains.fasta"
    with target_fasta.open("w", encoding="utf-8") as handle:
        if target_sequences:
            handle.write(">A6_TCR_chain_B\n")
            handle.write(target_sequences[0] + "\n")
            handle.write(">A6_TCR_chain_C\n")
            handle.write(target_sequences[1] + "\n")
        else:
            for entry in entries:
                entry_targets = entry.get("target_sequences", [])
                for idx, sequence in enumerate(entry_targets, start=1):
                    handle.write(f">{entry['entry_id']}_target_chain_{idx}\n")
                    handle.write(sequence + "\n")

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


def resolve_loss_weights(args: argparse.Namespace) -> Optional[Dict[str, float]]:
    preset_name = getattr(args, "loss_preset", "tuned")
    if preset_name not in LOSS_WEIGHT_PRESETS:
        raise ValueError(
            f"未知 loss_preset={preset_name!r}; 可选: {', '.join(sorted(LOSS_WEIGHT_PRESETS))}"
        )
    loss_weights = {}
    if preset_name != "default":
        loss_weights.update(LOSS_WEIGHT_PRESETS[preset_name])

    weight_arg_map = {
        "plddt": getattr(args, "w_plddt", None),
        "pae": getattr(args, "w_pae", None),
        "i_pae": getattr(args, "w_i_pae", None),
        "con": getattr(args, "w_con", None),
        "i_con": getattr(args, "w_i_con", None),
        "iptm": getattr(args, "w_iptm", None),
        "rg": getattr(args, "w_rg", None),
        "helix": getattr(args, "w_helix", None),
        "clash": getattr(args, "w_clash", None),
        "i_clash": getattr(args, "w_i_clash", None),
        "pde": getattr(args, "w_pde", None),
        "i_pde": getattr(args, "w_i_pde", None),
    }
    for key, value in weight_arg_map.items():
        if value is not None:
            loss_weights[key] = float(value)
    return loss_weights or None


def resolve_entry_template_path(entry_id: str, args: argparse.Namespace) -> Optional[str]:
    template_dir = getattr(args, "target_template_dir", None)
    if template_dir:
        template_root = Path(template_dir)
        candidates = [
            template_root / f"{entry_id}.pdb",
            template_root / f"{entry_id}.cif",
            template_root / f"{entry_id}.mmcif",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(
            f"{entry_id} 未找到 target template；已查找: "
            + ", ".join(str(candidate) for candidate in candidates)
        )

    return getattr(args, "target_template_path", None)


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
    if args.mode in {"hybrid", "mcts"} and args.save_checkpoints:
        optimization_checkpoint_dir = entry_results_dir / args.checkpoint_subdir_name
        optimization_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    hotspot_params = None
    hotspot_indices = entry.get("target_hotspot_concat_indices", [])
    if hotspot_indices:
        hotspot_params = {
            "target_hotspot_concat_indices": hotspot_indices,
        }
        logger.info(
            f"{entry['entry_id']} hotspot={entry.get('target_hotspot_labels', hotspot_indices)} "
            f"concat_idx={hotspot_indices}"
        )

    entry_target_sequences = entry.get("target_sequences") or target_sequences
    target_len_str = "+".join(str(len(seq)) for seq in entry_target_sequences)
    logger.info(
        f"开始 {entry['entry_id']} | binder_len={entry['binder_length']} | "
        f"target={target_len_str}"
    )
    if optimization_checkpoint_dir is not None:
        logger.info(f"{entry['entry_id']} checkpoint 输出目录: {optimization_checkpoint_dir}")

    loss_weights = resolve_loss_weights(args)
    if loss_weights is not None:
        logger.info(f"{entry['entry_id']} loss_weights_override={loss_weights}")
    target_template_path = resolve_entry_template_path(entry["entry_id"], args)
    if target_template_path:
        logger.info(f"{entry['entry_id']} target_template_path={target_template_path}")

    t0 = time.time()
    if args.mode == "first_order":
        best_seq, final_seq, losses_history = run_hallucination(
            model=model,
            target_seq=entry_target_sequences,
            L_binder=entry["binder_length"],
            N_steps=args.n_steps,
            learning_rate=args.learning_rate,
            tau_noise=args.tau_noise,
            device=device,
            loss_weights=loss_weights,
            init_binder_seq=entry["init_binder_seq"],
            hotspot_params=hotspot_params,
            seed=args.random_seed,
            target_template_path=target_template_path,
        )
    elif args.mode == "hybrid":
        best_seq, final_seq, losses_history = run_hybrid_hallucination(
            model=model,
            target_seq=entry_target_sequences,
            L_binder=entry["binder_length"],
            N_steps=args.n_steps,
            learning_rate=args.learning_rate,
            tau_noise=args.tau_noise,
            device=device,
            loss_weights=loss_weights,
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
            hotspot_params=hotspot_params,
            seed=args.random_seed,
            target_template_path=target_template_path,
        )
    else:
        best_seq, final_seq, losses_history = run_mcts_hallucination(
            model=model,
            target_seq=entry_target_sequences,
            L_binder=entry["binder_length"],
            N_steps=args.n_steps,
            learning_rate=args.learning_rate,
            tau_noise=args.tau_noise,
            device=device,
            loss_weights=loss_weights,
            K_top=args.k_top,
            B_candidates=args.b_candidates,
            T_sample=args.t_sample,
            max_mutations_per_seq=args.max_mutations_per_seq,
            use_annealing=args.use_annealing,
            T_init=args.t_init,
            T_min=args.t_min,
            num_beams=args.mcts_num_beams,
            use_prior=args.use_prior,
            prior_strength=args.prior_strength,
            checkpoint_dir=str(optimization_checkpoint_dir) if optimization_checkpoint_dir is not None else None,
            init_binder_seq=entry["init_binder_seq"],
            hotspot_params=hotspot_params,
            chunk_size=args.chunk_size,
            mcts_C=args.mcts_c,
            mcts_max_nodes=args.mcts_max_nodes,
            mcts_tau_fixed=args.mcts_tau_fixed,
            mcts_strategy=args.mcts_strategy,
            mcts_bootstrap_steps=args.mcts_bootstrap_steps,
            mcts_bootstrap_patience=args.mcts_bootstrap_patience,
            mcts_bootstrap_min_delta=args.mcts_bootstrap_min_delta,
            mcts_bootstrap_max_steps=args.mcts_bootstrap_max_steps,
            mcts_handoff_top_k=args.mcts_handoff_top_k,
            mcts_expand_actions_per_step=args.mcts_expand_actions_per_step,
            seed=args.random_seed,
            target_template_path=target_template_path,
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
        "target_chain_b_seq": entry_target_sequences[0] if entry_target_sequences else "",
        "target_chain_c_seq": entry_target_sequences[1] if len(entry_target_sequences) > 1 else "",
        "target_sequences": entry_target_sequences,
        "target_hotspot_residues": entry.get("target_hotspot_residues"),
        "target_hotspot_labels": entry.get("target_hotspot_labels", []),
        "target_hotspot_concat_indices": hotspot_indices,
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
            "mcts_c": args.mcts_c,
            "mcts_max_nodes": args.mcts_max_nodes,
            "mcts_tau_fixed": args.mcts_tau_fixed,
            "mcts_num_beams": args.mcts_num_beams,
            "mcts_strategy": args.mcts_strategy,
            "mcts_bootstrap_steps": args.mcts_bootstrap_steps,
            "mcts_bootstrap_patience": args.mcts_bootstrap_patience,
            "mcts_bootstrap_min_delta": args.mcts_bootstrap_min_delta,
            "mcts_bootstrap_max_steps": args.mcts_bootstrap_max_steps,
            "mcts_handoff_top_k": args.mcts_handoff_top_k,
            "mcts_expand_actions_per_step": args.mcts_expand_actions_per_step,
            "chunk_size": args.chunk_size,
            "use_prior": args.use_prior,
            "prior_strength": args.prior_strength,
            "target_hotspot_residues": args.target_hotspot_residues,
            "loss_preset": args.loss_preset,
            "target_template_path": target_template_path,
            "target_template_dir": args.target_template_dir,
            "loss_weights_override": loss_weights,
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
    parser.add_argument("--target-fasta-dir", dest="target_fasta_dir", type=str, default=argparse.SUPPRESS)
    parser.add_argument(
        "--target-fasta-residue-start",
        dest="target_fasta_residue_start",
        type=int,
        default=argparse.SUPPRESS,
        help="FASTA target 在结构/原始编号中的起始残基号；用于解析如 B64 的 hotspot",
    )
    parser.add_argument("--random-binder-length", dest="random_binder_length", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--random-binder-lengths", dest="random_binder_lengths", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--init-binder-fasta-dir", dest="init_binder_fasta_dir", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--random-seed", dest="random_seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--input-dir", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--results-dir", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--log-dir", type=str, default=argparse.SUPPRESS)
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
        choices=["first_order", "hybrid", "mcts"],
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
    parser.add_argument(
        "--loss-preset",
        dest="loss_preset",
        type=str,
        choices=sorted(LOSS_WEIGHT_PRESETS.keys()),
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--mcts-c", dest="mcts_c", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--mcts-max-nodes", dest="mcts_max_nodes", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--mcts-tau-fixed", dest="mcts_tau_fixed", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--mcts-num-beams", dest="mcts_num_beams", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--mcts-strategy",
        dest="mcts_strategy",
        choices=["legacy", "strict", "two_stage"],
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--mcts-bootstrap-steps", dest="mcts_bootstrap_steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--mcts-bootstrap-patience", dest="mcts_bootstrap_patience", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--mcts-bootstrap-min-delta", dest="mcts_bootstrap_min_delta", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--mcts-bootstrap-max-steps", dest="mcts_bootstrap_max_steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--mcts-handoff-top-k", dest="mcts_handoff_top_k", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--mcts-expand-actions-per-step", dest="mcts_expand_actions_per_step", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--chunk-size", dest="chunk_size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--prior-strength", dest="prior_strength", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--w-plddt", dest="w_plddt", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--w-pae", dest="w_pae", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--w-i-pae", dest="w_i_pae", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--w-con", dest="w_con", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--w-i-con", dest="w_i_con", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--w-iptm", dest="w_iptm", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--w-rg", dest="w_rg", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--w-helix", dest="w_helix", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--w-clash", dest="w_clash", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--w-i-clash", dest="w_i_clash", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--w-pde", dest="w_pde", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--w-i-pde", dest="w_i_pde", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--target-hotspot-residues",
        dest="target_hotspot_residues",
        type=str,
        default=argparse.SUPPRESS,
        help="指定 target hotspot，必须带链名前缀，如 B56,B57,C12-15",
    )
    parser.add_argument(
        "--target-template-path",
        dest="target_template_path",
        type=str,
        default=argparse.SUPPRESS,
        help="可选 target template 结构文件（.cif/.mmcif/.pdb），只作用于 target 链",
    )
    parser.add_argument(
        "--target-template-dir",
        dest="target_template_dir",
        type=str,
        default=argparse.SUPPRESS,
        help="可选 per-entry target template 目录，按 <entry_id>.pdb/.cif/.mmcif 查找",
    )
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
    add_bool_argument(parser, "use-prior", "MCTS 是否启用 BLOSUM62 先验")
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

    log_dir = Path(args.log_dir) if args.log_dir else (Path(PROJECT_ROOT) / "benchmark" / "log")
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

    selected_entry_ids = parse_entry_ids(args.entry_ids)
    if args.target_fasta_dir:
        entries, target_sequences = collect_fasta_random_entries(
            Path(args.target_fasta_dir),
            random_binder_length=args.random_binder_length,
            random_seed=args.random_seed,
            random_binder_lengths=args.random_binder_lengths,
            init_binder_fasta_dir=Path(args.init_binder_fasta_dir) if args.init_binder_fasta_dir else None,
            target_hotspot_residues=args.target_hotspot_residues,
            target_fasta_residue_start=args.target_fasta_residue_start,
            selected_entry_ids=selected_entry_ids,
        )
        logger.info(
            f"扫描到 {len(entries)} 个 FASTA target 条目 | "
            f"random_binder_length={args.random_binder_length} | "
            f"per_case_lengths={bool(args.random_binder_lengths)} | "
            f"init_binder_fasta_dir={args.init_binder_fasta_dir} | "
            f"random_seed={args.random_seed}"
        )
    else:
        source_dir = Path(args.source_dir)
        entries, target_sequences = collect_entries(
            source_dir,
            target_hotspot_residues=args.target_hotspot_residues,
        )
        logger.info(f"扫描到 {len(entries)} 个 source PDB 条目")
        logger.info(
            f"固定 target: chain_B={len(target_sequences[0])} aa, chain_C={len(target_sequences[1])} aa"
        )
    if args.target_hotspot_residues:
        logger.info(
            f"已启用 hotspot: spec={args.target_hotspot_residues} | "
            f"concat_idx={entries[0].get('target_hotspot_concat_indices', [])}"
        )

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
