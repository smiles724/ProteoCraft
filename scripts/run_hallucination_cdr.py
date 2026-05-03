#!/usr/bin/env python3
"""
BindCraft v2 - CDR-Aware Hallucination CLI

支持三种使用模式（自动检测，互斥）：

  Mode A — 一键自动（推荐）:
    python scripts/run_hallucination_cdr.py \
        --pdb_id 8PIH \
        --mode hybrid \
        --n_steps 200 \
        --learning_rate 0.05

    自动完成：下载 PDB → ANARCI CDR 识别 → 优化

  Mode B — 手动 JSON:
    python scripts/run_hallucination_cdr.py \
        --cdr_json outputs/8PIH_cdr.json \
        --mode hybrid --n_steps 200

  Mode C — 传统纯序列（与原 run_hallucination.py 等效）:
    python scripts/run_hallucination_cdr.py \
        --target GIVEQCCTSICSLYQLENYCN \
        --binder_length 40 \
        --mode hybrid --n_steps 200

三种模式的优先级：Mode A > Mode B > Mode C
（即同时指定 pdb_id 和 cdr_json 时，优先 Mode A）
"""

import os
import sys
import argparse
import logging
import torch

# 确保 HMMER 工具链（ANARCI 依赖 hmmscan/hmmpress 等）在 PATH 中
_HMMER_BIN = "/mnt/data/conda_envs/prot_und/bin"
if _HMMER_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + _HMMER_BIN

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from bindcraft.hallucination.pipeline import run_hallucination
from bindcraft.hallucination.pipeline_cdr import (
    run_hybrid_hallucination_cdr,
    run_mcts_hallucination_cdr,
)
from bindcraft.utils import LOSS_WEIGHT_PRESETS


def setup_logging(log_file: str = None):
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        fh.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO, handlers=[fh], force=True)
    else:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO, handlers=[sh], force=True)


def _parse_hotspot_positions_1based(raw_hotspots, target_len: int, parser: argparse.ArgumentParser, source: str):
    if raw_hotspots is None:
        return [], []

    if isinstance(raw_hotspots, (list, tuple)):
        tokens = list(raw_hotspots)
    else:
        raw_text = str(raw_hotspots).strip()
        if not raw_text:
            return [], []
        tokens = raw_text.replace(";", ",").replace(" ", ",").split(",")

    positions = []
    for token in tokens:
        if isinstance(token, int):
            positions.append(token)
            continue
        token = str(token).strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            if len(parts) != 2 or not parts[0].strip().isdigit() or not parts[1].strip().isdigit():
                parser.error(f"{source} 中无法解析 hotspot 范围: {token}")
            start = int(parts[0])
            end = int(parts[1])
            if start > end:
                parser.error(f"{source} hotspot 范围起点大于终点: {token}")
            positions.extend(range(start, end + 1))
        else:
            if not token.isdigit():
                parser.error(f"{source} 只能包含 1-based 正整数或范围；检测到: {token}")
            positions.append(int(token))

    seen = set()
    unique_positions = []
    for pos in positions:
        if pos < 1 or pos > target_len:
            parser.error(f"{source} 位置 {pos} 越界；当前 target 长度为 {target_len}。")
        if pos not in seen:
            seen.add(pos)
            unique_positions.append(pos)

    return unique_positions, [pos - 1 for pos in unique_positions]


def _extract_json_hotspots_1based(cdr_data):
    for key in (
        "target_hotspots",
        "target_hotspot_residues",
        "target_hotspot_residue_indices",
        "target_hotspot_residues_1based",
        "hotspots",
        "hotspot_residues",
    ):
        value = cdr_data.get(key)
        if value is not None:
            return value, key
    return None, None


def _build_hotspot_params(raw_hotspots, target_seq: str, parser: argparse.ArgumentParser, source: str):
    residues_1based, concat_indices = _parse_hotspot_positions_1based(
        raw_hotspots,
        len(target_seq or ""),
        parser,
        source,
    )
    if not concat_indices:
        return None
    return {
        "target_hotspot_residues_1based": residues_1based,
        "target_hotspot_concat_indices": concat_indices,
    }


def load_protenix_model(
    model_name: str = "protenix_base_default_v1.0.0",
    device: str = "cuda"
):
    import os as _os
    from collections.abc import Mapping
    from protenix.config.config import parse_configs
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from protenix.model.protenix import Protenix

    logging.info(f"加载模型: {model_name}")

    _script_dir = _os.path.dirname(_os.path.abspath(__file__))
    _project_dir = _os.path.dirname(_script_dir)
    _ckpt_dir = _os.path.join(_project_dir, "checkpoint")
    _os.makedirs(_ckpt_dir, exist_ok=True)
    checkpoint_path = _os.path.join(_ckpt_dir, f"{model_name}.pt")
    logging.info(f"Checkpoint: {checkpoint_path}")

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d

    base_configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    model_specific = model_configs[model_name]
    deep_update(base_configs, model_specific)

    arg_str = (
        f"--model_name {model_name} "
        f"--use_msa false "
        f"--dtype fp32 "
        f"--load_checkpoint_dir {_ckpt_dir} "
        f"--dump_dir /tmp/bindcraft_hallucination "
    )
    configs = parse_configs(
        configs=base_configs, arg_str=arg_str, fill_required_with_null=True,
    )

    device_obj = torch.device(device)
    model = Protenix(configs).to(device_obj)

    checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    state_dict = checkpoint["model"]
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    logging.info(f"模型加载完成")
    return model, configs


def main():
    parser = argparse.ArgumentParser(
        description="BindCraft v2 — CDR-Aware Hallucination Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── 基础参数 ──────────────────────────────────────────────────────────────
    parser.add_argument("--mode", type=str, default="hybrid",
                       choices=["first_order", "hybrid", "mcts"],
                       help="优化模式: first_order / hybrid / mcts（Mode A/B/C 均支持 hybrid 与 mcts）")
    parser.add_argument("--device", type=str, default="cuda",
                       help="计算设备（默认：cuda）")
    parser.add_argument("--model_name", type=str, default="protenix_base_default_v1.0.0",
                       help="模型名称")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="输出目录（默认：./outputs）")
    parser.add_argument("--log_file", type=str, default=None,
                       help="日志文件路径")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                       help="每个 step 保存 checkpoint 的根目录。不指定时默认自动推导；设为 '' 禁用。")

    # ── CDR Mode A：一键 PDB ─────────────────────────────────────────────────
    g = parser.add_argument_group("Mode A: 一键 PDB（自动识别 CDR）")
    g.add_argument("--pdb_id", default=None,
                   help="PDB ID（e.g. 8PIH）或 PDB 文件路径。"
                        "提供此参数后自动完成 CDR 识别，无需手动运行 identify_cdr.py。")
    g.add_argument("--binder_chain", default=None,
                   help="Binder chain ID（默认：第一条 heavy chain）")
    g.add_argument("--scheme", default="chothia",
                   choices=["chothia", "kabat", "imgt"],
                   help="ANARCI numbering scheme (default: chothia)")

    # ── CDR Mode B：手动 JSON ─────────────────────────────────────────────────
    g2 = parser.add_argument_group("Mode B: 手动 CDR JSON")
    g2.add_argument("--cdr_json", default=None,
                    help="预生成的 CDR JSON 路径（由 scripts/identify_cdr.py 生成）")

    # ── Fallback Mode C：纯序列 ────────────────────────────────────────────────
    g3 = parser.add_argument_group("Mode C: 纯序列（与原 run_hallucination.py 等效）")
    g3.add_argument("--target", type=str, default=None,
                    help="抗原序列（Mode C 必需，Mode A/B 可省略）")
    g3.add_argument("--binder_length", type=int, default=40,
                    help="Binder 长度（Mode C 使用）")
    g3.add_argument("--binder_sequence", default=None,
                    help="Binder 全长序列（Mode B 可选，用于长度推断）")

    # ── 优化参数 ──────────────────────────────────────────────────────────────
    g4 = parser.add_argument_group("优化参数（所有模式）")
    g4.add_argument("--n_steps", type=int, default=200,
                    help="优化步数（默认：200）")
    g4.add_argument("--learning_rate", type=float, default=0.05,
                    help="Adam 学习率（默认：0.05）")
    g4.add_argument("--tau_noise", type=float, default=4.0,
                    help="扩散噪声时间步（默认：4.0）")
    g4.add_argument("--k_top", type=int, default=10,
                    help="Top-K 突变候选数（默认：10）")
    g4.add_argument("--b_candidates", type=int, default=8,
                    help="零阶验证候选序列数（默认：8）")
    g4.add_argument("--t_sample", type=float, default=1.0,
                    help="突变采样温度（默认：1.0）")
    g4.add_argument("--max_mutations_per_seq", type=int, default=3,
                    help="每条序列的最大突变数（默认：3）")
    g4.add_argument("--use_annealing", type=lambda x: x.lower() == "true", default=True,
                    help="是否启用模拟退火（默认：true）")
    g4.add_argument("--t_init", type=float, default=1.0,
                    help="退火初始温度（默认：1.0）")
    g4.add_argument("--t_min", type=float, default=0.01,
                    help="退火最低温度（默认：0.01）")
    g4.add_argument("--use_tau_scheduling", type=lambda x: x.lower() == "true", default=False,
                    help="是否启用时间步退火调度（默认：false，改用固定 tau=4.0）")
    g4.add_argument("--tau_jitter", type=lambda x: x.lower() == "true", default=True,
                    help="是否启用时间步抖动（默认：true）")
    g4.add_argument("--use_prior", type=lambda x: x.lower() == "true", default=True,
                    help="启用 BLOSUM62 先验约束（默认：true）")
    g4.add_argument("--prior_strength", type=float, default=0.1,
                    help="先验权重系数（默认：0.1）")
    g4.add_argument("--chunk_size", type=int, default=None,
                    help="Attention 分块大小。当 binder>100 aa 时建议设为 256 避免 OOM（默认：None）")
    g4.add_argument("--target_hotspots", type=str, default=None,
                    help="Target hotspot residues，使用 1-based 序号；Mode B/C 支持逗号/空格/范围，如 10,11,45-47")
    g4.add_argument("--mcts_c", type=float, default=1.0, help="MCTS UCB 探索常数 C（默认：1.0）")
    g4.add_argument("--mcts_max_nodes", type=int, default=5000, help="MCTS 树最大节点数（默认：5000）")
    g4.add_argument("--mcts_tau_fixed", type=float, default=4.0, help="MCTS 固定扩散 tau（默认：4.0）")
    g4.add_argument("--mcts_num_beams", type=int, default=8, help="MCTS Phase0 根节点数（默认：8）")
    g4.add_argument("--mcts_strategy", type=str, default="two_stage",
                    choices=["legacy", "strict", "two_stage"],
                    help="CDR MCTS 策略：legacy 原版 / strict 文档版 / two_stage 两阶段（默认：two_stage）")
    g4.add_argument("--mcts_bootstrap_steps", type=int, default=50,
                    help="two_stage 中 StageA legacy 至少运行步数（默认：50）")
    g4.add_argument("--mcts_bootstrap_patience", type=int, default=10,
                    help="two_stage 中 StageA 停滞 patience（默认：10）")
    g4.add_argument("--mcts_bootstrap_min_delta", type=float, default=0.02,
                    help="two_stage 中判定 StageA 有效改善的最小 loss 降幅（默认：0.02）")
    g4.add_argument("--mcts_bootstrap_max_steps", type=int, default=80,
                    help="two_stage 中 StageA 最多运行步数（默认：80）")
    g4.add_argument("--mcts_handoff_top_k", type=int, default=None,
                    help="two_stage 交接给 StageB 的 root 数；默认等于 --mcts_num_beams")

    # ── 损失函数权重 ──────────────────────────────────────────────────────────
    g5 = parser.add_argument_group("损失函数权重")
    g5.add_argument("--loss_preset", type=str, default="tuned",
                    choices=sorted(LOSS_WEIGHT_PRESETS.keys()),
                    help="Loss 权重预设：default 原默认 / tuned 调优版（默认：tuned）")
    g5.add_argument("--w_plddt", type=float, default=None, help="pLDDT 权重")
    g5.add_argument("--w_pae", type=float, default=None, help="pAE (intra) 权重")
    g5.add_argument("--w_i_pae", type=float, default=None, help="i_pAE (inter) 权重")
    g5.add_argument("--w_con", type=float, default=None, help="con (intra contact) 权重")
    g5.add_argument("--w_i_con", type=float, default=None, help="i_con (inter contact) 权重")
    g5.add_argument("--w_iptm", type=float, default=None, help="iPTM 权重")
    g5.add_argument("--w_rg", type=float, default=None, help="Rg 权重")
    g5.add_argument("--w_helix", type=float, default=None, help="Helix 权重")
    g5.add_argument("--w_clash", type=float, default=None, help="clash (intra) 权重")
    g5.add_argument("--w_i_clash", type=float, default=None, help="i_clash (inter) 权重")
    g5.add_argument("--w_pde", type=float, default=None, help="PDE (intra) 权重")
    g5.add_argument("--w_i_pde", type=float, default=None, help="i_PDE (inter) 权重")

    args = parser.parse_args()

    if args.pdb_id is not None and args.mode == "first_order":
        parser.error("Mode A (--pdb_id) does not support --mode first_order; use hybrid or mcts")
    if args.pdb_id is not None and args.target_hotspots:
        parser.error(
            "--target_hotspots currently expects target sequence 1-based indices and is supported "
            "for Mode B/Mode C. Mode A PDB residue-number mapping is not implemented yet."
        )

    setup_logging(args.log_file)
    os.makedirs(args.output_dir, exist_ok=True)

    model, configs = load_protenix_model(args.model_name, args.device)

    # 组装 loss_weights: default 不覆盖；非 default preset 覆盖整套权重。
    # 单独的 --w_* 参数始终具有最高优先级。
    loss_weights = {}
    if args.loss_preset != "default":
        loss_weights.update(LOSS_WEIGHT_PRESETS[args.loss_preset])
    weight_arg_map = {
        "plddt": args.w_plddt, "pae": args.w_pae, "i_pae": args.w_i_pae,
        "con": args.w_con, "i_con": args.w_i_con, "iptm": args.w_iptm,
        "rg": args.w_rg, "helix": args.w_helix,
        "clash": args.w_clash, "i_clash": args.w_i_clash,
        "pde": args.w_pde, "i_pde": args.w_i_pde,
    }
    for key, val in weight_arg_map.items():
        if val is not None:
            loss_weights[key] = val

    hotspot_params = None

    # checkpoint_dir：None → 自动推导；"" → 禁用；显式路径 → 使用显式路径
    _ckpt_override = args.checkpoint_dir  # None / "" / str

    # ── 模式分支 ──────────────────────────────────────────────────────────────
    if args.pdb_id is not None:
        # Mode A：一键自动
        ckpt_dir = _ckpt_override if _ckpt_override is not None \
            else os.path.join(args.output_dir, f"{args.pdb_id}_cdr_checkpoints")
        _ckpt = ckpt_dir if ckpt_dir != "" else None
        logging.info(f"[CDR-Aware] Mode A — pdb_id={args.pdb_id} | checkpoint_dir={ckpt_dir}")
        if args.mode == "mcts":
            best_seq, final_seq, losses_history = run_mcts_hallucination_cdr(
                model=model,
                pdb_id=args.pdb_id,
                binder_chain=args.binder_chain,
                scheme=args.scheme,
                N_steps=args.n_steps,
                learning_rate=args.learning_rate,
                tau_noise=args.tau_noise,
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
                checkpoint_dir=_ckpt,
                hotspot_params=hotspot_params,
                mcts_C=args.mcts_c,
                mcts_max_nodes=args.mcts_max_nodes,
                mcts_tau_fixed=args.mcts_tau_fixed,
                mcts_strategy=args.mcts_strategy,
                mcts_bootstrap_steps=args.mcts_bootstrap_steps,
                mcts_bootstrap_patience=args.mcts_bootstrap_patience,
                mcts_bootstrap_min_delta=args.mcts_bootstrap_min_delta,
                mcts_bootstrap_max_steps=args.mcts_bootstrap_max_steps,
                mcts_handoff_top_k=args.mcts_handoff_top_k,
                chunk_size=args.chunk_size,
            )
        else:
            best_seq, final_seq, losses_history = run_hybrid_hallucination_cdr(
                model=model,
                pdb_id=args.pdb_id,
                binder_chain=args.binder_chain,
                scheme=args.scheme,
                N_steps=args.n_steps,
                learning_rate=args.learning_rate,
                tau_noise=args.tau_noise,
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
                use_prior=args.use_prior,
                prior_strength=args.prior_strength,
                checkpoint_dir=_ckpt,
                hotspot_params=hotspot_params,
                chunk_size=args.chunk_size,
            )
        run_label = f"ModeA_{args.pdb_id}"

    elif args.cdr_json is not None:
        # Mode B：CDR JSON
        import json
        with open(args.cdr_json) as f:
            cdr_data = json.load(f)

        binder_seq = args.binder_sequence or cdr_data.get("binder_sequence")
        cdr_mask = cdr_data.get("binder_cdr_mask")
        target_from_json = cdr_data.get("target_sequence", "")
        target_seq = args.target or target_from_json
        json_hotspots, json_hotspots_key = _extract_json_hotspots_1based(cdr_data)
        hotspot_source = "--target_hotspots" if args.target_hotspots else json_hotspots_key
        hotspot_input = args.target_hotspots if args.target_hotspots else json_hotspots

        json_basename = os.path.basename(args.cdr_json).replace('.json', '')
        ckpt_dir = _ckpt_override if _ckpt_override is not None \
            else os.path.join(args.output_dir, f"{json_basename}_checkpoints")
        _ckpt = ckpt_dir if ckpt_dir != "" else None
        logging.info(f"[CDR-Aware] Mode B — cdr_json={args.cdr_json} | checkpoint_dir={ckpt_dir}")
        if binder_seq is None:
            logging.error("--binder_sequence is required in Mode B when JSON lacks it")
            sys.exit(1)
        if cdr_mask is None:
            logging.error("CDR JSON missing 'binder_cdr_mask' field")
            sys.exit(1)
        hotspot_params = _build_hotspot_params(
            hotspot_input,
            target_seq,
            parser,
            hotspot_source or "CDR JSON hotspot field",
        )
        if hotspot_params is not None:
            logging.info(
                "[CDR-Aware] Hotspot routing enabled: target_hotspots_1based=%s target_concat_indices_0based=%s",
                hotspot_params["target_hotspot_residues_1based"],
                hotspot_params["target_hotspot_concat_indices"],
            )

        if args.mode == "mcts":
            best_seq, final_seq, losses_history = run_mcts_hallucination_cdr(
                model=model,
                binder_sequence=binder_seq,
                cdr_mask=cdr_mask,
                target_seq=target_seq,
                N_steps=args.n_steps,
                learning_rate=args.learning_rate,
                tau_noise=args.tau_noise,
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
                checkpoint_dir=_ckpt,
                hotspot_params=hotspot_params,
                mcts_C=args.mcts_c,
                mcts_max_nodes=args.mcts_max_nodes,
                mcts_tau_fixed=args.mcts_tau_fixed,
                mcts_strategy=args.mcts_strategy,
                mcts_bootstrap_steps=args.mcts_bootstrap_steps,
                mcts_bootstrap_patience=args.mcts_bootstrap_patience,
                mcts_bootstrap_min_delta=args.mcts_bootstrap_min_delta,
                mcts_bootstrap_max_steps=args.mcts_bootstrap_max_steps,
                mcts_handoff_top_k=args.mcts_handoff_top_k,
                chunk_size=args.chunk_size,
            )
        else:
            best_seq, final_seq, losses_history = run_hybrid_hallucination_cdr(
                model=model,
                binder_sequence=binder_seq,
                cdr_mask=cdr_mask,
                target_seq=target_seq,
                N_steps=args.n_steps,
                learning_rate=args.learning_rate,
                tau_noise=args.tau_noise,
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
                use_prior=args.use_prior,
                prior_strength=args.prior_strength,
                checkpoint_dir=_ckpt,
                hotspot_params=hotspot_params,
                chunk_size=args.chunk_size,
            )
        run_label = f"ModeB_{json_basename}"

    else:
        # Mode C：传统纯序列
        if args.target is None:
            parser.error("--pdb_id or --target is required (Mode C needs --target)")
        hotspot_params = _build_hotspot_params(
            args.target_hotspots,
            args.target,
            parser,
            "--target_hotspots",
        )
        if hotspot_params is not None:
            logging.info(
                "[CDR-Aware] Hotspot routing enabled: target_hotspots_1based=%s target_concat_indices_0based=%s",
                hotspot_params["target_hotspot_residues_1based"],
                hotspot_params["target_hotspot_concat_indices"],
            )
        ckpt_dir = _ckpt_override if _ckpt_override is not None \
            else os.path.join(args.output_dir, "ModeC_hybrid_checkpoints")
        _ckpt = ckpt_dir if ckpt_dir != "" else None
        if args.mode == "first_order":
            logging.info(f"[CDR-Aware] Mode C — first_order (纯一阶)")
            best_seq, final_seq, losses_history = run_hallucination(
                model=model,
                target_seq=args.target,
                L_binder=args.binder_length,
                N_steps=args.n_steps,
                learning_rate=args.learning_rate,
                tau_noise=args.tau_noise,
                device=torch.device(args.device),
                loss_weights=loss_weights,
                hotspot_params=hotspot_params,
            )
        elif args.mode == "mcts":
            logging.info(f"[CDR-Aware] Mode C — mcts (无 CDR mask)")
            best_seq, final_seq, losses_history = run_mcts_hallucination_cdr(
                model=model,
                target_seq=args.target,
                L_binder=args.binder_length,
                N_steps=args.n_steps,
                learning_rate=args.learning_rate,
                tau_noise=args.tau_noise,
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
                checkpoint_dir=_ckpt,
                hotspot_params=hotspot_params,
                mcts_C=args.mcts_c,
                mcts_max_nodes=args.mcts_max_nodes,
                mcts_tau_fixed=args.mcts_tau_fixed,
                mcts_strategy=args.mcts_strategy,
                mcts_bootstrap_steps=args.mcts_bootstrap_steps,
                mcts_bootstrap_patience=args.mcts_bootstrap_patience,
                mcts_bootstrap_min_delta=args.mcts_bootstrap_min_delta,
                mcts_bootstrap_max_steps=args.mcts_bootstrap_max_steps,
                mcts_handoff_top_k=args.mcts_handoff_top_k,
                chunk_size=args.chunk_size,
            )
        else:
            logging.info(f"[CDR-Aware] Mode C — hybrid (无 CDR mask)")
            best_seq, final_seq, losses_history = run_hybrid_hallucination_cdr(
                model=model,
                target_seq=args.target,
                L_binder=args.binder_length,
                N_steps=args.n_steps,
                learning_rate=args.learning_rate,
                tau_noise=args.tau_noise,
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
                use_prior=args.use_prior,
                prior_strength=args.prior_strength,
                checkpoint_dir=_ckpt,
                hotspot_params=hotspot_params,
                chunk_size=args.chunk_size,
            )
        run_label = f"ModeC_{args.mode}"

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"result_{run_label}_{timestamp}.txt")
    with open(output_file, "w") as f:
        f.write(f"Run Label: {run_label}\n")
        f.write(f"Mode: {args.mode}\n")
        if args.pdb_id:
            f.write(f"PDB ID: {args.pdb_id}\n")
        if args.cdr_json:
            f.write(f"CDR JSON: {args.cdr_json}\n")
        if args.target:
            f.write(f"Target: {args.target}\n")
        if hotspot_params is not None:
            f.write(f"Target Hotspots 1-based: {hotspot_params['target_hotspot_residues_1based']}\n")
            f.write(f"Target Hotspot Indices 0-based: {hotspot_params['target_hotspot_concat_indices']}\n")
        f.write(f"Binder Length: {args.binder_length}\n")
        f.write(f"Optimization Steps: {args.n_steps}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n")
        f.write(f"Tau Noise: {args.tau_noise}\n")
        if args.mode == "mcts":
            f.write(f"MCTS Strategy: cdr_{args.mcts_strategy}\n")
            f.write(f"MCTS Bootstrap Steps: {args.mcts_bootstrap_steps}\n")
            f.write(f"MCTS Bootstrap Max Steps: {args.mcts_bootstrap_max_steps}\n")
        f.write(f"\n")
        f.write(f"Best Sequence: {best_seq}\n")
        f.write(f"Final Sequence: {final_seq}\n")
        f.write(f"\n")
        f.write(f"Loss History:\n")
        for i, loss in enumerate(losses_history):
            f.write(f"Step {i}: {loss:.6f}\n")

    logging.info(f"结果已保存到: {output_file}")

    print("\n" + "=" * 70)
    print(" CDR-Aware Hallucination 完成！")
    print(f" 最佳序列: {best_seq}")
    print(f" 结果文件: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
