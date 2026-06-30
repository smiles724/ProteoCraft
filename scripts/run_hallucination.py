#!/usr/bin/env python3
"""
BindCraft v2 - 入口脚本

使用方法（纯一阶模式）:
    python scripts/run_hallucination.py \
        --target GIVEQCCTSICSLYQLENYCN \
        --binder_length 40 \
        --n_steps 200 \
        --learning_rate 0.05 \
        --tau_noise 4.0 \
        --w_i_con 3.0

使用方法（混合 1st+0th 模式）:
    python scripts/run_hallucination.py \
        --target GIVEQCCTSICSLYQLENYCN \
        --binder_length 40 \
        --mode hybrid \
        --n_steps 100 \
        --learning_rate 0.05 \
        --tau_noise 4.0 \
        --k_top 10 \
        --b_candidates 8 \
        --t_sample 1.0 \
        --max_mutations_per_seq 3 \
        --use_annealing true \
        --t_init 1.0 \
        --t_min 0.01 \
        --use_tau_scheduling true \
        --tau_jitter true \
        --w_i_pae 0.1 \
        --w_i_con 3.0
"""

import os
import sys
import argparse
import logging
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from bindcraft.hallucination.reproducibility import set_global_seed
from bindcraft.utils import LOSS_WEIGHT_PRESETS


def setup_logging(log_file: str = None, level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handlers = []
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        fh.setLevel(lvl)
        handlers.append(fh)
    else:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh.setLevel(lvl)
        handlers.append(sh)
    # force=True：避免被其它模块抢先 basicConfig 后级别不生效
    logging.basicConfig(level=lvl, handlers=handlers, force=True)


def resolve_checkpoint_dir(checkpoint_dir: str = None) -> str:
    """
    解析 checkpoint 目录优先级:
    1. CLI 显式传入
    2. 环境变量 PROTENIX_CHECKPOINT_DIR
    3. 项目根目录下的 checkpoint/（若存在）
    4. 环境变量 PROTENIX_ROOT_DIR/checkpoint
    """
    if checkpoint_dir:
        return checkpoint_dir

    env_checkpoint_dir = os.environ.get("PROTENIX_CHECKPOINT_DIR")
    if env_checkpoint_dir:
        return env_checkpoint_dir

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_dir = os.path.dirname(_script_dir)
    _local_ckpt = os.path.join(_project_dir, "checkpoint")
    if os.path.isdir(_local_ckpt):
        return _local_ckpt

    protenix_root = os.environ.get("PROTENIX_ROOT_DIR", os.path.expanduser("~"))
    return os.path.join(protenix_root, "checkpoint")


def _parse_target_hotspots_1based(raw_hotspots, target_len: int, parser: argparse.ArgumentParser):
    if raw_hotspots is None:
        return [], []
    raw_text = str(raw_hotspots).strip()
    if not raw_text:
        return [], []

    positions = []
    for token in raw_text.replace(";", ",").replace(" ", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            if len(parts) != 2 or not parts[0].strip().isdigit() or not parts[1].strip().isdigit():
                parser.error(f"--target_hotspots 中无法解析范围: {token}")
            start = int(parts[0])
            end = int(parts[1])
            if start > end:
                parser.error(f"--target_hotspots 范围起点大于终点: {token}")
            positions.extend(range(start, end + 1))
        else:
            if not token.isdigit():
                parser.error(f"--target_hotspots 只能包含 1-based 正整数或范围；检测到: {token}")
            positions.append(int(token))

    seen = set()
    unique_positions = []
    for pos in positions:
        if pos < 1 or pos > target_len:
            parser.error(
                f"--target_hotspots 位置 {pos} 越界；当前 target 长度为 {target_len}。"
            )
        if pos not in seen:
            seen.add(pos)
            unique_positions.append(pos)

    return unique_positions, [pos - 1 for pos in unique_positions]


def _build_hotspot_params(args, parser: argparse.ArgumentParser):
    residues_1based, concat_indices = _parse_target_hotspots_1based(
        args.target_hotspots,
        len(args.target),
        parser,
    )
    if not concat_indices:
        return None
    return {
        "target_hotspot_residues_1based": residues_1based,
        "target_hotspot_concat_indices": concat_indices,
    }


def parse_esmfold2_mutable_mask(raw_mask, binder_length: int, parser: argparse.ArgumentParser):
    if raw_mask is None:
        return None
    raw_text = str(raw_mask).strip()
    if not raw_text:
        return None

    true_tokens = {"1", "true", "t", "yes", "y"}
    false_tokens = {"0", "false", "f", "no", "n"}
    if all(ch in "01" for ch in raw_text):
        tokens = list(raw_text)
    else:
        tokens = [
            token.strip().lower()
            for token in raw_text.replace(";", ",").replace(" ", ",").split(",")
            if token.strip()
        ]

    mask = []
    for token in tokens:
        if token in true_tokens:
            mask.append(True)
        elif token in false_tokens:
            mask.append(False)
        else:
            parser.error(
                "--esmfold2_mutable_mask 只能包含 0/1 或 true/false；"
                f"检测到: {token}"
            )
    if len(mask) != int(binder_length):
        parser.error(
            f"--esmfold2_mutable_mask 长度为 {len(mask)}，"
            f"但 --binder_length 为 {binder_length}。"
        )
    return mask



def _cuequivariance_backend_available() -> bool:
    try:
        if not hasattr(torch.library, "custom_op"):
            logging.warning(
                "cuequivariance backend unavailable: torch.library.custom_op is missing; "
                "falling back to torch triangle kernels."
            )
            return False
        import cuequivariance_ops_torch  # noqa: F401
        import cuequivariance_torch  # noqa: F401
        return True
    except Exception as exc:
        logging.warning(
            "cuequivariance backend unavailable (%s); falling back to torch triangle kernels.",
            exc,
        )
        return False


def _apply_triangle_backend_fallback(configs: dict) -> None:
    needs_cue = (
        configs.get("triangle_attention") == "cuequivariance"
        or configs.get("triangle_multiplicative") == "cuequivariance"
    )
    if not needs_cue or _cuequivariance_backend_available():
        return
    if configs.get("triangle_attention") == "cuequivariance":
        configs["triangle_attention"] = "torch"
    if configs.get("triangle_multiplicative") == "cuequivariance":
        configs["triangle_multiplicative"] = "torch"

def load_protenix_model(
    model_name: str = "protenix_base_default_v1.0.0",
    device: str = "cuda",
    checkpoint_dir: str = None,
    model_dtype: str = "bf16",
):
    import copy
    from collections.abc import Mapping
    from protenix.config.config import parse_configs
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from protenix.model.protenix import Protenix

    logging.info(f"加载模型: {model_name} | dtype={model_dtype}")
    if model_name not in model_configs:
        supported = ", ".join(sorted(model_configs))
        raise ValueError(f"未知 Protenix model_name: {model_name}. Supported: {supported}")

    checkpoint_dir = resolve_checkpoint_dir(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}.pt")
    logging.info(f"Checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"未找到模型权重: {checkpoint_path}。"
            f"请检查 --model_name 或通过 --checkpoint_dir / PROTENIX_CHECKPOINT_DIR 指定正确目录。"
        )

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d

    base_configs = copy.deepcopy({**configs_base, **{"data": data_configs}, **inference_configs})
    model_specific = model_configs[model_name]
    deep_update(base_configs, model_specific)
    _apply_triangle_backend_fallback(base_configs)

    arg_str = (
        f"--model_name {model_name} "
        f"--use_msa false "
        f"--dtype {model_dtype} "
        f"--load_checkpoint_dir {checkpoint_dir} "
        f"--dump_dir /tmp/bindcraft_hallucination "
    )
    configs = parse_configs(
        configs=base_configs, arg_str=arg_str, fill_required_with_null=True,
    )
    logging.info(
        "Resolved Protenix config: c_z=%s diffusion_batch_size=%s N_cycle=%s "
        "pairformer_hidden_scale_up=%s confidence_hidden_scale_up=%s sample_N_step=%s",
        getattr(configs, "c_z", None),
        getattr(configs, "diffusion_batch_size", None),
        getattr(getattr(configs, "model", None), "N_cycle", None),
        getattr(getattr(getattr(configs, "model", None), "pairformer", None), "hidden_scale_up", None),
        getattr(getattr(getattr(configs, "model", None), "confidence_head", None), "hidden_scale_up", None),
        getattr(getattr(configs, "sample_diffusion", None), "N_step", None),
    )

    device_obj = torch.device(device)
    model = Protenix(configs).to(device_obj)
    model.bindcraft_model_name = model_name

    checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    state_dict = checkpoint["model"]
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        logging.warning(
            "Missing checkpoint keys (%d): %s",
            len(incompatible.missing_keys),
            incompatible.missing_keys[:20],
        )
    if incompatible.unexpected_keys:
        logging.warning(
            "Unexpected checkpoint keys (%d): %s",
            len(incompatible.unexpected_keys),
            incompatible.unexpected_keys[:20],
        )
    model.eval()

    logging.info(f"模型加载完成")
    return model, configs


def load_esmfold2_backend(args, device: str = "cuda"):
    """Lazy-load the real ESMFold2 backend only when explicitly requested."""
    from bindcraft.hallucination.esmfold2_backend import ESMFold2ProposalConfig
    from bindcraft.hallucination.esmfold2_runtime import ESMFold2RuntimeBackend

    config = ESMFold2ProposalConfig(
        experimental_model_name=args.esmfold2_experimental_model,
        verifier_model_name=args.esmfold2_verifier_model,
        verifier_num_loops=args.esmfold2_verifier_num_loops,
        verifier_num_sampling_steps=args.esmfold2_verifier_num_sampling_steps,
        soft_steps=args.esmfold2_soft_steps,
        lambda_esmc=args.esmfold2_lambda_esmc,
        num_loops=args.esmfold2_proposal_num_loops,
        num_sampling_steps=args.esmfold2_proposal_num_sampling_steps,
    )
    return ESMFold2RuntimeBackend(
        config=config,
        device=device,
        task_type=args.esmfold2_task_type,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BindCraft v2 — Hallucination Pipeline")

    # === 基础参数 ===
    parser.add_argument("--target", type=str, required=True,
                       help="Target 序列（单字母氨基酸序列）")
    parser.add_argument("--mode", type=str, default="first_order",
                       choices=["first_order", "hybrid", "mcts"],
                       help="优化模式: first_order / hybrid (1st+0th) / mcts")

    # === Binder 与优化参数 ===
    parser.add_argument("--binder_length", type=int, default=40,
                       help="Binder 长度（默认：40）")
    parser.add_argument("--n_steps", type=int, default=200,
                       help="优化步数（默认：200）")
    parser.add_argument("--learning_rate", type=float, default=0.05,
                       help="Adam 学习率（默认：0.05）")
    parser.add_argument("--tau_noise", type=float, default=4.0,
                       help="扩散噪声时间步（默认：4.0）")
    parser.add_argument("--device", type=str, default="cuda",
                       help="计算设备（默认：cuda）")
    parser.add_argument("--model_name", type=str, default="protenix_base_default_v1.0.0",
                       help="模型名称")
    parser.add_argument("--model_backend", type=str, default="protenix",
                       choices=["protenix", "esmfold2"],
                       help="模型 backend：protenix 保持原有 V1/V2 流程；esmfold2 启用 ESMFold2 第一版闭环")
    parser.add_argument("--esmfold2_verifier_model", type=str, default="biohub/ESMFold2",
                       help="ESMFold2 verifier 模型名（仅 --model_backend esmfold2 使用）")
    parser.add_argument("--esmfold2_experimental_model", type=str, default="biohub/ESMFold2-Experimental",
                       help="ESMFold2Experimental proposal 模型名（仅 --model_backend esmfold2 使用）")
    parser.add_argument("--esmfold2_soft_steps", type=int, default=5,
                       help="ESMFold2 proposal 每个 MCTS 节点的 soft gradient update 步数（默认：5）")
    parser.add_argument("--esmfold2_lambda_esmc", type=str, default="auto",
                       help="ESMC sequence prior 权重；auto 按任务类型选择，或传入数值字符串")
    parser.add_argument("--esmfold2_task_type", type=str, default="denovo",
                       choices=["denovo", "de_novo", "minibinder", "binder", "tcr", "cdr", "cdr_redesign", "antibody", "antibody_like"],
                       help="ESMFold2/ESMC 任务类型；用于解析 lambda_ESMC=auto（默认：denovo）")
    parser.add_argument("--esmfold2_mutable_mask", type=str, default=None,
                       help="ESMFold2 可变位点 mask；长度需等于 binder_length，支持 01101 或 0,1,1,true")
    parser.add_argument("--esmfold2_artifact_dir", type=str, default=None,
                       help="ESMFold2 每步结构化记录目录；默认写入 output_dir/esmfold2_artifacts")
    parser.add_argument("--esmfold2_verifier_num_loops", type=int, default=20,
                       help="ESMFold2 verifier full fold loop 数（默认：20；smoke test 可显式调小）")
    parser.add_argument("--esmfold2_verifier_num_sampling_steps", type=int, default=200,
                       help="ESMFold2 verifier diffusion sampling steps（默认：200；smoke test 可显式调小）")
    parser.add_argument("--esmfold2_proposal_num_loops", type=int, default=1,
                       help="ESMFold2 proposal distogram forward loop 数（默认：1）")
    parser.add_argument("--esmfold2_proposal_num_sampling_steps", type=int, default=1,
                       help="ESMFold2 proposal distogram forward diffusion steps（默认：1）")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                       help="Checkpoint 目录；默认依次读取 --checkpoint_dir、PROTENIX_CHECKPOINT_DIR、PROTENIX_ROOT_DIR/checkpoint")
    parser.add_argument("--save_checkpoint_dir", type=str, default=None,
                       help="中间优化 step 的 checkpoint 保存目录（仅用于优化过程保存，不用于模型权重加载）")
    parser.add_argument("--checkpoint_save_policy", type=str, default="all",
                       choices=["all", "best_only"],
                       help="checkpoint 保存策略：all 保存每次调用；best_only 只保存刷新 best loss 的帧")
    parser.add_argument("--dtype", type=str, default="bf16",
                       choices=["bf16", "fp32"],
                       help="模型计算精度（默认：bf16）")
    parser.add_argument("--proposal_amp", type=str, default="auto",
                       choices=["auto", "off", "bf16"],
                       help="MCTS proposal forward/backward autocast：auto 仅对 protenix-v2 启用 bf16（默认：auto）")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="输出目录（默认：./outputs）")
    parser.add_argument("--log_file", type=str, default=None,
                       help="日志文件路径")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别（默认：INFO；排错可用 DEBUG）")
    parser.add_argument("--init_binder_seq", type=str, default=None,
                       help="可选的初始 Binder 单字母序列；长度必须与 --binder_length 一致")
    parser.add_argument("--target_hotspots", type=str, default=None,
                       help="Target hotspot residues，使用 1-based 序号；支持逗号/空格/范围，如 10,11,45-47")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子；用于固定 Phase0、mutation sampling 和 verifier diffusion（默认：42）")
    parser.add_argument("--target_template_path", type=str, default=None,
                       help="可选 target template 结构文件（.cif/.mmcif/.pdb），只作用于 target 链")

    # === 混合模式专用参数 ===
    parser.add_argument("--k_top", type=int, default=10,
                       help="Top-K 突变候选数（默认：10）")
    parser.add_argument("--b_candidates", type=int, default=8,
                       help="零阶验证候选序列数（默认：8）")
    parser.add_argument("--t_sample", type=float, default=1.0,
                       help="突变采样温度（默认：1.0）")
    parser.add_argument("--max_mutations_per_seq", type=int, default=3,
                       help="每条序列的最大突变数（默认：3）")
    parser.add_argument("--stagea_max_mutations_per_seq", type=int, default=None,
                       help="ESMFold2/两阶段搜索 StageA 每条候选最大突变数；默认沿用 --max_mutations_per_seq")
    parser.add_argument("--stageb_max_mutations_per_seq", type=int, default=None,
                       help="ESMFold2/两阶段搜索 StageB 每条候选最大突变数；默认沿用 --max_mutations_per_seq")
    parser.add_argument("--use_annealing", type=lambda x: x.lower() == "true", default=True,
                       help="是否启用模拟退火（默认：true）")
    parser.add_argument("--t_init", type=float, default=1.0,
                       help="退火初始温度（默认：1.0）")
    parser.add_argument("--t_min", type=float, default=0.01,
                       help="退火最低温度（默认：0.01）")
    parser.add_argument("--use_tau_scheduling", type=lambda x: x.lower() == "true", default=False,
                       help="是否启用时间步退火调度（默认：false，改用固定 tau=4.0）")
    parser.add_argument("--tau_jitter", type=lambda x: x.lower() == "true", default=True,
                       help="是否启用时间步抖动（默认：true）")

    # === MCTS 模式参数 ===
    parser.add_argument("--mcts_c", type=float, default=1.0, help="MCTS UCB 探索常数 C（默认：1.0）")
    parser.add_argument("--mcts_max_nodes", type=int, default=5000, help="MCTS 树最大节点数（默认：5000）")
    parser.add_argument("--mcts_tau_fixed", type=float, default=4.0, help="MCTS 固定扩散 tau（默认：4.0）")
    parser.add_argument("--mcts_num_beams", type=int, default=8, help="MCTS Phase0 根节点数 / beam（默认：8）")
    parser.add_argument("--mcts_strategy", type=str, default="two_stage",
                       choices=["legacy", "stage_a_only", "strict", "two_stage"],
                       help="MCTS 策略：stage_a_only 只跑 StageA / legacy 兼容旧名 / strict 文档版 / two_stage 两阶段（默认：two_stage）")
    parser.add_argument("--mcts_bootstrap_steps", type=int, default=50,
                       help="two_stage: StageA 原版 MCTS 至少运行步数（默认：50）")
    parser.add_argument("--mcts_bootstrap_patience", type=int, default=10,
                       help="two_stage: StageA 停滞 patience（默认：10）")
    parser.add_argument("--mcts_bootstrap_min_delta", type=float, default=0.02,
                       help="two_stage: 判断 StageA 有效改善的 loss 阈值（默认：0.02）")
    parser.add_argument("--mcts_bootstrap_max_steps", type=int, default=80,
                       help="two_stage: StageA 最大运行步数（默认：80）")
    parser.add_argument("--mcts_handoff_top_k", type=int, default=None,
                       help="two_stage: 交给 StageB 的 top 序列数（默认：mcts_num_beams）")
    parser.add_argument("--mcts_expand_actions_per_step", type=int, default=2,
                       help="MCTS StageB 每步最多扩展并验证的 action 数（默认：2）")
    parser.add_argument("--mcts_prior_mode", type=str, default="full",
                       choices=["full", "uniform", "no_gradient", "no_group", "no_sequence"],
                       help="MCTS StageB prior 消融模式（默认：full，保持当前完整逻辑）")
    parser.add_argument("--mcts_proposal_method", type=str, default="one_step_fixed",
                       choices=["one_step_fixed", "one_step_adaptive", "two_step_truncated"],
                       help="MCTS StageB proposal 消融方法（默认：one_step_fixed，保持当前逻辑）")
    parser.add_argument("--mcts_adaptive_tau_start", type=float, default=160.0,
                       help="MCTS StageB one_step_adaptive 起始 tau（默认：160.0）")
    parser.add_argument("--mcts_adaptive_tau_end", type=float, default=0.1,
                       help="MCTS StageB one_step_adaptive 结束 tau；必须大于 0（默认：0.1）")
    parser.add_argument("--chunk_size", type=int, default=None,
                       help="Attention 分块（MCTS 一阶/零阶）；大复合物可设 256 避免 OOM（默认：None）")
    parser.add_argument("--use_prior", type=lambda x: x.lower() == "true", default=True,
                       help="MCTS：启用 BLOSUM62 先验（默认：true）")
    parser.add_argument("--prior_strength", type=float, default=0.1, help="MCTS：先验强度（默认：0.1）")

    # === 损失函数权重（key 必须与 DEFAULT_LOSS_WEIGHTS 一致） ===
    parser.add_argument("--loss_preset", type=str, default="tuned",
                       choices=sorted(LOSS_WEIGHT_PRESETS.keys()),
                       help="Loss 权重预设：default 原默认 / tuned 调优版（默认：tuned）")
    parser.add_argument("--w_plddt", type=float, default=None, help="pLDDT 权重")
    parser.add_argument("--w_pae", type=float, default=None, help="pAE (intra) 权重")
    parser.add_argument("--w_i_pae", type=float, default=None, help="i_pAE (inter) 权重")
    parser.add_argument("--w_con", type=float, default=None, help="con (intra contact) 权重")
    parser.add_argument("--w_i_con", type=float, default=None, help="i_con (inter contact) 权重")
    parser.add_argument("--w_iptm", type=float, default=None, help="iPTM 权重")
    parser.add_argument("--w_rg", type=float, default=None, help="Rg 权重")
    parser.add_argument("--w_helix", type=float, default=None, help="Helix 权重")
    parser.add_argument("--w_clash", type=float, default=None, help="clash (intra) 权重")
    parser.add_argument("--w_i_clash", type=float, default=None, help="i_clash (inter) 权重")
    parser.add_argument("--w_pde", type=float, default=None, help="PDE (intra) 权重")
    parser.add_argument("--w_i_pde", type=float, default=None, help="i_PDE (inter) 权重")

    return parser


def resolve_model_backend(args, parser: argparse.ArgumentParser = None) -> str:
    backend = str(getattr(args, "model_backend", "protenix") or "protenix").strip().lower()
    if backend not in {"protenix", "esmfold2"}:
        message = f"--model_backend must be one of protenix, esmfold2; got {backend!r}"
        if parser is not None:
            parser.error(message)
        raise ValueError(message)
    if backend == "esmfold2" and getattr(args, "mode", None) != "mcts":
        message = "ESMFold2 backend first version only supports --mode mcts"
        if parser is not None:
            parser.error(message)
        raise ValueError(message)
    if backend == "esmfold2" and int(getattr(args, "esmfold2_soft_steps", 0)) <= 0:
        message = "--esmfold2_soft_steps must be positive"
        if parser is not None:
            parser.error(message)
        raise ValueError(message)
    return backend


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    model_backend = resolve_model_backend(args, parser)

    if args.init_binder_seq is not None:
        args.init_binder_seq = args.init_binder_seq.strip().upper()
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        invalid = sorted(set(args.init_binder_seq) - valid_aas)
        if len(args.init_binder_seq) != args.binder_length:
            parser.error(
                f"--init_binder_seq 长度为 {len(args.init_binder_seq)}，"
                f"但 --binder_length 为 {args.binder_length}。"
            )
        if invalid:
            parser.error(
                "--init_binder_seq 只能包含 20 种标准氨基酸单字母代码；"
                f"检测到非法字符: {','.join(invalid)}"
            )
    esmfold2_mutable_mask = None
    if model_backend == "esmfold2":
        esmfold2_mutable_mask = parse_esmfold2_mutable_mask(
            args.esmfold2_mutable_mask,
            args.binder_length,
            parser,
        )

    hotspot_params = _build_hotspot_params(args, parser)

    setup_logging(args.log_file, level=args.log_level)
    os.makedirs(args.output_dir, exist_ok=True)
    set_global_seed(args.seed)
    logging.info("Random seed fixed to %s", args.seed)
    if hotspot_params is not None:
        if model_backend == "esmfold2":
            logging.info(
                "ESMFold2 backend received target_hotspots but does not use hotspot-aware routing yet: "
                "target_hotspots_1based=%s target_concat_indices_0based=%s",
                hotspot_params["target_hotspot_residues_1based"],
                hotspot_params["target_hotspot_concat_indices"],
            )
        else:
            logging.info(
                "Hotspot-aware routing enabled: target_hotspots_1based=%s target_concat_indices_0based=%s",
                hotspot_params["target_hotspot_residues_1based"],
                hotspot_params["target_hotspot_concat_indices"],
            )
    if model_backend == "esmfold2" and args.target_template_path:
        logging.info(
            "ESMFold2 backend received target_template_path but does not use templates yet: %s",
            args.target_template_path,
        )

    model = None
    configs = None
    esmfold2_backend = None
    if model_backend == "protenix":
        model, configs = load_protenix_model(
            args.model_name,
            args.device,
            checkpoint_dir=args.checkpoint_dir,
            model_dtype=args.dtype,
        )
    else:
        esmfold2_backend = load_esmfold2_backend(args, device=args.device)

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

    if model_backend == "esmfold2":
        from bindcraft.hallucination.esmfold2_pipeline import run_esmfold2_mcts_hallucination

        best_seq, final_seq, losses_history = run_esmfold2_mcts_hallucination(
            backend=esmfold2_backend,
            target_seq=args.target,
            L_binder=args.binder_length,
            N_steps=args.n_steps,
            K_top=args.k_top,
            B_candidates=args.b_candidates,
            T_sample=args.t_sample,
            max_mutations_per_seq=args.max_mutations_per_seq,
            stagea_max_mutations_per_seq=args.stagea_max_mutations_per_seq,
            stageb_max_mutations_per_seq=args.stageb_max_mutations_per_seq,
            num_beams=args.mcts_num_beams,
            init_binder_seq=args.init_binder_seq,
            mcts_C=args.mcts_c,
            mcts_max_nodes=args.mcts_max_nodes,
            mcts_strategy=args.mcts_strategy,
            mcts_bootstrap_steps=args.mcts_bootstrap_steps,
            mcts_bootstrap_patience=args.mcts_bootstrap_patience,
            mcts_bootstrap_min_delta=args.mcts_bootstrap_min_delta,
            mcts_bootstrap_max_steps=args.mcts_bootstrap_max_steps,
            mcts_handoff_top_k=args.mcts_handoff_top_k,
            mcts_expand_actions_per_step=args.mcts_expand_actions_per_step,
            mcts_prior_mode=args.mcts_prior_mode,
            seed=args.seed,
            mutable_mask=esmfold2_mutable_mask,
            artifact_dir=args.esmfold2_artifact_dir
            or os.path.join(args.output_dir, "esmfold2_artifacts"),
        )
    elif args.mode == "first_order":
        from bindcraft.hallucination.pipeline import run_hallucination

        best_seq, final_seq, losses_history = run_hallucination(
            model=model,
            target_seq=args.target,
            L_binder=args.binder_length,
            N_steps=args.n_steps,
            learning_rate=args.learning_rate,
            tau_noise=args.tau_noise,
            device=torch.device(args.device),
            loss_weights=loss_weights,
            init_binder_seq=args.init_binder_seq,
            hotspot_params=hotspot_params,
            seed=args.seed,
            target_template_path=args.target_template_path,
        )
    elif args.mode == "hybrid":
        from bindcraft.hallucination.pipeline import run_hybrid_hallucination

        best_seq, final_seq, losses_history = run_hybrid_hallucination(
            model=model,
            target_seq=args.target,
            L_binder=args.binder_length,
            N_steps=args.n_steps,
            learning_rate=args.learning_rate,
            tau_noise=args.tau_noise,
            device=torch.device(args.device),
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
            init_binder_seq=args.init_binder_seq,
            checkpoint_save_policy=args.checkpoint_save_policy,
            hotspot_params=hotspot_params,
            seed=args.seed,
            target_template_path=args.target_template_path,
        )
    else:
        from bindcraft.hallucination.pipeline import run_mcts_hallucination

        best_seq, final_seq, losses_history = run_mcts_hallucination(
            model=model,
            target_seq=args.target,
            L_binder=args.binder_length,
            N_steps=args.n_steps,
            learning_rate=args.learning_rate,
            tau_noise=args.tau_noise,
            device=torch.device(args.device),
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
            init_binder_seq=args.init_binder_seq,
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
            mcts_prior_mode=args.mcts_prior_mode,
            mcts_proposal_method=args.mcts_proposal_method,
            mcts_adaptive_tau_start=args.mcts_adaptive_tau_start,
            mcts_adaptive_tau_end=args.mcts_adaptive_tau_end,
            checkpoint_dir=args.save_checkpoint_dir,
            checkpoint_save_policy=args.checkpoint_save_policy,
            seed=args.seed,
            target_template_path=args.target_template_path,
            proposal_amp=args.proposal_amp,
        )

    output_file = os.path.join(args.output_dir, "hallucination_result.txt")
    with open(output_file, "w") as f:
        f.write(f"Target: {args.target}\n")
        f.write(f"Binder Length: {args.binder_length}\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Model Backend: {model_backend}\n")
        f.write(f"Model Name: {args.model_name if model_backend == 'protenix' else args.esmfold2_verifier_model}\n")
        if model_backend == "esmfold2":
            f.write(f"ESMFold2 Experimental Model: {args.esmfold2_experimental_model}\n")
            f.write(f"ESMFold2 Soft Steps: {args.esmfold2_soft_steps}\n")
            f.write(f"ESMFold2 Lambda ESMC: {args.esmfold2_lambda_esmc}\n")
            f.write(f"ESMFold2 Task Type: {args.esmfold2_task_type}\n")
            f.write(f"ESMFold2 Mutable Mask: {args.esmfold2_mutable_mask or 'ALL'}\n")
        f.write(f"Optimization Steps: {args.n_steps}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n")
        f.write(f"Tau Noise: {args.tau_noise}\n")
        f.write(f"Checkpoint Dir: {resolve_checkpoint_dir(args.checkpoint_dir)}\n")
        f.write(f"Save Checkpoint Dir: {args.save_checkpoint_dir or 'DISABLED'}\n")
        f.write(f"Checkpoint Save Policy: {args.checkpoint_save_policy}\n")
        f.write(f"MCTS Strategy: {args.mcts_strategy}\n")
        f.write(f"MCTS Bootstrap Steps: {args.mcts_bootstrap_steps}\n")
        f.write(f"MCTS Bootstrap Max Steps: {args.mcts_bootstrap_max_steps}\n")
        f.write(f"MCTS Expand Actions Per Step: {args.mcts_expand_actions_per_step}\n")
        f.write(f"MCTS Prior Mode: {args.mcts_prior_mode}\n")
        f.write(f"MCTS Proposal Method: {args.mcts_proposal_method}\n")
        f.write(f"MCTS Proposal AMP: {args.proposal_amp}\n")
        f.write(f"MCTS Adaptive Tau Start: {args.mcts_adaptive_tau_start}\n")
        f.write(f"MCTS Adaptive Tau End: {args.mcts_adaptive_tau_end}\n")
        f.write(f"Initial Binder Sequence: {args.init_binder_seq or 'RANDOM'}\n")
        f.write(f"Target Hotspots 1-based: {args.target_hotspots or 'NONE'}\n")
        f.write(f"Target Template Path: {args.target_template_path or 'None'}\n")
        f.write(f"\n")
        f.write(f"Best Sequence: {best_seq}\n")
        f.write(f"Final Sequence: {final_seq}\n")
        f.write(f"\n")
        f.write(f"Loss History:\n")
        for i, loss in enumerate(losses_history):
            f.write(f"Step {i}: {loss:.6f}\n")

    logging.info(f"结果已保存到: {output_file}")

    print("\n" + "=" * 70)
    print(" Hallucination 完成！")
    print(f" 最佳序列: {best_seq}")
    print(f" 结果文件: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    import faulthandler
    import traceback

    faulthandler.enable(all_threads=True, file=sys.stderr)

    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
