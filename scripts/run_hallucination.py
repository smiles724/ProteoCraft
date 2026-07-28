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
    target_length = sum(
        len(chain.strip())
        for chain in str(args.target).split("|")
        if chain.strip()
    )
    residues_1based, concat_indices = _parse_target_hotspots_1based(
        args.target_hotspots,
        target_length,
        parser,
    )
    if not concat_indices:
        return None
    return {
        "target_hotspot_residues_1based": residues_1based,
        "target_hotspot_concat_indices": concat_indices,
    }


def resolve_esmfold2_hotspot_defaults(
    args,
    *,
    model_backend: str,
    hotspot_params,
):
    """Resolve hotspot defaults without overriding explicit CLI values."""
    use_hotspot_defaults = model_backend == "esmfold2" and hotspot_params is not None
    if args.esmfold2_hotspot_weight is None:
        args.esmfold2_hotspot_weight = 0.5 if use_hotspot_defaults else 0.0
    if args.esmfold2_hotspot_interface_mode is None:
        args.esmfold2_hotspot_interface_mode = (
            "localized" if use_hotspot_defaults else "additive"
        )
    if args.esmfold2_hotspot_verify_weight is None:
        args.esmfold2_hotspot_verify_weight = 0.25 if use_hotspot_defaults else 0.0
    if args.esmfold2_hotspot_verify_too_close_weight is None:
        args.esmfold2_hotspot_verify_too_close_weight = (
            0.25 if use_hotspot_defaults else 0.0
        )


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


def parse_esmfold2_target_chain_roles(
    raw_roles,
    *,
    n_target_chains: int,
    parser: argparse.ArgumentParser = None,
):
    if raw_roles is None or not str(raw_roles).strip():
        return None
    roles = tuple(
        token.strip().lower()
        for token in str(raw_roles).replace(";", ",").replace(" ", ",").split(",")
        if token.strip()
    )
    error = None
    if len(roles) != int(n_target_chains):
        error = (
            "--esmfold2_target_chain_roles 数量必须与 --target 中的非 binder 链数一致；"
            f"检测到 {len(roles)} 个 role、{n_target_chains} 条链"
        )
    invalid = sorted(set(roles) - {"target", "scaffold"})
    if invalid:
        error = (
            "--esmfold2_target_chain_roles 只能包含 target/scaffold；"
            f"检测到: {','.join(invalid)}"
        )
    if "target" not in roles:
        error = "--esmfold2_target_chain_roles 至少需要一条 target 链"
    if error is not None:
        if parser is not None:
            parser.error(error)
        raise ValueError(error)
    return roles



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


def load_esmfold2_backend(
    args,
    device: str = "cuda",
    *,
    mutable_mask=None,
    target_chain_roles=None,
):
    """Lazy-load the real ESMFold2 backend only when explicitly requested."""
    from bindcraft.hallucination.esmfold2_backend import ESMFold2ProposalConfig
    from bindcraft.hallucination.esmfold2_runtime import ESMFold2RuntimeBackend

    stagea_soft_steps, _ = resolve_esmfold2_stage_soft_steps(args)
    config = ESMFold2ProposalConfig(
        experimental_model_name=args.esmfold2_experimental_model,
        verifier_model_name=args.esmfold2_verifier_model,
        verifier_num_loops=args.esmfold2_verifier_num_loops,
        verifier_num_sampling_steps=args.esmfold2_verifier_num_sampling_steps,
        soft_steps=stagea_soft_steps,
        temperature_min=args.esmfold2_temperature_min,
        temperature_max=args.esmfold2_temperature_max,
        lambda_esmc=args.esmfold2_lambda_esmc,
        num_loops=args.esmfold2_proposal_num_loops,
        num_sampling_steps=args.esmfold2_proposal_num_sampling_steps,
        proposal_load_internal_esmc=args.esmfold2_proposal_load_internal_esmc,
        proposal_lm_dropout=args.esmfold2_proposal_lm_dropout,
        force_proposal_lm_dropout=args.esmfold2_force_proposal_lm_dropout,
        one_step_diffusion_enabled=args.esmfold2_one_step_diffusion,
        one_step_diffusion_tau=args.esmfold2_one_step_diffusion_tau,
        one_step_diffusion_weight=args.esmfold2_one_step_diffusion_weight,
        one_step_diffusion_rg_weight=(
            args.esmfold2_one_step_diffusion_rg_weight
        ),
        one_step_diffusion_interval=(
            args.esmfold2_one_step_diffusion_interval
        ),
        one_step_diffusion_recycle_coordinates=(
            args.esmfold2_one_step_recycle_coordinates
        ),
        one_step_diffusion_hotspot_enabled=(
            args.esmfold2_one_step_diffusion_hotspot
        ),
        distogram_hotspot_enabled=args.esmfold2_distogram_hotspot,
        verifier_loss_mode=args.esmfold2_verifier_loss_mode,
        track_proposal_confidence=args.esmfold2_track_proposal_confidence,
        proposal_confidence_temperature=args.esmfold2_proposal_confidence_temperature,
        proposal_confidence_interval=args.esmfold2_proposal_confidence_interval,
        proposal_confidence_num_loops=args.esmfold2_proposal_confidence_num_loops,
        proposal_confidence_num_sampling_steps=args.esmfold2_proposal_confidence_num_sampling_steps,
        proposal_confidence_max_candidates=args.esmfold2_proposal_confidence_max_candidates,
        proposal_confidence_tail_steps=args.esmfold2_proposal_confidence_tail_steps,
        hotspot_weight=args.esmfold2_hotspot_weight,
        hotspot_satisfied_proposal_weight=(
            args.esmfold2_hotspot_satisfied_proposal_weight
        ),
        hotspot_cutoff=args.esmfold2_hotspot_cutoff,
        hotspot_interface_mode=args.esmfold2_hotspot_interface_mode,
        hotspot_verify_weight=args.esmfold2_hotspot_verify_weight,
        hotspot_verify_near_cutoff=args.esmfold2_hotspot_verify_near_cutoff,
        hotspot_verify_far_cutoff=args.esmfold2_hotspot_verify_far_cutoff,
        hotspot_verify_min_distance=args.esmfold2_hotspot_verify_min_distance,
        hotspot_verify_too_close_weight=(
            args.esmfold2_hotspot_verify_too_close_weight
        ),
        target_chain_roles=target_chain_roles,
        binder_mutable_mask=(
            None
            if mutable_mask is None
            else tuple(bool(value) for value in mutable_mask)
        ),
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
    parser.add_argument("--esmfold2_soft_steps", type=int, default=None,
                       help="兼容旧脚本：同时设置 ESMFold2 Stage A / Stage B 的 soft gradient update 步数")
    parser.add_argument("--esmfold2_stagea_soft_steps", type=int, default=None,
                       help="ESMFold2 Stage A 每个 MCTS 节点的 soft gradient update 步数（默认：30）")
    parser.add_argument(
        "--esmfold2_temperature_min",
        type=float,
        default=0.01,
        help="ESMFold2 soft optimization 最低温度（默认：0.01）",
    )
    parser.add_argument(
        "--esmfold2_temperature_max",
        type=float,
        default=1.0,
        help="ESMFold2 soft optimization 最高温度（默认：1.0；与最低温度相等时使用固定温度）",
    )
    parser.add_argument(
        "--esmfold2_stagea_tree_children",
        type=int,
        default=2,
        help=(
            "ESMFold2 Stage A 每轮 full verify 后进入 MCTS tree 的 child 数；"
            "默认 2（argmax soft trajectory + 其余候选中 verifier 最优者）"
        ),
    )
    parser.add_argument(
        "--esmfold2_stagea_root_selection_mode",
        type=str,
        choices=["global_ucb", "hierarchical_ucb"],
        default="global_ucb",
        help=(
            "ESMFold2 Stage A root/node 选择方式；global_ucb 保留原全树 UCB，"
            "hierarchical_ucb 先选 root 再在其子树内选 node"
        ),
    )
    parser.add_argument(
        "--esmfold2_stagea_min_root_expansions",
        type=int,
        default=0,
        help=(
            "hierarchical_ucb 下每个 Phase0 root 在自适应选择前至少获得的 "
            "outer-step 扩展次数（默认 0，旧行为不受影响）"
        ),
    )
    parser.add_argument(
        "--esmfold2_stagea_preserve_soft_state",
        type=lambda x: x.lower() == "true",
        default=True,
        help=(
            "Stage A 是否保留 child 的 soft logits：argmax child 不 anchor，"
            "mutation child 只在真实突变位做轻量 anchor（默认：true）"
        ),
    )
    parser.add_argument(
        "--esmfold2_stagea_cumulative_annealing",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Stage A 是否沿 MCTS 路径累计温度退火进度（默认：true）",
    )
    parser.add_argument(
        "--esmfold2_stagea_annealing_horizon",
        type=int,
        default=150,
        help="Stage A 路径累计温度退火的 soft-step horizon（默认：150）",
    )
    parser.add_argument(
        "--esmfold2_stagea_soft_anchor_margin",
        type=float,
        default=0.1,
        help="Stage A mutation child 在突变位超过当前最大 logit 的最小 margin（默认：0.1）",
    )
    parser.add_argument("--esmfold2_stageb_soft_steps", type=int, default=None,
                       help="ESMFold2 Stage B 每个 MCTS 节点的 soft gradient update 步数（默认：20）")
    parser.add_argument(
        "--esmfold2_stagea_paired_verifier_seed",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Stage A 是否让所有 sequence 使用同一个 run-level verifier seed（默认：true）",
    )
    parser.add_argument(
        "--esmfold2_stagea_verifier_seed",
        type=int,
        default=None,
        help="可选的 Stage A 固定 verifier seed；未设置时由 run seed 和 target 推导",
    )
    parser.add_argument(
        "--esmfold2_raw_lineage_annealing_horizon",
        type=int,
        default=150,
        help="raw_lineage_stage_a 连续退火总 soft-step 数（默认：150）",
    )
    parser.add_argument(
        "--esmfold2_best_of_n",
        type=int,
        default=8,
        help=(
            "ESMFold2 Best-of-N 独立轨迹数量；continuous 模式默认 8，"
            "chunked 对照建议显式设为 4"
        ),
    )
    parser.add_argument(
        "--esmfold2_best_of_n_pass_loss_threshold",
        type=float,
        default=None,
        help="可选的 Best-of-N verifier loss 通过阈值；未设置时不计算 pass@N",
    )
    parser.add_argument(
        "--esmfold2_best_of_n_resume",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Best-of-N 是否按 trajectory 断点续跑（默认：true）",
    )
    parser.add_argument(
        "--esmfold2_best_of_n_verify_candidates_per_outer_step",
        type=int,
        default=0,
        help=(
            "chunked Best-of-N 每个 outer step 生成并 full-verify 的候选数；"
            "raw_argmax 模式只用于最终排名，verifier_best 模式还用于"
            "选择下一轮 continuation（默认：0）"
        ),
    )
    parser.add_argument(
        "--esmfold2_best_of_n_continuation_policy",
        choices=["raw_argmax", "verifier_best"],
        default="raw_argmax",
        help=(
            "chunked Best-of-N 下一轮轨迹的延续方式：raw_argmax 保留"
            "未 anchor 的连续 logits；verifier_best 选择当前候选池中"
            " verifier loss 最低的序列，并仅在必要时做 soft anchor"
        ),
    )
    parser.add_argument(
        "--esmfold2_stageb_discrete_rerank",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Stage B 是否用轻量 proposal forward 重排离散单点候选（默认：true）",
    )
    parser.add_argument(
        "--esmfold2_stageb_rerank_pool_size",
        type=int,
        default=20,
        help="Stage B 每步进入轻量重排序的候选池大小（默认：20）",
    )
    parser.add_argument(
        "--esmfold2_stageb_exploration_actions",
        type=int,
        default=1,
        help="Stage B 每步 full verifier 保留的探索候选数（默认：1）",
    )
    parser.add_argument(
        "--esmfold2_stageb_verifier_candidates",
        type=int,
        default=8,
        help="Stage B 每步送入 full verifier 的候选数（默认：8）",
    )
    parser.add_argument(
        "--esmfold2_stageb_hotspot_coverage_weight",
        type=float,
        default=0.25,
        help="Stage B reward 中连续 hotspot coverage loss 的权重（默认：0.25）",
    )
    parser.add_argument(
        "--esmfold2_stageb_hotspot_distance_weight",
        type=float,
        default=0.25,
        help="Stage B reward 中 hotspot 距离 loss 的权重（默认：0.25）",
    )
    parser.add_argument(
        "--esmfold2_stageb_hotspot_miss_penalty",
        type=float,
        default=0.0,
        help="兼容旧实验的固定 hotspot miss penalty；V5 默认 0，改用连续 coverage loss",
    )
    parser.add_argument(
        "--esmfold2_stageb_min_pair_chain_iptm",
        type=float,
        default=0.0,
        help="Stage B reward 的 pair-chain iPTM 最低门槛；0 表示关闭（默认：0）",
    )
    parser.add_argument(
        "--esmfold2_stageb_handoff_quality_tolerance",
        type=float,
        default=0.05,
        help="Stage B branch 相对 handoff pair-chain iPTM 的最大允许下降（默认：0.05）",
    )
    parser.add_argument(
        "--esmfold2_stageb_paired_verifier_seed",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Stage B 是否让所有 sequence 使用同一个 run-level verifier seed（默认：true）",
    )
    parser.add_argument("--esmfold2_lambda_esmc", type=str, default="auto",
                       help="ESMC sequence prior 权重；auto 按任务类型选择，或传入数值字符串")
    parser.add_argument("--esmfold2_task_type", type=str, default="denovo",
                       choices=["denovo", "de_novo", "minibinder", "binder", "tcr", "cdr", "cdr_redesign", "antibody", "antibody_like"],
                       help="ESMFold2/ESMC 任务类型；用于解析 lambda_ESMC=auto（默认：denovo）")
    parser.add_argument("--esmfold2_logit_init_mode", type=str, default="official_exact",
                       choices=["official_free", "official_exact", "legacy_anchor"],
                       help="ESMFold2 root logits 初始化方式；official_exact 用官方 seed 直接生成诊断初始 logits")
    parser.add_argument(
        "--esmfold2_initial_logits_checkpoint",
        type=str,
        default=None,
        help=(
            "ESMFold2 MCTS Phase0 初始 logits checkpoint 文件或 Best-of-N "
            "artifact 目录；checkpoint 数量必须与 mcts_num_beams 完全一致"
        ),
    )
    parser.add_argument("--esmfold2_mutable_mask", type=str, default=None,
                       help="ESMFold2 可变位点 mask；长度需等于 binder_length，支持 01101 或 0,1,1,true")
    parser.add_argument(
        "--esmfold2_target_chain_roles",
        type=str,
        default=None,
        help=(
            "ESMFold2 非 binder 链角色，按 --target 的链顺序填写 target/scaffold；"
            "例如 target,scaffold,scaffold。缺省时所有链均视为 target"
        ),
    )
    parser.add_argument("--esmfold2_artifact_dir", type=str, default=None,
                       help="ESMFold2 每步结构化记录目录；默认写入 output_dir/esmfold2_artifacts")
    parser.add_argument("--esmfold2_stagea_handoff_checkpoint_interval", type=int, default=10,
                       help="Stage A handoff checkpoint 保存间隔；默认每 10 step，0 表示仅保存最终 handoff")
    parser.add_argument("--esmfold2_resume_handoff_checkpoint", type=str, default=None,
                       help="跳过 Phase0/Stage A，从指定 Stage A handoff checkpoint 直接运行 Stage B")
    parser.add_argument(
        "--esmfold2_resume_reverify_roots",
        type=lambda x: x.lower() == "true",
        default=False,
        help="恢复 handoff 时用当前 verifier 配置重新验证 root；verifier 配置改变时必须启用",
    )
    parser.add_argument("--esmfold2_verifier_num_loops", type=int, default=20,
                       help="ESMFold2 verifier full fold loop 数（默认：20；smoke test 可显式调小）")
    parser.add_argument("--esmfold2_verifier_num_sampling_steps", type=int, default=200,
                       help="ESMFold2 verifier diffusion sampling steps（默认：200；smoke test 可显式调小）")
    parser.add_argument("--esmfold2_proposal_num_loops", type=int, default=1,
                       help="ESMFold2 proposal distogram forward loop 数（默认：1）")
    parser.add_argument("--esmfold2_proposal_num_sampling_steps", type=int, default=1,
                       help="ESMFold2 proposal distogram forward diffusion steps（默认：1）")
    parser.add_argument("--esmfold2_proposal_load_internal_esmc", type=lambda x: x.lower() == "true", default=True,
                       help="是否为 ESMFold2Experimental proposal model 加载内部 ESMC（官方 binder design 为 true）")
    parser.add_argument("--esmfold2_proposal_lm_dropout", type=float, default=0.5,
                       help="ESMFold2Experimental proposal LM dropout；官方 binder design inversion/proposal 使用 0.5")
    parser.add_argument("--esmfold2_force_proposal_lm_dropout", type=lambda x: x.lower() == "true", default=True,
                       help="是否在 eval/inference 状态强制启用 proposal LM dropout（官方 binder design 为 true）")
    parser.add_argument(
        "--esmfold2_one_step_diffusion",
        type=lambda x: x.lower() == "true",
        default=False,
        help=(
            "是否启用可微 ESMFold2 one-step diffusion；默认 false。"
            "interval=0 时保持旧行为，在全部 soft steps 后额外更新一次"
        ),
    )
    parser.add_argument(
        "--esmfold2_one_step_diffusion_tau",
        type=float,
        default=4.0,
        help="ESMFold2 one-step diffusion 固定噪声尺度 tau（默认：4.0）",
    )
    parser.add_argument(
        "--esmfold2_one_step_diffusion_weight",
        type=float,
        default=1.0,
        help="归一化 one-step diffusion gradient 的组合权重（默认：1.0）",
    )
    parser.add_argument(
        "--esmfold2_one_step_diffusion_rg_weight",
        type=float,
        default=0.3,
        help=(
            "one-step diffusion common loss 中 Rg/compactness 项的权重"
            "（默认：0.3，保持原有行为）"
        ),
    )
    parser.add_argument(
        "--esmfold2_one_step_diffusion_interval",
        type=int,
        default=0,
        help=(
            "在 soft optimization 内每隔多少步联合加入一次 one-step "
            "diffusion；0 表示保持旧行为，在全部 soft steps 后额外更新一次"
        ),
    )
    parser.add_argument(
        "--esmfold2_one_step_recycle_coordinates",
        type=lambda x: x.lower() == "true",
        default=False,
        help=(
            "间隔式 one-step 中是否将上一次去噪坐标作为下一次的参考结构；"
            "仅在 interval > 0 时有效"
        ),
    )
    parser.add_argument(
        "--esmfold2_one_step_diffusion_hotspot",
        type=lambda x: x.lower() == "true",
        default=False,
        help=(
            "是否让 one-step diffusion 的 iPAE loss 使用 hotspot-aware "
            "localized 聚合；默认 false，保持现有 one-step 行为"
        ),
    )
    parser.add_argument(
        "--esmfold2_distogram_hotspot",
        type=lambda x: x.lower() == "true",
        default=True,
        help=(
            "是否在 ESMFold2 distogram proposal loss 中启用 hotspot；"
            "默认 true。设为 false 可单独消融 distogram hotspot，"
            "不会关闭 one-step diffusion hotspot 或 full-verifier hotspot"
        ),
    )
    parser.add_argument(
        "--esmfold2_verifier_loss_mode",
        choices=["pair_chain_iptm", "aligned_common"],
        default="pair_chain_iptm",
        help=(
            "ESMFold2 full verifier loss：旧版 pair-chain iPTM，或不含 PDE "
            "的 Protenix-aligned common loss；默认保留旧版"
        ),
    )
    parser.add_argument("--esmfold2_track_proposal_confidence", type=lambda x: x.lower() == "true", default=None,
                       help="ESMFold2 proposal 后期是否计算 confidence/iPTM 并加入候选（Best-of-N 默认 true，其余模式默认 false）")
    parser.add_argument("--esmfold2_proposal_confidence_temperature", type=float, default=0.05,
                       help="ESMFold2 proposal temperature 低于该值时记录 confidence/iPTM（默认：0.05）")
    parser.add_argument("--esmfold2_proposal_confidence_interval", type=int, default=1,
                       help="ESMFold2 proposal 后期每隔多少 local step 记录一次 confidence/iPTM（默认：1）")
    parser.add_argument("--esmfold2_proposal_confidence_num_loops", type=int, default=1,
                       help="ESMFold2 proposal confidence forward loop 数（默认：1）")
    parser.add_argument("--esmfold2_proposal_confidence_num_sampling_steps", type=int, default=50,
                       help="ESMFold2 proposal confidence forward sampling steps（默认：50）")
    parser.add_argument("--esmfold2_proposal_confidence_max_candidates", type=int, default=8,
                       help="每次 proposal 最多把多少个 late-confidence 序列加入 verifier 候选（默认：8）")
    parser.add_argument("--esmfold2_proposal_confidence_tail_steps", type=int, default=None,
                       help="ESMFold2 proposal 固定记录最后 N 个 local steps；Best-of-N 默认 20，其余模式默认 0")
    parser.add_argument("--esmfold2_final_polish", type=lambda x: x.lower() == "true", default=False,
                       help="ESMFold2 MCTS 搜索结束后是否从 best node logits 做 final continuous polish，并只在该阶段使用 late-confidence candidates（默认：false）")
    parser.add_argument(
        "--esmfold2_hotspot_weight",
        type=float,
        default=None,
        help=(
            "ESMFold2 proposal hotspot contact loss 权重；未显式设置时，"
            "ESMFold2 有 hotspot 为 0.5，否则为 0.0"
        ),
    )
    parser.add_argument("--esmfold2_hotspot_cutoff", type=float, default=8.0,
                       help="ESMFold2 proposal hotspot contact 距离阈值，单位 Å（默认：8.0）")
    parser.add_argument(
        "--esmfold2_hotspot_satisfied_proposal_weight",
        type=float,
        default=0.0,
        help=(
            "所有 hotspot 已在 verifier near cutoff 内命中后继续使用的 "
            "proposal hotspot 权重（默认：0.0，停止继续拉近）"
        ),
    )
    parser.add_argument(
        "--esmfold2_hotspot_interface_mode",
        type=str,
        choices=["additive", "localized"],
        default=None,
        help=(
            "hotspot proposal 聚合方式：additive 保持旧版额外加权；"
            "localized 用 hotspot_weight 在全局 inter 与 hotspot inter 之间插值；"
            "未显式设置时，ESMFold2 有 hotspot 为 localized，否则为 additive"
        ),
    )
    parser.add_argument(
        "--esmfold2_hotspot_verify_weight",
        type=float,
        default=None,
        help=(
            "ESMFold2 full verifier hotspot 距离 loss 权重；未显式设置时，"
            "ESMFold2 有 hotspot 为 0.25，否则为 0.0"
        ),
    )
    parser.add_argument("--esmfold2_hotspot_verify_near_cutoff", type=float, default=8.0,
                       help="ESMFold2 hotspot verifier 满足接触的重原子距离，单位 Å（默认：8.0；同时记录严格 5 Å coverage）")
    parser.add_argument("--esmfold2_hotspot_verify_far_cutoff", type=float, default=20.0,
                       help="ESMFold2 hotspot verifier 距离 loss 饱和上限，单位 Å（默认：20.0）")
    parser.add_argument(
        "--esmfold2_hotspot_verify_min_distance",
        type=float,
        default=2.5,
        help="hotspot 与 binder 的最小安全重原子距离，单位 Å（默认：2.5）",
    )
    parser.add_argument(
        "--esmfold2_hotspot_verify_too_close_weight",
        type=float,
        default=None,
        help=(
            "低于最小安全距离时的 verifier 软惩罚权重；未显式设置时，"
            "ESMFold2 有 hotspot 为 0.25，否则为 0.0"
        ),
    )
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
                       choices=["legacy", "stage_a_only", "strict", "two_stage", "continuous_stage_a", "raw_lineage_stage_a", "best_of_n", "best_of_n_chunked"],
                       help=(
                           "搜索策略：best_of_n 运行独立连续轨迹且不使用 MCTS；"
                           "best_of_n_chunked 将 --n_steps 总 outer-step 预算"
                           "均分给独立分段轨迹，不做逐步 verifier 选择；"
                           "其余为 StageA/两阶段 MCTS 模式（默认：two_stage）"
                       ))
    parser.add_argument("--mcts_bootstrap_steps", type=int, default=50,
                       help="two_stage: StageA 原版 MCTS 至少运行步数（默认：50）")
    parser.add_argument("--mcts_bootstrap_patience", type=int, default=10,
                       help="two_stage: StageA 停滞 patience（默认：10）")
    parser.add_argument("--mcts_bootstrap_min_delta", type=float, default=0.02,
                       help="two_stage: 判断 StageA 有效改善的 loss 阈值（默认：0.02）")
    parser.add_argument("--mcts_bootstrap_max_steps", type=int, default=80,
                       help="two_stage: StageA 最大运行步数（默认：80）")
    parser.add_argument("--mcts_handoff_top_k", type=int, default=None,
                       help="two_stage: 交给 StageB 的 top 序列数（Protenix 默认 mcts_num_beams；ESMFold2 默认 4）")
    parser.add_argument("--mcts_expand_actions_per_step", type=int, default=None,
                       help="MCTS StageB 每步最多扩展并验证的 action 数（Protenix 默认 2；ESMFold2 默认 4）")
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


def resolve_esmfold2_stage_soft_steps(args) -> tuple[int, int]:
    """Resolve ESMFold2 Stage A/B soft-step defaults with legacy compatibility."""

    strategy = str(getattr(args, "mcts_strategy", "") or "").strip().lower()
    legacy_soft_steps = getattr(args, "esmfold2_soft_steps", None)
    stagea_soft_steps = getattr(args, "esmfold2_stagea_soft_steps", None)
    stageb_soft_steps = getattr(args, "esmfold2_stageb_soft_steps", None)
    if legacy_soft_steps is not None:
        if stagea_soft_steps is None:
            stagea_soft_steps = legacy_soft_steps
        if stageb_soft_steps is None:
            stageb_soft_steps = legacy_soft_steps
    if stagea_soft_steps is None:
        stagea_soft_steps = 150 if strategy == "best_of_n" else 30
    if stageb_soft_steps is None:
        stageb_soft_steps = 20
    args.esmfold2_stagea_soft_steps = int(stagea_soft_steps)
    args.esmfold2_stageb_soft_steps = int(stageb_soft_steps)
    return args.esmfold2_stagea_soft_steps, args.esmfold2_stageb_soft_steps


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
    if backend == "esmfold2":
        try:
            stagea_soft_steps, stageb_soft_steps = resolve_esmfold2_stage_soft_steps(args)
        except (TypeError, ValueError) as exc:
            message = f"ESMFold2 soft-step arguments must be positive integers: {exc}"
            if parser is not None:
                parser.error(message)
            raise ValueError(message) from exc
        if stagea_soft_steps <= 0 or stageb_soft_steps <= 0:
            message = "--esmfold2_stagea_soft_steps and --esmfold2_stageb_soft_steps must be positive"
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
        if args.esmfold2_stagea_tree_children <= 0:
            message = "--esmfold2_stagea_tree_children must be positive"
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
        if args.esmfold2_stagea_min_root_expansions < 0:
            message = "--esmfold2_stagea_min_root_expansions must be non-negative"
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
        if args.esmfold2_stagea_annealing_horizon <= 0:
            message = "--esmfold2_stagea_annealing_horizon must be positive"
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
        if args.esmfold2_stagea_soft_anchor_margin <= 0.0:
            message = "--esmfold2_stagea_soft_anchor_margin must be positive"
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
        if args.esmfold2_temperature_min <= 0.0:
            message = "--esmfold2_temperature_min must be positive"
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
        if args.esmfold2_temperature_max < args.esmfold2_temperature_min:
            message = "--esmfold2_temperature_max must be >= --esmfold2_temperature_min"
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
        if args.esmfold2_raw_lineage_annealing_horizon <= 0:
            message = "--esmfold2_raw_lineage_annealing_horizon must be positive"
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
        if args.esmfold2_best_of_n <= 0:
            message = "--esmfold2_best_of_n must be positive"
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
        if args.esmfold2_best_of_n_verify_candidates_per_outer_step < 0:
            message = (
                "--esmfold2_best_of_n_verify_candidates_per_outer_step "
                "must be non-negative"
            )
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
        if args.esmfold2_track_proposal_confidence is None:
            args.esmfold2_track_proposal_confidence = (
                args.mcts_strategy == "best_of_n"
            )
        if args.esmfold2_proposal_confidence_tail_steps is None:
            args.esmfold2_proposal_confidence_tail_steps = (
                20
                if args.mcts_strategy in {"best_of_n", "raw_lineage_stage_a"}
                else 0
            )
        if args.esmfold2_proposal_confidence_tail_steps < 0:
            message = "--esmfold2_proposal_confidence_tail_steps must be non-negative"
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
        if getattr(args, "stageb_max_mutations_per_seq", None) is None:
            args.stageb_max_mutations_per_seq = 1
        if getattr(args, "mcts_handoff_top_k", None) is None:
            args.mcts_handoff_top_k = 2
        if args.esmfold2_stageb_rerank_pool_size <= 0:
            message = "--esmfold2_stageb_rerank_pool_size must be positive"
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
        if args.esmfold2_stageb_exploration_actions < 0:
            message = "--esmfold2_stageb_exploration_actions must be non-negative"
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
        if args.esmfold2_stageb_verifier_candidates <= 0:
            message = "--esmfold2_stageb_verifier_candidates must be positive"
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
        if args.esmfold2_stageb_hotspot_coverage_weight < 0.0:
            message = "--esmfold2_stageb_hotspot_coverage_weight must be non-negative"
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
        if args.esmfold2_stageb_hotspot_distance_weight < 0.0:
            message = "--esmfold2_stageb_hotspot_distance_weight must be non-negative"
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
        if args.esmfold2_stageb_hotspot_miss_penalty < 0.0:
            message = "--esmfold2_stageb_hotspot_miss_penalty must be non-negative"
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
        if not 0.0 <= args.esmfold2_stageb_min_pair_chain_iptm <= 1.0:
            message = "--esmfold2_stageb_min_pair_chain_iptm must be within [0, 1]"
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
        if not 0.0 <= args.esmfold2_stageb_handoff_quality_tolerance <= 1.0:
            message = "--esmfold2_stageb_handoff_quality_tolerance must be within [0, 1]"
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
        if args.esmfold2_stagea_handoff_checkpoint_interval < 0:
            message = "--esmfold2_stagea_handoff_checkpoint_interval must be non-negative"
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
        if args.esmfold2_resume_handoff_checkpoint and args.mcts_strategy not in {"two_stage", "strict"}:
            message = "--esmfold2_resume_handoff_checkpoint requires --mcts_strategy two_stage"
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
        if (
            args.esmfold2_initial_logits_checkpoint
            and args.esmfold2_resume_handoff_checkpoint
        ):
            message = (
                "--esmfold2_initial_logits_checkpoint and "
                "--esmfold2_resume_handoff_checkpoint are mutually exclusive"
            )
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
        if args.esmfold2_resume_reverify_roots and not args.esmfold2_resume_handoff_checkpoint:
            message = (
                "--esmfold2_resume_reverify_roots requires "
                "--esmfold2_resume_handoff_checkpoint"
            )
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
    if getattr(args, "mcts_expand_actions_per_step", None) is None:
        args.mcts_expand_actions_per_step = 2
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
    esmfold2_target_chain_roles = None
    if model_backend == "esmfold2":
        esmfold2_mutable_mask = parse_esmfold2_mutable_mask(
            args.esmfold2_mutable_mask,
            args.binder_length,
            parser,
        )
        esmfold2_target_chain_roles = parse_esmfold2_target_chain_roles(
            args.esmfold2_target_chain_roles,
            n_target_chains=len(
                [part for part in str(args.target).split("|") if part.strip()]
            ),
            parser=parser,
        )

    hotspot_params = _build_hotspot_params(args, parser)
    resolve_esmfold2_hotspot_defaults(
        args,
        model_backend=model_backend,
        hotspot_params=hotspot_params,
    )
    if args.esmfold2_hotspot_weight < 0.0:
        parser.error("--esmfold2_hotspot_weight 必须大于等于 0")
    if args.esmfold2_hotspot_cutoff <= 0.0:
        parser.error("--esmfold2_hotspot_cutoff 必须大于 0")
    if args.esmfold2_hotspot_satisfied_proposal_weight < 0.0:
        parser.error(
            "--esmfold2_hotspot_satisfied_proposal_weight 必须大于等于 0"
        )
    if (
        args.esmfold2_hotspot_interface_mode == "localized"
        and args.esmfold2_hotspot_weight > 1.0
    ):
        parser.error(
            "--esmfold2_hotspot_interface_mode localized 时，"
            "--esmfold2_hotspot_weight 必须位于 [0, 1]"
        )
    if (
        args.esmfold2_hotspot_interface_mode == "localized"
        and args.esmfold2_hotspot_satisfied_proposal_weight > 1.0
    ):
        parser.error(
            "--esmfold2_hotspot_interface_mode localized 时，"
            "--esmfold2_hotspot_satisfied_proposal_weight 必须位于 [0, 1]"
        )
    if args.esmfold2_hotspot_verify_weight < 0.0:
        parser.error("--esmfold2_hotspot_verify_weight 必须大于等于 0")
    if args.esmfold2_hotspot_verify_near_cutoff < 0.0:
        parser.error("--esmfold2_hotspot_verify_near_cutoff 必须大于等于 0")
    if (
        args.esmfold2_hotspot_verify_far_cutoff
        <= args.esmfold2_hotspot_verify_near_cutoff
    ):
        parser.error(
            "--esmfold2_hotspot_verify_far_cutoff 必须大于 "
            "--esmfold2_hotspot_verify_near_cutoff"
        )
    if args.esmfold2_hotspot_verify_min_distance < 0.0:
        parser.error("--esmfold2_hotspot_verify_min_distance 必须大于等于 0")
    if (
        args.esmfold2_hotspot_verify_min_distance
        > args.esmfold2_hotspot_verify_near_cutoff
    ):
        parser.error(
            "--esmfold2_hotspot_verify_min_distance 不能大于 "
            "--esmfold2_hotspot_verify_near_cutoff"
        )
    if args.esmfold2_hotspot_verify_too_close_weight < 0.0:
        parser.error(
            "--esmfold2_hotspot_verify_too_close_weight 必须大于等于 0"
        )
    if (
        model_backend == "esmfold2"
        and args.esmfold2_hotspot_weight > 0.0
        and hotspot_params is None
    ):
        parser.error(
            "--esmfold2_hotspot_weight > 0 时必须同时提供 --target_hotspots"
        )
    if (
        model_backend == "esmfold2"
        and args.esmfold2_hotspot_verify_weight > 0.0
        and hotspot_params is None
    ):
        parser.error(
            "--esmfold2_hotspot_verify_weight > 0 时必须同时提供 --target_hotspots"
        )
    if (
        model_backend == "esmfold2"
        and args.esmfold2_hotspot_verify_too_close_weight > 0.0
        and hotspot_params is None
    ):
        parser.error(
            "--esmfold2_hotspot_verify_too_close_weight > 0 时必须同时提供 "
            "--target_hotspots"
        )
    if (
        model_backend == "esmfold2"
        and args.esmfold2_one_step_diffusion_interval < 0
    ):
        parser.error(
            "--esmfold2_one_step_diffusion_interval must be non-negative"
        )
    if (
        model_backend == "esmfold2"
        and args.esmfold2_one_step_diffusion_interval > 0
        and not args.esmfold2_one_step_diffusion
    ):
        parser.error(
            "--esmfold2_one_step_diffusion_interval > 0 requires "
            "--esmfold2_one_step_diffusion true"
        )
    if (
        model_backend == "esmfold2"
        and args.esmfold2_one_step_recycle_coordinates
        and args.esmfold2_one_step_diffusion_interval <= 0
    ):
        parser.error(
            "--esmfold2_one_step_recycle_coordinates true requires "
            "--esmfold2_one_step_diffusion_interval > 0"
        )
    if (
        model_backend == "esmfold2"
        and args.esmfold2_one_step_diffusion_hotspot
        and not args.esmfold2_one_step_diffusion
    ):
        parser.error(
            "--esmfold2_one_step_diffusion_hotspot true requires "
            "--esmfold2_one_step_diffusion true"
        )
    if (
        model_backend == "esmfold2"
        and args.esmfold2_one_step_diffusion_hotspot
        and hotspot_params is None
    ):
        parser.error(
            "--esmfold2_one_step_diffusion_hotspot true requires "
            "--target_hotspots"
        )

    setup_logging(args.log_file, level=args.log_level)
    os.makedirs(args.output_dir, exist_ok=True)
    set_global_seed(args.seed)
    logging.info("Random seed fixed to %s", args.seed)
    if hotspot_params is not None:
        if model_backend == "esmfold2":
            logging.info(
                "ESMFold2 hotspot contact configured: target_hotspots_1based=%s "
                "target_concat_indices_0based=%s weight=%.4f "
                "satisfied_weight=%.4f cutoff=%.2fA interface_mode=%s enabled=%s",
                hotspot_params["target_hotspot_residues_1based"],
                hotspot_params["target_hotspot_concat_indices"],
                args.esmfold2_hotspot_weight,
                args.esmfold2_hotspot_satisfied_proposal_weight,
                args.esmfold2_hotspot_cutoff,
                args.esmfold2_hotspot_interface_mode,
                args.esmfold2_hotspot_weight > 0.0,
            )
        else:
            logging.info(
                "Hotspot-aware routing enabled: target_hotspots_1based=%s target_concat_indices_0based=%s",
                hotspot_params["target_hotspot_residues_1based"],
                hotspot_params["target_hotspot_concat_indices"],
            )
    if model_backend == "esmfold2":
        logging.info(
            "ESMFold2 hotspot verifier configured: attraction_weight=%.4f "
            "near=%.2fA far=%.2fA min_safe=%.2fA "
            "too_close_weight=%.4f enabled=%s",
            args.esmfold2_hotspot_verify_weight,
            args.esmfold2_hotspot_verify_near_cutoff,
            args.esmfold2_hotspot_verify_far_cutoff,
            args.esmfold2_hotspot_verify_min_distance,
            args.esmfold2_hotspot_verify_too_close_weight,
            bool(
                hotspot_params is not None
                and (
                    args.esmfold2_hotspot_verify_weight > 0.0
                    or args.esmfold2_hotspot_verify_too_close_weight > 0.0
                )
            ),
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
        esmfold2_backend = load_esmfold2_backend(
            args,
            device=args.device,
            mutable_mask=esmfold2_mutable_mask,
            target_chain_roles=esmfold2_target_chain_roles,
        )

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
        from bindcraft.hallucination.esmfold2_pipeline import (
            run_esmfold2_best_of_n,
            run_esmfold2_chunked_best_of_n,
            run_esmfold2_mcts_hallucination,
        )

        esmfold2_artifact_dir = (
            args.esmfold2_artifact_dir
            or os.path.join(args.output_dir, "esmfold2_artifacts")
        )
        if args.mcts_strategy == "best_of_n":
            best_seq, final_seq, losses_history = run_esmfold2_best_of_n(
                backend=esmfold2_backend,
                target_seq=args.target,
                L_binder=args.binder_length,
                best_of_n=args.esmfold2_best_of_n,
                init_binder_seq=args.init_binder_seq,
                seed=args.seed,
                mutable_mask=esmfold2_mutable_mask,
                hotspot_params=hotspot_params,
                artifact_dir=esmfold2_artifact_dir,
                esmfold2_initial_logits_checkpoint=(
                    args.esmfold2_initial_logits_checkpoint
                ),
                esmfold2_logit_init_mode=args.esmfold2_logit_init_mode,
                esmfold2_soft_steps=args.esmfold2_stagea_soft_steps,
                track_confidence=args.esmfold2_track_proposal_confidence,
                pass_loss_threshold=args.esmfold2_best_of_n_pass_loss_threshold,
                verifier_seed=args.esmfold2_stagea_verifier_seed,
                resume=args.esmfold2_best_of_n_resume,
            )
        elif args.mcts_strategy == "best_of_n_chunked":
            best_seq, final_seq, losses_history = (
                run_esmfold2_chunked_best_of_n(
                    backend=esmfold2_backend,
                    target_seq=args.target,
                    L_binder=args.binder_length,
                    best_of_n=args.esmfold2_best_of_n,
                    total_outer_steps=args.n_steps,
                    init_binder_seq=args.init_binder_seq,
                    seed=args.seed,
                    mutable_mask=esmfold2_mutable_mask,
                    hotspot_params=hotspot_params,
                    artifact_dir=esmfold2_artifact_dir,
                    esmfold2_initial_logits_checkpoint=(
                        args.esmfold2_initial_logits_checkpoint
                    ),
                    esmfold2_logit_init_mode=args.esmfold2_logit_init_mode,
                    esmfold2_soft_steps=args.esmfold2_stagea_soft_steps,
                    verifier_candidates_per_outer_step=(
                        args.esmfold2_best_of_n_verify_candidates_per_outer_step
                    ),
                    continuation_policy=(
                        args.esmfold2_best_of_n_continuation_policy
                    ),
                    continuation_soft_anchor_margin=(
                        args.esmfold2_stagea_soft_anchor_margin
                    ),
                    candidate_k_top=args.k_top,
                    candidate_temperature=args.t_sample,
                    candidate_max_mutations_per_seq=(
                        args.stagea_max_mutations_per_seq
                    ),
                    pass_loss_threshold=(
                        args.esmfold2_best_of_n_pass_loss_threshold
                    ),
                    verifier_seed=args.esmfold2_stagea_verifier_seed,
                    resume=args.esmfold2_best_of_n_resume,
                )
            )
        else:
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
                esmfold2_initial_logits_checkpoint=args.esmfold2_initial_logits_checkpoint,
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
                hotspot_params=hotspot_params,
                artifact_dir=esmfold2_artifact_dir,
                esmfold2_logit_init_mode=args.esmfold2_logit_init_mode,
                esmfold2_stagea_soft_steps=args.esmfold2_stagea_soft_steps,
                esmfold2_stagea_tree_children=args.esmfold2_stagea_tree_children,
                esmfold2_stagea_root_selection_mode=(
                    args.esmfold2_stagea_root_selection_mode
                ),
                esmfold2_stagea_min_root_expansions=(
                    args.esmfold2_stagea_min_root_expansions
                ),
                esmfold2_stagea_preserve_soft_state=args.esmfold2_stagea_preserve_soft_state,
                esmfold2_stagea_cumulative_annealing=args.esmfold2_stagea_cumulative_annealing,
                esmfold2_stagea_annealing_horizon=args.esmfold2_stagea_annealing_horizon,
                esmfold2_stagea_soft_anchor_margin=args.esmfold2_stagea_soft_anchor_margin,
                esmfold2_stagea_paired_verifier_seed=args.esmfold2_stagea_paired_verifier_seed,
                esmfold2_stagea_verifier_seed=args.esmfold2_stagea_verifier_seed,
                esmfold2_raw_lineage_annealing_horizon=args.esmfold2_raw_lineage_annealing_horizon,
                esmfold2_stageb_soft_steps=args.esmfold2_stageb_soft_steps,
                esmfold2_stageb_discrete_rerank=args.esmfold2_stageb_discrete_rerank,
                esmfold2_stageb_rerank_pool_size=args.esmfold2_stageb_rerank_pool_size,
                esmfold2_stageb_exploration_actions=args.esmfold2_stageb_exploration_actions,
                esmfold2_stageb_verifier_candidates=args.esmfold2_stageb_verifier_candidates,
                esmfold2_stageb_hotspot_coverage_weight=args.esmfold2_stageb_hotspot_coverage_weight,
                esmfold2_stageb_hotspot_distance_weight=args.esmfold2_stageb_hotspot_distance_weight,
                esmfold2_stageb_hotspot_miss_penalty=args.esmfold2_stageb_hotspot_miss_penalty,
                esmfold2_stageb_min_pair_chain_iptm=args.esmfold2_stageb_min_pair_chain_iptm,
                esmfold2_stageb_handoff_quality_tolerance=args.esmfold2_stageb_handoff_quality_tolerance,
                esmfold2_stageb_paired_verifier_seed=args.esmfold2_stageb_paired_verifier_seed,
                esmfold2_final_polish=args.esmfold2_final_polish,
                esmfold2_stagea_handoff_checkpoint_interval=args.esmfold2_stagea_handoff_checkpoint_interval,
                esmfold2_resume_handoff_checkpoint=args.esmfold2_resume_handoff_checkpoint,
                esmfold2_resume_reverify_roots=args.esmfold2_resume_reverify_roots,
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
            f.write(f"ESMFold2 Soft Steps Legacy Override: {args.esmfold2_soft_steps}\n")
            f.write(f"ESMFold2 Stage A Soft Steps: {args.esmfold2_stagea_soft_steps}\n")
            f.write(f"ESMFold2 Temperature Min: {args.esmfold2_temperature_min}\n")
            f.write(f"ESMFold2 Temperature Max: {args.esmfold2_temperature_max}\n")
            f.write(
                "ESMFold2 Stage A Tree Children: "
                f"{args.esmfold2_stagea_tree_children}\n"
            )
            f.write(
                "ESMFold2 Stage A Preserve Soft State: "
                f"{args.esmfold2_stagea_preserve_soft_state}\n"
            )
            f.write(
                "ESMFold2 Stage A Cumulative Annealing: "
                f"{args.esmfold2_stagea_cumulative_annealing}\n"
            )
            f.write(
                "ESMFold2 Stage A Annealing Horizon: "
                f"{args.esmfold2_stagea_annealing_horizon}\n"
            )
            f.write(
                "ESMFold2 Stage A Soft Anchor Margin: "
                f"{args.esmfold2_stagea_soft_anchor_margin}\n"
            )
            f.write(
                "ESMFold2 Stage A Paired Verifier Seed: "
                f"{args.esmfold2_stagea_paired_verifier_seed}\n"
            )
            f.write(
                "ESMFold2 Stage A Verifier Seed: "
                f"{args.esmfold2_stagea_verifier_seed or 'AUTO'}\n"
            )
            f.write(
                "ESMFold2 Raw Lineage Annealing Horizon: "
                f"{args.esmfold2_raw_lineage_annealing_horizon}\n"
            )
            f.write(f"ESMFold2 Stage B Soft Steps: {args.esmfold2_stageb_soft_steps}\n")
            f.write(f"ESMFold2 Best-of-N: {args.esmfold2_best_of_n}\n")
            if args.mcts_strategy == "best_of_n_chunked":
                f.write(
                    "ESMFold2 Chunked Best-of-N Total Outer Steps: "
                    f"{args.n_steps}\n"
                )
            f.write(
                "ESMFold2 Best-of-N Pass Loss Threshold: "
                f"{args.esmfold2_best_of_n_pass_loss_threshold}\n"
            )
            f.write(
                "ESMFold2 Best-of-N Verified Candidates Per Outer Step: "
                f"{args.esmfold2_best_of_n_verify_candidates_per_outer_step}\n"
            )
            f.write(
                "ESMFold2 Best-of-N Continuation Policy: "
                f"{args.esmfold2_best_of_n_continuation_policy}\n"
            )
            f.write(f"ESMFold2 Stage B Discrete Rerank: {args.esmfold2_stageb_discrete_rerank}\n")
            f.write(f"ESMFold2 Stage B Rerank Pool Size: {args.esmfold2_stageb_rerank_pool_size}\n")
            f.write(f"ESMFold2 Stage B Exploration Actions: {args.esmfold2_stageb_exploration_actions}\n")
            f.write(f"ESMFold2 Stage B Verifier Candidates: {args.esmfold2_stageb_verifier_candidates}\n")
            f.write(
                "ESMFold2 Stage B Hotspot Coverage Weight: "
                f"{args.esmfold2_stageb_hotspot_coverage_weight}\n"
            )
            f.write(
                "ESMFold2 Stage B Hotspot Distance Weight: "
                f"{args.esmfold2_stageb_hotspot_distance_weight}\n"
            )
            f.write(
                "ESMFold2 Stage B Hotspot Miss Penalty: "
                f"{args.esmfold2_stageb_hotspot_miss_penalty}\n"
            )
            f.write(
                "ESMFold2 Stage B Min Pair-chain iPTM: "
                f"{args.esmfold2_stageb_min_pair_chain_iptm}\n"
            )
            f.write(
                "ESMFold2 Stage B Handoff Quality Tolerance: "
                f"{args.esmfold2_stageb_handoff_quality_tolerance}\n"
            )
            f.write(
                "ESMFold2 Stage B Paired Verifier Seed: "
                f"{args.esmfold2_stageb_paired_verifier_seed}\n"
            )
            f.write(f"ESMFold2 Lambda ESMC: {args.esmfold2_lambda_esmc}\n")
            f.write(f"ESMFold2 Task Type: {args.esmfold2_task_type}\n")
            f.write(f"ESMFold2 Logit Init Mode: {args.esmfold2_logit_init_mode}\n")
            f.write(
                "ESMFold2 Initial Logits Checkpoint: "
                f"{args.esmfold2_initial_logits_checkpoint or 'NONE'}\n"
            )
            f.write(f"ESMFold2 Mutable Mask: {args.esmfold2_mutable_mask or 'ALL'}\n")
            f.write(
                "ESMFold2 Target Chain Roles: "
                f"{args.esmfold2_target_chain_roles or 'ALL_TARGET'}\n"
            )
            f.write(f"ESMFold2 Stage A Handoff Checkpoint Interval: {args.esmfold2_stagea_handoff_checkpoint_interval}\n")
            f.write(f"ESMFold2 Resume Handoff Checkpoint: {args.esmfold2_resume_handoff_checkpoint or 'NONE'}\n")
            f.write(f"ESMFold2 Resume Reverify Roots: {args.esmfold2_resume_reverify_roots}\n")
            f.write(f"ESMFold2 Proposal Load Internal ESMC: {args.esmfold2_proposal_load_internal_esmc}\n")
            f.write(f"ESMFold2 Proposal LM Dropout: {args.esmfold2_proposal_lm_dropout}\n")
            f.write(f"ESMFold2 Force Proposal LM Dropout: {args.esmfold2_force_proposal_lm_dropout}\n")
            f.write(f"ESMFold2 Track Proposal Confidence: {args.esmfold2_track_proposal_confidence}\n")
            f.write(f"ESMFold2 Proposal Confidence Temperature: {args.esmfold2_proposal_confidence_temperature}\n")
            f.write(f"ESMFold2 Proposal Confidence Interval: {args.esmfold2_proposal_confidence_interval}\n")
            f.write(f"ESMFold2 Proposal Confidence Loops: {args.esmfold2_proposal_confidence_num_loops}\n")
            f.write(f"ESMFold2 Proposal Confidence Sampling Steps: {args.esmfold2_proposal_confidence_num_sampling_steps}\n")
            f.write(f"ESMFold2 Proposal Confidence Max Candidates: {args.esmfold2_proposal_confidence_max_candidates}\n")
            f.write(f"ESMFold2 Proposal Confidence Tail Steps: {args.esmfold2_proposal_confidence_tail_steps}\n")
            f.write(f"ESMFold2 Final Polish: {args.esmfold2_final_polish}\n")
            f.write(
                "ESMFold2 One-step Diffusion: "
                f"{args.esmfold2_one_step_diffusion}\n"
            )
            f.write(
                "ESMFold2 One-step Diffusion Tau: "
                f"{args.esmfold2_one_step_diffusion_tau}\n"
            )
            f.write(
                "ESMFold2 One-step Diffusion Weight: "
                f"{args.esmfold2_one_step_diffusion_weight}\n"
            )
            f.write(
                "ESMFold2 One-step Diffusion Rg Weight: "
                f"{args.esmfold2_one_step_diffusion_rg_weight}\n"
            )
            f.write(
                "ESMFold2 One-step Diffusion Interval: "
                f"{args.esmfold2_one_step_diffusion_interval}\n"
            )
            f.write(
                "ESMFold2 One-step Recycle Coordinates: "
                f"{args.esmfold2_one_step_recycle_coordinates}\n"
            )
            f.write(
                "ESMFold2 One-step Diffusion Hotspot: "
                f"{args.esmfold2_one_step_diffusion_hotspot}\n"
            )
            f.write(
                "ESMFold2 Distogram Hotspot: "
                f"{args.esmfold2_distogram_hotspot}\n"
            )
            f.write(f"ESMFold2 Hotspot Weight: {args.esmfold2_hotspot_weight}\n")
            f.write(
                "ESMFold2 Hotspot Satisfied Proposal Weight: "
                f"{args.esmfold2_hotspot_satisfied_proposal_weight}\n"
            )
            f.write(f"ESMFold2 Hotspot Cutoff: {args.esmfold2_hotspot_cutoff} A\n")
            f.write(
                "ESMFold2 Hotspot Interface Mode: "
                f"{args.esmfold2_hotspot_interface_mode}\n"
            )
            f.write(
                "ESMFold2 Hotspot Verify Weight: "
                f"{args.esmfold2_hotspot_verify_weight}\n"
            )
            f.write(
                "ESMFold2 Hotspot Verify Near Cutoff: "
                f"{args.esmfold2_hotspot_verify_near_cutoff} A\n"
            )
            f.write(
                "ESMFold2 Hotspot Verify Far Cutoff: "
                f"{args.esmfold2_hotspot_verify_far_cutoff} A\n"
            )
            f.write(
                "ESMFold2 Hotspot Verify Min Distance: "
                f"{args.esmfold2_hotspot_verify_min_distance} A\n"
            )
            f.write(
                "ESMFold2 Hotspot Verify Too Close Weight: "
                f"{args.esmfold2_hotspot_verify_too_close_weight}\n"
            )
        f.write(f"Optimization Steps: {args.n_steps}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n")
        f.write(f"Tau Noise: {args.tau_noise}\n")
        f.write(f"Checkpoint Dir: {resolve_checkpoint_dir(args.checkpoint_dir)}\n")
        f.write(f"Save Checkpoint Dir: {args.save_checkpoint_dir or 'DISABLED'}\n")
        f.write(f"Checkpoint Save Policy: {args.checkpoint_save_policy}\n")
        f.write(f"MCTS Strategy: {args.mcts_strategy}\n")
        f.write(f"MCTS Handoff Top K: {args.mcts_handoff_top_k}\n")
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
        history_label = (
            "Trajectory Best Losses"
            if (
                model_backend == "esmfold2"
                and args.mcts_strategy in {"best_of_n", "best_of_n_chunked"}
            )
            else "Loss History"
        )
        item_label = (
            "Trajectory"
            if (
                model_backend == "esmfold2"
                and args.mcts_strategy in {"best_of_n", "best_of_n_chunked"}
            )
            else "Step"
        )
        f.write(f"{history_label}:\n")
        for i, loss in enumerate(losses_history):
            f.write(f"{item_label} {i}: {loss:.6f}\n")

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
