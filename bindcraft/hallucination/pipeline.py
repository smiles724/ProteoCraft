"""
BindCraft v2 - Hallucination 主循环

整合所有模块，实现完整的逆向梯度优化流程

流程：
1. 初始化 Logits (Theta)
2. 循环优化：
   - STE 桥接（模块 A）
   - 单步外推（模块 B）
   - 计算 Loss（模块 C - 对标原版 BindCraft）
   - 反向传播
   - 更新 Theta
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Union, Optional, Any
import logging
import os
import math
from contextlib import nullcontext
from dataclasses import replace

from bindcraft.hallucination.ste_bridge import (
    precompute_waa_embeddings,
    ste_bridge
)
from bindcraft.checkpoint import CheckpointManager
from bindcraft.utils import (
    build_single_chain_features,
    prepare_input_features as _prepare_input_features,
    freeze_model,
    log_step,
    NUM_AMINO_ACIDS,
    DEFAULT_LOSS_WEIGHTS,
    DEFAULT_CONTACT_PARAMS,
    DEFAULT_CLASH_PARAMS,
    _normalize_chain_sequences,
)
from bindcraft.hallucination.diffusion_one_step import (
    diffusion_one_step,
    diffusion_one_step_fused,
)
from bindcraft.hallucination.mutation_proposer import (
    AA_1_TO_IDX,
    propose_mutations,
    propose_mutations_legacy,
    propose_mutation_actions,
    normalize_mcts_prior_mode,
    normalize_mcts_proposal_method,
    select_de_novo_action_group,
    should_use_gradient_prior,
)
from bindcraft.hallucination.zeroth_order import zeroth_order_eval
from bindcraft.hallucination.reproducibility import set_global_seed, stable_int_seed
from bindcraft.losses.confidence import run_confidence_heads
from bindcraft.losses.total_loss import compute_all_losses
from protenix.model import sample_confidence

logger = logging.getLogger(__name__)
VALID_BINDER_AAS = "ACDEFGHIKLMNPQRSTVWY"


def _resolve_proposal_amp_dtype(model_name: Optional[str], proposal_amp: str = "auto"):
    """Resolve proposal-only autocast dtype without changing V1 defaults."""
    mode = (proposal_amp or "auto").strip().lower()
    if mode == "off":
        return None
    if mode == "bf16":
        return torch.bfloat16
    if mode == "auto":
        normalized_model = (model_name or "").strip().lower()
        return torch.bfloat16 if normalized_model == "protenix-v2" else None
    raise ValueError(
        "proposal_amp must be one of {'auto', 'off', 'bf16'}, "
        f"got {proposal_amp!r}"
    )


def _proposal_amp_context(device: torch.device, amp_dtype):
    if amp_dtype is None:
        return nullcontext()
    device_obj = torch.device(device)
    if device_obj.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def _verifier_seed(base_seed: Optional[int], target_seq: Union[str, List[str]], binder_seq: str) -> Optional[int]:
    """Stable seed for full verifier evaluation of one target/binder sequence."""
    if base_seed is None:
        return None
    target_key = "|".join(_normalize_chain_sequences(target_seq))
    return stable_int_seed(base_seed, "zeroth_order", target_key, binder_seq)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid boolean env %s=%r, using default=%s", name, value, default)
    return bool(default)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float env %s=%r, using default=%s", name, value, default)
        return float(default)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return int(default)
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid int env %s=%r, using default=%s", name, value, default)
        return int(default)


MCTS_STAGE2_PROGRESSIVE_WIDENING_K = _env_float("BINDCRAFT_MCTS_PW_K", 4.0)
MCTS_STAGE2_PROGRESSIVE_WIDENING_ALPHA = _env_float("BINDCRAFT_MCTS_PW_ALPHA", 0.5)
MCTS_STAGE2_MAX_CHILDREN_PER_NODE = _env_int("BINDCRAFT_MCTS_MAX_CHILDREN", 16)
MCTS_STAGE2_MIN_CHILDREN_BEFORE_DESCEND = _env_int("BINDCRAFT_MCTS_MIN_CHILDREN_BEFORE_DESCEND", 4)
MCTS_STAGE2_DESCEND_SCORE_MARGIN = _env_float("BINDCRAFT_MCTS_DESCEND_MARGIN", 0.0)
MCTS_STAGE2_EXPAND_ACTIONS_PER_STEP = _env_int("BINDCRAFT_MCTS_EXPAND_ACTIONS_PER_STEP", 2)
MCTS_STAGE2_GROUP_TEMPERATURE = _env_float("BINDCRAFT_MCTS_GROUP_TEMPERATURE", 1.0)
MCTS_STAGE2_HANDOFF_MIN_DISTANCE_FRAC = _env_float("BINDCRAFT_MCTS_HANDOFF_MIN_DISTANCE_FRAC", 0.03)
MCTS_STAGE2_STALE_EDIT_STEPS = _env_int("BINDCRAFT_MCTS_STALE_EDIT_STEPS", 5)
MCTS_STAGE3_USE_LOSS_GROUPING = _env_bool("BINDCRAFT_MCTS_USE_LOSS_GROUPING", True)
MCTS_STAGE3_USE_STRUCTURAL_GROUPING = _env_bool("BINDCRAFT_MCTS_USE_STRUCTURAL_GROUPING", True)
MCTS_STAGE3_USE_PARETO_PRUNING = _env_bool("BINDCRAFT_MCTS_USE_PARETO", True)
MCTS_STAGE3_USE_REWARD_NORMALIZATION = _env_bool("BINDCRAFT_MCTS_USE_REWARD_NORMALIZATION", True)
MCTS_STAGE3_USE_DIVERSITY_BONUS = _env_bool("BINDCRAFT_MCTS_USE_DIVERSITY_BONUS", True)
MCTS_STAGE3_REPRESENTATIVES_PER_GROUP = _env_int("BINDCRAFT_MCTS_REPRESENTATIVES_PER_GROUP", 5)
MCTS_STAGE3_LOSS_GROUP_BIN_WIDTH = _env_float("BINDCRAFT_MCTS_LOSS_GROUP_BIN_WIDTH", 1.0)
MCTS_STAGE3_DIVERSITY_BONUS_WEIGHT = _env_float("BINDCRAFT_MCTS_DIVERSITY_WEIGHT", 0.03)
MCTS_STAGE3_DIVERSITY_BONUS_MAX = _env_float("BINDCRAFT_MCTS_DIVERSITY_MAX", 1.0)


def _allocate_action_group_budgets(group_priors: dict, total_budget: int) -> dict:
    """
    Deterministically allocate a proposal-pool budget across action groups.

    This keeps the document's P_group(g|s) semantics without adding another
    source of stochasticity before the mutation sampler itself.
    """
    if total_budget <= 0:
        return {}

    valid = [
        (str(group), max(float(prior), 0.0))
        for group, prior in (group_priors or {}).items()
        if float(prior) > 0.0
    ]
    if not valid:
        return {}

    total_prior = sum(prior for _, prior in valid)
    if total_prior <= 0.0:
        return {}

    scaled = [
        (group, total_budget * prior / total_prior)
        for group, prior in valid
    ]
    budgets = {group: int(math.floor(raw)) for group, raw in scaled}
    used = sum(budgets.values())
    remainders = sorted(
        ((raw - math.floor(raw), group) for group, raw in scaled),
        reverse=True,
    )
    for _, group in remainders:
        if used >= total_budget:
            break
        budgets[group] += 1
        used += 1

    return {group: count for group, count in budgets.items() if count > 0}


def _uniform_de_novo_group_priors() -> Dict[str, float]:
    return {"unrouted": 1.0}


def _apply_stageb_prior_mode_to_groups(
    prior_mode: str,
    action_group: str,
    group_priors: Dict[str, float],
) -> Tuple[str, Dict[str, float]]:
    """
    Apply prior-ablation modes to StageB verifier-loss group priors.

    full/no_gradient/no_sequence keep verifier-loss group routing. no_group and
    uniform remove that signal and fall back to the unguided total-loss proposal.
    """
    prior_mode = normalize_mcts_prior_mode(prior_mode)
    if prior_mode in {"uniform", "no_group"}:
        return "unrouted", _uniform_de_novo_group_priors()
    return action_group, group_priors


def _renormalize_stageb_proposal_priors(proposals: list, prior_mode: str) -> list:
    """Make final PUCT action priors uniform when running the uniform ablation."""
    prior_mode = normalize_mcts_prior_mode(prior_mode)
    if prior_mode != "uniform" or not proposals:
        return proposals
    uniform_prior = 1.0 / max(len(proposals), 1)
    return [replace(proposal, prior=uniform_prior, group_prior=uniform_prior) for proposal in proposals]


def _normalize_init_binder_seq(init_binder_seq: str, L_binder: int) -> str:
    """
    校验并标准化用户提供的初始 Binder 序列。

    Args:
        init_binder_seq: 用户提供的单字母序列
        L_binder: 期望的 Binder 长度

    Returns:
        规范化后的大写单字母序列
    """
    seq = init_binder_seq.strip().upper()
    if len(seq) != L_binder:
        raise ValueError(
            f"初始 Binder 序列长度 {len(seq)} 与 binder_length={L_binder} 不一致"
        )

    invalid = sorted(set(seq) - set(VALID_BINDER_AAS))
    if invalid:
        raise ValueError(
            "初始 Binder 序列包含非法氨基酸字符: "
            + ",".join(invalid)
            + f"；仅支持 {VALID_BINDER_AAS}"
        )
    return seq


def _sample_random_binder_seq(L_binder: int) -> str:
    """
    生成随机初始 Binder 序列。
    """
    return "".join(np.random.choice(list(VALID_BINDER_AAS)) for _ in range(L_binder))


def _initialize_theta(
    L_binder: int,
    device: torch.device,
    init_binder_seq: str = None,
    logit_scale: float = 5.0,
) -> Tuple[torch.Tensor, str]:
    """
    根据初始 Binder 序列初始化 theta，使 argmax(theta) 与起始序列一致。
    """
    init_binder = (
        _sample_random_binder_seq(L_binder)
        if init_binder_seq is None
        else _normalize_init_binder_seq(init_binder_seq, L_binder)
    )

    theta = torch.randn(L_binder, NUM_AMINO_ACIDS, device=device) * 0.01
    with torch.no_grad():
        for i, aa in enumerate(init_binder):
            theta[i, AA_1_TO_IDX[aa]] += logit_scale
    theta.requires_grad_(True)
    return theta, init_binder


# ============================================================
# [已废弃] 纯一阶 Hallucination 循环
# 注意：本函数不再使用，当前算法为 run_hybrid_hallucination
# ============================================================
def run_hallucination(
    model,
    target_seq: Union[str, List[str]],
    L_binder: int = 40,
    N_steps: int = 200,
    learning_rate: float = 0.05,
    tau_noise: float = 4.0,
    device: torch.device = None,
    loss_weights: Dict[str, float] = None,
    init_binder_seq: str = None,
    hotspot_params: Optional[Dict] = None,
    seed: Optional[int] = 42,
    target_template_path: Optional[str] = None,
    target_chain_roles: Optional[Any] = None,
) -> Tuple[str, str, List[float]]:
    """
    运行 Hallucination 优化循环

    Args:
        model: Protenix 模型
        target_seq: Target 序列（单字母）
        L_binder: Binder 长度
        N_steps: 优化步数
        learning_rate: 学习率
        tau_noise: 噪声时间步
        device: 计算设备
        loss_weights: Loss 权重字典
        init_binder_seq: 可选的初始 Binder 单字母序列
        target_template_path: 可选 target template 结构文件

    Returns:
        best_seq: 最佳序列
        final_seq: 最终序列
        losses_history: Loss 历史
    """
    if device is None:
        device = next(model.parameters()).device
    set_global_seed(seed)

    weights = dict(DEFAULT_LOSS_WEIGHTS)
    if loss_weights is not None:
        weights.update(loss_weights)
    contact_params = dict(DEFAULT_CONTACT_PARAMS)
    target_sequences = _normalize_chain_sequences(target_seq)
    target_total_len = sum(len(seq) for seq in target_sequences)
    target_len_str = "+".join(str(len(seq)) for seq in target_sequences)

    logger.info("=" * 70)
    logger.info("开始 Hallucination 优化")
    logger.info(
        f"Target chains: {len(target_sequences)} | lengths={target_len_str} | total={target_total_len} aa"
    )
    logger.info(f"Binder 长度: {L_binder} aa")
    logger.info(f"优化步数: {N_steps}")
    logger.info(f"Loss 权重: {weights}")
    logger.info("=" * 70)

    freeze_model(model)

    # ── 预计算 W_AAE ──────────────────────────────────────────────────────
    logger.info("预计算 W_AAE...")
    W_AAE = precompute_waa_embeddings(model, device)

    # ── 初始化 Theta ──────────────────────────────────────────────────────
    set_global_seed(seed)
    theta, init_binder = _initialize_theta(
        L_binder=L_binder,
        device=device,
        init_binder_seq=init_binder_seq,
    )

    optimizer = torch.optim.Adam([theta], lr=learning_rate)

    # ── 获取冻结的 Target 特征 ────────────────────────────────────────────
    logger.info("获取 Target 特征（target-only 单链）...")
    with torch.no_grad():
        feat_target_only, _, _ = build_single_chain_features(target_seq, device)
        feat_target_only = _prepare_input_features(model, feat_target_only)
        s_target = model.input_embedder(feat_target_only, inplace_safe=False)
        s_target = s_target.clone()
    logger.info(f"s_target shape: {s_target.shape}")

    init_source = "指定" if init_binder_seq is not None else "随机"
    logger.info(f"初始 Binder 序列（{init_source}）: {init_binder}")

    # ── 获取 bin 参数（从 model configs 读取） ────────────────────────────
    plddt_params = sample_confidence.get_bin_params(model.configs.loss.plddt)
    pae_params = sample_confidence.get_bin_params(model.configs.loss.pae)
    distogram_params = sample_confidence.get_bin_params(model.configs.loss.distogram)
    pde_params = sample_confidence.get_bin_params(model.configs.loss.pde)

    # ── 优化循环 ──────────────────────────────────────────────────────────
    best_loss = float('inf')
    best_seq = init_binder
    best_step = 0
    losses_history = []
    no_improve_count = 0
    early_stop_patience = 60

    logger.info("开始优化循环...")

    for step in range(N_steps):
        optimizer.zero_grad()

        progress = step / max(N_steps - 1, 1)
        tau_temp = 2.0 * (1.0 - progress) + 0.05 * progress

        # ── 模块 A：STE 桥接 ─────────────────────────────────────────────
        s_binder, current_seq, input_feature_dict = ste_bridge(
            theta=theta,
            W_AAE=W_AAE,
            model=model,
            target_seq=target_seq,
            device=device,
            tau_temp=tau_temp,
            target_template_path=target_template_path,
        )

        # ── 模块 B：单步外推 ─────────────────────────────────────────────
        x0_hat, s_trunk, z_trunk = diffusion_one_step(
            model=model,
            s_binder=s_binder,
            s_target=s_target,
            input_feature_dict=input_feature_dict,
            x_coarse=None,
            tau=tau_noise,
            device=device
        )

        # ── 模块 C：计算 Loss（对标原版 BindCraft） ──────────────────────
        x0_flat = x0_hat.squeeze(0)  # [N_atoms, 3]

        s_inputs_full = torch.cat([s_binder, s_target], dim=0)
        heads = run_confidence_heads(
            model=model,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs_full,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            x0_hat=x0_hat,
            inplace_safe=False,
            chunk_size=None,
        )

        total_loss, losses = compute_all_losses(
            x0_flat=x0_flat,
            plddt_logits=heads["plddt_logits"],
            pae_logits=heads["pae_logits"],
            distogram_logits=heads["distogram_logits"],
            input_feature_dict=input_feature_dict,
            L_binder=L_binder,
            weights=weights,
            contact_params=contact_params,
            plddt_params=plddt_params,
            pae_params=pae_params,
            distogram_params=distogram_params,
            pde_logits=heads["pde_logits"],
            pde_params=pde_params,
            clash_params=dict(DEFAULT_CLASH_PARAMS),
            hotspot_params=hotspot_params,
            target_chain_roles=target_chain_roles,
        )

        # ── 反向传播 ─────────────────────────────────────────────────────
        total_loss.backward()

        grad_norm = theta.grad.norm().item() if theta.grad is not None else 0.0
        torch.nn.utils.clip_grad_norm_([theta], 1.0)

        optimizer.step()

        # ── 记录 ─────────────────────────────────────────────────────────
        loss_val = total_loss.item()
        losses_history.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_seq = current_seq
            best_step = step
            no_improve_count = 0
        else:
            no_improve_count += 1

        if step % 1 == 0 or step == N_steps - 1:
            loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v
                         for k, v in losses.items()}
            log_step(step, N_steps, loss_dict, current_seq,
                     f"grad: {grad_norm:.6f} | best@{best_step}")

        if no_improve_count >= early_stop_patience and step > 60:
            logger.info(f"早停：{early_stop_patience} 步无改进")
            break

    logger.info("=" * 70)
    logger.info("优化完成！")
    logger.info(f"最佳序列: {best_seq} (loss={best_loss:.4f} @step {best_step})")
    logger.info(f"最终序列: {current_seq}")
    logger.info("=" * 70)

    return best_seq, current_seq, losses_history


# ============================================================
# 混合优化循环（一阶提议 + 零阶验证）
# ============================================================

def _sync_theta_to_seq(
    theta: torch.Tensor,
    seq: str,
    logit_scale: float = 5.0,
) -> None:
    """
    将离散序列写回连续 Logits theta，使 argmax(theta) == seq。

    采用硬覆盖：完全丢弃当前 theta 值，用 one-hot 偏置重置。
    保证 argmax(theta) 必然等于目标序列，消除序列不一致风险。

    Args:
        theta: [L_binder, 20] 可学习 Logits（就地修改）
        seq: 目标离散序列（单字母 str）
        logit_scale: one-hot 偏置的强度
    """
    from bindcraft.hallucination.mutation_proposer import AA_1_TO_IDX
    indices = [AA_1_TO_IDX[aa] for aa in seq]
    with torch.no_grad():
        bias = torch.zeros_like(theta)
        for i, idx in enumerate(indices):
            bias[i, idx] = logit_scale
        theta.copy_(bias)


def run_hybrid_hallucination(
    model,
    target_seq: Union[str, List[str]],
    L_binder: int = 40,
    N_steps: int = 100,
    learning_rate: float = 0.05,
    tau_noise: float = 4.0,
    device: torch.device = None,
    loss_weights: Dict[str, float] = None,
    K_top: int = 10,
    B_candidates: int = 8,
    T_sample: float = 1.0,
    max_mutations_per_seq: int = 3,
    use_annealing: bool = True,
    T_init: float = 1.0,
    T_min: float = 0.01,
    use_tau_scheduling: bool = False,
    tau_jitter: bool = True,
    init_binder_seq: str = None,
    num_beams: int = 1,
    use_prior: bool = True,
    prior_strength: float = 0.1,
    use_tau_fusion: bool = False,
    tau_coarse: float = 0.8,
    tau_fine: float = 0.05,
    alpha: float = 0.5,
    use_macro_breathing: bool = False,
    breath_patience: int = 8,
    breath_tau_recovery: float = 0.9,
    breath_con_factor: float = 0.3,
    breath_recovery_steps: int = 3,
    checkpoint_dir: str = None,
    checkpoint_save_policy: str = "all",
    hotspot_params: Optional[Dict] = None,
    seed: Optional[int] = 42,
    target_template_path: Optional[str] = None,
    target_chain_roles: Optional[Any] = None,
) -> Tuple[str, str, List[float]]:
    """
    混合优化循环：一阶梯度引导离散候选 + 零阶全链路验证 + 接受准则。
    支持 Beam Search 多分支演化（num_beams >= 2）。

    当 num_beams=1（默认）时，退化为标准单路径混合优化：
      每轮：1阶反传 -> 候选生成(K条) -> 0阶验证(K条) -> 贪心/退火接受

    当 num_beams>=2 时，执行真正的 Beam Search：
      每轮：每条分支独立1阶反传 -> 全部B×K候选 -> 全局排序 -> 保留前B条候选到对应分支

    Args:
        model:         Protenix 模型
        target_seq:    Target 序列（单字母）
        L_binder:      Binder 长度
        N_steps:       混合优化总轮数
        learning_rate: 一阶优化学习率
        tau_noise:     固定噪声时间步（仅在 use_tau_scheduling=False 时生效）
        device:        计算设备
        loss_weights:  Loss 权重字典（仅在 use_tau_scheduling=False 时生效）
        K_top:         Top-K 候选突变数
        B_candidates:  每轮每条分支生成的候选序列数（即 K'
        T_sample:      候选采样温度
        max_mutations_per_seq: 每条候选最大突变数
        use_annealing: 启用模拟退火接受（否则纯贪心）
        T_init:        退火初始温度
        T_min:         退火最低温度
        use_tau_scheduling: 启用 Coarse-to-Fine tau 动态调度
        tau_jitter:    启用 tau 抖动
        init_binder_seq: 可选的初始 Binder 单字母序列
        num_beams:     Beam width（演化路径数量）；默认为 1（单路径）；>=2 时启用 Beam Search
        use_prior:     启用 BLOSUM62 先验约束掩码 M_prior
        prior_strength: 先验权重系数
        use_tau_fusion: 启用多频段 τ 融合（粗+细梯度融合）
        tau_coarse:     τ融合粗粒度时间步
        tau_fine:       τ融合细粒度时间步
        alpha:          τ融合粗粒度权重系数
        use_macro_breathing: 启用宏观呼吸机制（停滞检测 + τ 回退）
        breath_patience:       触发呼吸的连续停滞步数
        breath_tau_recovery:   呼吸状态下 tau 回退目标值
        breath_con_factor:     呼吸状态下 con/i_con 权重缩减系数
        breath_recovery_steps: 呼吸状态持续步数
        checkpoint_dir:         每步保存 checkpoint 的根目录，None 则不保存
        checkpoint_save_policy: checkpoint 保存策略；"all" 保存全部，
                                "best_only" 只保存刷新 best loss 的帧

    Returns:
        best_seq:        零阶验证下的最优序列
        final_seq:       最后一轮的全局最佳序列
        losses_history:  每轮零阶验证最优 loss 的历史
    """
    from bindcraft.hallucination.scheduler import AnnealingScheduler, TauScheduler, MacroBreathTracker

    if device is None:
        device = next(model.parameters()).device
    set_global_seed(seed)

    base_weights = dict(DEFAULT_LOSS_WEIGHTS)
    if loss_weights is not None:
        base_weights.update(loss_weights)
    contact_params = dict(DEFAULT_CONTACT_PARAMS)
    target_sequences = _normalize_chain_sequences(target_seq)
    target_total_len = sum(len(seq) for seq in target_sequences)
    target_len_str = "+".join(str(len(seq)) for seq in target_sequences)

    annealer = AnnealingScheduler(T_init=T_init, T_min=T_min) if use_annealing else None
    tau_sched = TauScheduler(jitter=tau_jitter) if use_tau_scheduling else None
    breath_tracker = MacroBreathTracker(
        patience=breath_patience,
        tau_recovery=breath_tau_recovery,
        con_factor=breath_con_factor,
        recovery_steps=breath_recovery_steps,
    ) if use_macro_breathing else None

    ckpt_mgr = None
    if checkpoint_dir is not None:
        ckpt_mgr = CheckpointManager(
            output_dir=checkpoint_dir,
            binder_seq="PLACEHOLDER",  # 实际序列每步不同，atom_array 动态构建
            target_seq=target_seq,
            save_policy=checkpoint_save_policy,
        )
        logger.info(f"Checkpoint 保存已启用: {checkpoint_dir} | policy={checkpoint_save_policy}")

    logger.info("=" * 70)
    logger.info("开始混合优化循环 (Hybrid: 1st-Order Proposal + 0th-Order Verification)")
    logger.info(
        f"Target chains: {len(target_sequences)} | lengths={target_len_str} | total={target_total_len} aa"
    )
    logger.info(f"Binder 长度: {L_binder} aa")
    logger.info(f"总轮数: {N_steps} | K_top={K_top} B={B_candidates} T_sample={T_sample}")
    logger.info(f"Annealing: {use_annealing} (T_init={T_init}, T_min={T_min})")
    logger.info(f"Tau scheduling: {use_tau_scheduling} (jitter={tau_jitter})")
    if use_macro_breathing:
        logger.info(
            f"MacroBreathing: enabled (patience={breath_patience}, "
            f"tau_recovery={breath_tau_recovery}, con_factor={breath_con_factor}, "
            f"recovery_steps={breath_recovery_steps})"
        )
    else:
        logger.info("MacroBreathing: disabled")
    if num_beams > 1:
        logger.info(f"Beam Search: num_beams={num_beams} (每轮产生 {num_beams}×{B_candidates}={num_beams*B_candidates} 条候选)")
    else:
        logger.info("Beam Search: 关闭（单路径模式）")
    logger.info(f"M_prior (BLOSUM62): use_prior={use_prior}, prior_strength={prior_strength}")
    if target_chain_roles is not None:
        logger.info("Target chain roles: %s", target_chain_roles)
    if use_tau_fusion:
        logger.info(f"τ-fusion: tau_coarse={tau_coarse}, tau_fine={tau_fine}, alpha={alpha}")
    else:
        logger.info("τ-fusion: 关闭")
    logger.info(f"Base loss 权重: {base_weights}")
    logger.info("Step 0: 零阶全链路初始化，将在优化循环前执行完整 AF3 推断获取初始锚点骨架")
    logger.info("=" * 70)

    freeze_model(model)

    # ── 预计算 W_AAE ──────────────────────────────────────────────────────
    logger.info("预计算 W_AAE...")
    W_AAE = precompute_waa_embeddings(model, device)

    # ── Beam 初始化 ─────────────────────────────────────────────────────────
    # 每个 beam 维护独立的 theta、optimizer、accepted 状态
    # num_beams=1 时退化为标准单路径（行为与之前完全一致）
    thetas = []
    optimizers = []
    beam_seqs = []        # 各 beam 当前接受的序列（str）
    beam_losses = []     # 各 beam 当前 accepted loss

    set_global_seed(seed)
    init_binder = None
    for b in range(num_beams):
        theta_b, init_binder_b = _initialize_theta(
            L_binder=L_binder,
            device=device,
            init_binder_seq=init_binder_seq,
        )
        thetas.append(theta_b)
        optimizers.append(torch.optim.Adam([theta_b], lr=learning_rate))
        if init_binder is None:
            init_binder = init_binder_b

    # ── 获取冻结的 Target 特征 ────────────────────────────────────────────
    logger.info("获取 Target 特征（target-only 单链）...")
    with torch.no_grad():
        feat_target_only, _, _ = build_single_chain_features(target_seq, device)
        feat_target_only = _prepare_input_features(model, feat_target_only)
        s_target = model.input_embedder(feat_target_only, inplace_safe=False)
        s_target = s_target.clone()
    logger.info(f"s_target shape: {s_target.shape}")

    init_source = "指定" if init_binder_seq is not None else "随机"
    logger.info(f"初始 Binder 序列（{init_source}）: {init_binder}")

    # 初始序列复制到所有 beam
    for b in range(num_beams):
        beam_seqs.append(init_binder)
        beam_losses.append(float("inf"))

    # ── 获取 bin 参数 ─────────────────────────────────────────────────────
    plddt_params = sample_confidence.get_bin_params(model.configs.loss.plddt)
    pae_params = sample_confidence.get_bin_params(model.configs.loss.pae)
    distogram_params = sample_confidence.get_bin_params(model.configs.loss.distogram)
    pde_params = sample_confidence.get_bin_params(model.configs.loss.pde)

    # ── Step 0: 零阶全链路初始化（与算法指导文档 5.1 节对齐）───────────────
    logger.info("=" * 70)
    logger.info("Step 0: 零阶全链路初始化，获取初始真实物理锚点骨架...")
    init_zeroth = zeroth_order_eval(
        binder_seq=init_binder,
        target_seq=target_seq,
        model=model,
        device=device,
        loss_weights=base_weights,
        contact_params=contact_params,
        hotspot_params=hotspot_params,
        seed=_verifier_seed(seed, target_seq, init_binder),
        target_template_path=target_template_path,
        target_chain_roles=target_chain_roles,
    )
    logger.info(
        f"  初始零阶 total_loss={init_zeroth.total_loss:.4f} | "
        f"pLDDT={init_zeroth.summary_confidence.get('plddt', 'N/A')} | "
        f"iPTM={init_zeroth.summary_confidence.get('i_ptm', 'N/A')}"
    )
    X_clean_0 = init_zeroth.coordinate[0:1]  # [1, N_atom, 3]
    # 每个 beam 维护独立的骨架缓存，第一轮从 Step 0 零阶骨架起步
    x_coarse_caches = [X_clean_0.clone().to(device) for _ in range(num_beams)]
    logger.info(f"X_clean_0 已注入骨架缓存，shape={X_clean_0.shape}，num_beams={num_beams}")
    logger.info("=" * 70)

    # 所有 beam 用同一个初始 loss 初始化
    for b in range(num_beams):
        beam_losses[b] = init_zeroth.total_loss
    if breath_tracker is not None:
        breath_tracker._best_loss = init_zeroth.total_loss

    # ── Step 0 Checkpoint ──────────────────────────────────────────────────
    if ckpt_mgr is not None:
        ckpt_mgr.save(
            step=0,
            beam_id=0,
            zeroth_result=init_zeroth,
            plddt_logits=init_zeroth.plddt_logits,
            accepted=True,
            is_global_best=True,
        )
        logger.info(f"[Checkpoint] Step 0 保存完毕: step_0000/")

    # ── 状态变量 ──────────────────────────────────────────────────────────
    best_zeroth_loss = init_zeroth.total_loss
    best_zeroth_seq = init_binder
    best_step = 0
    losses_history = []
    n_probabilistic_accepts = 0

    logger.info("开始混合优化主循环...")

    for step in range(N_steps):
        # ================================================================
        # 动态 tau 和 loss 权重
        # ================================================================
        if tau_sched is not None:
            step_tau = tau_sched.get_tau(step, N_steps)
            step_weights = tau_sched.get_dynamic_weights(step, N_steps, base_weights=base_weights)
            phase_name = tau_sched.get_phase_name(step, N_steps)
        else:
            step_tau = tau_noise
            step_weights = base_weights
            phase_name = "fixed"

        # ── 宏观呼吸覆盖（优先于一切调度）──────────────────────────────
        if breath_tracker is not None and breath_tracker.is_breathing:
            step_tau = breath_tracker.get_breathing_tau()
            step_weights = breath_tracker.apply_breathing_weights(step_weights)
            phase_name = "macro_breathing"

        anneal_T = annealer.get_temperature(step, N_steps) if annealer else 0.0

        # ================================================================
        # Step 1: 每个分支独立一阶反传（单路径时只跑第0个分支）
        # ================================================================
        step_grads = []      # 各 beam 的梯度
        step_fo_losses = []  # 各 beam 的一阶 loss
        step_current_seqs = []  # 各 beam 当前一阶序列

        for b in range(num_beams):
            optimizers[b].zero_grad()

            progress = step / max(N_steps - 1, 1)
            tau_temp = 2.0 * (1.0 - progress) + 0.05 * progress

            s_binder, current_seq, input_feature_dict = ste_bridge(
                theta=thetas[b], W_AAE=W_AAE, model=model,
                target_seq=target_seq, device=device, tau_temp=tau_temp,
                target_template_path=target_template_path,
            )

            if use_tau_fusion:
                # 多频段 τ 融合：粗粒度 + 细粒度双前向，融合梯度
                grad_fused, x0_fine = diffusion_one_step_fused(
                    model=model,
                    s_binder=s_binder,
                    s_target=s_target,
                    input_feature_dict=input_feature_dict,
                    x_coarse=None,
                    X_clean=x_coarse_caches[b],
                    theta=thetas[b],
                    step_weights=step_weights,
                    contact_params=contact_params,
                    plddt_params=plddt_params,
                    pae_params=pae_params,
                    distogram_params=distogram_params,
                    L_binder=L_binder,
                    tau_coarse=tau_coarse,
                    tau_fine=tau_fine,
                    alpha=alpha,
                    device=device,
                    pde_params=pde_params,
                    clash_params=dict(DEFAULT_CLASH_PARAMS),
                    hotspot_params=hotspot_params,
                )
                torch.nn.utils.clip_grad_norm_([thetas[b]], 1.0)
                optimizers[b].step()
                fo_loss_val = None  # tau_fusion 模式不单独记录单条 loss
                grad_for_proposal = grad_fused
            else:
                # 标准单前向模式
                x0_hat, s_trunk, z_trunk = diffusion_one_step(
                    model=model, s_binder=s_binder, s_target=s_target,
                    input_feature_dict=input_feature_dict,
                    x_coarse=None,
                    X_clean=x_coarse_caches[b],
                    tau=step_tau, device=device,
                )

                x0_flat = x0_hat.squeeze(0)
                s_inputs_full = torch.cat([s_binder, s_target], dim=0)
                heads = run_confidence_heads(
                    model=model, input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs_full, s_trunk=s_trunk, z_trunk=z_trunk,
                    x0_hat=x0_hat, inplace_safe=False, chunk_size=None,
                )

                first_order_loss, first_order_losses = compute_all_losses(
                    x0_flat=x0_flat,
                    plddt_logits=heads["plddt_logits"],
                    pae_logits=heads["pae_logits"],
                    distogram_logits=heads["distogram_logits"],
                    input_feature_dict=input_feature_dict,
                    L_binder=L_binder, weights=step_weights,
                    contact_params=contact_params,
                    plddt_params=plddt_params,
                    pae_params=pae_params,
                    distogram_params=distogram_params,
                    pde_logits=heads["pde_logits"],
                    pde_params=pde_params,
                    clash_params=dict(DEFAULT_CLASH_PARAMS),
                    hotspot_params=hotspot_params,
                    target_chain_roles=target_chain_roles,
                )

                first_order_loss.backward()
                grad_for_proposal = thetas[b].grad.clone() if thetas[b].grad is not None else None
                torch.nn.utils.clip_grad_norm_([thetas[b]], 1.0)
                optimizers[b].step()
                fo_loss_val = first_order_loss.item()

            step_grads.append(grad_for_proposal)
            step_fo_losses.append(fo_loss_val)
            step_current_seqs.append(current_seq)

        # ================================================================
        # Step 2: 收集全部候选（num_beams × B_candidates 条）
        # ================================================================
        all_candidates = []      # List[str]
        all_candidate_beams = [] # List[int] - 每条候选属于哪个 beam

        for b in range(num_beams):
            if step_grads[b] is None:
                continue
            grads_cpu = step_grads[b].detach().cpu()
            # 各分支独立采样时 current_seq 用该分支的
            cands = propose_mutations(
                grad_matrix=grads_cpu,
                current_seq=step_current_seqs[b],
                K_top=K_top,
                B_candidates=B_candidates,
                T_sample=T_sample,
                max_mutations_per_seq=max_mutations_per_seq,
                use_prior=use_prior,
                prior_strength=prior_strength,
            )
            for cand in cands:
                all_candidates.append(cand)
                all_candidate_beams.append(b)

        if not all_candidates:
            logger.warning(f"Step {step}: no candidates generated, skipping")
            losses_history.append(min(step_fo_losses) if step_fo_losses else float("inf"))
            continue

        # ================================================================
        # Step 3: 零阶验证全部候选
        # ================================================================
        all_zeroth_results = []  # List[ZerothOrderResult]

        for idx, cand_seq in enumerate(all_candidates):
            result = zeroth_order_eval(
                binder_seq=cand_seq,
                target_seq=target_seq,
                model=model,
                device=device,
                loss_weights=step_weights,
                contact_params=contact_params,
                hotspot_params=hotspot_params,
                seed=_verifier_seed(seed, target_seq, cand_seq),
                target_template_path=target_template_path,
                target_chain_roles=target_chain_roles,
            )
            all_zeroth_results.append(result)
            beam_src = all_candidate_beams[idx]
            logger.debug(
                f"  Step {step} | cand {idx}/{len(all_candidates)} "
                f"(beam={beam_src}) | seq={cand_seq[:15]}... | zo={result.total_loss:.4f}"
            )

        # ── 宏观呼吸检测（在下一轮开始前更新）─────────────────────────────
        # 用本轮最优候选 loss 更新呼吸跟踪器，决定下一轮是否触发呼吸状态
        if breath_tracker is not None:
            round_min_loss = min(r.total_loss for r in all_zeroth_results)
            breath_tracker.update(round_min_loss)

        # ================================================================
        # Step 4: Beam Search 全局排序（num_beams=1 时退化为贪心）
        # ================================================================
        if num_beams == 1:
            # 单路径：与原行为一致，在 step 0 的 zeroth 结果 + 当前候选中选最优
            round_best = min(all_zeroth_results, key=lambda r: r.total_loss)
            round_best_loss = round_best.total_loss

            accepted = False
            prob_accept = False

            if annealer is not None:
                accepted = annealer.accept(
                    beam_losses[0], round_best_loss, step, N_steps,
                )
                if accepted and round_best_loss >= beam_losses[0]:
                    prob_accept = True
                    n_probabilistic_accepts += 1
            else:
                accepted = round_best_loss < best_zeroth_loss

            if accepted:
                beam_losses[0] = round_best_loss
                beam_seqs[0] = round_best.binder_seq
                # 更新骨架缓存：使用被接受序列的零阶全链路骨架
                x_coarse_caches[0] = round_best.coordinate[0:1].clone().to(device)
                if round_best_loss < best_zeroth_loss:
                    best_zeroth_loss = round_best_loss
                    best_zeroth_seq = round_best.binder_seq
                    best_step = step
                _sync_theta_to_seq(thetas[0], beam_seqs[0])
                optimizers[0].state.clear()
                optimizers[0] = torch.optim.Adam([thetas[0]], lr=learning_rate)

            losses_history.append(round_best_loss if accepted else beam_losses[0])
            fo_val = step_fo_losses[0]
            logger.info(
                f"Step {step:3d}/{N_steps} | phase={phase_name} | tau={step_tau:.3f} | T={anneal_T:.4f} | "
                f"1st={'N/A(τ-fusion)' if fo_val is None else f'{fo_val:.4f}'} | 0th_best={round_best_loss:.4f} | "
                f"global_best={best_zeroth_loss:.4f}@{best_step} | "
                f"accepted={accepted}{' (prob)' if prob_accept else ''}"
            )

            # ── Checkpoint 保存（本轮最优候选，无论是否接受）───────────────
            if ckpt_mgr is not None:
                ckpt_mgr.save(
                    step=step,
                    beam_id=0,
                    zeroth_result=round_best,
                    plddt_logits=round_best.plddt_logits,
                    accepted=accepted,
                    is_global_best=(round_best_loss == best_zeroth_loss),
                )
                best_info = ckpt_mgr.get_best_info()
                logger.info(
                    f"  [Checkpoint] 保存 step={step} | best={best_info['loss']:.4f} @ "
                    f"step={best_info['step']} beam={best_info['beam']}"
                )

        else:
            # Beam Search：全部候选全局排序，保留前 num_beams 条
            # 按 zeroth_loss 排序（越小越好）
            sorted_indices = sorted(
                range(len(all_zeroth_results)),
                key=lambda i: all_zeroth_results[i].total_loss
            )
            top_beam_indices = sorted_indices[:num_beams]

            step_accepted_count = 0
            beam_accepted_map: Dict[int, bool] = {}   # beam_id → 是否被接受
            for rank, cand_idx in enumerate(top_beam_indices):
                src_beam = all_candidate_beams[cand_idx]
                result = all_zeroth_results[cand_idx]

                accepted = False
                if annealer is not None:
                    accepted = annealer.accept(
                        beam_losses[src_beam], result.total_loss, step, N_steps,
                    )
                    if accepted and result.total_loss >= beam_losses[src_beam]:
                        n_probabilistic_accepts += 1
                else:
                    accepted = result.total_loss < best_zeroth_loss

                beam_accepted_map[src_beam] = accepted

                if accepted:
                    beam_losses[src_beam] = result.total_loss
                    beam_seqs[src_beam] = result.binder_seq
                    # 更新该 beam 的骨架缓存：使用被接受序列的零阶全链路骨架
                    x_coarse_caches[src_beam] = result.coordinate[0:1].clone().to(device)
                    step_accepted_count += 1
                    if result.total_loss < best_zeroth_loss:
                        best_zeroth_loss = result.total_loss
                        best_zeroth_seq = result.binder_seq
                        best_step = step
                    _sync_theta_to_seq(thetas[src_beam], result.binder_seq)
                    optimizers[src_beam].state.clear()
                    optimizers[src_beam] = torch.optim.Adam([thetas[src_beam]], lr=learning_rate)

            losses_history.append(min(beam_losses))
            valid_fo = [v for v in step_fo_losses if v is not None]
            fo_val_str = "N/A(τ-fusion)" if not valid_fo else f"{min(valid_fo):.4f}"
            beam_loss_str = " | ".join(
                f"beam{b}={beam_losses[b]:.4f}" for b in range(num_beams)
            )
            logger.info(
                f"Step {step:3d}/{N_steps} | phase={phase_name} | tau={step_tau:.3f} | T={anneal_T:.4f} | "
                f"1st_best={fo_val_str} | global_best={best_zeroth_loss:.4f}@{best_step} | "
                f"top_k={len(all_candidates)} accepted={step_accepted_count}/{num_beams} | {beam_loss_str}"
            )

            # ── Checkpoint 保存（每个 top beam，不管是否接受都保存）─────────
            if ckpt_mgr is not None:
                for rank, cand_idx in enumerate(top_beam_indices):
                    src_beam = all_candidate_beams[cand_idx]
                    result = all_zeroth_results[cand_idx]
                    accepted = beam_accepted_map.get(src_beam, False)
                    is_best = (result.total_loss == best_zeroth_loss)
                    ckpt_mgr.save(
                        step=step,
                        beam_id=src_beam,
                        zeroth_result=result,
                        plddt_logits=result.plddt_logits,
                        accepted=accepted,
                        is_global_best=is_best,
                    )
                best_info = ckpt_mgr.get_best_info()
                logger.info(
                    f"  [Checkpoint] 保存 {step_accepted_count}/{num_beams} beam(s) | "
                    f"global_best={best_info['loss']:.4f} @ step={best_info['step']} beam={best_info['beam']}"
                )

    logger.info("=" * 70)
    logger.info("混合优化完成！")
    logger.info(f"零阶最佳序列: {best_zeroth_seq} (loss={best_zeroth_loss:.4f} @step {best_step})")
    logger.info(f"概率接受次数: {n_probabilistic_accepts}")
    if num_beams > 1:
        for b in range(num_beams):
            logger.info(f"  Beam {b}: {beam_seqs[b]} (loss={beam_losses[b]:.4f})")
    logger.info("=" * 70)

    return best_zeroth_seq, best_zeroth_seq, losses_history


# ============================================================
# MCTS 模式：全局资源调度器
# ============================================================


def _make_phase0_seeds(
    init_binder_seq: Optional[str],
    num_beams: int,
    L_binder: int,
    seed: Optional[int] = 42,
) -> List[str]:
    """
    Phase 0 冷启动序列列表。
    - 若提供 init_binder_seq：第 1 条为规范化种子，其余为全序列上随机 1–2 位突变拷贝。
    - 否则：与旧版一致，随机生成 num_beams 条全长序列。
    """
    rng = np.random.RandomState(int(seed) if seed is not None else None)
    if init_binder_seq is not None:
        seed = _normalize_init_binder_seq(init_binder_seq, L_binder)
        seeds = [seed]
        for _ in range(max(0, num_beams - 1)):
            cand = list(seed)
            n_mut = int(rng.randint(1, 3))  # 1 或 2
            positions = rng.choice(L_binder, size=min(n_mut, L_binder), replace=False)
            for pos in positions:
                cand[pos] = str(rng.choice(list(VALID_BINDER_AAS)))
            seeds.append("".join(cand))
        return seeds
    return [
        "".join(str(rng.choice(list(VALID_BINDER_AAS))) for _ in range(L_binder))
        for _ in range(num_beams)
    ]


def _init_child_logits_from_parent(
    parent_node: Any,
    child_seq: str,
    device: torch.device,
) -> torch.Tensor:
    """子节点 Logits：克隆父 logits 后同步到子离散序列。"""
    child_logits = parent_node.logits.clone().to(device)
    _sync_theta_to_seq(child_logits, child_seq)
    return child_logits


VALID_MCTS_STRATEGIES = {"legacy", "strict", "two_stage"}


def _normalize_mcts_strategy(strategy: str) -> str:
    normalized = str(strategy or "two_stage").strip().lower()
    if normalized not in VALID_MCTS_STRATEGIES:
        raise ValueError(
            f"Unsupported mcts_strategy={strategy!r}; "
            f"expected one of {sorted(VALID_MCTS_STRATEGIES)}"
        )
    return normalized


def _make_stage_checkpoint_manager(
    checkpoint_dir: Optional[str],
    stage_name: str,
    strategy: str,
    target_seq: Union[str, List[str]],
    checkpoint_save_policy: str = "all",
) -> Optional[CheckpointManager]:
    if checkpoint_dir is None:
        return None
    output_dir = checkpoint_dir
    if strategy == "two_stage":
        output_dir = os.path.join(checkpoint_dir, stage_name)
    manager = CheckpointManager(
        output_dir=output_dir,
        binder_seq="PLACEHOLDER",
        target_seq=target_seq,
        save_policy=checkpoint_save_policy,
    )
    logger.info(
        "Checkpoint 保存已启用[%s]: %s | policy=%s",
        stage_name,
        output_dir,
        checkpoint_save_policy,
    )
    return manager


def _select_top_unique_results(
    verifier_cache: Dict[str, Any],
    top_k: int,
) -> List[Any]:
    selected = []
    seen = set()
    for result in sorted(verifier_cache.values(), key=lambda item: float(item.total_loss)):
        seq = str(result.binder_seq)
        if seq in seen:
            continue
        seen.add(seq)
        selected.append(result)
        if len(selected) >= top_k:
            break
    return selected


def _hamming_distance(seq_a: str, seq_b: str) -> int:
    shared = min(len(seq_a), len(seq_b))
    mismatches = sum(1 for idx in range(shared) if seq_a[idx] != seq_b[idx])
    return mismatches + abs(len(seq_a) - len(seq_b))


def _handoff_structural_key(result: Any) -> Tuple[str, ...]:
    summary = getattr(result, "summary_confidence", None) or {}
    keys = (
        "hotspot_contact_bitset",
        "interface_patch_id",
        "secondary_structure_pattern",
        "contact_fingerprint",
    )
    return tuple(str(summary.get(key, "unknown")) for key in keys)


def _select_diverse_handoff_results(
    verifier_cache: Dict[str, Any],
    top_k: int,
    min_distance_frac: float = MCTS_STAGE2_HANDOFF_MIN_DISTANCE_FRAC,
) -> List[Any]:
    """
    Pick low-loss, non-duplicate StageA results for StageB roots.

    The first pass greedily enforces sequence distance and, when available,
    structural signature diversity. A second pass fills any remaining roots by
    loss so handoff never fails just because StageA explored one basin.
    """
    candidates = _select_top_unique_results(verifier_cache, max(len(verifier_cache), top_k))
    if top_k <= 0 or not candidates:
        return []

    selected: List[Any] = []
    selected_structural = set()
    seq_len = max(len(str(candidates[0].binder_seq)), 1)
    min_distance = max(1, int(math.ceil(seq_len * max(float(min_distance_frac), 0.0))))

    for result in candidates:
        seq = str(result.binder_seq)
        structural_key = _handoff_structural_key(result)
        distance_ok = all(
            _hamming_distance(seq, str(existing.binder_seq)) >= min_distance
            for existing in selected
        )
        structural_ok = (
            structural_key not in selected_structural
            or all(value == "unknown" for value in structural_key)
        )
        if selected and not (distance_ok or structural_ok):
            continue
        selected.append(result)
        selected_structural.add(structural_key)
        if len(selected) >= top_k:
            return selected

    selected_sequences = {str(result.binder_seq) for result in selected}
    for result in candidates:
        if str(result.binder_seq) in selected_sequences:
            continue
        selected.append(result)
        selected_sequences.add(str(result.binder_seq))
        if len(selected) >= top_k:
            break
    return selected


def _stageb_max_mutations_for_step(
    base_max_mutations: int,
    local_step: int,
    total_steps: int,
    stale_steps: int,
) -> int:
    """
    Document edit-size schedule for StageB de novo search.

    Early search uses 1/2-site edits, stale search can try 2/3-site edits, and
    the late phase returns to mostly 1-site refinement.
    """
    base = max(int(base_max_mutations), 1)
    if base <= 1:
        return 1

    progress = local_step / max(total_steps - 1, 1)
    if progress >= 0.75:
        return 1
    if stale_steps >= max(MCTS_STAGE2_STALE_EDIT_STEPS, 1):
        return min(base, 3)
    if progress < 0.40:
        return min(base, 2)
    return min(base, 2)


STAGEB_PROPOSAL_LOSS_WEIGHTS = {
    "core_stabilization": {
        "plddt": 0.20,
        "pae": 0.40,
        "pde": 0.05,
        "con": 1.00,
        "rg": 0.10,
    },
    "compactness_repair": {
        "rg": 0.60,
        "pae": 0.20,
        "con": 0.30,
    },
    "interface_formation": {
        "i_pae": 0.30,
        "i_pde": 0.05,
        "i_con": 1.50,
        "iptm": 0.10,
    },
    "clash_repair": {
        "clash": 1.00,
        "i_clash": 1.00,
    },
    "specificity_repair": {
        "i_con": 0.50,
        "i_pae": 0.20,
        "i_pde": 0.05,
    },
    "solubility_repair": {
        "rg": 0.40,
        "plddt": 0.10,
        "pae": 0.10,
    },
}


def _stageb_group_proposal_loss(
    losses: Dict[str, torch.Tensor],
    action_group: str,
    fallback_loss: torch.Tensor,
) -> Tuple[torch.Tensor, str]:
    """
    Build the one-step proposal loss used only for mutation gradients.

    Full verifier loss still supplies tree reward. This helper makes the
    gradient proposal match the selected repair direction: interface gradients
    emphasize interface losses, clash repair emphasizes clash terms, and core
    repair emphasizes fold/core terms.
    """
    weights = STAGEB_PROPOSAL_LOSS_WEIGHTS.get(action_group or "", {})
    selected_terms: List[str] = []
    proposal_loss: Optional[torch.Tensor] = None
    for key, weight in weights.items():
        if key not in losses:
            continue
        term = float(weight) * losses[key]
        proposal_loss = term if proposal_loss is None else proposal_loss + term
        selected_terms.append(key)

    if proposal_loss is None:
        return fallback_loss, "fallback_total"
    return proposal_loss, "+".join(selected_terms)


def _stageb_proposal_tau(
    proposal_method: str,
    base_tau: float,
    local_step: int,
    total_steps: int,
    stale_steps: int = 0,
    adaptive_tau_start: float = 160.0,
    adaptive_tau_end: float = 0.1,
) -> float:
    """
    Resolve the diffusion tau used by StageB proposal gradients.

    ``one_step_fixed`` and ``two_step_truncated`` preserve the historical
    ``base_tau`` behavior. ``one_step_adaptive`` follows the Protenix EDM noise
    scale more closely: it starts from a high-noise proposal and anneals to a
    small positive floor. The floor must stay above zero because Protenix
    embeds ``log(tau / sigma_data)`` inside the diffusion conditioning.
    """
    proposal_method = normalize_mcts_proposal_method(proposal_method)
    base_tau = float(base_tau)
    if proposal_method != "one_step_adaptive":
        return base_tau

    progress = float(local_step) / max(float(total_steps - 1), 1.0)
    progress = min(max(progress, 0.0), 1.0)

    tau_start = max(float(adaptive_tau_start), 1e-6)
    tau_end = max(float(adaptive_tau_end), 1e-6)
    tau_high = max(tau_start, tau_end)
    tau_low = min(tau_start, tau_end)

    tau = tau_start + (tau_end - tau_start) * progress
    return float(min(max(tau, tau_low), tau_high))


def _make_strict_mcts_tree(
    mcts_C: float,
    mcts_max_nodes: int,
    mcts_tau_fixed: float,
    expand_actions_per_step: int,
):
    from bindcraft.hallucination.mcts_controller import MCTSTree

    return MCTSTree(
        C=mcts_C,
        max_nodes=mcts_max_nodes,
        mcts_tau=mcts_tau_fixed,
        progressive_widening_k=MCTS_STAGE2_PROGRESSIVE_WIDENING_K,
        progressive_widening_alpha=MCTS_STAGE2_PROGRESSIVE_WIDENING_ALPHA,
        max_children_per_node=MCTS_STAGE2_MAX_CHILDREN_PER_NODE,
        min_children_before_descend=MCTS_STAGE2_MIN_CHILDREN_BEFORE_DESCEND,
        descend_score_margin=MCTS_STAGE2_DESCEND_SCORE_MARGIN,
        expand_actions_per_step=expand_actions_per_step,
        use_loss_grouping=MCTS_STAGE3_USE_LOSS_GROUPING,
        representatives_per_group=MCTS_STAGE3_REPRESENTATIVES_PER_GROUP,
        loss_group_bin_width=MCTS_STAGE3_LOSS_GROUP_BIN_WIDTH,
        use_reward_normalization=MCTS_STAGE3_USE_REWARD_NORMALIZATION,
        use_pareto_pruning=MCTS_STAGE3_USE_PARETO_PRUNING,
        use_structural_grouping=MCTS_STAGE3_USE_STRUCTURAL_GROUPING,
        use_diversity_bonus=MCTS_STAGE3_USE_DIVERSITY_BONUS,
        diversity_bonus_weight=MCTS_STAGE3_DIVERSITY_BONUS_WEIGHT,
        diversity_bonus_max=MCTS_STAGE3_DIVERSITY_BONUS_MAX,
    )


def run_mcts_hallucination(
    model,
    target_seq: Union[str, List[str]],
    L_binder: int = 40,
    N_steps: int = 100,
    learning_rate: float = 0.05,
    tau_noise: float = 4.0,
    device: torch.device = None,
    loss_weights: Dict[str, float] = None,
    K_top: int = 10,
    B_candidates: int = 8,
    T_sample: float = 1.0,
    max_mutations_per_seq: int = 3,
    use_annealing: bool = False,
    T_init: float = 1.0,
    T_min: float = 0.01,
    num_beams: int = 8,
    use_prior: bool = True,
    prior_strength: float = 0.1,
    checkpoint_dir: str = None,
    checkpoint_save_policy: str = "all",
    init_binder_seq: Optional[str] = None,
    hotspot_params: Optional[Dict] = None,
    chunk_size: int = None,
    mcts_C: float = 1.0,
    mcts_max_nodes: int = 5000,
    mcts_tau_fixed: float = 4.0,
    mcts_strategy: str = "two_stage",
    mcts_bootstrap_steps: int = 50,
    mcts_bootstrap_patience: int = 10,
    mcts_bootstrap_min_delta: float = 0.02,
    mcts_bootstrap_max_steps: int = 80,
    mcts_handoff_top_k: Optional[int] = None,
    mcts_expand_actions_per_step: int = MCTS_STAGE2_EXPAND_ACTIONS_PER_STEP,
    mcts_prior_mode: str = "full",
    mcts_proposal_method: str = "one_step_fixed",
    mcts_adaptive_tau_start: float = 160.0,
    mcts_adaptive_tau_end: float = 0.1,
    seed: Optional[int] = 42,
    target_template_path: Optional[str] = None,
    target_chain_roles: Optional[Any] = None,
    proposal_amp: str = "auto",
) -> Tuple[str, str, List[float]]:
    """
    MCTS 调度的混合优化：UCB 选点 + 一阶提议 + 零阶验证 + Expansion/Backprop/Prune。
    Phase 0 使用 ``--init_binder_seq`` 单种子 + 随机突变扩展（与 hybrid 的 init 语义对齐）。
    """
    from bindcraft.hallucination.mcts_legacy import LegacyMCTSTree
    from bindcraft.hallucination.scheduler import AnnealingScheduler

    if device is None:
        device = next(model.parameters()).device
    set_global_seed(seed)

    mcts_strategy = _normalize_mcts_strategy(mcts_strategy)
    mcts_bootstrap_steps = max(0, int(mcts_bootstrap_steps))
    mcts_bootstrap_patience = max(1, int(mcts_bootstrap_patience))
    mcts_bootstrap_min_delta = max(float(mcts_bootstrap_min_delta), 0.0)
    mcts_bootstrap_max_steps = max(mcts_bootstrap_steps, int(mcts_bootstrap_max_steps))
    mcts_handoff_top_k = max(1, int(mcts_handoff_top_k or num_beams))
    mcts_expand_actions_per_step = max(int(mcts_expand_actions_per_step), 1)
    mcts_prior_mode = normalize_mcts_prior_mode(mcts_prior_mode)
    mcts_proposal_method = normalize_mcts_proposal_method(mcts_proposal_method)
    mcts_adaptive_tau_start = float(mcts_adaptive_tau_start)
    mcts_adaptive_tau_end = float(mcts_adaptive_tau_end)
    model_name_for_amp = str(
        getattr(model, "bindcraft_model_name", None)
        or getattr(getattr(model, "configs", None), "model_name", "")
    )
    proposal_amp_dtype = _resolve_proposal_amp_dtype(model_name_for_amp, proposal_amp)

    base_weights = dict(DEFAULT_LOSS_WEIGHTS)
    if loss_weights is not None:
        base_weights.update(loss_weights)
    contact_params = dict(DEFAULT_CONTACT_PARAMS)

    target_sequences = _normalize_chain_sequences(target_seq)
    target_len_str = "+".join(str(len(s)) for s in target_sequences)
    target_total_len = sum(len(s) for s in target_sequences)

    annealer = AnnealingScheduler(T_init=T_init, T_min=T_min) if use_annealing else None

    if mcts_strategy == "two_stage":
        phase0_ckpt_mgr = _make_stage_checkpoint_manager(
            checkpoint_dir, "phase0", mcts_strategy, target_seq, checkpoint_save_policy
        )
        legacy_ckpt_mgr = _make_stage_checkpoint_manager(
            checkpoint_dir, "legacy_bootstrap", mcts_strategy, target_seq, checkpoint_save_policy
        )
        strict_ckpt_mgr = _make_stage_checkpoint_manager(
            checkpoint_dir, "document_puct_refine", mcts_strategy, target_seq, checkpoint_save_policy
        )
    elif mcts_strategy == "legacy":
        legacy_ckpt_mgr = _make_stage_checkpoint_manager(
            checkpoint_dir, "legacy_bootstrap", mcts_strategy, target_seq, checkpoint_save_policy
        )
        phase0_ckpt_mgr = legacy_ckpt_mgr
        strict_ckpt_mgr = None
    else:
        strict_ckpt_mgr = _make_stage_checkpoint_manager(
            checkpoint_dir, "document_puct_refine", mcts_strategy, target_seq, checkpoint_save_policy
        )
        phase0_ckpt_mgr = strict_ckpt_mgr
        legacy_ckpt_mgr = None

    logger.info("=" * 70)
    logger.info("MCTS 调度混合优化 (1st-Order Proposal + 0th-Order Verification)")
    logger.info(
        f"Target chains: {len(target_sequences)} | lengths={target_len_str} | total={target_total_len} aa"
    )
    logger.info(f"Binder 长度: {L_binder} aa")
    logger.info(f"总轮数: {N_steps} | K_top={K_top} B={B_candidates} T_sample={T_sample}")
    logger.info(f"MCTS: C={mcts_C}, max_nodes={mcts_max_nodes}, tau_fixed={mcts_tau_fixed}")
    logger.info(
        "MCTS strategy=%s | bootstrap_steps=%d patience=%d min_delta=%.4f "
        "max_bootstrap=%d handoff_top_k=%d",
        mcts_strategy,
        mcts_bootstrap_steps,
        mcts_bootstrap_patience,
        mcts_bootstrap_min_delta,
        mcts_bootstrap_max_steps,
        mcts_handoff_top_k,
    )
    logger.info("Random seed: %s", seed)
    logger.info(
        "MCTS Stage2: progressive_widening(k=%.1f, alpha=%.2f, max_children=%d) "
        "+ action_puct(expand_per_step=%d) "
        "+ de novo action routing(T=%.2f)",
        MCTS_STAGE2_PROGRESSIVE_WIDENING_K,
        MCTS_STAGE2_PROGRESSIVE_WIDENING_ALPHA,
        MCTS_STAGE2_MAX_CHILDREN_PER_NODE,
        mcts_expand_actions_per_step,
        MCTS_STAGE2_GROUP_TEMPERATURE,
    )
    logger.info(
        "MCTS Stage3-lite: loss_bin_grouping(bin_width=%.2f, representatives_per_group=%d) "
        "+ diversity_bonus(weight=%.3f, max=%.1f)",
        MCTS_STAGE3_LOSS_GROUP_BIN_WIDTH,
        MCTS_STAGE3_REPRESENTATIVES_PER_GROUP,
        MCTS_STAGE3_DIVERSITY_BONUS_WEIGHT,
        MCTS_STAGE3_DIVERSITY_BONUS_MAX,
    )
    logger.info(
        "MCTS switches: loss_grouping=%s structural_grouping=%s pareto=%s "
        "reward_normalization=%s diversity_bonus=%s",
        MCTS_STAGE3_USE_LOSS_GROUPING,
        MCTS_STAGE3_USE_STRUCTURAL_GROUPING,
        MCTS_STAGE3_USE_PARETO_PRUNING,
        MCTS_STAGE3_USE_REWARD_NORMALIZATION,
        MCTS_STAGE3_USE_DIVERSITY_BONUS,
    )
    logger.info(f"Annealing: {use_annealing} (C={mcts_C} is primary exploration control)")
    logger.info(
        "MCTS prior mode: %s | M_prior(BLOSUM62) requested=%s prior_strength=%.3f",
        mcts_prior_mode,
        use_prior,
        prior_strength,
    )
    logger.info("MCTS StageB proposal method: %s", mcts_proposal_method)
    logger.info(
        "Proposal AMP: mode=%s | model=%s | resolved_dtype=%s",
        proposal_amp,
        model_name_for_amp or "unknown",
        proposal_amp_dtype,
    )
    if mcts_proposal_method == "one_step_adaptive":
        logger.info(
            "MCTS StageB adaptive tau schedule: %.3f -> %.3f",
            mcts_adaptive_tau_start,
            mcts_adaptive_tau_end,
        )
    if target_chain_roles is not None:
        logger.info("Target chain roles: %s", target_chain_roles)
    logger.info(f"Base loss 权重: {base_weights}")
    logger.info("=" * 70)

    freeze_model(model)

    logger.info("预计算 W_AAE...")
    W_AAE = precompute_waa_embeddings(model, device)

    logger.info("获取 Target 特征（target-only）...")
    with torch.no_grad():
        feat_target_only, _, _ = build_single_chain_features(target_seq, device)
        feat_target_only = _prepare_input_features(model, feat_target_only)
        s_target = model.input_embedder(feat_target_only, inplace_safe=False)
        s_target = s_target.clone()
    logger.info(f"s_target shape: {s_target.shape}")

    plddt_params = sample_confidence.get_bin_params(model.configs.loss.plddt)
    pae_params = sample_confidence.get_bin_params(model.configs.loss.pae)
    distogram_params = sample_confidence.get_bin_params(model.configs.loss.distogram)
    pde_params = sample_confidence.get_bin_params(model.configs.loss.pde)

    mcts_tree = (
        _make_strict_mcts_tree(
            mcts_C, mcts_max_nodes, mcts_tau_fixed, mcts_expand_actions_per_step
        )
        if mcts_strategy == "strict"
        else None
    )
    if mcts_tree is not None:
        logger.info(
            "Document PUCT MCTS 树已创建: C=%s, max_nodes=%s, tau_fixed=%s",
            mcts_C,
            mcts_max_nodes,
            mcts_tau_fixed,
        )

    # ── Phase 0：全局冷启动 ─────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Phase 0: 全局冷启动 (Best-of-N)")
    template_seqs = _make_phase0_seeds(init_binder_seq, num_beams, L_binder, seed=seed)
    if init_binder_seq is not None:
        logger.info(
            f"使用 init_binder_seq 种子 + 突变扩展，共 {len(template_seqs)} 条 Phase0 候选"
        )
    else:
        logger.info(f"无 init_binder_seq，随机生成 {len(template_seqs)} 条初始序列")

    logger.info(f"对 {len(template_seqs)} 条模板执行 Step3 零阶验证...")
    phase0_results = []
    verifier_cache: Dict[str, Any] = {}
    for idx, tseq in enumerate(template_seqs):
        logger.info(f"  Phase0 [{idx+1}/{len(template_seqs)}]: {tseq[:20]}...")
        result = zeroth_order_eval(
            binder_seq=tseq,
            target_seq=target_seq,
            model=model,
            device=device,
            loss_weights=base_weights,
            contact_params=contact_params,
            chunk_size=chunk_size,
            hotspot_params=hotspot_params,
            seed=_verifier_seed(seed, target_seq, tseq),
            target_template_path=target_template_path,
            target_chain_roles=target_chain_roles,
        )
        phase0_results.append((tseq, result))
        verifier_cache[result.binder_seq] = result
        if mcts_tree is not None:
            mcts_tree.store_verifier_result(result)
        logger.info(
            f"    L_real={result.total_loss:.4f} "
            f"pLDDT={result.summary_confidence.get('plddt', 'N/A')} "
            f"iPTM={result.summary_confidence.get('i_ptm', 'N/A')}"
        )

    phase0_results.sort(key=lambda x: x[1].total_loss)
    top_beam = min(num_beams, len(phase0_results))

    set_global_seed(stable_int_seed(seed, "mcts_after_phase0"))
    for i in range(top_beam):
        seq, result = phase0_results[i]
        theta_root = torch.zeros(L_binder, NUM_AMINO_ACIDS, device=device)
        _sync_theta_to_seq(theta_root, seq)
        theta_root.requires_grad_(True)

        if mcts_tree is not None:
            root_node = mcts_tree.add_root_node(
                sequence=seq,
                logits=theta_root,
                L_real=result.total_loss,
                coordinate=result.coordinate,
                loss_vector=result.loss_dict,
                prior=1.0 / max(top_beam, 1),
                structural_signature=mcts_tree.extract_structural_signature(
                    result=result,
                    sequence=seq,
                    loss_vector=result.loss_dict,
                ),
            )
            logger.info(
                f"  根节点 {i+1}: {seq[:20]}... "
                f"L={result.total_loss:.4f} R={root_node.tree_reward:.4f} "
                f"norm={root_node.status_flags.get('normalized_loss', 'N/A')} "
                f"mode={root_node.status_flags.get('reward_mode', 'raw')}"
            )
        else:
            logger.info(f"  Phase0 candidate {i+1}: {seq[:20]}... L={result.total_loss:.4f}")

    best_zeroth_loss = phase0_results[0][1].total_loss
    best_zeroth_seq = phase0_results[0][0]
    best_step = 0
    losses_history = []

    if phase0_ckpt_mgr is not None:
        _, best_result = phase0_results[0]
        phase0_ckpt_mgr.save(
            step=0,
            beam_id=0,
            zeroth_result=best_result,
            plddt_logits=best_result.plddt_logits,
            accepted=True,
            is_global_best=True,
        )
        logger.info("[Checkpoint] Phase 0 保存完毕")

    phase0_root_count = mcts_tree.total_nodes if mcts_tree is not None else top_beam
    logger.info(f"Phase 0 完成：{phase0_root_count} 个根候选, best_L={best_zeroth_loss:.4f}")
    logger.info("=" * 70)

    legacy_steps_done = 0
    step_offset = 0
    strict_loop_steps = N_steps
    ckpt_mgr = strict_ckpt_mgr

    if mcts_strategy in {"legacy", "two_stage"}:
        legacy_tree = LegacyMCTSTree(
            C=mcts_C,
            max_nodes=mcts_max_nodes,
            mcts_tau=mcts_tau_fixed,
        )
        for i in range(top_beam):
            seq, result = phase0_results[i]
            theta_root = torch.zeros(L_binder, NUM_AMINO_ACIDS, device=device)
            _sync_theta_to_seq(theta_root, seq)
            theta_root.requires_grad_(True)
            legacy_tree.add_root_node(
                sequence=seq,
                logits=theta_root,
                L_real=result.total_loss,
                coordinate=result.coordinate,
            )

        legacy_limit = N_steps if mcts_strategy == "legacy" else min(N_steps, mcts_bootstrap_max_steps)
        legacy_min_steps = min(N_steps, mcts_bootstrap_steps)
        significant_best_loss = best_zeroth_loss
        stale_steps = 0

        logger.info("=" * 70)
        logger.info(
            "StageA legacy_bootstrap 启动: max_steps=%d min_steps=%d patience=%d min_delta=%.4f",
            legacy_limit,
            legacy_min_steps,
            mcts_bootstrap_patience,
            mcts_bootstrap_min_delta,
        )

        for legacy_step in range(legacy_limit):
            selected_node = legacy_tree.select()
            selected_seq = selected_node.sequence
            selected_logits = selected_node.logits.clone().to(device)

            with torch.no_grad():
                feat_binder_only, _, _ = build_single_chain_features(selected_seq, device)
                feat_binder_only = _prepare_input_features(model, feat_binder_only)
                if selected_node.coordinate is not None:
                    x_coarse_cached = selected_node.coordinate.clone().to(device)
                    if x_coarse_cached.dim() >= 3 and x_coarse_cached.size(-3) > 1:
                        x_coarse_cached = x_coarse_cached.narrow(-3, 0, 1).contiguous()
                else:
                    ref_pos = feat_binder_only["ref_pos"]
                    x_coarse_cached = ref_pos.clone().unsqueeze(0).to(device)
                    x_coarse_cached = x_coarse_cached + torch.randn_like(x_coarse_cached) * 0.1

            theta_step2 = selected_logits.detach().clone().requires_grad_(True)
            optimizer_step2 = torch.optim.Adam([theta_step2], lr=learning_rate)

            progress = legacy_step / max(N_steps - 1, 1)
            tau_temp = 2.0 * (1.0 - progress) + 0.05 * progress

            s_binder_step2, current_seq_step2, input_feature_dict_step2 = ste_bridge(
                theta=theta_step2,
                W_AAE=W_AAE,
                model=model,
                target_seq=target_seq,
                device=device,
                tau_temp=tau_temp,
                target_template_path=target_template_path,
            )

            with _proposal_amp_context(device, proposal_amp_dtype):
                x0_hat_step2, s_trunk_step2, z_trunk_step2 = diffusion_one_step(
                    model=model,
                    s_binder=s_binder_step2,
                    s_target=s_target,
                    input_feature_dict=input_feature_dict_step2,
                    x_coarse=x_coarse_cached,
                    X_clean=selected_node.coordinate,
                    tau=mcts_tau_fixed,
                    device=device,
                    chunk_size=chunk_size,
                )

                x0_flat = x0_hat_step2.squeeze(0)
                s_inputs_full = torch.cat([s_binder_step2, s_target], dim=0)
                heads = run_confidence_heads(
                    model=model,
                    input_feature_dict=input_feature_dict_step2,
                    s_inputs=s_inputs_full,
                    s_trunk=s_trunk_step2,
                    z_trunk=z_trunk_step2,
                    x0_hat=x0_hat_step2,
                    inplace_safe=False,
                    chunk_size=chunk_size,
                )

                first_order_loss, first_order_losses = compute_all_losses(
                    x0_flat=x0_flat,
                    plddt_logits=heads["plddt_logits"],
                    pae_logits=heads["pae_logits"],
                    distogram_logits=heads["distogram_logits"],
                    input_feature_dict=input_feature_dict_step2,
                    L_binder=L_binder,
                    weights=base_weights,
                    contact_params=contact_params,
                    plddt_params=plddt_params,
                    pae_params=pae_params,
                    distogram_params=distogram_params,
                    pde_logits=heads["pde_logits"],
                    pde_params=pde_params,
                    clash_params=dict(DEFAULT_CLASH_PARAMS),
                    hotspot_params=hotspot_params,
                    target_chain_roles=target_chain_roles,
                )

            first_order_loss.backward()
            grad_for_proposal = theta_step2.grad.clone() if theta_step2.grad is not None else None
            torch.nn.utils.clip_grad_norm_([theta_step2], 1.0)
            optimizer_step2.step()
            fo_loss_val = first_order_loss.item()

            if grad_for_proposal is None:
                logger.warning("StageA Step %d: grad is None, skipping candidate generation", legacy_step)
                losses_history.append(best_zeroth_loss)
                stale_steps += 1
                legacy_steps_done += 1
                continue

            candidates = propose_mutations_legacy(
                grad_matrix=grad_for_proposal.detach().cpu(),
                current_seq=current_seq_step2,
                K_top=K_top,
                B_candidates=B_candidates,
                T_sample=T_sample,
                max_mutations_per_seq=max_mutations_per_seq,
                use_prior=use_prior,
                prior_strength=prior_strength,
            )

            if not candidates:
                logger.warning("StageA Step %d: no candidates generated, skipping", legacy_step)
                losses_history.append(best_zeroth_loss)
                stale_steps += 1
                legacy_steps_done += 1
                continue

            all_zeroth_results = []
            for cand_seq in candidates:
                result = zeroth_order_eval(
                    binder_seq=cand_seq,
                    target_seq=target_seq,
                    model=model,
                    device=device,
                    loss_weights=base_weights,
                    contact_params=contact_params,
                    chunk_size=chunk_size,
                    hotspot_params=hotspot_params,
                    seed=_verifier_seed(seed, target_seq, cand_seq),
                    target_template_path=target_template_path,
                    target_chain_roles=target_chain_roles,
                )
                verifier_cache[result.binder_seq] = result
                all_zeroth_results.append(result)

            new_nodes = legacy_tree.expand(selected_node, all_zeroth_results)
            for new_node in new_nodes:
                new_node.logits = _init_child_logits_from_parent(
                    selected_node,
                    new_node.sequence,
                    device,
                ).detach().cpu()

            losses_for_backprop = [result.total_loss for result in all_zeroth_results]
            legacy_tree.backpropagate(selected_node, losses_for_backprop)

            n_pruned = legacy_tree.prune()
            if n_pruned > 0:
                logger.debug("StageA Step %d: pruned %d nodes", legacy_step, n_pruned)

            round_best = min(all_zeroth_results, key=lambda result: result.total_loss)
            round_best_loss = round_best.total_loss

            accepted = False
            prob_accept = False
            if annealer is not None:
                current_node_loss = selected_node.L_real
                accepted = annealer.accept(current_node_loss, round_best_loss, legacy_step, N_steps)
                if accepted and round_best_loss >= current_node_loss:
                    prob_accept = True
            else:
                accepted = round_best_loss <= best_zeroth_loss

            if accepted and round_best_loss <= best_zeroth_loss:
                best_zeroth_loss = round_best_loss
                best_zeroth_seq = round_best.binder_seq
                best_step = legacy_step

            if best_zeroth_loss <= significant_best_loss - mcts_bootstrap_min_delta:
                significant_best_loss = best_zeroth_loss
                stale_steps = 0
            else:
                stale_steps += 1

            # Public MCTS loss history is the global best trace. The per-round
            # verifier best is still logged as round_best for diagnostics.
            losses_history.append(best_zeroth_loss)
            legacy_steps_done += 1
            stats = legacy_tree.get_node_stats()

            logger.info(
                "StageA Step %3d/%d | sel_node=%d(d=%d) L_real=%.4f 1st=%.4f | "
                "K=%d round_best=%.4f global_best=%.4f@%d stale=%d "
                "tree=%dnodes max_depth=%d N_total=%d | accepted=%s%s",
                legacy_step,
                legacy_limit,
                selected_node.node_id,
                selected_node.depth,
                selected_node.L_real,
                fo_loss_val,
                len(candidates),
                round_best_loss,
                best_zeroth_loss,
                best_step,
                stale_steps,
                stats["total_nodes"],
                stats["max_depth"],
                stats["N_total"],
                accepted,
                "(prob)" if prob_accept else "",
            )

            if legacy_ckpt_mgr is not None:
                legacy_ckpt_mgr.save(
                    step=legacy_step,
                    beam_id=0,
                    zeroth_result=round_best,
                    plddt_logits=round_best.plddt_logits,
                    accepted=accepted,
                    is_global_best=(round_best_loss == best_zeroth_loss),
                )
                best_info = legacy_ckpt_mgr.get_best_info()
                logger.info(
                    "  [Legacy Checkpoint] best=%.4f @ step=%s beam=%s",
                    best_info["loss"],
                    best_info["step"],
                    best_info["beam"],
                )

            if (
                mcts_strategy == "two_stage"
                and legacy_steps_done >= legacy_min_steps
                and stale_steps >= mcts_bootstrap_patience
            ):
                logger.info(
                    "StageA plateau detected: steps=%d stale=%d best=%.4f; switching to StageB",
                    legacy_steps_done,
                    stale_steps,
                    best_zeroth_loss,
                )
                break

        legacy_tree.log_tree_summary()

        if mcts_strategy == "legacy" or legacy_steps_done >= N_steps:
            logger.info("=" * 70)
            logger.info(
                "Legacy MCTS 完成: best_seq=%s loss=%.4f @step %d",
                best_zeroth_seq,
                best_zeroth_loss,
                best_step,
            )
            logger.info("=" * 70)
            return best_zeroth_seq, best_zeroth_seq, losses_history

        handoff_results = _select_diverse_handoff_results(verifier_cache, mcts_handoff_top_k)
        if not handoff_results:
            raise RuntimeError("[MCTS] two_stage handoff has no verifier results")

        mcts_tree = _make_strict_mcts_tree(
            mcts_C, mcts_max_nodes, mcts_tau_fixed, mcts_expand_actions_per_step
        )
        for cached_result in verifier_cache.values():
            mcts_tree.store_verifier_result(cached_result)

        for i, result in enumerate(handoff_results):
            theta_root = torch.zeros(L_binder, NUM_AMINO_ACIDS, device=device)
            _sync_theta_to_seq(theta_root, result.binder_seq)
            theta_root.requires_grad_(True)
            root_node = mcts_tree.add_root_node(
                sequence=result.binder_seq,
                logits=theta_root,
                L_real=result.total_loss,
                coordinate=result.coordinate,
                loss_vector=result.loss_dict,
                prior=1.0 / max(len(handoff_results), 1),
                structural_signature=mcts_tree.extract_structural_signature(
                    result=result,
                    sequence=result.binder_seq,
                    loss_vector=result.loss_dict,
                ),
            )
            logger.info(
                "StageB root %d/%d: seq=%s L=%.4f R=%.4f structural=%s",
                i + 1,
                len(handoff_results),
                result.binder_seq[:20],
                result.total_loss,
                root_node.tree_reward,
                _handoff_structural_key(result),
            )

        step_offset = legacy_steps_done
        strict_loop_steps = max(0, N_steps - legacy_steps_done)
        ckpt_mgr = strict_ckpt_mgr
        logger.info(
            "Stage switch: legacy_steps=%d strict_steps=%d legacy_best_loss=%.4f",
            legacy_steps_done,
            strict_loop_steps,
            best_zeroth_loss,
        )

    logger.info("开始 Document PUCT MCTS 主循环: steps=%d offset=%d", strict_loop_steps, step_offset)
    strict_stage_best_loss = float(best_zeroth_loss)
    strict_stale_steps = 0

    for step in range(strict_loop_steps):
        global_step = step_offset + step
        selection = mcts_tree.select_decision()
        selected_node = selection.node
        if selection.decision_type.endswith("terminal"):
            logger.warning(
                f"Step {global_step}: selected node {selected_node.node_id} is terminal, skipping"
            )
            losses_history.append(best_zeroth_loss)
            continue

        selected_seq = selected_node.sequence
        selected_logits = selected_node.logits.clone().to(device)

        with torch.no_grad():
            feat_binder_only, _, _ = build_single_chain_features(selected_seq, device)
            feat_binder_only = _prepare_input_features(model, feat_binder_only)
            if selected_node.coordinate is not None:
                x_coarse_cached = selected_node.coordinate.clone().to(device)
                # zeroth_order 可能保存 [N_sample, N_atom, 3]（默认 N_sample>1），
                # 单步外推只用首个 sample，避免 t_hat/r_noisy 形状不匹配
                if x_coarse_cached.dim() >= 3 and x_coarse_cached.size(-3) > 1:
                    x_coarse_cached = x_coarse_cached.narrow(-3, 0, 1).contiguous()
            else:
                ref_pos = feat_binder_only["ref_pos"]
                x_coarse_cached = ref_pos.clone().unsqueeze(0).to(device)
                x_coarse_cached = x_coarse_cached + torch.randn_like(x_coarse_cached) * 0.1

        # detach：避免 clone(logits) 仍为非 leaf，Adam 报 can't optimize a non-leaf Tensor
        theta_step2 = selected_logits.detach().clone().requires_grad_(True)
        optimizer_step2 = torch.optim.Adam([theta_step2], lr=learning_rate)

        progress = global_step / max(N_steps - 1, 1)
        tau_temp = 2.0 * (1.0 - progress) + 0.05 * progress

        s_binder_step2, current_seq_step2, input_feature_dict_step2 = ste_bridge(
            theta=theta_step2,
            W_AAE=W_AAE,
            model=model,
            target_seq=target_seq,
            device=device,
            tau_temp=tau_temp,
            target_template_path=target_template_path,
        )

        proposal_tau = _stageb_proposal_tau(
            mcts_proposal_method,
            mcts_tau_fixed,
            local_step=step,
            total_steps=strict_loop_steps,
            stale_steps=strict_stale_steps,
            adaptive_tau_start=mcts_adaptive_tau_start,
            adaptive_tau_end=mcts_adaptive_tau_end,
        )
        with _proposal_amp_context(device, proposal_amp_dtype):
            x0_hat_step2, s_trunk_step2, z_trunk_step2 = diffusion_one_step(
                model=model,
                s_binder=s_binder_step2,
                s_target=s_target,
                input_feature_dict=input_feature_dict_step2,
                x_coarse=x_coarse_cached,
                X_clean=selected_node.coordinate,
                tau=proposal_tau,
                device=device,
                chunk_size=chunk_size,
            )

            x0_flat = x0_hat_step2.squeeze(0)
            s_inputs_full = torch.cat([s_binder_step2, s_target], dim=0)
            heads = run_confidence_heads(
                model=model,
                input_feature_dict=input_feature_dict_step2,
                s_inputs=s_inputs_full,
                s_trunk=s_trunk_step2,
                z_trunk=z_trunk_step2,
                x0_hat=x0_hat_step2,
                inplace_safe=False,
                chunk_size=chunk_size,
            )

            routing_info = select_de_novo_action_group(
                selected_node.loss_vector,
                temperature=MCTS_STAGE2_GROUP_TEMPERATURE,
                loss_weights=base_weights,
                binder_length=L_binder,
            )
            action_group = str(routing_info["action_group"])
            group_prior = float(routing_info["group_prior"])
            group_priors = {
                str(group): float(prior)
                for group, prior in dict(routing_info.get("group_priors", {})).items()
            }
            if not group_priors:
                group_priors = {action_group: group_prior}
            action_group, group_priors = _apply_stageb_prior_mode_to_groups(
                mcts_prior_mode,
                action_group,
                group_priors,
            )
            group_prior = float(
                group_priors.get(
                    action_group,
                    max(group_priors.values()) if group_priors else 1.0,
                )
            )

            first_order_loss, first_order_losses = compute_all_losses(
                x0_flat=x0_flat,
                plddt_logits=heads["plddt_logits"],
                pae_logits=heads["pae_logits"],
                distogram_logits=heads["distogram_logits"],
                input_feature_dict=input_feature_dict_step2,
                L_binder=L_binder,
                weights=base_weights,
                contact_params=contact_params,
                plddt_params=plddt_params,
                pae_params=pae_params,
                distogram_params=distogram_params,
                pde_logits=heads["pde_logits"],
                pde_params=pde_params,
                clash_params=dict(DEFAULT_CLASH_PARAMS),
                hotspot_params=hotspot_params,
                target_chain_roles=target_chain_roles,
            )
            proposal_loss, proposal_loss_mode = _stageb_group_proposal_loss(
                first_order_losses,
                action_group,
                first_order_loss,
            )

        use_gradient_proposal = should_use_gradient_prior(mcts_prior_mode)
        if use_gradient_proposal:
            proposal_loss.backward()
            grad_for_proposal = theta_step2.grad.clone() if theta_step2.grad is not None else None
            torch.nn.utils.clip_grad_norm_([theta_step2], 1.0)
            optimizer_step2.step()
        else:
            grad_for_proposal = torch.zeros_like(theta_step2.detach())
        fo_loss_val = first_order_loss.item()
        proposal_loss_val = float(proposal_loss.detach().item())
        proposal_passes = 1 if use_gradient_proposal else 0

        if (
            use_gradient_proposal
            and mcts_proposal_method == "two_step_truncated"
            and grad_for_proposal is not None
        ):
            optimizer_step2.zero_grad(set_to_none=True)
            s_binder_step2, current_seq_step2, input_feature_dict_step2 = ste_bridge(
                theta=theta_step2,
                W_AAE=W_AAE,
                model=model,
                target_seq=target_seq,
                device=device,
                tau_temp=tau_temp,
                target_template_path=target_template_path,
            )
            with _proposal_amp_context(device, proposal_amp_dtype):
                x0_hat_step2, s_trunk_step2, z_trunk_step2 = diffusion_one_step(
                    model=model,
                    s_binder=s_binder_step2,
                    s_target=s_target,
                    input_feature_dict=input_feature_dict_step2,
                    x_coarse=x_coarse_cached,
                    X_clean=selected_node.coordinate,
                    tau=proposal_tau,
                    device=device,
                    chunk_size=chunk_size,
                )
                x0_flat = x0_hat_step2.squeeze(0)
                s_inputs_full = torch.cat([s_binder_step2, s_target], dim=0)
                heads = run_confidence_heads(
                    model=model,
                    input_feature_dict=input_feature_dict_step2,
                    s_inputs=s_inputs_full,
                    s_trunk=s_trunk_step2,
                    z_trunk=z_trunk_step2,
                    x0_hat=x0_hat_step2,
                    inplace_safe=False,
                    chunk_size=chunk_size,
                )
                first_order_loss, first_order_losses = compute_all_losses(
                    x0_flat=x0_flat,
                    plddt_logits=heads["plddt_logits"],
                    pae_logits=heads["pae_logits"],
                    distogram_logits=heads["distogram_logits"],
                    input_feature_dict=input_feature_dict_step2,
                    L_binder=L_binder,
                    weights=base_weights,
                    contact_params=contact_params,
                    plddt_params=plddt_params,
                    pae_params=pae_params,
                    distogram_params=distogram_params,
                    pde_logits=heads["pde_logits"],
                    pde_params=pde_params,
                    clash_params=dict(DEFAULT_CLASH_PARAMS),
                    hotspot_params=hotspot_params,
                    target_chain_roles=target_chain_roles,
                )
                proposal_loss, proposal_loss_mode = _stageb_group_proposal_loss(
                    first_order_losses,
                    action_group,
                    first_order_loss,
                )
            proposal_loss.backward()
            grad_for_proposal = theta_step2.grad.clone() if theta_step2.grad is not None else None
            torch.nn.utils.clip_grad_norm_([theta_step2], 1.0)
            optimizer_step2.step()
            fo_loss_val = first_order_loss.item()
            proposal_loss_val = float(proposal_loss.detach().item())
            proposal_passes = 2

        if grad_for_proposal is None:
            logger.warning(f"Step {global_step}: grad is None, skipping candidate generation")
            losses_history.append(best_zeroth_loss)
            continue

        grads_cpu = grad_for_proposal.detach().cpu()

        stageb_max_mutations = _stageb_max_mutations_for_step(
            base_max_mutations=max_mutations_per_seq,
            local_step=step,
            total_steps=strict_loop_steps,
            stale_steps=strict_stale_steps,
        )

        widening_slots = mcts_tree.remaining_expansion_slots(selected_node)
        expand_budget = min(
            mcts_expand_actions_per_step,
            widening_slots,
        )
        if expand_budget <= 0:
            logger.warning(
                f"Step {global_step}: selected node {selected_node.node_id} has no progressive widening slots"
            )
            losses_history.append(best_zeroth_loss)
            continue

        proposal_pool_budget = max(B_candidates, expand_budget)
        group_budgets = _allocate_action_group_budgets(
            group_priors,
            proposal_pool_budget,
        )
        proposal_pool = []
        for group_name, group_budget in group_budgets.items():
            proposal_pool.extend(
                propose_mutation_actions(
                    grad_matrix=grads_cpu,
                    current_seq=current_seq_step2,
                    K_top=K_top,
                    B_candidates=group_budget,
                    T_sample=T_sample,
                    max_mutations_per_seq=stageb_max_mutations,
                    use_prior=use_prior,
                    prior_strength=prior_strength,
                    action_group=group_name,
                    group_prior=group_priors.get(group_name, 0.0),
                    structural_signature=selected_node.structural_signature,
                    prior_mode=mcts_prior_mode,
                )
            )
        proposal_pool = _renormalize_stageb_proposal_priors(
            proposal_pool,
            mcts_prior_mode,
        )

        if not proposal_pool:
            logger.warning(f"Step {global_step}: no candidates generated, skipping")
            losses_history.append(best_zeroth_loss)
            continue

        registered_actions = mcts_tree.register_action_proposals(
            selected_node,
            proposal_pool,
        )
        selected_actions = mcts_tree.select_unexpanded_actions(
            selected_node,
            max_actions=expand_budget,
            candidates=registered_actions,
        )
        if not selected_actions:
            logger.warning(
                f"Step {global_step}: no fresh unexpanded action selected from {len(proposal_pool)} proposals"
            )
            losses_history.append(best_zeroth_loss)
            continue

        selected_proposals = [
            action.proposal
            for action in selected_actions
            if action.proposal is not None
        ]
        action_by_sequence = {
            action.sequence: action
            for action in selected_actions
        }

        fresh_proposals = []
        cached_proposals = []
        cached_results = []
        duplicate_results = []
        reward_by_sequence = {}
        seen_step = set()
        duplicates_skipped = 0
        cache_hits_before = mcts_tree.get_node_stats()["verifier_cache_hits"]
        duplicate_merges_before = mcts_tree.get_node_stats()["duplicate_merges"]
        for proposal in selected_proposals:
            if proposal.sequence in seen_step:
                duplicates_skipped += 1
                continue
            seen_step.add(proposal.sequence)
            if mcts_tree.has_sequence(proposal.sequence):
                duplicates_skipped += 1
                action = action_by_sequence.get(proposal.sequence)
                if action is not None:
                    action.status = "duplicate"
                cached = mcts_tree.get_cached_verifier_result(proposal.sequence)
                existing = mcts_tree.get_node_by_sequence(proposal.sequence)
                if cached is None and existing is not None:
                    cached = mcts_tree.node_to_cached_result(existing)
                reward = mcts_tree.merge_duplicate(selected_node, proposal, cached)
                reward_by_sequence[proposal.sequence] = reward
                if cached is not None:
                    duplicate_results.append(cached)
                continue

            cached = mcts_tree.get_cached_verifier_result(proposal.sequence)
            if cached is not None:
                cached_proposals.append(proposal)
                cached_results.append(cached)
                continue

            fresh_proposals.append(proposal)

        if not fresh_proposals and not cached_results and not reward_by_sequence:
            logger.warning(
                f"Step {global_step}: all {len(selected_proposals)} selected actions were duplicate, skipping verifier"
            )
            losses_history.append(best_zeroth_loss)
            continue

        all_zeroth_results = []
        for proposal in fresh_proposals:
            result = zeroth_order_eval(
                binder_seq=proposal.sequence,
                target_seq=target_seq,
                model=model,
                device=device,
                loss_weights=base_weights,
                contact_params=contact_params,
                chunk_size=chunk_size,
                hotspot_params=hotspot_params,
                seed=_verifier_seed(seed, target_seq, proposal.sequence),
                target_template_path=target_template_path,
                target_chain_roles=target_chain_roles,
            )
            mcts_tree.store_verifier_result(result)
            all_zeroth_results.append(result)

        expand_proposals = cached_proposals + fresh_proposals
        expand_results = cached_results + all_zeroth_results
        round_results = duplicate_results + expand_results

        new_nodes = mcts_tree.expand(selected_node, expand_proposals, expand_results)
        inserted_rewards = {
            new_node.sequence: new_node.tree_reward
            for new_node in new_nodes
        }

        for new_node in new_nodes:
            child_logits = _init_child_logits_from_parent(
                selected_node, new_node.sequence, device
            )
            new_node.logits = child_logits.detach().cpu()

        for result in expand_results:
            reward_by_sequence[result.binder_seq] = inserted_rewards.get(
                result.binder_seq,
                mcts_tree.result_reward(result),
            )

        for action in selected_actions:
            if action.sequence not in reward_by_sequence:
                continue
            mcts_tree.backpropagate_action(
                selection,
                action,
                reward_by_sequence[action.sequence],
            )

        n_pruned = mcts_tree.prune()
        if n_pruned > 0:
            logger.debug(f"Step {global_step}: pruned {n_pruned} nodes")

        round_best = min(round_results, key=lambda r: r.total_loss)
        round_best_loss = round_best.total_loss
        round_best_node = mcts_tree.get_node_by_sequence(round_best.binder_seq)
        round_best_reward = (
            round_best_node.tree_reward
            if round_best_node is not None
            else mcts_tree.result_reward(round_best)
        )

        accepted = False
        prob_accept = False
        if annealer is not None:
            current_node_loss = selected_node.L_real
            accepted = annealer.accept(current_node_loss, round_best_loss, global_step, N_steps)
            if accepted and round_best_loss >= current_node_loss:
                prob_accept = True
        else:
            accepted = round_best_loss <= best_zeroth_loss

        previous_global_best_loss = best_zeroth_loss
        if accepted and round_best_loss <= best_zeroth_loss:
            best_zeroth_loss = round_best_loss
            best_zeroth_seq = round_best.binder_seq
            best_step = global_step

        if best_zeroth_loss < strict_stage_best_loss - mcts_bootstrap_min_delta:
            strict_stage_best_loss = float(best_zeroth_loss)
            strict_stale_steps = 0
        elif best_zeroth_loss < previous_global_best_loss:
            strict_stage_best_loss = min(strict_stage_best_loss, float(best_zeroth_loss))
            strict_stale_steps = 0
        else:
            strict_stale_steps += 1

        # Public MCTS loss history is the global best trace. The per-round
        # verifier best is still logged as round_best for diagnostics.
        losses_history.append(best_zeroth_loss)

        stats = mcts_tree.get_node_stats()
        verifier_cache_hits = stats["verifier_cache_hits"] - cache_hits_before
        duplicate_merges = stats["duplicate_merges"] - duplicate_merges_before
        best_action_score_str = (
            "NA" if selection.best_action_score is None
            else f"{selection.best_action_score:.4f}"
        )
        frontier_score_str = (
            "NA" if selection.frontier_score is None
            else f"{selection.frontier_score:.4f}"
        )

        logger.info(
            f"Step {global_step:3d}/{N_steps} | "
            f"sel_node={selected_node.node_id}(d={selected_node.depth}) "
            f"L_real={selected_node.L_real:.4f} "
            f"R={selected_node.tree_reward:.4f} Q={selected_node.q_value:.4f} "
            f"P={selected_node.prior:.4f} "
            f"1st={fo_loss_val:.4f} prop={proposal_loss_val:.4f}({proposal_loss_mode}) | "
            f"proposal={mcts_proposal_method} pass={proposal_passes} tau={proposal_tau:.3f} "
            f"group={action_group} Pg={group_prior:.3f} "
            f"edit_max={stageb_max_mutations} stale={strict_stale_steps} "
            f"pw={len(selected_node.children)}/{mcts_tree.progressive_widening_limit(selected_node)} "
            f"decision={selection.decision_type} path={len(selection.action_path)} "
            f"score={best_action_score_str}/{frontier_score_str} "
            f"pool={len(proposal_pool)} groups_sampled={len(group_budgets)} "
            f"selected_actions={len(selected_actions)} fresh={len(fresh_proposals)} cached={len(cached_results)} "
            f"dup={duplicates_skipped} dmerge={duplicate_merges} cache_hit={verifier_cache_hits} "
            f"groups={stats['num_groups']} invalid={stats['num_invalid']} "
            f"rmode={selected_node.status_flags.get('reward_mode', 'raw')} "
            f"gskip={stats['group_rejections']} "
            f"round_best={round_best_loss:.4f} "
            f"round_R={round_best_reward:.4f} "
            f"global_best={best_zeroth_loss:.4f}@{best_step} "
            f"tree={stats['total_nodes']}nodes "
            f"max_depth={stats['max_depth']} "
            f"N_total={stats['N_total']} | "
            f"accepted={accepted}{'(prob)' if prob_accept else ''}"
        )

        if ckpt_mgr is not None:
            ckpt_mgr.save(
                step=global_step,
                beam_id=0,
                zeroth_result=round_best,
                plddt_logits=round_best.plddt_logits,
                accepted=accepted,
                is_global_best=(round_best_loss == best_zeroth_loss),
            )
            best_info = ckpt_mgr.get_best_info()
            logger.info(
                f"  [Checkpoint] best={best_info['loss']:.4f} @ "
                f"step={best_info['step']} beam={best_info['beam']}"
            )

    mcts_tree.log_tree_summary()

    best_final = mcts_tree.get_global_best()
    logger.info("=" * 70)
    logger.info("MCTS 混合优化完成！")
    logger.info(
        f"全局最优序列(reward): {best_final.sequence} "
        f"(R={best_final.tree_reward:.4f}, L={best_final.L_real:.4f})"
    )
    logger.info(f"最低 total_loss 序列: {best_zeroth_seq} (loss={best_zeroth_loss:.4f} @step {best_step})")
    logger.info(f"MCTS 树最终节点数: {mcts_tree.total_nodes}")
    logger.info(
        f"MCTS 全局最优节点: id={best_final.node_id} depth={best_final.depth} "
        f"seq={best_final.sequence[:20]}... L={best_final.L_real:.4f} R={best_final.tree_reward:.4f}"
    )
    logger.info("=" * 70)

    # Keep the public "best sequence" compatible with loss-based reports while
    # still exposing the reward-best tree node as the final MCTS choice.
    return best_zeroth_seq, best_final.sequence, losses_history
