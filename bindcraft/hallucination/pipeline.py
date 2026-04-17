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
from typing import Dict, Tuple, List, Union
import logging

from bindcraft.hallucination.ste_bridge import (
    precompute_waa_embeddings,
    ste_bridge
)
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
from bindcraft.hallucination.diffusion_one_step import diffusion_one_step
from bindcraft.hallucination.mutation_proposer import AA_1_TO_IDX, propose_mutations
from bindcraft.hallucination.zeroth_order import zeroth_order_eval
from bindcraft.losses.confidence import run_confidence_heads
from bindcraft.losses.total_loss import compute_all_losses
from protenix.model import sample_confidence

logger = logging.getLogger(__name__)
VALID_BINDER_AAS = "ACDEFGHIKLMNPQRSTVWY"


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
# 主 Hallucination 循环（纯一阶模式）
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

    Returns:
        best_seq: 最佳序列
        final_seq: 最终序列
        losses_history: Loss 历史
    """
    if device is None:
        device = next(model.parameters()).device

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
    torch.manual_seed(42)
    np.random.seed(42)
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
            tau_temp=tau_temp
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
            pde_logits=heads["pde_logits"],
            distogram_logits=heads["distogram_logits"],
            input_feature_dict=input_feature_dict,
            L_binder=L_binder,
            weights=weights,
            contact_params=contact_params,
            clash_params=dict(DEFAULT_CLASH_PARAMS),
            plddt_params=plddt_params,
            pae_params=pae_params,
            pde_params=pde_params,
            distogram_params=distogram_params,
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

    采用 soft reset：在当前 theta 基础上叠加一个较强的 one-hot 偏置，
    而非完全覆盖，以保留 Adam 优化器中积累的动量信息。

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
        theta.copy_(theta * 0.3 + bias)


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
    use_tau_scheduling: bool = True,
    tau_jitter: bool = True,
    init_binder_seq: str = None,
) -> Tuple[str, str, List[float]]:
    """
    混合优化循环：一阶梯度引导离散候选 + 零阶全链路验证 + 接受准则。

    每轮四步：
    1. 一阶反传：STE 桥接 -> 单步外推 -> loss -> backward -> Adam 更新 theta
    2. 候选生成：从 theta.grad 提取突变潜力，采样 B 条离散候选
    3. 零阶验证：对 B 条候选做完整 AF3 推断，计算真实 loss
    4. 接受准则：贪心或模拟退火接受最优候选

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
        B_candidates:  每轮生成的候选序列数
        T_sample:      候选采样温度
        max_mutations_per_seq: 每条候选最大突变数
        use_annealing: 启用模拟退火接受（否则纯贪心）
        T_init:        退火初始温度
        T_min:         退火最低温度
        use_tau_scheduling: 启用 Coarse-to-Fine tau 动态调度
        tau_jitter:    启用 tau 抖动
        init_binder_seq: 可选的初始 Binder 单字母序列

    Returns:
        best_seq:        零阶验证下的最优序列
        final_seq:       最后一轮的一阶当前序列
        losses_history:  每轮零阶验证最优 loss 的历史
    """
    from bindcraft.hallucination.scheduler import AnnealingScheduler, TauScheduler

    if device is None:
        device = next(model.parameters()).device

    base_weights = dict(DEFAULT_LOSS_WEIGHTS)
    if loss_weights is not None:
        base_weights.update(loss_weights)
    contact_params = dict(DEFAULT_CONTACT_PARAMS)
    target_sequences = _normalize_chain_sequences(target_seq)
    target_total_len = sum(len(seq) for seq in target_sequences)
    target_len_str = "+".join(str(len(seq)) for seq in target_sequences)

    annealer = AnnealingScheduler(T_init=T_init, T_min=T_min) if use_annealing else None
    tau_sched = TauScheduler(jitter=tau_jitter) if use_tau_scheduling else None

    logger.info("=" * 70)
    logger.info("开始混合优化循环 (Hybrid: 1st-Order Proposal + 0th-Order Verification)")
    logger.info(
        f"Target chains: {len(target_sequences)} | lengths={target_len_str} | total={target_total_len} aa"
    )
    logger.info(f"Binder 长度: {L_binder} aa")
    logger.info(f"总轮数: {N_steps} | K_top={K_top} B={B_candidates} T_sample={T_sample}")
    logger.info(f"Annealing: {use_annealing} (T_init={T_init}, T_min={T_min})")
    logger.info(f"Tau scheduling: {use_tau_scheduling} (jitter={tau_jitter})")
    logger.info(f"Base loss 权重: {base_weights}")
    logger.info("=" * 70)

    freeze_model(model)

    # ── 预计算 W_AAE ──────────────────────────────────────────────────────
    logger.info("预计算 W_AAE...")
    W_AAE = precompute_waa_embeddings(model, device)

    # ── 初始化 Theta ──────────────────────────────────────────────────────
    torch.manual_seed(42)
    np.random.seed(42)
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

    # ── 获取 bin 参数 ─────────────────────────────────────────────────────
    plddt_params = sample_confidence.get_bin_params(model.configs.loss.plddt)
    pae_params = sample_confidence.get_bin_params(model.configs.loss.pae)
    distogram_params = sample_confidence.get_bin_params(model.configs.loss.distogram)
    pde_params = sample_confidence.get_bin_params(model.configs.loss.pde)

    # ── 状态变量 ──────────────────────────────────────────────────────────
    best_zeroth_loss = float("inf")
    current_accepted_loss = float("inf")
    best_zeroth_seq = init_binder
    current_accepted_seq = init_binder
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
            step_weights = tau_sched.get_dynamic_weights(step, N_steps)
            phase_name = tau_sched.get_phase_name(step, N_steps)
        else:
            step_tau = tau_noise
            step_weights = base_weights
            phase_name = "fixed"

        anneal_T = annealer.get_temperature(step, N_steps) if annealer else 0.0

        # ================================================================
        # Step 1: 一阶反向传播
        # ================================================================
        optimizer.zero_grad()

        progress = step / max(N_steps - 1, 1)
        tau_temp = 2.0 * (1.0 - progress) + 0.05 * progress

        s_binder, current_seq, input_feature_dict = ste_bridge(
            theta=theta, W_AAE=W_AAE, model=model,
            target_seq=target_seq, device=device, tau_temp=tau_temp,
        )

        x0_hat, s_trunk, z_trunk = diffusion_one_step(
            model=model, s_binder=s_binder, s_target=s_target,
            input_feature_dict=input_feature_dict, x_coarse=None,
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
            pde_logits=heads["pde_logits"],
            distogram_logits=heads["distogram_logits"],
            input_feature_dict=input_feature_dict,
            L_binder=L_binder, weights=step_weights,
            contact_params=contact_params,
            clash_params=dict(DEFAULT_CLASH_PARAMS),
            plddt_params=plddt_params,
            pae_params=pae_params,
            pde_params=pde_params,
            distogram_params=distogram_params,
        )

        first_order_loss.backward()

        grad_saved = theta.grad.clone() if theta.grad is not None else None

        torch.nn.utils.clip_grad_norm_([theta], 1.0)
        optimizer.step()

        fo_loss_val = first_order_loss.item()

        # ================================================================
        # Step 2: 梯度引导候选生成
        # ================================================================
        if grad_saved is None:
            logger.warning(f"Step {step}: theta.grad is None, skipping mutation proposal")
            losses_history.append(fo_loss_val)
            continue

        candidates = propose_mutations(
            grad_matrix=grad_saved,
            current_seq=current_seq,
            K_top=K_top,
            B_candidates=B_candidates,
            T_sample=T_sample,
            max_mutations_per_seq=max_mutations_per_seq,
        )

        # ================================================================
        # Step 3: 零阶验证 B 条候选
        # ================================================================
        zeroth_results = []
        for b, cand_seq in enumerate(candidates):
            result = zeroth_order_eval(
                binder_seq=cand_seq,
                target_seq=target_seq,
                model=model,
                device=device,
                loss_weights=step_weights,
                contact_params=contact_params,
            )
            zeroth_results.append(result)
            logger.info(
                f"  Step {step} | candidate {b}/{B_candidates}: "
                f"seq={cand_seq[:15]}... | zeroth_loss={result.total_loss:.4f}"
            )

        # ================================================================
        # Step 4: 接受准则（贪心或模拟退火）
        # ================================================================
        round_best = min(zeroth_results, key=lambda r: r.total_loss)
        round_best_loss = round_best.total_loss

        accepted = False
        prob_accept = False

        if annealer is not None:
            accepted = annealer.accept(
                current_accepted_loss, round_best_loss, step, N_steps,
            )
            if accepted and round_best_loss >= current_accepted_loss:
                prob_accept = True
                n_probabilistic_accepts += 1
        else:
            accepted = round_best_loss < best_zeroth_loss

        if accepted:
            current_accepted_loss = round_best_loss
            current_accepted_seq = round_best.binder_seq

            if round_best_loss < best_zeroth_loss:
                best_zeroth_loss = round_best_loss
                best_zeroth_seq = round_best.binder_seq
                best_step = step

            _sync_theta_to_seq(theta, current_accepted_seq)
            optimizer.state.clear()
            optimizer = torch.optim.Adam([theta], lr=learning_rate)

        losses_history.append(round_best_loss)

        fo_dict = {k: v.item() if isinstance(v, torch.Tensor) else v
                   for k, v in first_order_losses.items()}
        logger.info(
            f"Step {step:3d}/{N_steps} | phase={phase_name} | tau={step_tau:.3f} | T={anneal_T:.4f} | "
            f"1st={fo_loss_val:.4f} | 0th_best={round_best_loss:.4f} | "
            f"global_best={best_zeroth_loss:.4f}@{best_step} | "
            f"accepted={accepted}{' (prob)' if prob_accept else ''} | seq={current_seq[:20]}..."
        )

    logger.info("=" * 70)
    logger.info("混合优化完成！")
    logger.info(f"零阶最佳序列: {best_zeroth_seq} (loss={best_zeroth_loss:.4f} @step {best_step})")
    logger.info(f"一阶最终序列: {current_seq}")
    logger.info(f"概率接受次数: {n_probabilistic_accepts}")
    logger.info("=" * 70)

    return best_zeroth_seq, current_seq, losses_history
