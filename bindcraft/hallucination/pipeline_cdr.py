"""
BindCraft v2 - CDR-Aware Hallucination Pipeline

克隆自 pipeline.py，扩展支持 CDR（互补决定区）自动识别与保护。
支持三种使用模式：
  A — 一键自动: pdb_id 自动下载 + ANARCI 识别 CDR
  B — 手动 JSON: 复用预先生成的 CDR JSON
  C — 传统纯序列: 无 CDR mask，等效于原 pipeline

CDR mask 注入的两个位置：
  ① propose_mutations 前 — 框架区梯度置零，阻止突变候选
  ③ zeroth_order_eval 前  — 安全网，强制不变框架区
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional, Any, Union
import logging
from pathlib import Path

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
    MutationProposal,
    extract_action_group_position_weights,
    propose_mutations,
    propose_mutations_legacy,
    propose_mutation_actions,
    normalize_mcts_prior_mode,
    select_de_novo_action_group,
)
from bindcraft.hallucination.zeroth_order import zeroth_order_eval
from bindcraft.hallucination.pipeline import (
    MCTS_STAGE2_EXPAND_ACTIONS_PER_STEP,
    MCTS_STAGE2_GROUP_TEMPERATURE,
    MCTS_STAGE2_PROGRESSIVE_WIDENING_ALPHA,
    MCTS_STAGE2_PROGRESSIVE_WIDENING_K,
    MCTS_STAGE2_MAX_CHILDREN_PER_NODE,
    MCTS_STAGE3_DIVERSITY_BONUS_MAX,
    MCTS_STAGE3_DIVERSITY_BONUS_WEIGHT,
    MCTS_STAGE3_LOSS_GROUP_BIN_WIDTH,
    MCTS_STAGE3_REPRESENTATIVES_PER_GROUP,
    MCTS_STAGE3_USE_DIVERSITY_BONUS,
    MCTS_STAGE3_USE_LOSS_GROUPING,
    MCTS_STAGE3_USE_PARETO_PRUNING,
    MCTS_STAGE3_USE_REWARD_NORMALIZATION,
    MCTS_STAGE3_USE_STRUCTURAL_GROUPING,
    _allocate_action_group_budgets,
    _apply_stageb_prior_mode_to_groups,
    _handoff_structural_key,
    _make_strict_mcts_tree,
    _make_stage_checkpoint_manager,
    _normalize_mcts_strategy,
    _renormalize_stageb_proposal_priors,
    _select_diverse_handoff_results,
    _stageb_group_proposal_loss,
    _stageb_max_mutations_for_step,
)
from bindcraft.losses.confidence import run_confidence_heads
from bindcraft.losses.total_loss import compute_all_losses
from protenix.model import sample_confidence

logger = logging.getLogger(__name__)


# ============================================================
# CDR 自动识别 — 内嵌 identify_cdr.py 核心逻辑
# ============================================================

def _identify_cdr_from_pdb(
    pdb_input: str,
    binder_chain: str = None,
    scheme: str = "chothia",
    cdr_def: str = "contact",
) -> dict:
    """
    给定 PDB ID 或 PDB 文件路径，自动完成 CDR 识别。
    返回 dict：binder_sequence, binder_cdr_mask, target_sequence, binder_chain_id
    """
    import logging as _logging
    from scripts.identify_cdr import (
        download_pdb,
        extract_sequences_from_pdb,
        classify_chains,
        format_chain_result,
    )

    _logger = _logging.getLogger(__name__)

    pdb_path = Path(pdb_input)
    if not pdb_path.is_file():
        if len(pdb_input) == 4 and pdb_input.isalnum():
            _logger.info(f"Downloading PDB {pdb_input} ...")
            pdb_path = download_pdb(pdb_input.upper())
        else:
            raise ValueError(f"Invalid PDB input: {pdb_input}")

    chains = extract_sequences_from_pdb(pdb_path)
    antibody_chains, antigen_chains = classify_chains(chains, scheme=scheme)

    if binder_chain:
        assert binder_chain in antibody_chains, \
            f"Chain {binder_chain} not an antibody chain"
        binder_chain_id = binder_chain
    else:
        heavy = [c for c, i in antibody_chains.items() if i["chain_type"] == "H"]
        binder_chain_id = heavy[0] if heavy else list(antibody_chains.keys())[0]

    ab_results = {}
    for chain_id, chain_info in antibody_chains.items():
        ab_results[chain_id] = format_chain_result(chain_info, cdr_def)

    binder_info = ab_results[binder_chain_id]
    # For crystallographic ASU with duplicated chains (e.g. 8YAF has 6 identical copies),
    # deduplicate identical antigen sequences so the target isn't artificially inflated
    unique_antigen_seqs = list(dict.fromkeys(antigen_chains.values()))
    target_sequence = "".join(unique_antigen_seqs)

    return {
        "binder_chain_id": binder_chain_id,
        "binder_sequence": binder_info["sequence"],
        "binder_cdr_mask": binder_info["cdr_mask"],
        "target_sequence": target_sequence,
    }


# ============================================================
# [已废弃] 纯一阶 Hallucination 循环
# 注意：本函数不再使用，当前算法为 run_hybrid_hallucination
# ============================================================
def run_hallucination(
    model,
    target_seq: str,
    L_binder: int = 40,
    N_steps: int = 200,
    learning_rate: float = 0.05,
    tau_noise: float = 4.0,
    device: torch.device = None,
    loss_weights: Dict[str, float] = None,
    hotspot_params: Optional[Dict] = None,
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

    logger.info("=" * 70)
    logger.info("开始 Hallucination 优化")
    logger.info(f"Target: {target_seq} ({len(target_seq)} aa)")
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
    theta = torch.randn(L_binder, NUM_AMINO_ACIDS, device=device) * 0.01
    theta.requires_grad_(True)

    optimizer = torch.optim.Adam([theta], lr=learning_rate)

    # ── 获取冻结的 Target 特征 ────────────────────────────────────────────
    logger.info("获取 Target 特征（target-only 单链）...")
    with torch.no_grad():
        feat_target_only, _, _ = build_single_chain_features(target_seq, device)
        feat_target_only = _prepare_input_features(model, feat_target_only)
        s_target = model.input_embedder(feat_target_only, inplace_safe=False)
        s_target = s_target.clone()
    logger.info(f"s_target shape: {s_target.shape}")

    init_binder = "".join(
        np.random.choice(list("ACDEFGHIKLMNPQRSTVWY")) for _ in range(L_binder)
    )
    logger.info(f"初始 Binder 序列（随机）: {init_binder}")

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
            device=device,
            chunk_size=None,
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


def run_hybrid_hallucination_cdr(
    model,
    pdb_id: str = None,
    binder_sequence: str = None,
    cdr_mask: List[bool] = None,
    target_seq: Union[str, List[str]] = None,
    target_chain_roles: Optional[Any] = None,
    L_binder: int = None,
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
    hotspot_params: Optional[Dict] = None,
    target_template_path: Optional[str] = None,
    # ── Mode A 专属参数 ───────────────────────────────────────────
    binder_chain: str = None,
    scheme: str = "chothia",
    chunk_size: int = None,
) -> Tuple[str, str, List[float]]:
    """
    CDR-aware 混合优化循环：支持三种模式自动切换。

    模式 A — 一键自动: pdb_id 非 None，自动下载 PDB → ANARCI → 开始优化
    模式 B — 手动 JSON: cdr_mask 非 None（binder_sequence 也需提供）
    模式 C — 传统纯序列: 两者均为 None，cdr_mask = [True]*L_binder（全可突变）

    CDR mask 注入的两个位置：
      ① propose_mutations 前 — 框架区梯度置零，阻止突变候选
      ③ zeroth_order_eval 前  — 安全网，强制不变框架区

    Args:
        model:         Protenix 模型
        pdb_id:        PDB ID 或文件路径（Mode A）
        binder_sequence: Binder 全长序列（Mode B/C）
        cdr_mask:      CDR mask List[bool]（Mode B）
        target_seq:    抗原/partner 序列（Mode B/C 必需，Mode A 可省略）
        target_chain_roles: 可选 partner 链角色，按 target_seq 顺序传入，
                          例如 ["target", "scaffold"]。
        L_binder:      Binder 长度（可省略，自动从 binder_sequence 推断）
        N_steps:       混合优化总轮数
        learning_rate: 一阶优化学习率
        tau_noise:     固定噪声时间步（仅在 use_tau_scheduling=False 时生效）
        device:        计算设备
        loss_weights:  Loss 权重字典（仅在 use_tau_scheduling=False 时生效）
        K_top:         Top-K 候选突变数
        B_candidates:  每轮每条分支生成的候选序列数（即 K'）
        T_sample:      候选采样温度
        max_mutations_per_seq: 每条候选最大突变数
        use_annealing: 启用模拟退火接受（否则纯贪心）
        T_init:        退火初始温度
        T_min:         退火最低温度
        use_tau_scheduling: 启用 Coarse-to-Fine tau 动态调度
        tau_jitter:    启用 tau 抖动
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
        target_template_path:   可选复合物 template 文件；CDR 模式下 framework
                                binder residues 作为 template context 保留
        binder_chain:   指定 binder chain ID（Mode A，默认为第一条 heavy chain）
        scheme:         ANARCI numbering scheme（Mode A，默认 chothia）

    Returns:
        best_seq:        零阶验证下的最优序列
        final_seq:       最后一轮的全局最佳序列
        losses_history:  每轮零阶验证最优 loss 的历史
    """
    from bindcraft.hallucination.scheduler import AnnealingScheduler, TauScheduler, MacroBreathTracker

    if device is None:
        device = next(model.parameters()).device

    # ── CDR 自动推断 ────────────────────────────────────────────────────────────
    if pdb_id is not None:
        # 模式 A：一键从 PDB 开始
        cdr_info = _identify_cdr_from_pdb(pdb_id, binder_chain=binder_chain, scheme=scheme)
        binder_sequence = cdr_info["binder_sequence"]
        cdr_mask = cdr_info["binder_cdr_mask"]
        inferred_target = cdr_info["target_sequence"]
        if target_seq is None:
            target_seq = inferred_target
        elif target_seq != inferred_target:
            logger.warning(
                f"Target seq mismatch: CLI='{target_seq}' vs PDB antigen='{inferred_target}'"
                f" — using CLI value."
            )
        L_binder = len(binder_sequence)
        logger.info(
            f"[CDR-auto] Mode A — PDB={pdb_id} | binder={cdr_info['binder_chain_id']}"
            f" ({L_binder} aa) | CDR={sum(cdr_mask)}/{L_binder}"
        )
    elif cdr_mask is not None:
        # 模式 B：CDR mask 手动提供
        if binder_sequence is None and L_binder is None:
            raise ValueError("Mode B requires --binder_sequence or --L_binder")
        L_binder = L_binder or len(binder_sequence)
        logger.info(
            f"[CDR-auto] Mode B — CDR={sum(cdr_mask)}/{L_binder} residues mutable"
        )
    else:
        # 模式 C：传统纯序列模式（cdr_mask = 全 True）
        L_binder = L_binder or 40
        cdr_mask = [True] * L_binder
        logger.info(f"[CDR-auto] Mode C — no CDR mask, full-length optimization")

    assert len(cdr_mask) == L_binder, \
        f"cdr_mask length {len(cdr_mask)} != L_binder {L_binder}"

    base_weights = dict(DEFAULT_LOSS_WEIGHTS)
    if loss_weights is not None:
        base_weights.update(loss_weights)
    contact_params = dict(DEFAULT_CONTACT_PARAMS)
    partner_roles = _normalize_partner_chain_roles(target_seq, target_chain_roles)
    binder_template_mask = _cdr_binder_template_mask(cdr_mask, target_template_path)

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
            binder_seq="PLACEHOLDER",
            target_seq=target_seq,
        )
        logger.info(f"Checkpoint 保存已启用: {checkpoint_dir}")

    logger.info("=" * 70)
    logger.info("CDR-Aware 混合优化 (Mode A/B/C auto-detected above)")
    tgt_seqs = _normalize_chain_sequences(target_seq)
    logger.info(
        "Target/partner chains: %d | lengths=%s | total=%d aa",
        len(tgt_seqs),
        "+".join(str(len(seq)) for seq in tgt_seqs),
        sum(len(seq) for seq in tgt_seqs),
    )
    if partner_roles is not None:
        logger.info("Partner chain roles: %s", partner_roles)
    logger.info(f"Binder 长度: {L_binder} aa | CDR residues: {sum(cdr_mask)}/{L_binder}")
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
        logger.info(f"Beam Search: num_beams={num_beams} (每轮 {num_beams}×{B_candidates}={num_beams*B_candidates} 条候选)")
    else:
        logger.info("Beam Search: 关闭（单路径模式）")
    logger.info(f"M_prior (BLOSUM62): use_prior={use_prior}, prior_strength={prior_strength}")
    if use_tau_fusion:
        logger.info(f"τ-fusion: tau_coarse={tau_coarse}, tau_fine={tau_fine}, alpha={alpha}")
    else:
        logger.info("τ-fusion: 关闭")
    logger.info("Template: %s | binder_template_mask=%s", target_template_path, binder_template_mask is not None)
    logger.info(f"Base loss 权重: {base_weights}")
    logger.info("Step 0: 零阶全链路初始化，将在优化循环前执行完整 AF3 推断获取初始锚点骨架")
    logger.info("=" * 70)

    freeze_model(model)

    # ── 预计算 W_AAE ──────────────────────────────────────────────────────────
    logger.info("预计算 W_AAE...")
    W_AAE = precompute_waa_embeddings(model, device)

    # ── Beam 初始化 ────────────────────────────────────────────────────────────
    thetas = []
    optimizers = []
    beam_seqs = []
    beam_losses = []

    torch.manual_seed(42)
    np.random.seed(42)
    for b in range(num_beams):
        theta_b = torch.randn(L_binder, NUM_AMINO_ACIDS, device=device) * 0.01
        theta_b.requires_grad_(True)
        thetas.append(theta_b)
        optimizers.append(torch.optim.Adam([theta_b], lr=learning_rate))

    # ── 获取冻结的 Target 特征 ────────────────────────────────────────────────
    logger.info("获取 Target 特征（target-only 单链）...")
    with torch.no_grad():
        feat_target_only, _, _ = build_single_chain_features(target_seq, device)
        feat_target_only = _prepare_input_features(model, feat_target_only)
        s_target = model.input_embedder(feat_target_only, inplace_safe=False)
        s_target = s_target.clone()
    logger.info(f"s_target shape: {s_target.shape}")

    # Mode A/B：用真实抗体序列作为起点；Mode C 随机初始化
    if binder_sequence is not None and not all(cdr_mask):
        # Mode A/B：提供了真实 binder 序列且非全可突变
        init_binder = binder_sequence
        logger.info(f"初始 Binder 序列（真实抗体）: {init_binder}")
    else:
        # Mode C 或全 True：随机初始化
        init_binder = "".join(
            np.random.choice(list("ACDEFGHIKLMNPQRSTVWY")) for _ in range(L_binder)
        )
        logger.info(f"初始 Binder 序列（随机）: {init_binder}")

    # 初始序列复制到所有 beam
    for b in range(num_beams):
        beam_seqs.append(init_binder)
        beam_losses.append(float("inf"))

    # ── 获取 bin 参数 ─────────────────────────────────────────────────────
    plddt_params = sample_confidence.get_bin_params(model.configs.loss.plddt)
    pae_params = sample_confidence.get_bin_params(model.configs.loss.pae)
    distogram_params = sample_confidence.get_bin_params(model.configs.loss.distogram)
    pde_params = sample_confidence.get_bin_params(model.configs.loss.pde)

    # ── Step 0: 零阶全链路初始化 ─────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Step 0: 零阶全链路初始化，获取初始真实物理锚点骨架...")
    init_zeroth = zeroth_order_eval(
        binder_seq=init_binder,
        target_seq=target_seq,
        model=model,
        device=device,
        loss_weights=base_weights,
        contact_params=contact_params,
        chunk_size=None,
        hotspot_params=hotspot_params,
        opt_token_mask=cdr_mask,
        target_template_path=target_template_path,
        binder_template_mask=binder_template_mask,
        target_chain_roles=partner_roles,
    )
    logger.info(
        f"  初始零阶 total_loss={init_zeroth.total_loss:.4f} | "
        f"pLDDT={init_zeroth.summary_confidence.get('plddt', 'N/A')} | "
        f"iPTM={init_zeroth.summary_confidence.get('i_ptm', 'N/A')}"
    )
    X_clean_0 = init_zeroth.coordinate[0:1]
    x_coarse_caches = [X_clean_0.clone().to(device) for _ in range(num_beams)]
    logger.info(f"X_clean_0 已注入骨架缓存，shape={X_clean_0.shape}，num_beams={num_beams}")
    logger.info("=" * 70)

    # ── 状态变量 ──────────────────────────────────────────────────────────
    # 用 Phase 0 的权重体系建立基准，使 global_best 与主循环所有候选在同一套权重下比较。
    # 主循环 step=0 使用 tau_sched.get_dynamic_weights(step=0)，即 WEIGHT_PROFILES[0]（Phase 0 权重）。
    # 如果 tau_sched 未启用，则退回到 base_weights（与 step_weights 一致）。
    if tau_sched is not None:
        phase0_weights = tau_sched.get_dynamic_weights(step=0, total_steps=N_steps, base_weights=base_weights)
    else:
        phase0_weights = base_weights
    logger.info(f"建立基准 (Phase 0 权重体系): {phase0_weights}")
    baseline_for_best = zeroth_order_eval(
        binder_seq=init_binder,
        target_seq=target_seq,
        model=model,
        device=device,
        loss_weights=phase0_weights,
        contact_params=contact_params,
        chunk_size=None,
        hotspot_params=hotspot_params,
        opt_token_mask=cdr_mask,
        target_template_path=target_template_path,
        binder_template_mask=binder_template_mask,
        target_chain_roles=partner_roles,
    )
    logger.info(f"  基准 loss={baseline_for_best.total_loss:.4f} | pLDDT={baseline_for_best.summary_confidence.get('plddt', 'N/A')}")
    best_zeroth_loss = baseline_for_best.total_loss
    best_zeroth_seq = init_binder
    best_step = 0
    losses_history = []
    n_probabilistic_accepts = 0

    for b in range(num_beams):
        beam_losses[b] = baseline_for_best.total_loss
    if breath_tracker is not None:
        breath_tracker._best_loss = baseline_for_best.total_loss

    # ── Step 0 Checkpoint ──────────────────────────────────────────────────
    # 使用 baseline_for_best（Phase 0 权重）而非 init_zeroth（base_weights），
    # 使 ckpt_mgr 内部的 best_loss 与主循环 Phase 0 权重体系完全一致。
    if ckpt_mgr is not None:
        ckpt_mgr.save(
            step=0, beam_id=0, zeroth_result=baseline_for_best,
            plddt_logits=baseline_for_best.plddt_logits,
            accepted=True, is_global_best=True,
        )
        logger.info(f"[Checkpoint] Step 0 保存完毕: step_0000/")

    logger.info("开始 CDR-Aware 混合优化主循环...")

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

        # ── 宏观呼吸覆盖 ─────────────────────────────────────────────────
        if breath_tracker is not None and breath_tracker.is_breathing:
            step_tau = breath_tracker.get_breathing_tau()
            step_weights = breath_tracker.apply_breathing_weights(step_weights)
            phase_name = "macro_breathing"

        anneal_T = annealer.get_temperature(step, N_steps) if annealer else 0.0

        # ================================================================
        # Step 1: 每个分支独立一阶反传
        # ================================================================
        step_grads = []
        step_fo_losses = []
        step_current_seqs = []

        for b in range(num_beams):
            optimizers[b].zero_grad()

            # 在 ste_bridge 前强制同步：保证 argmax 与骨架缓存的序列完全一致，
            # 避免骨架回退。CDR 保护由注入点①（梯度掩码）和③（安全网）保证。
            _sync_theta_to_seq(thetas[b], beam_seqs[b])

            progress = step / max(N_steps - 1, 1)
            tau_temp = 2.0 * (1.0 - progress) + 0.05 * progress

            s_binder, current_seq, input_feature_dict = ste_bridge(
                theta=thetas[b], W_AAE=W_AAE, model=model,
                target_seq=target_seq, device=device, tau_temp=tau_temp,
                target_template_path=target_template_path,
                binder_template_mask=binder_template_mask,
            )

            if use_tau_fusion:
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
                    opt_token_mask=cdr_mask,
                    target_chain_roles=partner_roles,
                    chunk_size=None,
                )
                torch.nn.utils.clip_grad_norm_([thetas[b]], 1.0)
                optimizers[b].step()
                fo_loss_val = None
                grad_for_proposal = grad_fused
            else:
                x0_hat, s_trunk, z_trunk = diffusion_one_step(
                    model=model, s_binder=s_binder, s_target=s_target,
                    input_feature_dict=input_feature_dict,
                    x_coarse=None,
                    X_clean=x_coarse_caches[b],
                    tau=step_tau, device=device,
                    chunk_size=None,
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
                    opt_token_mask=cdr_mask,
                    target_chain_roles=partner_roles,
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
        # Step 2: 收集全部候选（注入点①：框架区梯度置零）
        # ================================================================
        all_candidates = []
        all_candidate_beams = []

        for b in range(num_beams):
            if step_grads[b] is None:
                continue
            grads_cpu = step_grads[b].detach().cpu()

            # ① CDR-aware：将框架区的梯度置零，这些位置不会产生突变候选
            grad_masked = grads_cpu.clone()
            framework_indices = [i for i, is_cdr in enumerate(cdr_mask) if not is_cdr]
            grad_masked[framework_indices, :] = 0.0

            cands = propose_mutations(
                grad_matrix=grad_masked,
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
        # Step 3: 零阶验证全部候选（注入点③：CDR 安全网）
        # ================================================================
        all_zeroth_results = []

        for idx, cand_seq in enumerate(all_candidates):
            beam_src = all_candidate_beams[idx]

            # ③ CDR 安全网：确保候选不变框架区
            framework_mismatch = [
                i for i, is_cdr in enumerate(cdr_mask) if not is_cdr
                and cand_seq[i] != step_current_seqs[beam_src][i]
            ]
            if framework_mismatch:
                logger.warning(
                    f"Step {step} | beam {beam_src}: CDR guard clipped framework at {framework_mismatch}"
                )
                cand_seq = list(cand_seq)
                for i in framework_mismatch:
                    cand_seq[i] = step_current_seqs[beam_src][i]
                cand_seq = "".join(cand_seq)

            result = zeroth_order_eval(
                binder_seq=cand_seq,
                target_seq=target_seq,
                model=model,
                device=device,
                loss_weights=step_weights,
                contact_params=contact_params,
                chunk_size=None,
                hotspot_params=hotspot_params,
                opt_token_mask=cdr_mask,
                target_template_path=target_template_path,
                binder_template_mask=binder_template_mask,
                target_chain_roles=partner_roles,
            )
            all_zeroth_results.append(result)
            logger.debug(
                f"  Step {step} | cand {idx}/{len(all_candidates)} "
                f"(beam={beam_src}) | seq={cand_seq[:15]}... | zo={result.total_loss:.4f}"
            )

        # ── 宏观呼吸检测 ────────────────────────────────────────────────
        if breath_tracker is not None:
            round_min_loss = min(r.total_loss for r in all_zeroth_results)
            breath_tracker.update(round_min_loss)

        # ================================================================
        # Step 4: Beam Search 全局排序
        # ================================================================
        if num_beams == 1:
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
                # <= 而非 <：第一轮候选（best=inf 时）自动接受；
                # 后续轮次仅当新候选优于等于当前 global_best 时接受
                accepted = round_best_loss <= best_zeroth_loss

            if accepted:
                beam_losses[0] = round_best_loss
                beam_seqs[0] = round_best.binder_seq
                x_coarse_caches[0] = round_best.coordinate[0:1].clone().to(device)
                if round_best_loss < best_zeroth_loss:
                    best_zeroth_loss = round_best_loss
                    best_zeroth_seq = round_best.binder_seq
                    best_step = step
                # 接受后同步：全序列 one-hot（框架区和 CDR 区都用目标序列覆盖），
                # 保证 ste_bridge 的 argmax 与 zeroth_order 完全一致
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

            if ckpt_mgr is not None:
                ckpt_mgr.save(
                    step=step, beam_id=0, zeroth_result=round_best,
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
            # Beam Search
            sorted_indices = sorted(
                range(len(all_zeroth_results)),
                key=lambda i: all_zeroth_results[i].total_loss
            )
            top_beam_indices = sorted_indices[:num_beams]

            step_accepted_count = 0
            beam_accepted_map: Dict[int, bool] = {}
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
                    # <= 而非 <：第一轮候选（best=inf 时）自动接受；
                    # 后续轮次仅当新候选优于等于当前 global_best 时接受
                    accepted = result.total_loss <= best_zeroth_loss

                beam_accepted_map[src_beam] = accepted

                if accepted:
                    beam_losses[src_beam] = result.total_loss
                    beam_seqs[src_beam] = result.binder_seq
                    x_coarse_caches[src_beam] = result.coordinate[0:1].clone().to(device)
                    step_accepted_count += 1
                    if result.total_loss < best_zeroth_loss:
                        best_zeroth_loss = result.total_loss
                        best_zeroth_seq = result.binder_seq
                        best_step = step
                    # 接受后同步：全序列 one-hot
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

            if ckpt_mgr is not None:
                for rank, cand_idx in enumerate(top_beam_indices):
                    src_beam = all_candidate_beams[cand_idx]
                    result = all_zeroth_results[cand_idx]
                    accepted = beam_accepted_map.get(src_beam, False)
                    is_best = (result.total_loss == best_zeroth_loss)
                    ckpt_mgr.save(
                        step=step, beam_id=src_beam, zeroth_result=result,
                        plddt_logits=result.plddt_logits,
                        accepted=accepted, is_global_best=is_best,
                    )
                best_info = ckpt_mgr.get_best_info()
                logger.info(
                    f"  [Checkpoint] 保存 {step_accepted_count}/{num_beams} beam(s) | "
                    f"global_best={best_info['loss']:.4f} @ step={best_info['step']} beam={best_info['beam']}"
                )

    logger.info("=" * 70)
    logger.info("CDR-Aware 混合优化完成！")
    logger.info(f"零阶最佳序列: {best_zeroth_seq} (loss={best_zeroth_loss:.4f} @step {best_step})")
    logger.info(f"CDR residues: {sum(cdr_mask)}/{L_binder} | Framework: {L_binder - sum(cdr_mask)}/{L_binder}")
    logger.info(f"概率接受次数: {n_probabilistic_accepts}")
    if num_beams > 1:
        for b in range(num_beams):
            logger.info(f"  Beam {b}: {beam_seqs[b]} (loss={beam_losses[b]:.4f})")
    logger.info("=" * 70)

    return best_zeroth_seq, best_zeroth_seq, losses_history


# ============================================================
# MCTS 模式：CDR-Aware 版本
# ============================================================


def _init_child_logits_for_mcts_cdr(
    parent_node: Any,
    child_seq: str,
    cdr_mask: List[bool],
    device: torch.device,
) -> torch.Tensor:
    """CDR-aware MCTS：子节点 Logits 初始化，框架区与全序列 one-hot 对齐。"""
    if len(cdr_mask) != len(child_seq):
        raise ValueError("cdr_mask length must match child_seq length")
    child_logits = parent_node.logits.clone().to(device)
    _sync_theta_to_seq(child_logits, child_seq)
    return child_logits


def _clip_seq_by_cdr_mask(
    seq: str,
    cdr_mask: List[bool],
    reference_seq: str,
) -> str:
    """
    CDR 安全网：将序列中非 CDR 区的位置强制还原为参考序列。
    用于候选提议后和零阶验证前的双重保护。
    """
    if len(seq) != len(cdr_mask):
        raise ValueError(f"seq len {len(seq)} != cdr_mask len {len(cdr_mask)}")
    if len(seq) != len(reference_seq):
        raise ValueError(f"seq len {len(seq)} != ref len {len(reference_seq)}")

    seq_list = list(seq)
    clipped = False
    for i, is_cdr in enumerate(cdr_mask):
        if not is_cdr and seq_list[i] != reference_seq[i]:
            seq_list[i] = reference_seq[i]
            clipped = True
    if clipped:
        logger.debug("CDR guard: clipped framework to reference")
    return "".join(seq_list)


def _normalize_partner_chain_roles(
    target_seq: Union[str, List[str]],
    target_chain_roles: Optional[Any],
) -> Optional[List[str]]:
    """Normalize optional CDR/TCR partner roles ordered like target_seq chains."""
    if target_chain_roles is None:
        return None
    if isinstance(target_chain_roles, str):
        roles = [part.strip() for part in target_chain_roles.split(",") if part.strip()]
    else:
        roles = [str(role).strip() for role in target_chain_roles]
    target_sequences = _normalize_chain_sequences(target_seq)
    if len(roles) != len(target_sequences):
        raise ValueError(
            f"target_chain_roles length {len(roles)} must equal target_seq chain count "
            f"{len(target_sequences)}"
        )
    return roles


def _cdr_binder_template_mask(
    cdr_mask: List[bool],
    target_template_path: Optional[str],
) -> Optional[List[bool]]:
    """
    Keep framework template coordinates while masking mutable CDR residues.

    Returning None preserves the de novo target-only template path.
    """
    if target_template_path is None:
        return None
    if not cdr_mask or all(cdr_mask):
        return None
    return [not is_cdr for is_cdr in cdr_mask]


def _cdr_stageb_position_weights(
    sequence: str,
    cdr_mask: List[bool],
    action_group: Optional[str],
    structural_signature: Optional[Dict[str, Any]],
) -> torch.Tensor:
    """
    Combine document-style structural routing with the antibody CDR mask.

    The CDR mask is hard: framework positions receive zero weight even when
    structural routing points there. If no structural residue map is available,
    all CDR residues remain eligible.
    """
    if len(sequence) != len(cdr_mask):
        raise ValueError("sequence and cdr_mask lengths must match")

    cdr_weights = torch.tensor(
        [1.0 if is_cdr else 0.0 for is_cdr in cdr_mask],
        dtype=torch.float32,
    )
    routed = extract_action_group_position_weights(
        sequence,
        action_group,
        structural_signature=structural_signature,
    )
    if routed is None or routed.numel() != cdr_weights.numel():
        return cdr_weights

    routed = torch.clamp(routed.detach().cpu().flatten().float(), min=0.0)
    combined = routed * cdr_weights
    if float(combined.max().item()) <= 0.0:
        return cdr_weights
    return combined


def _clip_mcts_proposal_by_cdr_mask(
    proposal: MutationProposal,
    cdr_mask: List[bool],
    reference_seq: str,
) -> MutationProposal:
    clipped_sequence = _clip_seq_by_cdr_mask(
        proposal.sequence,
        cdr_mask,
        reference_seq,
    )
    if clipped_sequence == proposal.sequence:
        return proposal

    clipped_edits = [
        edit for edit in proposal.edits
        if 0 <= int(edit[0]) < len(cdr_mask) and cdr_mask[int(edit[0])]
    ]
    return MutationProposal(
        sequence=clipped_sequence,
        edits=clipped_edits,
        prior=proposal.prior,
        benefit=proposal.benefit,
        action_group=proposal.action_group,
        group_prior=proposal.group_prior,
    )


def run_mcts_hallucination_cdr(
    model,
    pdb_id: str = None,
    binder_sequence: str = None,
    cdr_mask: List[bool] = None,
    target_seq: Union[str, List[str]] = None,
    target_chain_roles: Optional[Any] = None,
    L_binder: int = None,
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
    hotspot_params: Optional[Dict] = None,
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
    binder_chain: str = None,
    scheme: str = "chothia",
    chunk_size: int = None,
    target_template_path: Optional[str] = None,
) -> Tuple[str, str, List[float]]:
    """
    CDR-Aware MCTS：与 ``run_hybrid_hallucination_cdr`` 相同的 Mode A/B/C 检测，
    Phase 0 与主循环中透传 ``hotspot_params`` / ``chunk_size``。
    """
    from bindcraft.hallucination.scheduler import AnnealingScheduler

    if device is None:
        device = next(model.parameters()).device

    mcts_strategy = _normalize_mcts_strategy(mcts_strategy)
    mcts_bootstrap_steps = max(0, int(mcts_bootstrap_steps))
    mcts_bootstrap_patience = max(1, int(mcts_bootstrap_patience))
    mcts_bootstrap_min_delta = max(float(mcts_bootstrap_min_delta), 0.0)
    mcts_bootstrap_max_steps = max(mcts_bootstrap_steps, int(mcts_bootstrap_max_steps))
    mcts_handoff_top_k = max(1, int(mcts_handoff_top_k or num_beams))
    mcts_expand_actions_per_step = max(int(mcts_expand_actions_per_step), 1)
    mcts_prior_mode = normalize_mcts_prior_mode(mcts_prior_mode)

    # ── CDR 自动推断（与 hybrid 对齐）─────────────────────────────────────
    if pdb_id is not None:
        cdr_info = _identify_cdr_from_pdb(pdb_id, binder_chain=binder_chain, scheme=scheme)
        binder_sequence = cdr_info["binder_sequence"]
        cdr_mask = cdr_info["binder_cdr_mask"]
        inferred_target = cdr_info["target_sequence"]
        if target_seq is None:
            target_seq = inferred_target
        elif target_seq != inferred_target:
            logger.warning(
                f"Target seq mismatch: CLI='{target_seq}' vs PDB antigen='{inferred_target}'"
                f" — using CLI value."
            )
        L_binder = len(binder_sequence)
        logger.info(
            f"[CDR-auto] Mode A — PDB={pdb_id} | binder={cdr_info['binder_chain_id']}"
            f" ({L_binder} aa) | CDR={sum(cdr_mask)}/{L_binder}"
        )
    elif cdr_mask is not None:
        if binder_sequence is None and L_binder is None:
            raise ValueError("Mode B requires --binder_sequence or --L_binder")
        L_binder = L_binder or len(binder_sequence)
        logger.info(
            f"[CDR-auto] Mode B — CDR={sum(cdr_mask)}/{L_binder} residues mutable"
        )
    else:
        L_binder = L_binder or 40
        cdr_mask = [True] * L_binder
        logger.info("[CDR-auto] Mode C — no CDR mask, full-length optimization")

    assert len(cdr_mask) == L_binder, (
        f"cdr_mask length {len(cdr_mask)} != L_binder {L_binder}"
    )
    partner_roles = _normalize_partner_chain_roles(target_seq, target_chain_roles)
    binder_template_mask = _cdr_binder_template_mask(cdr_mask, target_template_path)

    base_weights = dict(DEFAULT_LOSS_WEIGHTS)
    if loss_weights is not None:
        base_weights.update(loss_weights)
    contact_params = dict(DEFAULT_CONTACT_PARAMS)

    framework_indices = [i for i, is_cdr in enumerate(cdr_mask) if not is_cdr]
    has_cdr_protection = len(framework_indices) > 0 and any(not m for m in cdr_mask)

    annealer = AnnealingScheduler(T_init=T_init, T_min=T_min) if use_annealing else None

    if mcts_strategy == "two_stage":
        phase0_ckpt_mgr = _make_stage_checkpoint_manager(
            checkpoint_dir, "phase0", mcts_strategy, target_seq
        )
        legacy_ckpt_mgr = _make_stage_checkpoint_manager(
            checkpoint_dir, "legacy_bootstrap", mcts_strategy, target_seq
        )
        strict_ckpt_mgr = _make_stage_checkpoint_manager(
            checkpoint_dir, "document_puct_refine", mcts_strategy, target_seq
        )
    elif mcts_strategy == "legacy":
        legacy_ckpt_mgr = _make_stage_checkpoint_manager(
            checkpoint_dir, "legacy_bootstrap", mcts_strategy, target_seq
        )
        phase0_ckpt_mgr = legacy_ckpt_mgr
        strict_ckpt_mgr = None
    else:
        strict_ckpt_mgr = _make_stage_checkpoint_manager(
            checkpoint_dir, "document_puct_refine", mcts_strategy, target_seq
        )
        phase0_ckpt_mgr = strict_ckpt_mgr
        legacy_ckpt_mgr = None

    tgt_seqs = _normalize_chain_sequences(target_seq)
    tgt_len_str = "+".join(str(len(s)) for s in tgt_seqs)
    tgt_total = sum(len(s) for s in tgt_seqs)

    logger.info("=" * 70)
    logger.info("CDR-Aware MCTS 混合优化 (Mode A/B/C auto-detected above)")
    logger.info(
        f"Target chains: {len(tgt_seqs)} | lengths={tgt_len_str} | total={tgt_total} aa"
    )
    logger.info(f"Binder 长度: {L_binder} aa | CDR residues: {sum(cdr_mask)}/{L_binder}")
    logger.info("Partner roles: %s", partner_roles or "all_target")
    logger.info("Template: %s | binder_template_mask=%s", target_template_path, binder_template_mask is not None)
    logger.info(f"总轮数: {N_steps} | K_top={K_top} B={B_candidates} T_sample={T_sample}")
    logger.info(f"MCTS: C={mcts_C}, max_nodes={mcts_max_nodes}, tau_fixed={mcts_tau_fixed}")
    logger.info(
        "CDR MCTS strategy=%s | bootstrap_steps=%d patience=%d min_delta=%.4f "
        "max_bootstrap=%d handoff_top_k=%d",
        mcts_strategy,
        mcts_bootstrap_steps,
        mcts_bootstrap_patience,
        mcts_bootstrap_min_delta,
        mcts_bootstrap_max_steps,
        mcts_handoff_top_k,
    )
    logger.info(
        "CDR Document PUCT: progressive_widening(k=%.1f, alpha=%.2f, max_children=%d) "
        "+ action_puct(expand_per_step=%d) + CDR hard mask",
        MCTS_STAGE2_PROGRESSIVE_WIDENING_K,
        MCTS_STAGE2_PROGRESSIVE_WIDENING_ALPHA,
        MCTS_STAGE2_MAX_CHILDREN_PER_NODE,
        mcts_expand_actions_per_step,
    )
    logger.info("CDR MCTS prior mode: %s", mcts_prior_mode)
    logger.info(
        "CDR Stage3-lite: loss_grouping=%s(bin_width=%.2f, reps=%d) "
        "structural_grouping=%s pareto=%s reward_normalization=%s "
        "diversity_bonus=%s(weight=%.3f, max=%.1f)",
        MCTS_STAGE3_USE_LOSS_GROUPING,
        MCTS_STAGE3_LOSS_GROUP_BIN_WIDTH,
        MCTS_STAGE3_REPRESENTATIVES_PER_GROUP,
        MCTS_STAGE3_USE_STRUCTURAL_GROUPING,
        MCTS_STAGE3_USE_PARETO_PRUNING,
        MCTS_STAGE3_USE_REWARD_NORMALIZATION,
        MCTS_STAGE3_USE_DIVERSITY_BONUS,
        MCTS_STAGE3_DIVERSITY_BONUS_WEIGHT,
        MCTS_STAGE3_DIVERSITY_BONUS_MAX,
    )
    logger.info(f"Annealing: {use_annealing}")
    logger.info(f"CDR protection: {len(framework_indices)} framework residues")
    logger.info(f"M_prior (BLOSUM62): use_prior={use_prior}, prior_strength={prior_strength}")
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
            mcts_C,
            mcts_max_nodes,
            mcts_tau_fixed,
            mcts_expand_actions_per_step,
        )
        if mcts_strategy == "strict"
        else None
    )
    if mcts_tree is not None:
        logger.info(
            "CDR Document PUCT MCTS 树已创建: C=%s, max_nodes=%s, tau_fixed=%s",
            mcts_C,
            mcts_max_nodes,
            mcts_tau_fixed,
        )

    logger.info("=" * 70)
    logger.info("Phase 0: 全局冷启动 (Best-of-N)")

    if binder_sequence is not None and not all(cdr_mask):
        init_binder = binder_sequence
        logger.info(f"初始 Binder（真实抗体）: {init_binder}")
    else:
        torch.manual_seed(42)
        np.random.seed(42)
        init_binder = "".join(
            np.random.choice(list("ACDEFGHIKLMNPQRSTVWY")) for _ in range(L_binder)
        )
        logger.info(f"初始 Binder（随机）: {init_binder}")

    torch.manual_seed(42)
    np.random.seed(42)

    if binder_sequence is not None and not all(cdr_mask):
        phase0_candidates = [init_binder]
        cdr_indices = [i for i, m in enumerate(cdr_mask) if m]
        for _ in range(num_beams - 1):
            cand_list = list(init_binder)
            n_mut = min(len(cdr_indices), np.random.randint(1, 3))
            for pos in np.random.choice(cdr_indices, n_mut, replace=False):
                cand_list[pos] = np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"))
            phase0_candidates.append("".join(cand_list))
    else:
        phase0_candidates = []
        for _ in range(num_beams):
            seq = "".join(
                np.random.choice(list("ACDEFGHIKLMNPQRSTVWY")) for _ in range(L_binder)
            )
            phase0_candidates.append(seq)

    logger.info(f"Phase 0：对 {len(phase0_candidates)} 条候选执行 Step3 零阶验证...")
    phase0_results = []
    verifier_cache: Dict[str, Any] = {}
    for idx, cand_seq in enumerate(phase0_candidates):
        if has_cdr_protection:
            cand_seq = _clip_seq_by_cdr_mask(cand_seq, cdr_mask, init_binder)

        logger.info(f"  Phase0 [{idx+1}/{len(phase0_candidates)}]: {cand_seq[:20]}...")
        result = zeroth_order_eval(
            binder_seq=cand_seq,
            target_seq=target_seq,
            model=model,
            device=device,
            loss_weights=base_weights,
            contact_params=contact_params,
            chunk_size=chunk_size,
            hotspot_params=hotspot_params,
            opt_token_mask=cdr_mask,
            target_template_path=target_template_path,
            binder_template_mask=binder_template_mask,
            target_chain_roles=partner_roles,
        )
        phase0_results.append((cand_seq, result))
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

    torch.manual_seed(42)
    np.random.seed(42)
    for i in range(top_beam):
        seq, result = phase0_results[i]
        theta_root = torch.zeros(L_binder, NUM_AMINO_ACIDS, device=device)
        _sync_theta_to_seq(theta_root, seq)
        theta_root.requires_grad_(True)

        if mcts_tree is not None:
            mcts_tree.add_root_node(
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
            root_node = mcts_tree.get_node_by_sequence(seq)
            root_reward = root_node.tree_reward if root_node is not None else mcts_tree.result_reward(result)
            logger.info(
                f"  根节点 {i+1}: {seq[:20]}... "
                f"L={result.total_loss:.4f} R={root_reward:.4f}"
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
        from bindcraft.hallucination.mcts_legacy import LegacyMCTSTree

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
            "CDR StageA legacy_bootstrap 启动: max_steps=%d min_steps=%d patience=%d min_delta=%.4f",
            legacy_limit,
            legacy_min_steps,
            mcts_bootstrap_patience,
            mcts_bootstrap_min_delta,
        )

        for legacy_step in range(legacy_limit):
            selected_node = legacy_tree.select()
            selected_seq = selected_node.sequence
            selected_logits = selected_node.logits.clone().to(device)

            if has_cdr_protection:
                selected_seq_clipped = _clip_seq_by_cdr_mask(selected_seq, cdr_mask, init_binder)
                if selected_seq_clipped != selected_seq:
                    logger.debug(
                        "StageA Step %d: CDR guard clipped selected_seq framework, id=%d",
                        legacy_step,
                        selected_node.node_id,
                    )
                    selected_seq = selected_seq_clipped
                    _sync_theta_to_seq(selected_logits, selected_seq)

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
                binder_template_mask=binder_template_mask,
            )

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

            first_order_loss, _ = compute_all_losses(
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
                opt_token_mask=cdr_mask,
                target_chain_roles=partner_roles,
            )

            first_order_loss.backward()
            grad_for_proposal = theta_step2.grad.clone() if theta_step2.grad is not None else None
            torch.nn.utils.clip_grad_norm_([theta_step2], 1.0)
            optimizer_step2.step()
            fo_loss_val = first_order_loss.item()

            if grad_for_proposal is None:
                logger.warning("CDR StageA Step %d: grad is None, skipping candidate generation", legacy_step)
                losses_history.append(best_zeroth_loss)
                stale_steps += 1
                legacy_steps_done += 1
                continue

            grads_cpu = grad_for_proposal.detach().cpu()
            if has_cdr_protection:
                grads_cpu = grads_cpu.clone()
                grads_cpu[framework_indices, :] = 0.0

            candidates = propose_mutations_legacy(
                grad_matrix=grads_cpu,
                current_seq=current_seq_step2,
                K_top=K_top,
                B_candidates=B_candidates,
                T_sample=T_sample,
                max_mutations_per_seq=max_mutations_per_seq,
                use_prior=use_prior,
                prior_strength=prior_strength,
            )

            if not candidates:
                logger.warning("CDR StageA Step %d: no candidates generated, skipping", legacy_step)
                losses_history.append(best_zeroth_loss)
                stale_steps += 1
                legacy_steps_done += 1
                continue

            all_zeroth_results = []
            for cand_seq in candidates:
                if has_cdr_protection:
                    cand_seq = _clip_seq_by_cdr_mask(cand_seq, cdr_mask, init_binder)

                result = zeroth_order_eval(
                    binder_seq=cand_seq,
                    target_seq=target_seq,
                    model=model,
                    device=device,
                    loss_weights=base_weights,
                    contact_params=contact_params,
                    chunk_size=chunk_size,
                    hotspot_params=hotspot_params,
                    opt_token_mask=cdr_mask,
                    target_template_path=target_template_path,
                    binder_template_mask=binder_template_mask,
                    target_chain_roles=partner_roles,
                )
                verifier_cache[result.binder_seq] = result
                all_zeroth_results.append(result)

            new_nodes = legacy_tree.expand(selected_node, all_zeroth_results)
            for new_node in new_nodes:
                new_node.logits = _init_child_logits_for_mcts_cdr(
                    selected_node,
                    new_node.sequence,
                    cdr_mask,
                    device,
                ).detach().cpu()

            losses_for_backprop = [result.total_loss for result in all_zeroth_results]
            legacy_tree.backpropagate(selected_node, losses_for_backprop)

            n_pruned = legacy_tree.prune()
            if n_pruned > 0:
                logger.debug("CDR StageA Step %d: pruned %d nodes", legacy_step, n_pruned)

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

            losses_history.append(round_best_loss)
            legacy_steps_done += 1
            stats = legacy_tree.get_node_stats()

            logger.info(
                "CDR StageA Step %3d/%d | sel_node=%d(d=%d) L_real=%.4f 1st=%.4f | "
                "K=%d round_best=%.4f global_best=%.4f@%d stale=%d "
                "tree=%dnodes max_depth=%d N_total=%d | CDR=%d/%d accepted=%s%s",
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
                sum(cdr_mask),
                L_binder,
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
                    "  [CDR Legacy Checkpoint] best=%.4f @ step=%s beam=%s",
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
                    "CDR StageA plateau detected: steps=%d stale=%d best=%.4f; switching to StageB",
                    legacy_steps_done,
                    stale_steps,
                    best_zeroth_loss,
                )
                break

        legacy_tree.log_tree_summary()

        if mcts_strategy == "legacy" or legacy_steps_done >= N_steps:
            logger.info("=" * 70)
            logger.info(
                "CDR Legacy MCTS 完成: best_seq=%s loss=%.4f @step %d",
                best_zeroth_seq,
                best_zeroth_loss,
                best_step,
            )
            logger.info("=" * 70)
            return best_zeroth_seq, best_zeroth_seq, losses_history

        handoff_results = _select_diverse_handoff_results(verifier_cache, mcts_handoff_top_k)
        if not handoff_results:
            raise RuntimeError("[CDR MCTS] two_stage handoff has no verifier results")

        mcts_tree = _make_strict_mcts_tree(
            mcts_C,
            mcts_max_nodes,
            mcts_tau_fixed,
            mcts_expand_actions_per_step,
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
                "CDR StageB root %d/%d: seq=%s L=%.4f R=%.4f structural=%s",
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
            "CDR Stage switch: legacy_steps=%d strict_steps=%d legacy_best_loss=%.4f",
            legacy_steps_done,
            strict_loop_steps,
            best_zeroth_loss,
        )

    logger.info("开始 CDR Document PUCT MCTS 主循环: steps=%d offset=%d", strict_loop_steps, step_offset)
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

        if has_cdr_protection:
            selected_seq_clipped = _clip_seq_by_cdr_mask(selected_seq, cdr_mask, init_binder)
            if selected_seq_clipped != selected_seq:
                logger.debug(
                    f"Step {global_step}: CDR guard clipped selected_seq framework, "
                    f"id={selected_node.node_id}"
                )
                selected_seq = selected_seq_clipped
                _sync_theta_to_seq(selected_logits, selected_seq)

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
            binder_template_mask=binder_template_mask,
        )

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

        routing_info = select_de_novo_action_group(
            selected_node.loss_vector,
            temperature=MCTS_STAGE2_GROUP_TEMPERATURE,
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
            opt_token_mask=cdr_mask,
            target_chain_roles=partner_roles,
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

        if grad_for_proposal is None:
            logger.warning(f"Step {global_step}: grad is None, skipping candidate generation")
            losses_history.append(best_zeroth_loss)
            continue

        grads_cpu = grad_for_proposal.detach().cpu()
        if has_cdr_protection:
            grads_cpu = grads_cpu.clone()
            grads_cpu[framework_indices, :] = 0.0

        stageb_max_mutations = _stageb_max_mutations_for_step(
            base_max_mutations=max_mutations_per_seq,
            local_step=step,
            total_steps=strict_loop_steps,
            stale_steps=strict_stale_steps,
        )

        widening_slots = mcts_tree.remaining_expansion_slots(selected_node)
        expand_budget = min(mcts_expand_actions_per_step, widening_slots)
        if expand_budget <= 0:
            logger.warning(
                f"Step {global_step}: selected node {selected_node.node_id} has no progressive widening slots"
            )
            losses_history.append(best_zeroth_loss)
            continue

        proposal_pool_budget = max(B_candidates, expand_budget)
        group_budgets = _allocate_action_group_budgets(group_priors, proposal_pool_budget)
        proposal_pool: List[MutationProposal] = []
        for group_name, group_budget in group_budgets.items():
            position_weights = _cdr_stageb_position_weights(
                current_seq_step2,
                cdr_mask,
                group_name,
                selected_node.structural_signature,
            )
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
                    position_weights=position_weights,
                    enforce_position_mask=True,
                    prior_mode=mcts_prior_mode,
                )
            )
        proposal_pool = _renormalize_stageb_proposal_priors(
            proposal_pool,
            mcts_prior_mode,
        )

        proposal_dedup_skipped = 0
        proposal_seen = set()
        clipped_pool: List[MutationProposal] = []
        for proposal in proposal_pool:
            clipped_proposal = (
                _clip_mcts_proposal_by_cdr_mask(proposal, cdr_mask, init_binder)
                if has_cdr_protection else proposal
            )
            if clipped_proposal.sequence == selected_seq:
                proposal_dedup_skipped += 1
                continue
            if clipped_proposal.sequence in proposal_seen:
                proposal_dedup_skipped += 1
                continue
            proposal_seen.add(clipped_proposal.sequence)
            clipped_pool.append(clipped_proposal)
        proposal_pool = clipped_pool

        if not proposal_pool:
            logger.warning(f"Step {global_step}: no fresh CDR-valid candidates generated, skipping")
            losses_history.append(best_zeroth_loss)
            continue

        registered_actions = mcts_tree.register_action_proposals(selected_node, proposal_pool)
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
        action_by_sequence = {action.sequence: action for action in selected_actions}

        fresh_proposals: List[MutationProposal] = []
        cached_proposals: List[MutationProposal] = []
        cached_results = []
        duplicate_results = []
        reward_by_sequence: Dict[str, float] = {}
        seen_step = set()
        duplicates_skipped = proposal_dedup_skipped
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
            cand_seq = (
                _clip_seq_by_cdr_mask(proposal.sequence, cdr_mask, init_binder)
                if has_cdr_protection else proposal.sequence
            )
            result = zeroth_order_eval(
                binder_seq=cand_seq,
                target_seq=target_seq,
                model=model,
                device=device,
                loss_weights=base_weights,
                contact_params=contact_params,
                chunk_size=chunk_size,
                hotspot_params=hotspot_params,
                opt_token_mask=cdr_mask,
                target_template_path=target_template_path,
                binder_template_mask=binder_template_mask,
                target_chain_roles=partner_roles,
            )
            verifier_cache[result.binder_seq] = result
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
            new_node.logits = _init_child_logits_for_mcts_cdr(
                selected_node,
                new_node.sequence,
                cdr_mask,
                device,
            ).detach().cpu()

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

        if not round_results:
            logger.warning(f"Step {global_step}: no verifier result available after duplicate merge")
            losses_history.append(best_zeroth_loss)
            continue

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

        if best_zeroth_loss < strict_stage_best_loss - 1e-12:
            strict_stage_best_loss = float(best_zeroth_loss)
            strict_stale_steps = 0
        elif best_zeroth_loss < previous_global_best_loss:
            strict_stage_best_loss = min(strict_stage_best_loss, float(best_zeroth_loss))
            strict_stale_steps = 0
        else:
            strict_stale_steps += 1

        losses_history.append(round_best_loss)

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
            f"round_best={round_best_loss:.4f} "
            f"round_R={round_best_reward:.4f} "
            f"global_best={best_zeroth_loss:.4f}@{best_step} "
            f"tree={stats['total_nodes']}nodes "
            f"max_depth={stats['max_depth']} "
            f"N_total={stats['N_total']} | "
            f"CDR={sum(cdr_mask)}/{L_binder} | "
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
    logger.info("CDR-Aware MCTS 混合优化完成！")
    logger.info(
        f"全局最优序列(loss): {best_zeroth_seq} "
        f"(loss={best_zeroth_loss:.4f} @step {best_step})"
    )
    logger.info(
        f"全局最优序列(reward): {best_final.sequence} "
        f"(R={best_final.tree_reward:.4f}, L={best_final.L_real:.4f})"
    )
    logger.info(f"CDR residues: {sum(cdr_mask)}/{L_binder} | Framework: {L_binder - sum(cdr_mask)}/{L_binder}")
    logger.info(f"MCTS 树最终节点数: {mcts_tree.total_nodes}")
    logger.info(
        f"MCTS 全局最优节点: id={best_final.node_id} depth={best_final.depth} "
        f"seq={best_final.sequence[:20]}... L={best_final.L_real:.4f} R={best_final.tree_reward:.4f}"
    )
    logger.info("=" * 70)

    return best_zeroth_seq, best_final.sequence, losses_history
