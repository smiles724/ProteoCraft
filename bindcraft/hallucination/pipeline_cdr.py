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
from typing import Dict, Tuple, List, Optional
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
)
from bindcraft.hallucination.diffusion_one_step import (
    diffusion_one_step,
    diffusion_one_step_fused,
)
from bindcraft.hallucination.mutation_proposer import propose_mutations
from bindcraft.hallucination.zeroth_order import zeroth_order_eval
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
    target_seq: str = None,
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
        target_seq:    抗原序列（Mode B/C 必需，Mode A 可省略）
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
    logger.info(f"Target: {target_seq} ({len(target_seq)} aa)")
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
