"""
BindCraft v2 - 单步外推模块（模块 B）

实现算法指导中的"截断式单步 x̂_0 外推预测"
解决多步 Diffusion ODE 的显存和梯度衰减问题

算法指导（flow matching 视角）：
  v_pred = Diffusion(x_τ, τ, s, z)
  x̂_0 = x_τ - τ · v_pred

Protenix 实现（EDM 视角，等价）：
  r_update = F_θ(c_in · x_noisy)        ← f_forward 的输出
  x_denoised = c_skip·x_noisy + c_out·r_update  ← forward 的输出，即 x̂_0

对应关系：DiffusionModule.forward 直接返回 x̂_0（x_denoised），
不需要再做 x_τ - τ·v_pred 的外推步骤。
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
import logging

from bindcraft.hallucination.pairformer_wrapper import pairformer_forward_no_ckpt

logger = logging.getLogger(__name__)


# ============================================================
# 初始化粗结构（x_coarse）
# ============================================================

def get_coarse_structure(
    model,
    input_feature_dict: Dict,
    s_inputs: torch.Tensor,
    device: torch.device,
    X_clean: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    获取粗结构 x_coarse，用作后续单步外推的起点。

    优先级：
    1. 若传入了 X_clean（Step 0/Step 3 得到的全链路零阶物理骨架），直接使用
       - X_clean 的形状为 [1, N_atom, 3]（经 ODESolve 完整去噪后的真实物理结构）
       - 这避免了每次从随机起点出发，让一阶前向的起点更有物理意义
    2. 否则从 ref_pos + 随机噪声初始化

    Args:
        model: Protenix 模型
        input_feature_dict: 输入特征字典
        s_inputs: 输入的单体特征，[N_token, c_s]
        device: 计算设备
        X_clean: 可选，零阶全链路骨架，[1, N_atom, 3]

    Returns:
        x_coarse: 粗结构坐标，[1, N_atoms, 3]
    """
    N_atoms_dict = input_feature_dict["ref_pos"].shape[0]

    if X_clean is not None:
        # 分支1：使用零阶物理骨架（优先路径）
        # 验证原子数匹配（序列不变时原子数相同）
        if X_clean.shape[0] == 1 and X_clean.shape[1] == N_atoms_dict:
            x_coarse = X_clean.clone().to(device)
            logger.info(f"[骨架] 使用零阶骨架，shape={x_coarse.shape}")
        else:
            # 原子数不匹配（序列变化导致原子数变化），回退到随机初始化
            logger.info(
                f"[骨架] X_clean 原子数不匹配（{X_clean.shape[1]} vs {N_atoms_dict}），回退随机初始化"
            )
            x_coarse = None

    if X_clean is None or x_coarse is None:
        # 分支2：随机初始化（首次或序列变化后）
        logger.info("[骨架] 随机初始化（ref_pos + 小随机扰动）")
        x_coarse = input_feature_dict["ref_pos"].clone().unsqueeze(0).to(device)  # [1, N_atoms, 3]
        x_coarse = x_coarse + torch.randn_like(x_coarse) * 0.1

    logger.debug(f"粗结构 shape: {x_coarse.shape}")
    return x_coarse


# ============================================================
# 单步去噪（核心）
# ============================================================

def one_step_denoise(
    model,
    x_coarse: torch.Tensor,
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    input_feature_dict: Dict,
    tau: float = 4.0,
    device: torch.device = None
) -> torch.Tensor:
    """
    单步预测无噪声结构 x̂_0。

    算法指导（flow matching 视角）：
      x_τ = x_coarse + τ·ε
      v_pred = Diffusion(x_τ, τ, s, z)
      x̂_0 = x_τ - τ·v_pred

    Protenix 等价实现（EDM 视角）：
      DiffusionModule.forward 直接返回 x_denoised = x̂_0，
      内部已完成 EDM 的 c_skip/c_out 缩放，无需再做外推。

    注意：这个函数需要使用当前的 input_feature_dict，
    因为不同的序列会有不同数量的原子。
    
    Args:
        model: Protenix 模型
        x_coarse: 粗结构坐标，[1, N_atoms, 3]
        s_inputs: 输入的单体特征，[N_token, c_s]
        s_trunk: Pairformer 输出的单体特征，[N_token, c_s]
        z_trunk: Pairformer 输出的对特征，[N_token, N_token, c_z]
        input_feature_dict: 输入特征字典（必须与当前序列匹配）
        tau: 噪声时间步（标量）
        device: 计算设备
        
    Returns:
        x0_hat: 外推的无噪声结构，[1, N_atoms, 3]
    """
    if device is None:
        device = x_coarse.device

    # 单步外推只需要 1 个 sample；上游可能传入 [N_sample, N_atom, 3]
    # （如 zeroth_order_eval 缓存的 N_sample=5），先收缩到首个 sample，
    # 否则会触发 protenix 的 t_hat.size(-1) == r_noisy.size(-3) 断言。
    if x_coarse.dim() >= 3 and x_coarse.size(-3) > 1:
        x_coarse = x_coarse.narrow(-3, 0, 1).contiguous()

    # 重要：检查 x_coarse 和 input_feature_dict 的原子数是否匹配
    N_atoms_coarse = x_coarse.shape[-2]
    N_atoms_dict = input_feature_dict["ref_pos"].shape[0]
    
    # 如果原子数不匹配（例如跨步复用的 x_coarse 与当前序列原子数不同），
    # 使用 input_feature_dict 的 ref_pos 重新初始化
    if N_atoms_coarse != N_atoms_dict:
        logger.debug(
            f"原子数不匹配：x_coarse={N_atoms_coarse}, input_feature_dict={N_atoms_dict}，"
            "使用 ref_pos 重新初始化（序列变化导致原子数变化属正常）"
        )
        x_coarse = input_feature_dict["ref_pos"].clone().unsqueeze(0)
        x_coarse = x_coarse + torch.randn_like(x_coarse) * 0.1
    
    # 步骤 1：添加噪声，得到 x_τ
    noise = tau * torch.randn_like(x_coarse)
    x_noisy = x_coarse.detach() + noise  # [1, N_atoms, 3]，detach 使 x_τ 不参与梯度

    # 步骤 2：通过 DiffusionModule 直接得到 x̂_0（无噪声预测）
    #
    # 算法指导公式（flow matching 视角）：
    #   v_pred = Diffusion(x_τ, τ, s, z)
    #   x̂_0 = x_τ - τ · v_pred
    #
    # Protenix 实现（EDM 视角）：
    #   r_update = F_θ(c_in · x_noisy, c_noise(τ))
    #   x_denoised = c_skip · x_noisy + c_out · r_update
    #   其中 c_skip = 1/(1+s²)，c_out = τ/√(1+s²)，s = τ/σ_data
    #
    # 两者等价：DiffusionModule.forward 返回的 x_denoised 就是 x̂_0。
    # 注意：不能用 x_noisy - tau * x_denoised，那是错误的二次外推。
    #
    # 与 x_noisy 的 N_sample 维对齐，满足 protenix 的形状断言
    n_sample = x_noisy.size(-3) if x_noisy.dim() >= 3 else 1
    t_hat = torch.full((n_sample,), float(tau), device=device, dtype=x_noisy.dtype)

    # 准备 diffusion conditioning 参数（pair_z 含梯度，来自 z_trunk）
    pair_z = model.diffusion_module.diffusion_conditioning.prepare_cache(
        input_feature_dict["relp"], z_trunk
    )

    # 准备原子级特征（p_lm, c_l 含梯度，来自 pair_z / z_trunk）
    p_lm, c_l = model.diffusion_module.atom_attention_encoder.prepare_cache(
        ref_pos=input_feature_dict["ref_pos"],
        ref_charge=input_feature_dict["ref_charge"],
        ref_mask=input_feature_dict["ref_mask"],
        ref_element=input_feature_dict["ref_element"],
        ref_atom_name_chars=input_feature_dict["ref_atom_name_chars"],
        atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
        d_lm=input_feature_dict["d_lm"],
        v_lm=input_feature_dict["v_lm"],
        pad_info=input_feature_dict["pad_info"],
        r_l=True,
        z=pair_z,
        inplace_safe=False,
    )

    # 调用 DiffusionModule.forward，直接返回 x̂_0（EDM 的 x_denoised）
    x0_hat = model.diffusion_module(
        x_noisy=x_noisy,
        t_hat_noise_level=t_hat,
        input_feature_dict=input_feature_dict,
        s_inputs=s_inputs,
        s_trunk=s_trunk,
        z_trunk=z_trunk,
        pair_z=pair_z,
        p_lm=p_lm,
        c_l=c_l,
        inplace_safe=False,
    )  # [1, N_atoms, 3]，即 x̂_0

    return x0_hat


# ============================================================
# 完整的单步外推流程
# ============================================================

def diffusion_one_step(
    model,
    s_binder: torch.Tensor,
    s_target: torch.Tensor,
    input_feature_dict: Dict,
    x_coarse: Optional[torch.Tensor] = None,
    X_clean: Optional[torch.Tensor] = None,
    tau: float = 4.0,
    device: torch.device = None,
    chunk_size: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    完整的单步外推流程（模块 B）

    流程：
    1. 拼接 Binder（STE 软特征）和 Target 特征 → s_inputs_ste
    2. 通过无 checkpoint 的 Pairformer 获取 s_trunk 和 z_trunk
       （梯度可从 s_trunk/z_trunk 反传至 s_binder → Theta）
    3. 获取 x_coarse（优先用 X_clean 零阶骨架，回退到随机初始化）
    4. 单步外推预测 x0_hat

    Args:
        model: Protenix 模型（权重已冻结）
        s_binder: Binder 特征（来自 STE，含梯度），[L_binder, c_s]
        s_target: Target 特征（冻结），[L_target, c_s]
        input_feature_dict: 输入特征字典
        x_coarse: 粗结构（内部使用，优先由 X_clean 决定；若为 None 则从 X_clean 初始化）
        X_clean: 零阶全链路骨架，[1, N_atom, 3]（由 Step 0 或 Step 3 获得）
        tau: 噪声时间步
        device: 计算设备
        chunk_size: Attention 分块大小。114+ aa 建议 256，避免 OOM。

    Returns:
        x0_hat: 外推的无噪声结构，[1, N_atoms, 3]
        s_trunk: Pairformer 输出的单体特征，[N_token, c_s]
        z_trunk: Pairformer 输出的对特征，[N_token, N_token, c_z]
    """
    if device is None:
        device = s_binder.device

    # 步骤 1：拼接 STE Binder 特征与冻结 Target 特征
    s_inputs_ste = torch.cat([s_binder, s_target], dim=0)  # [N_token, c_s]

    # 步骤 2：无 checkpoint 的 Pairformer 前向（阶段 3 核心）
    # 传入 s_init_override=s_inputs_ste，使梯度从 s_trunk/z_trunk 反传至 s_binder
    _, s_trunk, z_trunk = pairformer_forward_no_ckpt(
        model=model,
        input_feature_dict=input_feature_dict,
        s_init_override=s_inputs_ste,
        N_cycle=4,
        chunk_size=chunk_size,
    )

    # 步骤 3：获取 x_coarse（优先用 X_clean，回退到随机初始化）
    if x_coarse is None:
        x_coarse = get_coarse_structure(
            model, input_feature_dict, s_inputs_ste, device, X_clean=X_clean
        )

    # 步骤 4：单步外推
    x0_hat = one_step_denoise(
        model=model,
        x_coarse=x_coarse,
        s_inputs=s_inputs_ste,
        s_trunk=s_trunk,
        z_trunk=z_trunk,
        input_feature_dict=input_feature_dict,
        tau=tau,
        device=device,
    )

    return x0_hat, s_trunk, z_trunk


# ============================================================
# 多频段 τ 融合（Multi-Frequency τ Fusion）
# ============================================================

def diffusion_one_step_fused(
    model,
    s_binder: torch.Tensor,
    s_target: torch.Tensor,
    input_feature_dict: Dict,
    x_coarse: Optional[torch.Tensor],
    X_clean: Optional[torch.Tensor],
    theta: torch.Tensor,
    step_weights: Dict,
    contact_params: Dict,
    plddt_params,
    pae_params,
    distogram_params,
    L_binder: int,
    tau_coarse: float = 0.8,
    tau_fine: float = 0.05,
    alpha: float = 0.5,
    device: torch.device = None,
    pde_params=None,
    clash_params=None,
    hotspot_params=None,
    chunk_size: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    多频段 τ 融合：在同一优化步中，分别用粗粒度 tau 和细粒度 tau 各执行一次
    一阶前向，融合两次梯度后再更新 theta。

    公式：
      G_coarse = nabla_Theta L(tau=tau_coarse)
      G_fine   = nabla_Theta L(tau=tau_fine)
      G_fused  = alpha * G_coarse + (1 - alpha) * G_fine

    其中 alpha 越大越偏向宏观探索（粗粒度），越小越偏向局部精修（细粒度）。

    注意：该函数假设 optimizer.zero_grad() 已在外层调用，
    caller 负责在返回值后执行 optimizer.step()。

    Args:
        model, s_binder, s_target, input_feature_dict, x_coarse:
            与 diffusion_one_step 相同
        X_clean: 零阶全链路骨架，[1, N_atom, 3]
        theta:           [L_binder, 20] 可学习 Logits（需要就地清零梯度）
        step_weights:   loss 权重字典
        contact_params:  contact loss 参数
        plddt_params / pae_params / distogram_params: bin 参数
        L_binder:        binder 长度
        tau_coarse:      粗粒度时间步（建议 0.6~0.9）
        tau_fine:        细粒度时间步（建议 0.01~0.2）
        alpha:           融合系数（粗粒度权重，0~1）
        device:          计算设备

    Returns:
        grad_fused: [L_binder, 20] 融合后的梯度
        x0_fine:    [1, N_atom, 3] 细粒度预测的结构（用于后续候选生成参考）
    """
    if device is None:
        device = s_binder.device

    s_inputs_ste = torch.cat([s_binder, s_target], dim=0)

    # ── 获取 x_coarse（优先用 X_clean，回退到随机初始化）────────────────────
    if x_coarse is None:
        x_coarse = get_coarse_structure(
            model, input_feature_dict, s_inputs_ste, device, X_clean=X_clean
        )

    # ── 粗粒度前向（tau_coarse）───────────────────────────────────────────
    theta.grad.zero_()
    _, s_trunk_coarse, z_trunk_coarse = pairformer_forward_no_ckpt(
        model=model, input_feature_dict=input_feature_dict,
        s_init_override=s_inputs_ste, N_cycle=4, chunk_size=chunk_size,
    )
    x0_coarse = one_step_denoise(
        model=model, x_coarse=x_coarse, s_inputs=s_inputs_ste,
        s_trunk=s_trunk_coarse, z_trunk=z_trunk_coarse,
        input_feature_dict=input_feature_dict, tau=tau_coarse, device=device,
    )

    heads_coarse = _run_confidence_heads(
        model, input_feature_dict, s_inputs_ste, s_trunk_coarse, z_trunk_coarse, x0_coarse,
        chunk_size=chunk_size,
    )
    loss_coarse, _ = _compute_loss(
        x0_coarse.squeeze(0), heads_coarse, input_feature_dict,
        L_binder, step_weights, contact_params,
        plddt_params, pae_params, distogram_params,
        pde_params=pde_params, clash_params=clash_params,
        hotspot_params=hotspot_params,
    )
    loss_coarse.backward(retain_graph=True)
    G_coarse = theta.grad.clone()
    theta.grad.zero_()

    # ── 细粒度前向（tau_fine）─────────────────────────────────────────────
    # Pairformer 结果可复用（不依赖 tau，只依赖序列特征）
    # 重新构建原子级 cache（pair_z 依赖 z_trunk，但粗细粒度用的是同一个 z_trunk）
    pair_z = model.diffusion_module.diffusion_conditioning.prepare_cache(
        input_feature_dict["relp"], z_trunk_coarse
    )
    p_lm, c_l = model.diffusion_module.atom_attention_encoder.prepare_cache(
        ref_pos=input_feature_dict["ref_pos"],
        ref_charge=input_feature_dict["ref_charge"],
        ref_mask=input_feature_dict["ref_mask"],
        ref_element=input_feature_dict["ref_element"],
        ref_atom_name_chars=input_feature_dict["ref_atom_name_chars"],
        atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
        d_lm=input_feature_dict["d_lm"],
        v_lm=input_feature_dict["v_lm"],
        pad_info=input_feature_dict["pad_info"],
        r_l=True, z=pair_z, inplace_safe=False,
    )
    x_noisy_fine = x_coarse.detach() + tau_fine * torch.randn_like(x_coarse)
    n_sample_fine = x_noisy_fine.size(-3) if x_noisy_fine.dim() >= 3 else 1
    t_hat_fine = torch.full((n_sample_fine,), float(tau_fine), device=device, dtype=x_noisy_fine.dtype)
    x0_fine = model.diffusion_module(
        x_noisy=x_noisy_fine, t_hat_noise_level=t_hat_fine,
        input_feature_dict=input_feature_dict,
        s_inputs=s_inputs_ste, s_trunk=s_trunk_coarse, z_trunk=z_trunk_coarse,
        pair_z=pair_z, p_lm=p_lm, c_l=c_l, inplace_safe=False,
    )
    heads_fine = _run_confidence_heads(
        model, input_feature_dict, s_inputs_ste, s_trunk_coarse, z_trunk_coarse, x0_fine,
        chunk_size=chunk_size,
    )
    loss_fine, _ = _compute_loss(
        x0_fine.squeeze(0), heads_fine, input_feature_dict,
        L_binder, step_weights, contact_params,
        plddt_params, pae_params, distogram_params,
        pde_params=pde_params, clash_params=clash_params,
        hotspot_params=hotspot_params,
    )
    loss_fine.backward()
    G_fine = theta.grad.clone()

    # ── 梯度融合 ──────────────────────────────────────────────────────────
    grad_fused = alpha * G_coarse + (1.0 - alpha) * G_fine
    theta.grad.copy_(grad_fused)

    logger.debug(
        f"  τ-fusion: coarse_loss={loss_coarse.item():.4f} fine_loss={loss_fine.item():.4f} "
        f"α={alpha} G_coarse_norm={G_coarse.norm().item():.4f} "
        f"G_fine_norm={G_fine.norm().item():.4f} G_fused_norm={grad_fused.norm().item():.4f}"
    )

    return grad_fused, x0_fine


# ── 内部辅助函数（避免在 fusion 函数中重复大段代码）───────────────────────

def _run_confidence_heads(model, input_feature_dict, s_inputs, s_trunk, z_trunk, x0_hat, chunk_size=None):
    from bindcraft.losses.confidence import run_confidence_heads
    return run_confidence_heads(
        model=model, input_feature_dict=input_feature_dict,
        s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk,
        x0_hat=x0_hat, inplace_safe=False, chunk_size=chunk_size,
    )


def _compute_loss(x0_flat, heads, input_feature_dict, L_binder, weights,
                   contact_params, plddt_params, pae_params, distogram_params,
                   pde_params=None, clash_params=None, hotspot_params=None):
    from bindcraft.losses.total_loss import compute_all_losses
    return compute_all_losses(
        x0_flat=x0_flat,
        plddt_logits=heads["plddt_logits"],
        pae_logits=heads["pae_logits"],
        distogram_logits=heads["distogram_logits"],
        input_feature_dict=input_feature_dict,
        L_binder=L_binder, weights=weights,
        contact_params=contact_params,
        plddt_params=plddt_params, pae_params=pae_params, distogram_params=distogram_params,
        pde_logits=heads.get("pde_logits"),
        pde_params=pde_params, clash_params=clash_params,
        hotspot_params=hotspot_params,
    )
