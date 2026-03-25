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
    device: torch.device
) -> torch.Tensor:
    """
    通过简单的随机初始化获取粗结构
    
    这个粗结构用作后续单步外推的起点
    
    Args:
        model: Protenix 模型
        input_feature_dict: 输入特征字典
        s_inputs: 输入的单体特征，[N_token, c_s]
        device: 计算设备
        
    Returns:
        x_coarse: 粗结构坐标，[1, N_atoms, 3]
    """
    logger.info("初始化粗结构（随机初始化）...")
    
    # 获取原子数量
    N_atoms = input_feature_dict["ref_pos"].shape[0]
    
    # 使用 ref_pos 作为初始粗结构（加上小的随机扰动）
    with torch.no_grad():
        x_coarse = input_feature_dict["ref_pos"].clone().unsqueeze(0)  # [1, N_atoms, 3]
        # 添加小的随机扰动
        x_coarse = x_coarse + torch.randn_like(x_coarse) * 0.1
    
    logger.info(f"粗结构初始化完成，shape: {x_coarse.shape}")
    
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
    
    # 重要：检查 x_coarse 和 input_feature_dict 的原子数是否匹配
    N_atoms_coarse = x_coarse.shape[1]
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
    t_hat = torch.tensor([tau], device=device)

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
    tau: float = 4.0,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    完整的单步外推流程（模块 B）

    流程：
    1. 拼接 Binder（STE 软特征）和 Target 特征 → s_inputs_ste
    2. 通过无 checkpoint 的 Pairformer 获取 s_trunk 和 z_trunk
       （梯度可从 s_trunk/z_trunk 反传至 s_binder → Theta）
    3. 单步外推预测 x0_hat

    Args:
        model: Protenix 模型（权重已冻结）
        s_binder: Binder 特征（来自 STE，含梯度），[L_binder, c_s]
        s_target: Target 特征（冻结），[L_target, c_s]
        input_feature_dict: 输入特征字典
        x_coarse: 粗结构（如果为 None 则初始化），[1, N_atoms, 3]
        tau: 噪声时间步
        device: 计算设备

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
    )

    # 步骤 3：初始化 x_coarse（如果需要）
    if x_coarse is None:
        x_coarse = get_coarse_structure(model, input_feature_dict, s_inputs_ste, device)

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
