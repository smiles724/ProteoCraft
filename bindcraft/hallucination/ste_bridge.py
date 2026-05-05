"""
BindCraft v2 - STE 桥接模块（模块 A）

实现算法指导中的"连续重参数化与 STE 桥接"
解决 InputEmbedder 的离散不可导问题

核心思路：
1. 连续 Logits (Theta) → 软概率 (P) → 软特征 (s_soft)
2. 硬采样 → 硬特征 (s_hard)
3. STE 融合：s_binder = s_hard.detach() + s_soft - s_soft.detach()
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


# ============================================================
# W_AAE 预计算（特征字典）
# ============================================================

def precompute_waa_embeddings(model, device: torch.device) -> torch.Tensor:
    """
    预计算 20 种氨基酸的特征字典 W_AAE

    算法指导：W_AAE ∈ R^{20 × c_s_inputs}
    "由 20 种单体氨基酸通过 AtomAttentionEncoder 提取的基础特征"

    实现策略：为每种氨基酸构建**单残基单链**系统（无任何 context 污染），
    调用 model.input_embedder 获取其 449 维特征（c_s_inputs）。
    这样 d_lm/v_lm 只包含该氨基酸自身原子的局部特征，与算法指导一致。

    Args:
        model: Protenix 模型
        device: 计算设备

    Returns:
        W_AAE: [20, c_s_inputs] 的特征矩阵（c_s_inputs=449）
    """
    from bindcraft.utils import (
        PROTEIN_RESIDUE_NAMES,
        AA_1LETTER,
        build_single_chain_features,
        prepare_input_features,
    )

    logger.info("预计算 W_AAE（20 种氨基酸的单残基特征字典）...")

    W_AAE_list = []

    with torch.no_grad():
        for aa_3letter in PROTEIN_RESIDUE_NAMES:
            aa_1letter = AA_1LETTER[aa_3letter]

            # 构建单残基单链特征（无 context 污染）
            feat, _, _ = build_single_chain_features(aa_1letter, device)
            feat = prepare_input_features(model, feat)

            # 通过 InputEmbedder 获取特征，输出 [1, c_s_inputs]（单残基）
            s_inputs = model.input_embedder(feat, inplace_safe=False)  # [1, c_s_inputs]

            # 提取唯一的 token 特征
            aa_embedding = s_inputs[0].clone()  # [c_s_inputs]
            W_AAE_list.append(aa_embedding)

    W_AAE = torch.stack(W_AAE_list, dim=0)  # [20, c_s_inputs]
    logger.info(f"W_AAE 预计算完成，shape: {W_AAE.shape}")

    return W_AAE


# ============================================================
# 软特征计算
# ============================================================

def compute_soft_features(
    theta: torch.Tensor,
    W_AAE: torch.Tensor,
    tau_temp: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从连续 Logits 计算软特征
    
    算法指导公式：
    P = Softmax(Theta / tau_temp)  ∈ R^{L_binder × 20}
    s_soft = P · W_AAE  ∈ R^{L_binder × c_s}
    
    Args:
        theta: 连续 Logits，[L_binder, 20]
        W_AAE: 氨基酸特征字典，[20, c_s]
        tau_temp: 温度参数（控制平滑度）
        
    Returns:
        P: 软概率分布，[L_binder, 20]
        s_soft: 软特征，[L_binder, c_s]
    """
    # 计算软概率
    P = F.softmax(theta / tau_temp, dim=-1)  # [L_binder, 20]
    
    # 计算软特征
    s_soft = P @ W_AAE  # [L_binder, c_s]
    
    return P, s_soft


# ============================================================
# 硬特征计算
# ============================================================

def compute_hard_features(
    P: torch.Tensor,
    model,
    target_seq: str,
    device: torch.device,
    target_template_path: str = None,
    binder_template_mask=None,
) -> Tuple[torch.Tensor, str, Dict]:
    """
    从概率分布采样硬序列，并获取硬特征
    
    Args:
        P: 软概率分布，[L_binder, 20]
        model: Protenix 模型
        target_seq: Target 序列
        device: 计算设备
        target_template_path: 可选 target template 结构文件
        binder_template_mask: 可选 binder template token mask；CDR redesign 时
                              用于保留 framework template、屏蔽 CDR 区域
        
    Returns:
        s_hard_binder: 硬特征（仅 Binder 部分），[L_binder, c_s]
        binder_seq: 采样的 Binder 序列（单字母）
        input_feature_dict: 完整的输入特征字典
    """
    from bindcraft.utils import (
        PROTEIN_RESIDUE_NAMES,
        sequence_to_one_letter,
        build_complex_features,
        prepare_input_features
    )
    
    # 硬采样：取概率最大的氨基酸
    hard_indices = P.argmax(dim=-1)  # [L_binder]
    hard_seq_3letter = [PROTEIN_RESIDUE_NAMES[i] for i in hard_indices.cpu().tolist()]
    binder_seq = sequence_to_one_letter(hard_seq_3letter)
    
    # 构建完整的 Binder-Target 特征
    with torch.no_grad():
        input_feature_dict, N_token, N_atom = build_complex_features(
            binder_seq,
            target_seq,
            device,
            target_template_path=target_template_path,
            binder_template_mask=binder_template_mask,
        )
        input_feature_dict = prepare_input_features(model, input_feature_dict)
        
        # 通过 InputEmbedder 获取完整特征
        s_inputs_full = model.input_embedder(input_feature_dict, inplace_safe=False)  # [N_token, c_s_inputs]
        
        # 提取 Binder 部分
        L_binder = len(binder_seq)
        s_hard_binder = s_inputs_full[:L_binder, :].clone()  # [L_binder, c_s]
    
    return s_hard_binder, binder_seq, input_feature_dict


# ============================================================
# STE 融合
# ============================================================

def apply_ste(
    s_hard: torch.Tensor,
    s_soft: torch.Tensor
) -> torch.Tensor:
    """
    应用直通估计器（Straight-Through Estimator）
    
    算法指导公式：
    s_binder = s_hard.detach() + s_soft - s_soft.detach()
    
    机制解析：
    - 前向传播：数值等于 s_hard（合法的硬特征）
    - 反向传播：梯度流向 s_soft（连续可导）
    
    Args:
        s_hard: 硬特征，[L_binder, c_s]
        s_soft: 软特征，[L_binder, c_s]
        
    Returns:
        s_binder: STE 融合后的特征，[L_binder, c_s]
    """
    s_binder = s_hard.detach() + s_soft - s_soft.detach()
    return s_binder


# ============================================================
# 完整的 STE 桥接流程
# ============================================================

def ste_bridge(
    theta: torch.Tensor,
    W_AAE: torch.Tensor,
    model,
    target_seq: str,
    device: torch.device,
    tau_temp: float = 1.0,
    target_template_path: str = None,
    binder_template_mask=None,
) -> Tuple[torch.Tensor, str, Dict]:
    """
    完整的 STE 桥接流程（模块 A）
    
    流程：
    1. 计算软特征：P = Softmax(Theta / tau), s_soft = P @ W_AAE
    2. 计算硬特征：采样序列 → InputEmbedder → s_hard
    3. STE 融合：s_binder = s_hard.detach() + s_soft - s_soft.detach()
    
    Args:
        theta: 连续 Logits，[L_binder, 20]
        W_AAE: 氨基酸特征字典，[20, c_s]
        model: Protenix 模型
        target_seq: Target 序列
        device: 计算设备
        tau_temp: 温度参数
        target_template_path: 可选 target template 结构文件
        binder_template_mask: 可选 binder template token mask；CDR redesign 时
                              用于保留 framework template、屏蔽 CDR 区域
        
    Returns:
        s_binder: STE 融合后的 Binder 特征，[L_binder, c_s]
        binder_seq: 当前的 Binder 序列
        input_feature_dict: 输入特征字典（用于后续计算）
    """
    # 步骤 1：计算软特征
    P, s_soft = compute_soft_features(theta, W_AAE, tau_temp)
    
    # 步骤 2：计算硬特征
    s_hard_binder, binder_seq, input_feature_dict = compute_hard_features(
        P,
        model,
        target_seq,
        device,
        target_template_path=target_template_path,
        binder_template_mask=binder_template_mask,
    )
    
    # 步骤 3：STE 融合
    s_binder = apply_ste(s_hard_binder, s_soft)
    
    return s_binder, binder_seq, input_feature_dict
