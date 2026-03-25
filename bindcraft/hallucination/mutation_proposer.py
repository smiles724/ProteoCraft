"""
BindCraft v2 - 梯度引导的离散候选生成器

根据一阶梯度矩阵 G = nabla_Theta L，通过一阶泰勒近似计算每个位置
从当前氨基酸突变为其他氨基酸的收益，选取 Top-K 候选突变，
并采样组合出 B 条离散候选序列。

对应算法指导文档：
  - 算法指导2.md 第六章 Step 2
  - Bindcraft v2零阶更新.md 第五章 5.3 Step 2

核心公式：
  Delta_L[i, c->a] ≈ dL/dTheta[i,a] - dL/dTheta[i,c]
  P_mut(i, a) = Softmax(-Delta_L[i, c->a] / T_sample)
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple
import logging

from bindcraft.utils import (
    PROTEIN_RESIDUE_NAMES,
    AA_1LETTER,
    NUM_AMINO_ACIDS,
)

logger = logging.getLogger(__name__)

AA_1_TO_IDX = {AA_1LETTER[name]: idx for idx, name in enumerate(PROTEIN_RESIDUE_NAMES)}


def _seq_to_indices(seq: str) -> List[int]:
    """单字母序列 -> 索引列表"""
    return [AA_1_TO_IDX[aa] for aa in seq]


def _indices_to_seq(indices: List[int]) -> str:
    """索引列表 -> 单字母序列"""
    return "".join(AA_1LETTER[PROTEIN_RESIDUE_NAMES[i]] for i in indices)


def compute_mutation_benefit(
    grad_matrix: torch.Tensor,
    current_indices: List[int],
) -> torch.Tensor:
    """
    计算突变收益矩阵 Delta_L。

    Delta_L[i, a] = G[i, a] - G[i, c_i]
    其中 c_i 是位置 i 当前的氨基酸索引。
    负值表示突变为 a 可降低 loss（有利突变）。

    Args:
        grad_matrix: [L_binder, 20] 梯度矩阵
        current_indices: 长度 L_binder 的当前氨基酸索引列表

    Returns:
        delta_L: [L_binder, 20] 突变收益矩阵
    """
    L = grad_matrix.shape[0]
    current_grad = grad_matrix[torch.arange(L), current_indices]  # [L]
    delta_L = grad_matrix - current_grad.unsqueeze(1)  # [L, 20]
    return delta_L


def select_top_k_mutations(
    delta_L: torch.Tensor,
    current_indices: List[int],
    K_top: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    选取收益最大的 Top-K 候选突变（排除自身突变）。

    Args:
        delta_L: [L_binder, 20] 突变收益矩阵
        current_indices: 当前氨基酸索引
        K_top: 选取数量

    Returns:
        positions: [K] 突变位置
        new_aas:   [K] 突变后的氨基酸索引
        benefits:  [K] 对应的 -delta_L 值（越大越好）
    """
    L = delta_L.shape[0]
    benefit = -delta_L.clone()  # [L, 20]，正值 = 有利

    # 将自身突变的收益设为 -inf，使其不会被选中
    for i, c in enumerate(current_indices):
        benefit[i, c] = float("-inf")

    flat_benefit = benefit.reshape(-1)  # [L * 20]
    K_actual = min(K_top, int((flat_benefit > float("-inf")).sum().item()))
    if K_actual == 0:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long), torch.tensor([])

    topk_vals, topk_flat_idx = torch.topk(flat_benefit, K_actual)
    positions = topk_flat_idx // NUM_AMINO_ACIDS
    new_aas = topk_flat_idx % NUM_AMINO_ACIDS

    return positions, new_aas, topk_vals


def sample_candidate_sequences(
    current_indices: List[int],
    positions: torch.Tensor,
    new_aas: torch.Tensor,
    benefits: torch.Tensor,
    B_candidates: int,
    T_sample: float,
    max_mutations_per_seq: int,
) -> List[List[int]]:
    """
    从 Top-K 候选突变中采样组合，生成 B 条候选序列。

    采样策略：
    1. 按 Softmax(benefits / T_sample) 构造突变采样分布
    2. 每条候选随机选择 1~max_mutations_per_seq 个突变
    3. 同一位置只保留最后一个突变（避免冲突）

    Args:
        current_indices: 当前序列索引
        positions:   [K] 突变位置
        new_aas:     [K] 突变氨基酸
        benefits:    [K] 突变收益
        B_candidates: 生成候选数
        T_sample:     采样温度
        max_mutations_per_seq: 每条候选最大突变数

    Returns:
        candidates: B 条候选序列（索引列表的列表）
    """
    K = positions.shape[0]
    if K == 0:
        return [list(current_indices)] * B_candidates

    probs = F.softmax(benefits / max(T_sample, 1e-8), dim=0)  # [K]

    candidates = []
    seen = set()

    for _ in range(B_candidates * 5):  # 多试几次以去重
        if len(candidates) >= B_candidates:
            break

        n_mut = torch.randint(1, max_mutations_per_seq + 1, (1,)).item()
        n_mut = min(n_mut, K)

        chosen_idx = torch.multinomial(probs, n_mut, replacement=False)

        seq = list(current_indices)
        for ci in chosen_idx.tolist():
            pos = positions[ci].item()
            aa = new_aas[ci].item()
            seq[pos] = aa

        key = tuple(seq)
        if key not in seen and key != tuple(current_indices):
            seen.add(key)
            candidates.append(seq)

    # 如果去重后不够 B 条，用已有的填充
    while len(candidates) < B_candidates:
        if candidates:
            candidates.append(list(candidates[len(candidates) % len(candidates)]))
        else:
            candidates.append(list(current_indices))

    return candidates[:B_candidates]


def propose_mutations(
    grad_matrix: torch.Tensor,
    current_seq: str,
    K_top: int = 10,
    B_candidates: int = 8,
    T_sample: float = 1.0,
    max_mutations_per_seq: int = 3,
) -> List[str]:
    """
    梯度引导的离散候选序列生成。

    完整流程：
    1. 将梯度矩阵转换为突变收益矩阵 Delta_L
    2. 选取 Top-K 候选突变
    3. 按收益分布采样组合，生成 B 条离散候选序列

    Args:
        grad_matrix:  [L_binder, 20] 梯度矩阵 G = nabla_Theta L
        current_seq:  当前离散序列（单字母 str）
        K_top:        选取 Top-K 候选突变数
        B_candidates: 生成 B 条候选序列
        T_sample:     采样温度（越小越确定性，越大越随机）
        max_mutations_per_seq: 每条候选的最大突变数

    Returns:
        B 条离散候选序列（单字母 str 列表）
    """
    current_indices = _seq_to_indices(current_seq)

    grad_cpu = grad_matrix.detach().cpu()

    delta_L = compute_mutation_benefit(grad_cpu, current_indices)

    positions, new_aas, benefits = select_top_k_mutations(
        delta_L, current_indices, K_top
    )

    if positions.shape[0] == 0:
        logger.warning("No valid mutations found, returning current sequence copies")
        return [current_seq] * B_candidates

    candidate_indices = sample_candidate_sequences(
        current_indices, positions, new_aas, benefits,
        B_candidates, T_sample, max_mutations_per_seq,
    )

    candidate_seqs = [_indices_to_seq(idx_list) for idx_list in candidate_indices]

    logger.info(
        f"Proposed {len(candidate_seqs)} candidates | "
        f"Top-K={positions.shape[0]} mutations | "
        f"Best benefit={benefits[0]:.4f} at pos {positions[0].item()}"
    )

    return candidate_seqs
