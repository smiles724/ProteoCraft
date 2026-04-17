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
import numpy as np
from typing import List, Tuple
import logging

from bindcraft.utils import (
    PROTEIN_RESIDUE_NAMES,
    AA_1LETTER,
    NUM_AMINO_ACIDS,
)

logger = logging.getLogger(__name__)

AA_1_TO_IDX = {AA_1LETTER[name]: idx for idx, name in enumerate(PROTEIN_RESIDUE_NAMES)}

# ── BLOSUM62 替换矩阵（20×20，对应 PROTEIN_RESIDUE_NAMES 顺序）─────────────
# 行 = 目标氨基酸 (to)，列 = 源氨基酸 (from)
# 正值：演化上保守的替换（高分）；负值：罕见替换（低分）
# 矩阵来源：NCBI BLAST BLOSUM62，对角线为最大正值，非对角线反映替换频率
# 顺序：ALA, ARG, ASN, ASP, CYS, GLN, GLU, GLY, HIS, ILE, LEU, LYS, MET, PHE, PRO, SER, THR, TRP, TYR, VAL
_BLOSUM62_ROWS = [
    # A    R    N    D    C    Q    E    G    H    I    L    K    M    F    P    S    T    W    Y    V
    [ 4,  -1,  -2,  -2,   0,  -1,  -1,   0,  -2,  -1,  -1,  -1,  -1,  -2,  -1,   1,   0,  -3,  -2,   0],  # A
    [-1,   5,   0,  -2,  -3,   1,   0,  -2,   0,  -3,  -2,   2,  -1,  -3,  -2,  -1,  -1,  -3,  -2,  -3],  # R
    [-2,   0,   6,   1,  -3,   0,   0,   0,   1,  -3,  -3,   0,  -2,  -3,  -2,   1,   0,  -4,  -2,  -3],  # N
    [-2,  -2,   1,   6,  -3,   0,   2,  -1,  -1,  -3,  -4,  -1,  -3,  -3,  -1,   0,  -1,  -4,  -3,  -3],  # D
    [ 0,  -3,  -3,  -3,   9,  -3,  -4,  -3,  -3,  -1,  -1,  -3,  -1,  -2,  -3,  -1,  -1,  -2,  -2,  -1],  # C
    [-1,   1,   0,   0,  -3,   5,   2,  -2,   0,  -3,  -2,   1,   0,  -3,  -1,   0,  -1,  -2,  -1,  -2],  # Q
    [-1,   0,   0,   2,  -4,   2,   5,  -2,   0,  -3,  -3,   1,  -2,  -3,  -1,   0,  -1,  -3,  -2,  -2],  # E
    [ 0,  -2,   0,  -1,  -3,  -2,  -2,   6,  -2,  -4,  -4,  -2,  -3,  -3,  -2,   0,  -2,  -2,  -3,  -3],  # G
    [-2,   0,   1,  -1,  -3,   0,   0,  -2,   8,  -3,  -3,  -1,  -2,  -1,  -2,  -1,  -2,  -2,   2,  -3],  # H
    [-1,  -3,  -3,  -3,  -1,  -3,  -3,  -4,  -3,   4,   2,  -3,   1,   0,  -3,  -2,  -1,  -3,  -1,   3],  # I
    [-1,  -2,  -3,  -4,  -1,  -2,  -3,  -4,  -3,   2,   4,  -2,   2,   0,  -3,  -2,  -1,  -2,  -1,   1],  # L
    [-1,   2,   0,  -1,  -3,   1,   1,  -2,  -1,  -3,  -2,   5,  -1,  -3,  -1,   0,  -1,  -3,  -2,  -2],  # K
    [-1,  -1,  -2,  -3,  -1,   0,  -2,  -3,  -2,   1,   2,  -1,   5,   0,  -2,  -1,  -1,  -1,  -1,   1],  # M
    [-2,  -3,  -3,  -3,  -2,  -3,  -3,  -3,  -1,   0,   0,  -3,   0,   6,  -4,  -2,  -2,   1,   3,  -1],  # F
    [-1,  -2,  -2,  -1,  -3,  -1,  -1,  -2,  -2,  -3,  -3,  -1,  -2,  -4,   7,  -1,  -1,  -4,  -3,  -2],  # P
    [ 1,  -1,   1,   0,  -1,   0,   0,   0,  -1,  -2,  -2,   0,  -1,  -2,  -1,   4,   1,  -3,  -2,  -2],  # S
    [ 0,  -1,   0,  -1,  -1,  -1,  -1,  -2,  -2,  -1,  -1,  -1,  -1,  -2,  -1,   1,   5,  -2,  -2,   0],  # T
    [-3,  -3,  -4,  -4,  -2,  -2,  -3,  -2,  -2,  -3,  -2,  -3,  -1,   1,  -4,  -3,  -2,  11,   2,  -3],  # W
    [-2,  -2,  -2,  -3,  -2,  -1,  -2,  -3,   2,  -1,  -1,  -2,  -1,   3,  -3,  -2,  -2,   2,   7,  -1],  # Y
    [ 0,  -3,  -3,  -3,  -1,  -2,  -2,  -3,  -3,   3,   1,  -2,   1,  -1,  -2,  -2,   0,  -3,  -1,   4],  # V
]
# to_idx = row (目标氨基酸 to), from_idx = col (源氨基酸 from)
# BLOSUM62[a, c] = 从 c 替换为 a 的得分（c→a）
BLOSUM62: np.ndarray = np.array(_BLOSUM62_ROWS, dtype=np.float32)


def _seq_to_indices(seq: str) -> List[int]:
    """单字母序列 -> 索引列表"""
    return [AA_1_TO_IDX[aa] for aa in seq]


def _indices_to_seq(indices: List[int]) -> str:
    """索引列表 -> 单字母序列"""
    return "".join(AA_1LETTER[PROTEIN_RESIDUE_NAMES[i]] for i in indices)


def compute_mutation_benefit(
    grad_matrix: torch.Tensor,
    current_indices: List[int],
    use_prior: bool = True,
    prior_strength: float = 0.1,
) -> torch.Tensor:
    """
    计算突变收益矩阵 Delta_L，并可选地引入 BLOSUM62 生物物理先验。

    Delta_L[i, a] = G[i, a] - G[i, c_i]
    其中 c_i 是位置 i 当前的氨基酸索引。
    负值表示突变为 a 可降低 loss（有利突变）。

    当 use_prior=True 时，在梯度收益上叠加 BLOSUM62 替换得分：
      adjusted[i, a] = delta_L[i, a] + prior_strength * BLOSUM62[a, c_i]
    BLOSUM62[a, c] 为从 c 替换为 a 的演化保守性得分（正值=保守）。
    这会使演化上合理的突变（高 BLOSUM62 分）更容易被选中，
    而不合理的突变（低分或负分）被抑制。

    Args:
        grad_matrix:      [L_binder, 20] 梯度矩阵
        current_indices:  长度 L_binder 的当前氨基酸索引列表
        use_prior:        是否应用 BLOSUM62 先验约束掩码
        prior_strength:   先验权重系数（推荐 0.05~0.2，太大则梯度信息被淹没）

    Returns:
        delta_L: [L_binder, 20] 突变收益矩阵
    """
    L = grad_matrix.shape[0]
    current_grad = grad_matrix[torch.arange(L), current_indices]  # [L]
    delta_L = grad_matrix - current_grad.unsqueeze(1)  # [L, 20]

    if use_prior:
        # BLOSUM62[current_a, current_c] — 从 c_i 替换为 a 的保守性得分
        current_indices_arr = np.array(current_indices, dtype=np.int64)  # [L]
        # 从 BLOSUM62 中按行（to）和列（from）索引
        prior_scores = BLOSUM62[:, current_indices_arr]  # [20, L]
        prior_scores = prior_scores.T  # [L, 20]
        delta_L = delta_L + prior_strength * torch.from_numpy(prior_scores).to(delta_L)

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
    use_prior: bool = True,
    prior_strength: float = 0.1,
) -> List[str]:
    """
    梯度引导的离散候选序列生成（含可选 BLOSUM62 先验约束）。

    完整流程：
    1. 将梯度矩阵转换为突变收益矩阵 Delta_L
    2. （可选）叠加 BLOSUM62 先验约束掩码 M_prior
    3. 选取 Top-K 候选突变
    4. 按收益分布采样组合，生成 B 条离散候选序列

    Args:
        grad_matrix:          [L_binder, 20] 梯度矩阵 G = nabla_Theta L
        current_seq:         当前离散序列（单字母 str）
        K_top:              选取 Top-K 候选突变数
        B_candidates:       生成 B 条候选序列
        T_sample:           采样温度（越小越确定性，越大越随机）
        max_mutations_per_seq: 每条候选的最大突变数
        use_prior:          是否启用 BLOSUM62 先验约束掩码（M_prior）
        prior_strength:     先验权重系数（推荐 0.05~0.2）

    Returns:
        B 条离散候选序列（单字母 str 列表）
    """
    current_indices = _seq_to_indices(current_seq)

    grad_cpu = grad_matrix.detach().cpu()

    delta_L = compute_mutation_benefit(
        grad_cpu, current_indices,
        use_prior=use_prior, prior_strength=prior_strength,
    )

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
