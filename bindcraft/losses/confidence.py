"""
BindCraft v2 - Confidence Loss 函数

对标原版 BindCraft 的置信度 loss 体系：
- pLDDT: binder 的 pLDDT 置信度
- pAE (intra): binder 内部 PAE
- i_pAE (inter): binder-target 界面 PAE
- con (intra): binder 内部接触数
- i_con (inter): binder-target 界面接触数
- iPTM: 界面 pTM
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


# ── pLDDT Loss ────────────────────────────────────────────────────────────────

def compute_plddt_loss(
    plddt_logits: torch.Tensor,
    binder_atom_mask: torch.Tensor,
    plddt_params: dict,
) -> torch.Tensor:
    """
    pLDDT loss: 1 - mean(plddt_binder)

    Args:
        plddt_logits: [N_sample, N_atom, plddt_bins]
        binder_atom_mask: [N_atom] bool
        plddt_params: dict with min_bin, max_bin, no_bins
    """
    from protenix.model.sample_confidence import logits_to_score
    plddt_score = logits_to_score(plddt_logits, **plddt_params)  # [N_sample, N_atom]
    binder_plddt = plddt_score[:, binder_atom_mask].mean()
    return 1.0 - binder_plddt


# ── PAE Loss (intra / inter) ─────────────────────────────────────────────────

def compute_pae_loss(
    pae_logits: torch.Tensor,
    asym_id: torch.Tensor,
    pae_params: dict,
    mode: str = "intra",
    hotspot_token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    PAE loss: mean PAE score over intra-chain or inter-chain token pairs.

    mode="intra": binder 内部（同链）的 PAE -> 越低越好
    mode="inter": binder-target 界面（跨链）的 PAE -> 越低越好
      若提供 hotspot_token_mask，则 inter 模式只计算 binder x hotspot 的 PAE（仿 v1）。

    Args:
        pae_logits: [N_sample, N_token, N_token, pae_bins]
        asym_id: [N_token] chain ID
        pae_params: dict with min_bin, max_bin, no_bins
        mode: "intra" or "inter"
        hotspot_token_mask: [N_token] bool, optional hotspot target token mask
    """
    from protenix.model.sample_confidence import logits_to_score
    pae_score = logits_to_score(pae_logits, **pae_params)  # [N_sample, N_token, N_token]
    # 对齐 ColabDesign: 归一化到约 [0,1] 并对称化
    pae_score = pae_score / 31.0
    pae_score = 0.5 * (pae_score + pae_score.transpose(-1, -2))

    # 以第一个 token 所在链作为 binder 链，避免 unique() 排序带来的链顺序歧义
    binder_id = asym_id[0]
    binder_mask = asym_id == binder_id

    if mode == "intra":
        mask = binder_mask.unsqueeze(0) & binder_mask.unsqueeze(1)
        n = mask.sum()
        if n == 0:
            return torch.tensor(0.0, device=pae_score.device, requires_grad=True)
        return (pae_score * mask.unsqueeze(0).float()).sum() / (n * pae_score.shape[0])
    else:
        if hotspot_token_mask is not None:
            # v1 style: only compute PAE between binder tokens and hotspot target tokens
            mask_inter = binder_mask.unsqueeze(1) & hotspot_token_mask.unsqueeze(0)
        else:
            mask_inter = binder_mask.unsqueeze(0) != binder_mask.unsqueeze(1)
        n = mask_inter.sum()
        if n == 0:
            return torch.tensor(0.0, device=pae_score.device, requires_grad=True)
        return (pae_score * mask_inter.unsqueeze(0).float()).sum() / (n * pae_score.shape[0])


# ── Contact Loss (con / i_con) ───────────────────────────────────────────────

def compute_contact_loss(
    distogram_logits: torch.Tensor,
    binder_token_mask: torch.Tensor,
    target_token_mask: torch.Tensor,
    distogram_params: dict,
    mode: str = "intra",
    cutoff: float = 14.0,
    num: int = 2,
    num_pos: int = 2,
    seqsep: int = 9,
    binary: bool = False,
    residue_index: torch.Tensor = None,
    hotspot_token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Contact loss（对齐 ColabDesign get_con_loss / _get_con_loss）。

    Args:
        distogram_logits: [N_token, N_token, no_bins]
        binder_token_mask: [N_token] bool
        target_token_mask: [N_token] bool
        distogram_params: dict with min_bin, max_bin, no_bins
        mode: "intra" (binder internal) or "inter" (binder-target interface)
        cutoff: contact distance threshold (Angstrom)
        num: target number of contacts per residue
        seqsep: minimum sequence separation for intra contacts
        hotspot_token_mask: [N_token] bool, optional; when set in inter mode,
            only contacts against hotspot target tokens are counted (v1 style)
    """
    def _min_k(x: torch.Tensor, k: int, mask: torch.Tensor = None) -> torch.Tensor:
        """沿最后一维取最小 k 个（忽略 mask=False 位置），并取均值。"""
        if mask is not None:
            x = torch.where(mask, x, torch.full_like(x, float("inf")))
        y, _ = torch.sort(x, dim=-1)
        if y.shape[-1] == 0:
            return torch.zeros(y.shape[:-1], device=y.device, dtype=y.dtype)
        k = max(1, min(k, y.shape[-1]))
        yk = y[..., :k]
        finite = torch.isfinite(yk)
        denom = finite.float().sum(dim=-1).clamp_min(1.0)
        yk = torch.where(finite, yk, torch.zeros_like(yk))
        return yk.sum(dim=-1) / denom

    prob = F.softmax(distogram_logits, dim=-1)
    min_bin = distogram_params["min_bin"]
    max_bin = distogram_params["max_bin"]
    no_bins = distogram_params["no_bins"]
    bin_width = (max_bin - min_bin) / no_bins
    boundaries = torch.linspace(min_bin, max_bin - bin_width, no_bins,
                                device=prob.device)
    bin_centers = boundaries + 0.5 * bin_width
    thres_idx = int((bin_centers < cutoff).sum().item())
    bins = (bin_centers < cutoff).to(distogram_logits.dtype)  # [B]
    log_prob = F.log_softmax(distogram_logits, dim=-1)
    masked_logits = distogram_logits + (1.0 - bins) * (-1e7)
    prob_cut = F.softmax(masked_logits, dim=-1)
    con_loss_cat_ent = -(prob_cut * log_prob).sum(dim=-1)  # [N, N]
    con_loss_bin_ent = -torch.log((prob * bins).sum(dim=-1) + 1e-8)  # [N, N]
    pair_loss = con_loss_bin_ent if binary else con_loss_cat_ent  # [N, N]

    N = pair_loss.shape[0]
    if residue_index is not None:
        offset = residue_index.unsqueeze(1) - residue_index.unsqueeze(0)
    else:
        idx = torch.arange(N, device=pair_loss.device)
        offset = idx.unsqueeze(1) - idx.unsqueeze(0)

    # ColabDesign 中 m 初始由 seqsep 控制
    if mode == "intra":
        m = (offset.abs() >= seqsep)
    else:
        m = torch.ones_like(offset, dtype=torch.bool)

    if mode == "intra":
        # binder-binder
        mask_2d = binder_token_mask.unsqueeze(0) & binder_token_mask.unsqueeze(1)
        m = m & mask_2d
        mask_1d = binder_token_mask
        effective_num_pos = num_pos
    else:
        if hotspot_token_mask is not None:
            # v1 style: 行 = hotspot 残基，列 = binder，num_pos = 全部 hotspot（等效 inf）
            m = m & binder_token_mask.unsqueeze(0)
            mask_1d = hotspot_token_mask
            effective_num_pos = int(hotspot_token_mask.sum().item())
        else:
            # 默认: 行 = binder，列 = target
            m = m & target_token_mask.unsqueeze(0)
            mask_1d = binder_token_mask
            effective_num_pos = num_pos

    per_pos = _min_k(pair_loss, k=num, mask=m)
    return _min_k(per_pos, k=effective_num_pos, mask=mask_1d)


# ── iPTM Loss ─────────────────────────────────────────────────────────────────

def compute_iptm_loss(
    model,
    input_feature_dict: Dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    x0_hat: torch.Tensor,
    inplace_safe: bool = True,
    chunk_size: int = None,
) -> torch.Tensor:
    """
    iPTM loss: 1 - iPTM（最小化此值 = 最大化 iPTM）。

    调用 Protenix 的 confidence head 和 sample_confidence 计算 iPTM。
    """
    from protenix.model import sample_confidence

    if x0_hat.dim() == 2:
        x0_hat = x0_hat.unsqueeze(0)

    plddt, pae, pde, resolved = model.run_confidence_head(
        input_feature_dict=input_feature_dict,
        s_inputs=s_inputs,
        s_trunk=s_trunk,
        z_trunk=z_trunk,
        pair_mask=None,
        x_pred_coords=x0_hat,
        use_embedding=True,
        triangle_multiplicative=model.configs.triangle_multiplicative,
        triangle_attention=model.configs.triangle_attention,
        inplace_safe=inplace_safe,
        chunk_size=chunk_size,
    )

    pae_params = sample_confidence.get_bin_params(model.configs.loss.pae)
    _, pae_prob = sample_confidence.logits_to_score(pae, **pae_params, return_prob=True)

    token_asym_id = input_feature_dict["asym_id"]
    token_has_frame = input_feature_dict["has_frame"]
    if token_has_frame.dtype != torch.bool:
        token_has_frame = token_has_frame.bool()

    iptm = sample_confidence.calculate_iptm(
        pae_prob=pae_prob,
        has_frame=token_has_frame,
        asym_id=token_asym_id,
        **pae_params,
    )

    if iptm.dim() > 0:
        iptm = iptm.mean()

    return 1.0 - iptm


# ── 一次性获取所有 confidence 输出 ────────────────────────────────────────────

def run_confidence_heads(
    model,
    input_feature_dict: Dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    x0_hat: torch.Tensor,
    inplace_safe: bool = False,
    chunk_size: int = None,
) -> dict:
    """
    一次调用 confidence head，返回所有 logits，供各 loss 函数复用。

    Returns:
        dict with keys: plddt_logits, pae_logits, pde_logits, distogram_logits
    """
    from protenix.model import sample_confidence

    if x0_hat.dim() == 2:
        x0_hat = x0_hat.unsqueeze(0)

    distogram_logits = model.distogram_head(z_trunk)

    plddt, pae, pde, resolved = model.run_confidence_head(
        input_feature_dict=input_feature_dict,
        s_inputs=s_inputs,
        s_trunk=s_trunk,
        z_trunk=z_trunk,
        pair_mask=None,
        x_pred_coords=x0_hat,
        use_embedding=True,
        triangle_multiplicative=model.configs.triangle_multiplicative,
        triangle_attention=model.configs.triangle_attention,
        inplace_safe=inplace_safe,
        chunk_size=chunk_size,
    )

    return {
        "plddt_logits": plddt,
        "pae_logits": pae,
        "pde_logits": pde,
        "distogram_logits": distogram_logits,
    }


# ── PDE Loss (intra / inter) ────────────────────────────────────────────────

def compute_pde_loss(
    pde_logits: torch.Tensor,
    asym_id: torch.Tensor,
    pde_params: dict,
    mode: str = "intra",
    contact_probs: torch.Tensor = None,
) -> torch.Tensor:
    """
    PDE loss (Predicted Distance Error).

    mode="intra": binder 内部 pair 的 raw mean（与 PAE intra 模式一致）。
    mode="inter": contact-weighted gPDE（AF3 rank_metric 方式）。
        用 contact_probs 加权，只聚焦有接触的 binder-target pair，
        避免被大量非界面 pair 稀释。

    Args:
        pde_logits: [N_sample, N_token, N_token, pde_bins]
        asym_id: [N_token] chain ID
        pde_params: dict with min_bin, max_bin, no_bins
        mode: "intra" or "inter"
        contact_probs: [N_token, N_token] contact probability (inter only)
    """
    from protenix.model.sample_confidence import logits_to_score
    pde_score = logits_to_score(pde_logits, **pde_params)
    pde_score = pde_score / pde_params["max_bin"]
    pde_score = 0.5 * (pde_score + pde_score.transpose(-1, -2))

    binder_id = asym_id[0]

    if mode == "intra":
        mask = (asym_id == binder_id).unsqueeze(0) & (asym_id == binder_id).unsqueeze(1)
        n = mask.sum()
        if n == 0:
            return torch.tensor(0.0, device=pde_score.device, requires_grad=True)
        return (pde_score * mask.unsqueeze(0).float()).sum() / (n * pde_score.shape[0])
    else:
        # contact-weighted gPDE: sum(PDE * contact_probs) / sum(contact_probs)
        assert contact_probs is not None, "inter mode requires contact_probs"
        mask_inter = (asym_id == binder_id).unsqueeze(0) != (asym_id == binder_id).unsqueeze(1)
        weighted_mask = contact_probs * mask_inter.float()
        denom = weighted_mask.sum()
        if denom < 1e-8:
            return torch.tensor(0.0, device=pde_score.device, requires_grad=True)
        return (pde_score * weighted_mask.unsqueeze(0)).sum() / (denom * pde_score.shape[0])
