"""
BindCraft v2 - 统一 Loss 计算

将 12 项 loss 的计算逻辑集中到一个函数中，供一阶优化（pipeline.py）
和零阶验证（zeroth_order.py）共同调用，避免两处逻辑不一致。

原有 8 项: pLDDT / pAE / i_pAE / con / i_con / iPTM / Rg / Helix
新增 4 项: clash / i_clash / pde / i_pde
"""

import torch
from typing import Dict, Tuple

from protenix.model import sample_confidence

from bindcraft.losses.confidence import (
    compute_plddt_loss,
    compute_pae_loss,
    compute_contact_loss,
    compute_pde_loss,
)
from bindcraft.losses.geometry import (
    compute_rg_loss,
    compute_helix_loss,
    compute_soft_clash_loss,
)


def compute_all_losses(
    x0_flat: torch.Tensor,
    plddt_logits: torch.Tensor,
    pae_logits: torch.Tensor,
    pde_logits: torch.Tensor,
    distogram_logits: torch.Tensor,
    input_feature_dict: Dict,
    L_binder: int,
    weights: Dict[str, float],
    contact_params: Dict[str, float],
    clash_params: Dict[str, float],
    plddt_params: dict,
    pae_params: dict,
    pde_params: dict,
    distogram_params: dict,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    在给定的全原子坐标和 confidence logits 上计算全部 12 项 loss。

    Args:
        x0_flat:           全原子坐标 [N_atoms, 3]
        plddt_logits:      pLDDT logits [N_sample, N_atom, plddt_bins]
        pae_logits:        PAE logits [N_sample, N_token, N_token, pae_bins]
        pde_logits:        PDE logits [N_sample, N_token, N_token, pde_bins] (None to skip)
        distogram_logits:  Distogram logits [N_token, N_token, no_bins]
        input_feature_dict: 输入特征字典
        L_binder:          Binder 残基数
        weights:           各项 loss 权重
        contact_params:    接触 loss 参数
        clash_params:      碰撞 loss 参数 (r_clash, bond_sep)
        plddt_params:      pLDDT bin 参数
        pae_params:        PAE bin 参数
        pde_params:        PDE bin 参数
        distogram_params:  Distogram bin 参数

    Returns:
        total_loss: 加权总 loss
        losses:     各项分项 loss 的字典（含 "total" 键）
    """
    device = x0_flat.device

    # ── 构建 mask ────────────────────────────────────────────────────────
    atom_to_token = input_feature_dict["atom_to_token_idx"]
    asym_id = input_feature_dict["asym_id"]
    binder_chain_id = asym_id[0]
    binder_token_mask = asym_id == binder_chain_id
    target_token_mask = ~binder_token_mask

    binder_atom_mask = binder_token_mask[atom_to_token]
    target_atom_mask = target_token_mask[atom_to_token]

    rep_atom_mask = input_feature_dict.get("distogram_rep_atom_mask", None)
    if rep_atom_mask is not None:
        rep_atom_mask = rep_atom_mask.bool()
        binder_rep_atom_mask = binder_atom_mask & rep_atom_mask
        if binder_rep_atom_mask.sum() == 0:
            binder_rep_atom_mask = binder_atom_mask
    else:
        binder_rep_atom_mask = binder_atom_mask

    # ── 逐项计算 loss ───────────────────────────────────────────────────
    losses: Dict[str, torch.Tensor] = {}
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    # pLDDT
    if weights.get("plddt", 0) != 0:
        l_plddt = compute_plddt_loss(plddt_logits, binder_rep_atom_mask, plddt_params)
        losses["plddt"] = l_plddt
        total_loss = total_loss + weights["plddt"] * l_plddt

    # pAE (intra)
    if weights.get("pae", 0) != 0:
        l_pae = compute_pae_loss(pae_logits, asym_id, pae_params, mode="intra")
        losses["pae"] = l_pae
        total_loss = total_loss + weights["pae"] * l_pae

    # i_pAE (inter)
    if weights.get("i_pae", 0) != 0:
        l_ipae = compute_pae_loss(pae_logits, asym_id, pae_params, mode="inter")
        losses["i_pae"] = l_ipae
        total_loss = total_loss + weights["i_pae"] * l_ipae

    # con (intra contacts)
    if weights.get("con", 0) != 0:
        l_con = compute_contact_loss(
            distogram_logits, binder_token_mask, target_token_mask,
            distogram_params, mode="intra",
            cutoff=contact_params["intra_cutoff"],
            num=contact_params["intra_num"],
            num_pos=contact_params["intra_num"],
            seqsep=contact_params["seqsep"],
            binary=False,
            residue_index=input_feature_dict.get("residue_index"),
        )
        losses["con"] = l_con
        total_loss = total_loss + weights["con"] * l_con

    # i_con (inter contacts)
    if weights.get("i_con", 0) != 0:
        l_icon = compute_contact_loss(
            distogram_logits, binder_token_mask, target_token_mask,
            distogram_params, mode="inter",
            cutoff=contact_params["inter_cutoff"],
            num=contact_params["inter_num"],
            num_pos=contact_params["inter_num"],
            binary=False,
            residue_index=input_feature_dict.get("residue_index"),
        )
        losses["i_con"] = l_icon
        total_loss = total_loss + weights["i_con"] * l_icon

    # iPTM
    if weights.get("iptm", 0) != 0:
        _, pae_prob = sample_confidence.logits_to_score(
            pae_logits, **pae_params, return_prob=True
        )
        token_has_frame = input_feature_dict["has_frame"]
        if token_has_frame.dtype != torch.bool:
            token_has_frame = token_has_frame.bool()
        iptm_val = sample_confidence.calculate_iptm(
            pae_prob=pae_prob,
            has_frame=token_has_frame,
            asym_id=asym_id,
            **pae_params,
        )
        if iptm_val.dim() > 0:
            iptm_val = iptm_val.mean()
        l_iptm = 1.0 - iptm_val
        losses["iptm"] = l_iptm
        total_loss = total_loss + weights["iptm"] * l_iptm

    # Rg
    if weights.get("rg", 0) != 0:
        binder_coords = x0_flat[binder_rep_atom_mask]
        l_rg = compute_rg_loss(binder_coords, L_binder)
        losses["rg"] = l_rg
        total_loss = total_loss + weights["rg"] * l_rg

    # Helix
    if weights.get("helix", 0) != 0:
        l_helix = compute_helix_loss(
            distogram_logits, binder_token_mask, distogram_params
        )
        losses["helix"] = l_helix
        total_loss = total_loss + weights["helix"] * l_helix

    # ── 新增: Clash ──────────────────────────────────────────────────────

    # clash (intra)
    if weights.get("clash", 0) != 0:
        l_clash = compute_soft_clash_loss(
            x0_flat, binder_atom_mask, target_atom_mask,
            atom_to_token, input_feature_dict.get("residue_index", asym_id),
            mode="intra",
            r_clash=clash_params.get("r_clash", 2.0),
            bond_sep=clash_params.get("bond_sep", 1),
        )
        losses["clash"] = l_clash
        total_loss = total_loss + weights["clash"] * l_clash

    # i_clash (inter)
    if weights.get("i_clash", 0) != 0:
        l_iclash = compute_soft_clash_loss(
            x0_flat, binder_atom_mask, target_atom_mask,
            atom_to_token, input_feature_dict.get("residue_index", asym_id),
            mode="inter",
            r_clash=clash_params.get("r_clash", 2.0),
            bond_sep=clash_params.get("bond_sep", 1),
        )
        losses["i_clash"] = l_iclash
        total_loss = total_loss + weights["i_clash"] * l_iclash

    # ── 新增: PDE ────────────────────────────────────────────────────────

    # pde (intra)
    if weights.get("pde", 0) != 0 and pde_logits is not None:
        l_pde = compute_pde_loss(pde_logits, asym_id, pde_params, mode="intra")
        losses["pde"] = l_pde
        total_loss = total_loss + weights["pde"] * l_pde

    # i_pde (inter, contact-weighted)
    if weights.get("i_pde", 0) != 0 and pde_logits is not None:
        contact_probs = sample_confidence.compute_contact_prob(
            distogram_logits=distogram_logits, **distogram_params,
        )
        l_ipde = compute_pde_loss(
            pde_logits, asym_id, pde_params, mode="inter",
            contact_probs=contact_probs,
        )
        losses["i_pde"] = l_ipde
        total_loss = total_loss + weights["i_pde"] * l_ipde

    losses["total"] = total_loss
    return total_loss, losses
