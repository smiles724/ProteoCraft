"""
BindCraft v2 - 统一 Loss 计算

将 12 项 loss 的计算逻辑集中到一个函数中，供一阶优化（pipeline.py）
和零阶验证（zeroth_order.py）共同调用，避免两处逻辑不一致。

原有 8 项: pLDDT / pAE / i_pAE / con / i_con / iPTM / Rg / Helix
新增 4 项: clash / i_clash / pde / i_pde
"""

import torch
from typing import Dict, Tuple, Any, List, Optional

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


def _normalize_task_type(task_type: Optional[str]) -> Optional[str]:
    if task_type is None:
        return None
    normalized = str(task_type).strip().lower()
    aliases = {
        "de_novo": "type2",
        "denovo": "type2",
        "binder": "type2",
        "scaffold_redesign": "type1",
        "cdr": "type1",
        "pmhc": "type3",
        "tcr_mimic": "type3",
        "tcr-mimic": "type3",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"type1", "type2", "type3"}:
        raise ValueError(f"未知 task_type: {task_type}")
    return normalized


def _target_chain_masks_in_order(
    asym_id: torch.Tensor,
    binder_chain_id: torch.Tensor,
) -> List[torch.Tensor]:
    chain_ids: List[int] = []
    for value in asym_id.detach().cpu().tolist():
        chain_id = int(value)
        if chain_id == int(binder_chain_id.detach().cpu().item()):
            continue
        if chain_id not in chain_ids:
            chain_ids.append(chain_id)
    return [asym_id == chain_id for chain_id in chain_ids]


def _normalize_target_chain_roles(target_chain_roles: Any) -> List[str]:
    if target_chain_roles is None:
        raise ValueError("task_type='type3' requires target_chain_roles")
    if isinstance(target_chain_roles, str):
        raw_roles = [part.strip() for part in target_chain_roles.split(",") if part.strip()]
    else:
        raw_roles = [str(role).strip() for role in target_chain_roles]

    normalized_roles = []
    for role in raw_roles:
        role_l = role.lower()
        if role_l in {"p", "pep", "peptide"}:
            normalized_roles.append("peptide")
        elif role_l in {"h", "hla", "mhc", "m", "background"}:
            normalized_roles.append("hla")
        else:
            raise ValueError(f"未知 target chain role: {role}")
    return normalized_roles


def _normalize_type1_partner_chain_roles(target_chain_roles: Any) -> List[str]:
    if target_chain_roles is None:
        return []
    if isinstance(target_chain_roles, str):
        raw_roles = [part.strip() for part in target_chain_roles.split(",") if part.strip()]
    else:
        raw_roles = [str(role).strip() for role in target_chain_roles]

    normalized_roles = []
    for role in raw_roles:
        role_l = role.lower()
        if role_l in {"a", "ag", "antigen", "target", "t", "pep", "peptide"}:
            normalized_roles.append("target")
        elif role_l in {"s", "scaffold", "context", "fixed", "background"}:
            normalized_roles.append("scaffold")
        else:
            raise ValueError(f"未知 target/scaffold chain role: {role}")
    return normalized_roles


def compute_all_losses(
    x0_flat: torch.Tensor,
    plddt_logits: torch.Tensor,
    pae_logits: torch.Tensor,
    distogram_logits: torch.Tensor,
    input_feature_dict: Dict,
    L_binder: int,
    weights: Dict[str, float],
    contact_params: Dict[str, float],
    plddt_params: dict,
    pae_params: dict,
    distogram_params: dict,
    pde_logits: torch.Tensor = None,
    pde_params: dict = None,
    clash_params: Dict[str, float] = None,
    hotspot_params: Dict[str, Any] = None,
    opt_token_mask: torch.Tensor = None,
    task_type: str = None,
    target_chain_roles: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    在给定的全原子坐标和 confidence logits 上计算全部 12 项 loss。

    Args:
        x0_flat:           全原子坐标 [N_atoms, 3]（squeeze 后，无 batch 维）
        plddt_logits:      pLDDT logits [N_sample, N_atom, plddt_bins]
        pae_logits:        PAE logits [N_sample, N_token, N_token, pae_bins]
        distogram_logits:  Distogram logits [N_token, N_token, no_bins]
        input_feature_dict: 输入特征字典（含 asym_id / atom_to_token_idx / has_frame 等）
        L_binder:          Binder 残基数
        weights:           各项 loss 权重
        contact_params:    接触 loss 参数（cutoff / num / seqsep）
        plddt_params:      pLDDT 的 bin 参数 (min_bin, max_bin, no_bins)
        pae_params:        PAE 的 bin 参数
        distogram_params:  Distogram 的 bin 参数
        pde_logits:        PDE logits [N_sample, N_token, N_token, pde_bins] (None to skip)
        pde_params:        PDE 的 bin 参数 (None to skip)
        clash_params:      碰撞 loss 参数 (r_clash, bond_sep) (None to skip)
        opt_token_mask:    可选优化区域 mask。None 表示整个 binder 参与 loss；
                           长度可为 binder token 数或全 token 数。
        task_type:         可选任务类型；"type3" 启用 pMHC peptide/HLA 边界。
        target_chain_roles: Type III 时为 target 链角色，如 ["peptide", "hla"]；
                           CDR/Type I 时可传 ["target", "scaffold"]，按非 binder 链顺序
                           把抗原和固定外部 scaffold 拆开。

    Returns:
        total_loss: 加权总 loss（标量 tensor）
        losses:     各项分项 loss 的字典（含 "total" 键）
    """
    device = x0_flat.device

    # ── 构建 mask ────────────────────────────────────────────────────────
    atom_to_token = input_feature_dict["atom_to_token_idx"]
    asym_id = input_feature_dict["asym_id"]
    binder_chain_id = asym_id[0]
    binder_token_mask = asym_id == binder_chain_id
    partner_token_mask = ~binder_token_mask
    target_token_mask = partner_token_mask
    normalized_task = _normalize_task_type(task_type)
    is_type3 = normalized_task == "type3"
    partner_chain_masks = _target_chain_masks_in_order(asym_id, binder_chain_id)

    external_scaffold_token_mask = torch.zeros_like(target_token_mask, dtype=torch.bool)
    has_type1_partner_roles = False
    if target_chain_roles is not None and not is_type3:
        roles = _normalize_type1_partner_chain_roles(target_chain_roles)
        if len(roles) != len(partner_chain_masks):
            raise ValueError(
                f"target_chain_roles length {len(roles)} must equal non-binder chain count "
                f"{len(partner_chain_masks)}"
            )
        target_token_mask = torch.zeros_like(partner_token_mask, dtype=torch.bool)
        for role, chain_mask in zip(roles, partner_chain_masks):
            if role == "target":
                target_token_mask = target_token_mask | chain_mask
            elif role == "scaffold":
                external_scaffold_token_mask = external_scaffold_token_mask | chain_mask
        if target_token_mask.sum() == 0:
            raise ValueError("target_chain_roles must mark at least one non-binder chain as target")
        has_type1_partner_roles = True

    design_token_mask = binder_token_mask
    local_redesign = False
    if opt_token_mask is not None:
        opt_token_mask = torch.as_tensor(opt_token_mask, device=device, dtype=torch.bool)
        if opt_token_mask.numel() == asym_id.numel():
            if (opt_token_mask & ~binder_token_mask).any():
                raise ValueError("opt_token_mask may only select binder tokens")
            design_token_mask = opt_token_mask & binder_token_mask
        elif opt_token_mask.numel() == int(binder_token_mask.sum().item()):
            design_token_mask = torch.zeros_like(binder_token_mask, dtype=torch.bool)
            binder_indices = binder_token_mask.nonzero(as_tuple=True)[0]
            design_token_mask[binder_indices] = opt_token_mask
        else:
            raise ValueError(
                f"opt_token_mask length {opt_token_mask.numel()} must equal binder token count "
                f"{int(binder_token_mask.sum().item())} or total token count {asym_id.numel()}"
            )
        if design_token_mask.sum() == 0:
            raise ValueError("opt_token_mask selects no binder tokens")
        if (design_token_mask & ~binder_token_mask).any():
            raise ValueError("opt_token_mask may only select binder tokens")
        local_redesign = bool((design_token_mask != binder_token_mask).any().item())
    if is_type3 and local_redesign:
        raise ValueError("task_type='type3' does not support opt_token_mask; Type III optimizes the full binder")
    binder_scaffold_token_mask = binder_token_mask & ~design_token_mask
    scaffold_token_mask = binder_scaffold_token_mask | external_scaffold_token_mask

    binder_atom_mask = binder_token_mask[atom_to_token]
    target_atom_mask = target_token_mask[atom_to_token]
    design_atom_mask = design_token_mask[atom_to_token]
    scaffold_atom_mask = scaffold_token_mask[atom_to_token]

    rep_atom_mask = input_feature_dict.get("distogram_rep_atom_mask", None)
    if rep_atom_mask is not None:
        rep_atom_mask = rep_atom_mask.bool()
        binder_rep_atom_mask = binder_atom_mask & rep_atom_mask
        design_rep_atom_mask = design_atom_mask & rep_atom_mask
        if binder_rep_atom_mask.sum() == 0:
            binder_rep_atom_mask = binder_atom_mask
        if design_rep_atom_mask.sum() == 0:
            design_rep_atom_mask = design_atom_mask
    else:
        binder_rep_atom_mask = binder_atom_mask
        design_rep_atom_mask = design_atom_mask

    hotspot_token_mask = None
    _hotspot_params = hotspot_params or {}
    hotspot_concat_indices = _hotspot_params.get("target_hotspot_concat_indices") or []
    if hotspot_concat_indices:
        target_token_indices = partner_token_mask.nonzero(as_tuple=True)[0]
        invalid_indices = [
            int(index) for index in hotspot_concat_indices
            if int(index) < 0 or int(index) >= target_token_indices.shape[0]
        ]
        if invalid_indices:
            raise ValueError(
                f"hotspot target concat indices 越界: {invalid_indices}; target_len={target_token_indices.shape[0]}"
            )

        hotspot_index_tensor = torch.as_tensor(
            [int(index) for index in hotspot_concat_indices],
            device=device,
            dtype=torch.long,
        )
        hotspot_token_indices = target_token_indices[hotspot_index_tensor]
        hotspot_token_mask = torch.zeros_like(target_token_mask, dtype=torch.bool)
        hotspot_token_mask[hotspot_token_indices] = True

    peptide_token_mask = None
    hla_token_mask = None
    hla_pair_mask = None
    if is_type3:
        roles = _normalize_target_chain_roles(target_chain_roles)
        if len(roles) != len(partner_chain_masks):
            raise ValueError(
                f"target_chain_roles length {len(roles)} must equal target chain count {len(partner_chain_masks)}"
            )
        peptide_token_mask = torch.zeros_like(partner_token_mask, dtype=torch.bool)
        hla_token_mask = torch.zeros_like(partner_token_mask, dtype=torch.bool)
        for role, chain_mask in zip(roles, partner_chain_masks):
            if role == "peptide":
                peptide_token_mask = peptide_token_mask | chain_mask
            elif role == "hla":
                hla_token_mask = hla_token_mask | chain_mask
        if peptide_token_mask.sum() == 0:
            raise ValueError("task_type='type3' requires at least one peptide chain")
        if hla_token_mask.sum() == 0:
            raise ValueError("task_type='type3' requires at least one HLA/MHC chain")
        if hotspot_token_mask is not None and (hotspot_token_mask & ~peptide_token_mask).any():
            raise ValueError("Type III hotspot indices must fall on peptide chains")
    elif hotspot_token_mask is not None and (hotspot_token_mask & ~target_token_mask).any():
        raise ValueError("hotspot target concat indices must fall on target chains")

    interface_target_mask = hotspot_token_mask if hotspot_token_mask is not None else target_token_mask
    if is_type3:
        interface_target_mask = hotspot_token_mask if hotspot_token_mask is not None else peptide_token_mask

    pae_intra_pair_mask = None
    pae_inter_pair_mask = None
    pde_intra_pair_mask = None
    pde_inter_pair_mask = None
    if local_redesign or has_type1_partner_roles:
        design_x_design = design_token_mask.unsqueeze(0) & design_token_mask.unsqueeze(1)
        design_x_scaffold = design_token_mask.unsqueeze(1) & scaffold_token_mask.unsqueeze(0)
        design_x_target = design_token_mask.unsqueeze(1) & interface_target_mask.unsqueeze(0)
        pae_intra_pair_mask = design_x_design | design_x_scaffold
        pae_inter_pair_mask = design_x_target
        pde_intra_pair_mask = pae_intra_pair_mask
        pde_inter_pair_mask = design_x_target
    elif is_type3:
        binder_x_peptide = binder_token_mask.unsqueeze(1) & interface_target_mask.unsqueeze(0)
        hla_pair_mask = binder_token_mask.unsqueeze(1) & hla_token_mask.unsqueeze(0)
        pae_inter_pair_mask = binder_x_peptide
        pde_inter_pair_mask = binder_x_peptide

    # ── 逐项计算 loss ───────────────────────────────────────────────────
    losses: Dict[str, torch.Tensor] = {}
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    # pLDDT
    if weights.get("plddt", 0) != 0:
        plddt_atom_mask = design_rep_atom_mask if local_redesign else binder_rep_atom_mask
        l_plddt = compute_plddt_loss(plddt_logits, plddt_atom_mask, plddt_params)
        losses["plddt"] = l_plddt
        total_loss = total_loss + weights["plddt"] * l_plddt

    # pAE (intra)
    if weights.get("pae", 0) != 0:
        l_pae = compute_pae_loss(
            pae_logits, asym_id, pae_params, mode="intra",
            pair_mask=pae_intra_pair_mask,
        )
        losses["pae"] = l_pae
        total_loss = total_loss + weights["pae"] * l_pae

    # i_pAE (inter)
    if weights.get("i_pae", 0) != 0:
        l_ipae = compute_pae_loss(
            pae_logits, asym_id, pae_params, mode="inter",
            hotspot_token_mask=hotspot_token_mask,
            pair_mask=pae_inter_pair_mask,
        )
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
            row_token_mask=design_token_mask if local_redesign else None,
            col_token_mask=design_token_mask if local_redesign else None,
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
            num_pos=contact_params.get("inter_num_pos", contact_params["inter_num"]),
            binary=False,
            residue_index=input_feature_dict.get("residue_index"),
            hotspot_token_mask=hotspot_token_mask,
            row_token_mask=design_token_mask if local_redesign else (binder_token_mask if is_type3 else None),
            col_token_mask=interface_target_mask if (local_redesign or is_type3) else None,
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
        rg_atom_mask = design_rep_atom_mask if local_redesign else binder_rep_atom_mask
        binder_coords = x0_flat[rg_atom_mask]
        rg_len = int(design_token_mask.sum().item()) if local_redesign else L_binder
        l_rg = compute_rg_loss(binder_coords, rg_len)
        losses["rg"] = l_rg
        total_loss = total_loss + weights["rg"] * l_rg

    # Helix（负权重 = 鼓励螺旋）
    if weights.get("helix", 0) != 0:
        l_helix = compute_helix_loss(
            distogram_logits,
            design_token_mask if local_redesign else binder_token_mask,
            distogram_params,
            residue_index=input_feature_dict.get("residue_index"),
        )
        losses["helix"] = l_helix
        total_loss = total_loss + weights["helix"] * l_helix

    # ── 新增: Clash ──────────────────────────────────────────────────────

    _clash_params = clash_params or {}

    # clash (intra)
    if weights.get("clash", 0) != 0:
        if local_redesign or has_type1_partner_roles:
            residue_index = input_feature_dict.get("residue_index", asym_id)
            l_clash_oo = compute_soft_clash_loss(
                x0_flat, binder_atom_mask, target_atom_mask,
                atom_to_token, residue_index,
                mode="intra",
                r_clash=_clash_params.get("r_clash", 2.0),
                bond_sep=_clash_params.get("bond_sep", 1),
                atom_mask_a=design_atom_mask,
                atom_mask_b=design_atom_mask,
                exclude_bonded=True,
            )
            l_clash_os = compute_soft_clash_loss(
                x0_flat, binder_atom_mask, target_atom_mask,
                atom_to_token, residue_index,
                mode="intra",
                r_clash=_clash_params.get("r_clash", 2.0),
                bond_sep=_clash_params.get("bond_sep", 1),
                atom_mask_a=design_atom_mask,
                atom_mask_b=scaffold_atom_mask,
                detach_b=True,
                exclude_bonded=True,
            )
            l_clash = l_clash_oo + l_clash_os
        else:
            l_clash = compute_soft_clash_loss(
                x0_flat, binder_atom_mask, target_atom_mask,
                atom_to_token, input_feature_dict.get("residue_index", asym_id),
                mode="intra",
                r_clash=_clash_params.get("r_clash", 2.0),
                bond_sep=_clash_params.get("bond_sep", 1),
            )
        losses["clash"] = l_clash
        total_loss = total_loss + weights["clash"] * l_clash

    # i_clash (inter)
    if weights.get("i_clash", 0) != 0:
        if local_redesign or has_type1_partner_roles:
            l_iclash = compute_soft_clash_loss(
                x0_flat, binder_atom_mask, target_atom_mask,
                atom_to_token, input_feature_dict.get("residue_index", asym_id),
                mode="inter",
                r_clash=_clash_params.get("r_clash", 2.0),
                bond_sep=_clash_params.get("bond_sep", 1),
                atom_mask_a=design_atom_mask,
                atom_mask_b=target_atom_mask,
                detach_b=True,
                exclude_bonded=False,
            )
        elif is_type3:
            l_iclash = compute_soft_clash_loss(
                x0_flat, binder_atom_mask, target_atom_mask,
                atom_to_token, input_feature_dict.get("residue_index", asym_id),
                mode="inter",
                r_clash=_clash_params.get("r_clash", 2.0),
                bond_sep=_clash_params.get("bond_sep", 1),
                atom_mask_a=binder_atom_mask,
                atom_mask_b=target_atom_mask,
                detach_b=True,
                exclude_bonded=False,
            )
        else:
            l_iclash = compute_soft_clash_loss(
                x0_flat, binder_atom_mask, target_atom_mask,
                atom_to_token, input_feature_dict.get("residue_index", asym_id),
                mode="inter",
                r_clash=_clash_params.get("r_clash", 2.0),
                bond_sep=_clash_params.get("bond_sep", 1),
            )
        losses["i_clash"] = l_iclash
        total_loss = total_loss + weights["i_clash"] * l_iclash

    # ── 新增: PDE ────────────────────────────────────────────────────────

    # pde (intra)
    if weights.get("pde", 0) != 0 and pde_logits is not None and pde_params is not None:
        l_pde = compute_pde_loss(
            pde_logits, asym_id, pde_params, mode="intra",
            pair_mask=pde_intra_pair_mask,
        )
        losses["pde"] = l_pde
        total_loss = total_loss + weights["pde"] * l_pde

    # i_pde (inter, contact-weighted gPDE)
    if weights.get("i_pde", 0) != 0 and pde_logits is not None and pde_params is not None:
        contact_probs = sample_confidence.compute_contact_prob(
            distogram_logits=distogram_logits, **distogram_params,
        )
        l_ipde = compute_pde_loss(
            pde_logits, asym_id, pde_params, mode="inter",
            contact_probs=contact_probs,
            pair_mask=pde_inter_pair_mask,
        )
        losses["i_pde"] = l_ipde
        total_loss = total_loss + weights["i_pde"] * l_ipde

    if is_type3:
        l_ipae_hla = compute_pae_loss(
            pae_logits, asym_id, pae_params, mode="inter",
            pair_mask=hla_pair_mask,
        )
        losses["i_pae_hla"] = l_ipae_hla
        if weights.get("i_pae_hla", 0) != 0:
            total_loss = total_loss + weights["i_pae_hla"] * l_ipae_hla

        l_icon_hla = compute_contact_loss(
            distogram_logits, binder_token_mask, target_token_mask,
            distogram_params, mode="inter",
            cutoff=contact_params["inter_cutoff"],
            num=contact_params["inter_num"],
            num_pos=contact_params.get("inter_num_pos", contact_params["inter_num"]),
            binary=False,
            residue_index=input_feature_dict.get("residue_index"),
            row_token_mask=binder_token_mask,
            col_token_mask=hla_token_mask,
        )
        losses["i_con_hla"] = l_icon_hla
        if weights.get("i_con_hla", 0) != 0:
            total_loss = total_loss + weights["i_con_hla"] * l_icon_hla

        if pde_logits is not None and pde_params is not None:
            contact_probs = sample_confidence.compute_contact_prob(
                distogram_logits=distogram_logits, **distogram_params,
            )
            l_ipde_hla = compute_pde_loss(
                pde_logits, asym_id, pde_params, mode="inter",
                contact_probs=contact_probs,
                pair_mask=hla_pair_mask,
            )
            losses["i_pde_hla"] = l_ipde_hla
            if weights.get("i_pde_hla", 0) != 0:
                total_loss = total_loss + weights["i_pde_hla"] * l_ipde_hla

    losses["total"] = total_loss
    return total_loss, losses
