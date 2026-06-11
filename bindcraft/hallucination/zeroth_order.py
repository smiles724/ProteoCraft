"""
BindCraft v2 - 零阶全链路推断模块

给定一条离散氨基酸序列，执行完整的 AF3 多步 ODE 扩散采样（非单步近似），
在最终生成的全原子结构上计算与一阶优化完全相同的 8 项 loss。

这是"一阶提议 + 零阶验证"混合框架的基础组件：
    - 一阶梯度（pipeline.py）用于提议候选突变方向
    - 零阶推断（本模块）用于在物理真实结构上验收候选序列

与 run_full_eval.py 的区别：
    run_full_eval.py 只提取 Protenix 原生的 summary_confidence（pLDDT/iPTM 等），
    本模块额外计算与 pipeline.py 完全对齐的 8 项 loss（通过共享的 compute_all_losses），
    使得零阶验证的 loss 可以直接与一阶优化的 loss 进行比较和排序。

设计要点：
    - 不调用 _main_inference_loop（因其不暴露 z_trunk 和 distogram_logits），
      而是手动分步组装推断流程，保留对中间变量的完全控制。
    - 全程 torch.no_grad()，不构建计算图，显存开销远小于一阶反传。
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import logging

from protenix.model import sample_confidence

from bindcraft.utils import (
    build_complex_features,
    prepare_input_features,
    DEFAULT_LOSS_WEIGHTS,
    DEFAULT_CONTACT_PARAMS,
    DEFAULT_CLASH_PARAMS,
)
from bindcraft.losses.total_loss import compute_all_losses
from bindcraft.hallucination.reproducibility import temporary_seed

logger = logging.getLogger(__name__)


def _resolve_model_n_cycle(model, fallback: int = 4) -> int:
    """Return the model-configured Pairformer recycle count when available."""
    model_cfg = getattr(getattr(model, "configs", None), "model", None)
    n_cycle = getattr(model_cfg, "N_cycle", None)
    if n_cycle is None:
        n_cycle = getattr(model, "N_cycle", None)
    if n_cycle is None:
        return fallback
    return int(n_cycle)


@dataclass
class ZerothOrderResult:
    """零阶推断的结构化返回值"""

    binder_seq: str
    target_seq: str
    total_loss: float
    loss_dict: Dict[str, float]
    summary_confidence: dict
    coordinate: Optional[torch.Tensor] = field(default=None, repr=False)
    plddt_logits: Optional[torch.Tensor] = field(default=None, repr=False)  # [1, N_atom, bins]


def _to_python_list(values: torch.Tensor) -> list:
    return [int(v) for v in values.detach().cpu().tolist()]


def _local_token_positions(mask: torch.Tensor) -> torch.Tensor:
    return torch.arange(int(mask.sum().item()), device=mask.device, dtype=torch.long)


def _binder_token_to_local_index(
    atom_to_token_idx: torch.Tensor,
    binder_token_mask: torch.Tensor,
) -> Dict[int, int]:
    binder_token_indices = binder_token_mask.nonzero(as_tuple=True)[0]
    return {
        int(token_idx.item()): int(local_idx)
        for local_idx, token_idx in enumerate(binder_token_indices)
    }


def _secondary_structure_pattern_from_coords(
    binder_rep_coords: torch.Tensor,
) -> str:
    n = binder_rep_coords.shape[0]
    if n < 4:
        return "mixed"

    local = binder_rep_coords.float()
    i3_dist = torch.linalg.norm(local[:-3] - local[3:], dim=-1)
    helix_fraction = float((i3_dist < 6.5).float().mean().item())
    end_to_end = float(torch.linalg.norm(local[-1] - local[0]).item())
    rg = float(torch.sqrt(((local - local.mean(dim=0, keepdim=True)) ** 2).sum(dim=-1).mean() + 1e-8).item())
    extension = end_to_end / max(rg, 1e-8)

    if helix_fraction >= 0.55:
        return "helix-rich"
    if extension >= 3.5:
        return "extended"
    if rg < 0.9 * (2.38 * (n ** 0.365)):
        return "collapsed"
    return "mixed"


def compute_mcts_structural_metadata(
    x0_flat: torch.Tensor,
    contact_probs: torch.Tensor,
    input_feature_dict: Dict[str, torch.Tensor],
    L_binder: int,
    binder_seq: Optional[str] = None,
    hotspot_params: Optional[Dict[str, Any]] = None,
    contact_prob_threshold: float = 0.20,
    patch_size: int = 10,
    clash_distance: float = 2.0,
) -> Dict[str, Any]:
    """
    Convert verifier structure outputs into MCTS grouping/routing metadata.

    The returned keys match the structural grouping and action-routing fields
    consumed by MCTSTree / mutation_proposer:
      - hotspot_contact_bitset, contact_fingerprint, interface_patch_id
      - secondary_structure_pattern
      - interface/target-facing/hotspot-facing/background/clash residue indices
    All residue indices are binder-local 0-based positions.
    """
    device = x0_flat.device
    asym_id = input_feature_dict["asym_id"]
    atom_to_token = input_feature_dict["atom_to_token_idx"]
    binder_chain_id = asym_id[0]
    binder_token_mask = asym_id == binder_chain_id
    target_token_mask = ~binder_token_mask
    binder_token_indices = binder_token_mask.nonzero(as_tuple=True)[0]
    target_token_indices = target_token_mask.nonzero(as_tuple=True)[0]

    metadata: Dict[str, Any] = {
        "hotspot_contact_bitset": "unknown",
        "interface_patch_id": "unknown",
        "secondary_structure_pattern": "unknown",
        "contact_fingerprint": "unknown",
        "interface_residue_indices": [],
        "target_facing_residue_indices": [],
        "hotspot_facing_residue_indices": [],
        "background_contact_residue_indices": [],
        "non_hotspot_contact_residue_indices": [],
        "clash_residue_indices": [],
        "clash_near_residue_indices": [],
        "surface_residue_indices": [],
        "non_interface_residue_indices": [],
        "core_residue_indices": [],
        "buried_residue_indices": [],
        "exposed_hydrophobic_residue_indices": [],
    }

    if binder_token_indices.numel() == 0 or target_token_indices.numel() == 0:
        return metadata

    inter_contact = contact_probs[binder_token_indices][:, target_token_indices]
    if inter_contact.numel() > 0:
        binder_contact_strength = inter_contact.max(dim=1).values
        interface_local = (binder_contact_strength >= contact_prob_threshold).nonzero(as_tuple=True)[0]
        metadata["interface_residue_indices"] = _to_python_list(interface_local)
        metadata["target_facing_residue_indices"] = metadata["interface_residue_indices"]

        target_contact_strength = inter_contact.max(dim=0).values
        target_contact_mask = target_contact_strength >= contact_prob_threshold
        contact_fingerprint = "".join(
            "1" if bool(v) else "0"
            for v in target_contact_mask.detach().cpu().tolist()
        )
        metadata["contact_fingerprint"] = contact_fingerprint

        if target_contact_strength.numel() > 0 and float(target_contact_strength.max().item()) > 0.0:
            dominant_target = int(target_contact_strength.argmax().item())
            patch_id = dominant_target // max(int(patch_size), 1)
            metadata["interface_patch_id"] = f"patch_{patch_id}"

    hotspot_params = hotspot_params or {}
    hotspot_concat_indices = hotspot_params.get("target_hotspot_concat_indices") or []
    valid_hotspot_indices = [
        int(index)
        for index in hotspot_concat_indices
        if 0 <= int(index) < target_token_indices.shape[0]
    ]
    if valid_hotspot_indices and inter_contact.numel() > 0:
        hotspot_cols = torch.as_tensor(valid_hotspot_indices, device=device, dtype=torch.long)
        hotspot_contact = inter_contact[:, hotspot_cols]
        hotspot_bitset_tensor = hotspot_contact.max(dim=0).values >= contact_prob_threshold
        metadata["hotspot_contact_bitset"] = "".join(
            "1" if bool(v) else "0"
            for v in hotspot_bitset_tensor.detach().cpu().tolist()
        )

        hotspot_local = (hotspot_contact.max(dim=1).values >= contact_prob_threshold).nonzero(as_tuple=True)[0]
        metadata["hotspot_facing_residue_indices"] = _to_python_list(hotspot_local)

        background_contact = inter_contact.clone()
        background_contact[:, hotspot_cols] = 0.0
        background_local = (background_contact.max(dim=1).values >= contact_prob_threshold).nonzero(as_tuple=True)[0]
        metadata["background_contact_residue_indices"] = _to_python_list(background_local)
        metadata["non_hotspot_contact_residue_indices"] = metadata["background_contact_residue_indices"]

    atom_binder_mask = binder_token_mask[atom_to_token]
    atom_target_mask = target_token_mask[atom_to_token]
    binder_to_local = _binder_token_to_local_index(atom_to_token, binder_token_mask)

    rep_atom_mask = input_feature_dict.get("distogram_rep_atom_mask", None)
    if rep_atom_mask is not None:
        rep_atom_mask = rep_atom_mask.bool()
        binder_rep_atom_mask = atom_binder_mask & rep_atom_mask
    else:
        binder_rep_atom_mask = atom_binder_mask
    binder_rep_coords = x0_flat[binder_rep_atom_mask]
    if binder_rep_coords.shape[0] >= min(max(L_binder, 1), 4):
        binder_rep_coords = binder_rep_coords[:L_binder]
        metadata["secondary_structure_pattern"] = _secondary_structure_pattern_from_coords(binder_rep_coords)

        centroid = binder_rep_coords.mean(dim=0, keepdim=True)
        radial_distance = torch.linalg.norm(binder_rep_coords - centroid, dim=-1)
        pair_dist = torch.cdist(binder_rep_coords.float(), binder_rep_coords.float())
        neighbor_count = ((pair_dist < 8.0) & (pair_dist > 1e-6)).sum(dim=1)

        surface_score = radial_distance
        core_score = neighbor_count.float() - 0.1 * radial_distance
        n_surface = max(1, int(round(0.35 * L_binder)))
        n_core = max(1, int(round(0.35 * L_binder)))
        surface_local = torch.topk(surface_score, k=min(n_surface, L_binder), largest=True).indices
        core_local = torch.topk(core_score, k=min(n_core, L_binder), largest=True).indices

        metadata["surface_residue_indices"] = sorted(_to_python_list(surface_local))
        metadata["core_residue_indices"] = sorted(_to_python_list(core_local))
        metadata["buried_residue_indices"] = metadata["core_residue_indices"]

        interface_set = set(metadata["interface_residue_indices"])
        metadata["non_interface_residue_indices"] = [
            idx for idx in range(L_binder) if idx not in interface_set
        ]

        if binder_seq:
            hydrophobic = set("AVLIMFWYC")
            surface_set = set(metadata["surface_residue_indices"])
            metadata["exposed_hydrophobic_residue_indices"] = [
                idx for idx, aa in enumerate(binder_seq[:L_binder])
                if idx in surface_set and aa in hydrophobic
            ]

    binder_atom_indices = atom_binder_mask.nonzero(as_tuple=True)[0]
    target_atom_indices = atom_target_mask.nonzero(as_tuple=True)[0]
    if binder_atom_indices.numel() > 0 and target_atom_indices.numel() > 0:
        binder_coords = x0_flat[binder_atom_indices]
        target_coords = x0_flat[target_atom_indices]
        dist = torch.cdist(binder_coords, target_coords)
        clashing_binder_atoms = (dist < clash_distance).any(dim=1)
        clash_atom_indices = binder_atom_indices[clashing_binder_atoms]
        clash_local = set()
        for atom_idx in clash_atom_indices.detach().cpu().tolist():
            token_idx = int(atom_to_token[int(atom_idx)].item())
            if token_idx in binder_to_local:
                clash_local.add(binder_to_local[token_idx])
        clash_list = sorted(clash_local)
        metadata["clash_residue_indices"] = clash_list
        metadata["clash_near_residue_indices"] = clash_list

    return metadata


def _run_full_forward(
    model,
    input_feature_dict: Dict,
    device: torch.device,
    N_cycle: Optional[int] = None,
    chunk_size: int = None,
) -> tuple:
    """
    手动分步执行 Protenix 完整推断流程。

    与 _main_inference_loop 等价，但保留对 z_trunk 的引用，
    以便后续直接计算 distogram_logits。

    Args:
        chunk_size: Attention 分块大小（None 表示不分块）。
                   当 complex 较大（如 114+ aa binder）时建议设为 256，
                   避免 OOM。

    Returns:
        coordinate:       [N_sample, N_atom, 3]
        s_inputs:         [N_token, c_s_inputs]
        s_trunk:          [N_token, c_s]
        z_trunk:          [N_token, N_token, c_z]
        plddt_logits:     [N_sample, N_atom, plddt_bins]
        pae_logits:       [N_sample, N_token, N_token, pae_bins]
        pde_logits:       [N_sample, N_token, N_token, pde_bins]
        distogram_logits: [N_token, N_token, no_bins]
        contact_probs:    [N_token, N_token]
    """
    if N_cycle is None:
        N_cycle = _resolve_model_n_cycle(model)

    # Step 1: Pairformer
    s_inputs, s_trunk, z_trunk = model.get_pairformer_output(
        input_feature_dict=input_feature_dict,
        N_cycle=N_cycle,
        inplace_safe=True,
        chunk_size=chunk_size,
    )

    # Step 2: 清理不再需要的特征（与 _main_inference_loop 一致）
    keys_to_delete = [
        k for k in input_feature_dict.keys()
        if "template_" in k or k in [
            "msa", "has_deletion", "deletion_value", "profile", "deletion_mean",
        ]
    ]
    for k in keys_to_delete:
        del input_feature_dict[k]

    # Step 3: Diffusion 采样（完整多步 ODE）
    N_sample = model.configs.sample_diffusion["N_sample"]
    N_step = model.configs.sample_diffusion["N_step"]

    noise_schedule = model.inference_noise_scheduler(
        N_step=N_step, device=device, dtype=s_inputs.dtype
    )

    cache = {}
    if model.enable_diffusion_shared_vars_cache:
        cache["pair_z"] = model.diffusion_module.diffusion_conditioning.prepare_cache(
            input_feature_dict["relp"], z_trunk, False
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
            r_l=True,
            z=cache["pair_z"],
            inplace_safe=False,
        )
        cache["p_lm"] = p_lm
        cache["c_l"] = c_l
    else:
        cache["pair_z"] = None
        cache["p_lm"] = None
        cache["c_l"] = None

    coordinate = model.sample_diffusion(
        denoise_net=model.diffusion_module,
        input_feature_dict=input_feature_dict,
        s_inputs=s_inputs,
        s_trunk=s_trunk,
        z_trunk=None if cache["pair_z"] is not None else z_trunk,
        pair_z=cache["pair_z"],
        p_lm=cache["p_lm"],
        c_l=cache["c_l"],
        N_sample=N_sample,
        noise_schedule=noise_schedule,
        inplace_safe=True,
        enable_efficient_fusion=model.enable_efficient_fusion,
    )  # [N_sample, N_atom, 3]

    # Step 4: Distogram（直接从 z_trunk 计算，保留完整 logits）
    distogram_logits = model.distogram_head(z_trunk)  # [N_token, N_token, no_bins]

    contact_probs = sample_confidence.compute_contact_prob(
        distogram_logits=distogram_logits,
        **sample_confidence.get_bin_params(model.configs.loss.distogram),
    )  # [N_token, N_token]

    # Step 5: Confidence heads
    plddt_logits, pae_logits, pde_logits, _ = model.run_confidence_head(
        input_feature_dict=input_feature_dict,
        s_inputs=s_inputs,
        s_trunk=s_trunk,
        z_trunk=z_trunk,
        pair_mask=None,
        x_pred_coords=coordinate,
        triangle_multiplicative=model.configs.triangle_multiplicative,
        triangle_attention=model.configs.triangle_attention,
        inplace_safe=True,
        chunk_size=chunk_size,
    )

    return (
        coordinate, s_inputs, s_trunk, z_trunk,
        plddt_logits, pae_logits, pde_logits,
        distogram_logits, contact_probs,
    )


def zeroth_order_eval(
    binder_seq: str,
    target_seq: str,
    model,
    device: torch.device,
    loss_weights: Dict[str, float] = None,
    contact_params: Dict[str, float] = None,
    N_cycle: Optional[int] = None,
    chunk_size: int = None,
    hotspot_params: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    target_template_path: Optional[str] = None,
    opt_token_mask: Optional[Any] = None,
    target_chain_roles: Optional[Any] = None,
    binder_template_mask: Optional[Any] = None,
) -> ZerothOrderResult:
    """
    对一条离散候选序列执行完整 AF3 推断，并在最终结构上计算全部 8 项 loss。

    Args:
        binder_seq:     Binder 序列（单字母 str，如 "ACDEF..."）
        target_seq:     Target 序列（单字母 str）
        model:          Protenix 模型（权重已冻结）
        device:         计算设备
        loss_weights:   各项 loss 权重（默认使用 DEFAULT_LOSS_WEIGHTS）
        contact_params: 接触 loss 参数（默认使用 DEFAULT_CONTACT_PARAMS）
        N_cycle:        Pairformer 循环次数；默认 None 时使用模型配置
                        model.configs.model.N_cycle
        chunk_size:     Attention 分块大小。当 complex 较大（如 114+ aa）
                       时建议设为 256，避免 OOM。默认为 None（不分块）。
        seed:           可选 verifier 随机种子；同一序列/输入可复现 AF3 sampling。
        target_template_path: 可选 target template 结构文件
        opt_token_mask: 可选优化区域 mask；CDR redesign 时为 binder 长度 bool mask
        target_chain_roles: 可选非 binder 链角色，例如 ["target", "scaffold"]
        binder_template_mask: 可选 binder template token mask；CDR redesign 时
                              用于保留 framework template、屏蔽设计区域

    Returns:
        ZerothOrderResult: 包含 total_loss / loss_dict / summary_confidence / coordinate
    """
    weights = dict(DEFAULT_LOSS_WEIGHTS)
    if loss_weights is not None:
        weights.update(loss_weights)
    c_params = dict(DEFAULT_CONTACT_PARAMS)
    if contact_params is not None:
        c_params.update(contact_params)
    if N_cycle is None:
        N_cycle = _resolve_model_n_cycle(model)

    L_binder = len(binder_seq)

    plddt_params = sample_confidence.get_bin_params(model.configs.loss.plddt)
    pae_params = sample_confidence.get_bin_params(model.configs.loss.pae)
    distogram_params = sample_confidence.get_bin_params(model.configs.loss.distogram)
    pde_params = sample_confidence.get_bin_params(model.configs.loss.pde)

    with temporary_seed(seed, device=device), torch.no_grad():
        # ── 构建特征 ─────────────────────────────────────────────────────
        input_feature_dict, N_token, N_atom = build_complex_features(
            binder_seq,
            target_seq,
            device,
            target_template_path=target_template_path,
            binder_template_mask=binder_template_mask,
        )
        input_feature_dict = prepare_input_features(model, input_feature_dict)

        # ── 完整推断 ─────────────────────────────────────────────────────
        (
            coordinate, s_inputs, s_trunk, z_trunk,
            plddt_logits, pae_logits, pde_logits,
            distogram_logits, contact_probs,
        ) = _run_full_forward(model, input_feature_dict, device, N_cycle, chunk_size=chunk_size)

        # ── Summary Confidence（Protenix 原生指标）────────────────────────
        summary_confidence_list, _ = sample_confidence.compute_full_data_and_summary(
            configs=model.configs,
            pae_logits=pae_logits,
            plddt_logits=plddt_logits,
            pde_logits=pde_logits,
            contact_probs=contact_probs,
            token_asym_id=input_feature_dict["asym_id"],
            token_has_frame=input_feature_dict["has_frame"],
            atom_coordinate=coordinate,
            atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
            atom_is_polymer=1 - input_feature_dict["is_ligand"],
            N_recycle=N_cycle,
            return_full_data=False,
        )

        # 基于 Protenix ranking_score 在 N_sample 中选择最佳结构样本
        # ranking_score 缺失时回退 0.0；summary 列表为空时回退 index 0（等价历史行为）
        scores = []
        for sc in summary_confidence_list:
            score = sc.get("ranking_score", 0.0)
            if isinstance(score, torch.Tensor):
                score = float(score.flatten()[0].item())
            else:
                try:
                    score = float(score)
                except (TypeError, ValueError):
                    score = 0.0
            scores.append(score)

        if scores:
            best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
            best_ranking = scores[best_idx]
            summary_confidence = summary_confidence_list[best_idx]
        else:
            best_idx = 0
            best_ranking = 0.0
            summary_confidence = {}

        logger.info(
            "[zeroth] best_sample_idx=%d ranking_score=%.4f",
            best_idx,
            best_ranking,
        )

        # ── 计算 12 项 loss ──────────────────────────────────────────────
        # coordinate 形状: [N_sample, N_atom, 3]
        # compute_all_losses 期望 x0_flat: [N_atom, 3]
        # plddt_logits: [N_sample, N_atom, bins]
        # pae_logits: [N_sample, N_token, N_token, bins]
        # 避免 CUDA 上高级索引路径潜在卡死：改用 narrow + squeeze
        coordinate_sel = coordinate.narrow(0, best_idx, 1).contiguous()
        x0_flat = coordinate_sel.squeeze(0)  # [N_atom, 3]
        plddt_for_loss = plddt_logits[best_idx : best_idx + 1]  # [1, N_atom, bins]
        pae_for_loss = pae_logits[best_idx : best_idx + 1]      # [1, N_token, N_token, bins]
        pde_for_loss = pde_logits[best_idx : best_idx + 1]      # [1, N_token, N_token, bins]

        total_loss, losses = compute_all_losses(
            x0_flat=x0_flat,
            plddt_logits=plddt_for_loss,
            pae_logits=pae_for_loss,
            distogram_logits=distogram_logits,
            input_feature_dict=input_feature_dict,
            L_binder=L_binder,
            weights=weights,
            contact_params=c_params,
            plddt_params=plddt_params,
            pae_params=pae_params,
            distogram_params=distogram_params,
            pde_logits=pde_for_loss,
            pde_params=pde_params,
            clash_params=dict(DEFAULT_CLASH_PARAMS),
            hotspot_params=hotspot_params,
            opt_token_mask=opt_token_mask,
            target_chain_roles=target_chain_roles,
        )

        structural_metadata = compute_mcts_structural_metadata(
            x0_flat=x0_flat,
            contact_probs=contact_probs,
            input_feature_dict=input_feature_dict,
            L_binder=L_binder,
            binder_seq=binder_seq,
            hotspot_params=hotspot_params,
            contact_prob_threshold=float(
                (hotspot_params or {}).get("mcts_contact_prob_threshold", 0.20)
            ),
            patch_size=int((hotspot_params or {}).get("mcts_interface_patch_size", 10)),
            clash_distance=float(dict(DEFAULT_CLASH_PARAMS).get("r_clash", 2.0)),
        )
        logger.info(
            "[zeroth] mcts_struct ss=%s patch=%s interface=%d core=%d surface=%d clash=%d hotspot=%s",
            structural_metadata.get("secondary_structure_pattern", "unknown"),
            structural_metadata.get("interface_patch_id", "unknown"),
            len(structural_metadata.get("interface_residue_indices", [])),
            len(structural_metadata.get("core_residue_indices", [])),
            len(structural_metadata.get("surface_residue_indices", [])),
            len(structural_metadata.get("clash_residue_indices", [])),
            structural_metadata.get("hotspot_contact_bitset", "unknown"),
        )

    # ── 组装结果 ─────────────────────────────────────────────────────────
    loss_dict = {
        k: v.item() if isinstance(v, torch.Tensor) else v
        for k, v in losses.items()
    }

    def _scalar(t):
        if isinstance(t, torch.Tensor):
            return float(t.flatten()[0].item())
        return float(t)

    sc_flat = {}
    for k, v in summary_confidence.items():
        try:
            sc_flat[k] = _scalar(v)
        except (TypeError, ValueError):
            sc_flat[k] = v
    sc_flat.update(structural_metadata)

    coordinate_cpu = coordinate_sel.cpu()
    plddt_logits_cpu = plddt_logits[best_idx : best_idx + 1].cpu()  # [1, N_atom, bins]

    return ZerothOrderResult(
        binder_seq=binder_seq,
        target_seq=target_seq,
        total_loss=total_loss.item(),
        loss_dict=loss_dict,
        summary_confidence=sc_flat,
        coordinate=coordinate_cpu,
        plddt_logits=plddt_logits_cpu,
    )
