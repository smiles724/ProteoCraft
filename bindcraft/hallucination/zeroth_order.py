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
from typing import Dict, Optional
import logging

from protenix.model import sample_confidence

from bindcraft.utils import (
    build_complex_features,
    prepare_input_features,
    DEFAULT_LOSS_WEIGHTS,
    DEFAULT_CONTACT_PARAMS,
)
from bindcraft.losses.total_loss import compute_all_losses

logger = logging.getLogger(__name__)


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


def _run_full_forward(
    model,
    input_feature_dict: Dict,
    device: torch.device,
    N_cycle: int = 4,
) -> tuple:
    """
    手动分步执行 Protenix 完整推断流程。

    与 _main_inference_loop 等价，但保留对 z_trunk 的引用，
    以便后续直接计算 distogram_logits。

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
    # Step 1: Pairformer
    s_inputs, s_trunk, z_trunk = model.get_pairformer_output(
        input_feature_dict=input_feature_dict,
        N_cycle=N_cycle,
        inplace_safe=True,
        chunk_size=None,
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
        chunk_size=None,
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
    N_cycle: int = 4,
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
        N_cycle:        Pairformer 循环次数

    Returns:
        ZerothOrderResult: 包含 total_loss / loss_dict / summary_confidence / coordinate
    """
    weights = dict(DEFAULT_LOSS_WEIGHTS)
    if loss_weights is not None:
        weights.update(loss_weights)
    c_params = dict(DEFAULT_CONTACT_PARAMS)
    if contact_params is not None:
        c_params.update(contact_params)

    L_binder = len(binder_seq)

    plddt_params = sample_confidence.get_bin_params(model.configs.loss.plddt)
    pae_params = sample_confidence.get_bin_params(model.configs.loss.pae)
    distogram_params = sample_confidence.get_bin_params(model.configs.loss.distogram)

    with torch.no_grad():
        # ── 构建特征 ─────────────────────────────────────────────────────
        input_feature_dict, N_token, N_atom = build_complex_features(
            binder_seq, target_seq, device
        )
        input_feature_dict = prepare_input_features(model, input_feature_dict)

        # ── 完整推断 ─────────────────────────────────────────────────────
        (
            coordinate, s_inputs, s_trunk, z_trunk,
            plddt_logits, pae_logits, pde_logits,
            distogram_logits, contact_probs,
        ) = _run_full_forward(model, input_feature_dict, device, N_cycle)

        # ── 计算 8 项 loss ───────────────────────────────────────────────
        # coordinate 形状: [N_sample, N_atom, 3]
        # compute_all_losses 期望 x0_flat: [N_atom, 3]
        # plddt_logits: [N_sample, N_atom, bins]
        # pae_logits: [N_sample, N_token, N_token, bins]
        # 取第一个 sample 用于 loss 计算（与 run_full_eval 一致）
        x0_flat = coordinate[0]  # [N_atom, 3]
        plddt_for_loss = plddt_logits[0:1]  # [1, N_atom, bins]，保留 N_sample 维
        pae_for_loss = pae_logits[0:1]      # [1, N_token, N_token, bins]

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
        )

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
        summary_confidence = summary_confidence_list[0] if summary_confidence_list else {}

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

    return ZerothOrderResult(
        binder_seq=binder_seq,
        target_seq=target_seq,
        total_loss=total_loss.item(),
        loss_dict=loss_dict,
        summary_confidence=sc_flat,
        coordinate=coordinate.cpu(),
        plddt_logits=plddt_logits[0:1].cpu(),  # [1, N_atom, bins]
    )
