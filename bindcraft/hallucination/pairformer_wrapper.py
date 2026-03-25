"""
BindCraft v2 - 无 Gradient Checkpoint 的 Pairformer 前向（阶段 3）

问题背景：
  Protenix 的 PairformerStack.forward() 和 MSAModule.forward() 均调用 checkpoint_blocks()。
  - PairformerStack：blocks_per_ckpt 默认 None，但若配置文件设置了非 None 值则触发 checkpointing。
  - MSAModule：blocks_per_ckpt 默认 1，只要 grad 启用就必然触发 checkpointing。
  Gradient checkpointing 要求前向中不能有 inplace 操作，否则反向传播时
  重新计算会读到被修改后的张量，导致 RuntimeError。

解决方案：
  直接复用 Protenix 模型中已加载的权重（不重新初始化），
  但手动逐块调用 PairformerBlock 和 MSABlock，完全跳过 checkpoint_blocks。
  这样梯度可以安全地穿过 48 层 Pairformer，同时不触发任何 inplace 问题。

算法指导对应：
  s, z = Pairformer(s̃_init, z_init | Φ_pair)
  梯度链中的 ∂s/∂s̃_init 层。
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


def _msa_module_forward_no_ckpt(
    msa_module,
    input_feature_dict: Dict,
    z: torch.Tensor,
    s_inputs: torch.Tensor,
) -> torch.Tensor:
    """
    手动执行 MSAModule 前向，绕过内部的 checkpoint_blocks。

    MSAModule.forward() 中 blocks_per_ckpt 默认为 1，只要 grad 启用就触发 checkpointing。
    这里复制其前向逻辑，但将 checkpoint_blocks 替换为直接逐块调用。

    Args:
        msa_module: model.msa_module
        input_feature_dict: 输入特征字典
        z: 对特征，[N_token, N_token, c_z]
        s_inputs: 单体特征（来自 input_embedder），[N_token, c_s_inputs]

    Returns:
        z: 更新后的对特征，[N_token, N_token, c_z]
    """
    if msa_module.n_blocks < 1:
        return z
    if "msa" not in input_feature_dict:
        return z
    if input_feature_dict["msa"].dim() < 2:
        return z

    from protenix.model.utils import sample_msa_feature_dict_random_without_replacement

    msa_feat = sample_msa_feature_dict_random_without_replacement(
        feat_dict=input_feature_dict,
        dim_dict={feat_name: -2 for feat_name in msa_module.input_feature},
        cutoff=(
            msa_module.msa_configs["train_cutoff"]
            if msa_module.training
            else msa_module.msa_configs["test_cutoff"]
        ),
        lower_bound=(
            msa_module.msa_configs["train_lowerb"]
            if msa_module.training
            else msa_module.msa_configs["test_lowerb"]
        ),
        strategy=msa_module.msa_configs["strategy"],
    )

    msa_feat["msa"] = F.one_hot(
        msa_feat["msa"],
        num_classes=msa_module.input_feature["msa"],
    ).float()

    target_shape = msa_feat["msa"].shape[:-1]
    msa_sample = torch.cat(
        [
            msa_feat[name].reshape(*target_shape, d)
            for name, d in msa_module.input_feature.items()
        ],
        dim=-1,
    )

    msa_sample = msa_module.linear_no_bias_m(msa_sample)
    msa_sample = msa_sample + msa_module.linear_no_bias_s(s_inputs)

    # 手动逐块调用，绕过 checkpoint_blocks
    for block in msa_module.blocks:
        msa_sample, z = block(
            m=msa_sample,
            z=z,
            pair_mask=None,
            triangle_multiplicative="torch",
            triangle_attention="torch",
            inplace_safe=False,
            chunk_size=None,
        )

    return z


def pairformer_forward_no_ckpt(
    model,
    input_feature_dict: Dict,
    s_init_override: Optional[torch.Tensor] = None,
    N_cycle: int = 4,
) -> tuple:
    """
    无 Gradient Checkpoint 的完整 Pairformer 前向。

    直接复用 model 中已加载的权重（linear_no_bias_sinit、pairformer_stack 等），
    但手动逐块调用，绕过 PairformerStack 和 MSAModule 中的 checkpoint_blocks，
    保证梯度正常反传。

    Args:
        model: 已加载权重的 Protenix 模型（权重已冻结）
        input_feature_dict: 输入特征字典（含 relp、token_bonds 等）
        s_init_override: 若不为 None，用此张量替换 model.input_embedder 的输出
                         作为 s_inputs（即 STE 桥接后的 s̃_init 拼接 s_target）。
                         shape: [N_token, c_s]
        N_cycle: 循环次数，默认 4

    Returns:
        s_inputs: 原始 input_embedder 输出（或 s_init_override），[N_token, c_s]
        s: Pairformer 输出的单体特征，[N_token, c_s]
        z: Pairformer 输出的对特征，[N_token, N_token, c_z]
    """
    # ── 1. 获取 s_inputs ──────────────────────────────────────────────────────
    if s_init_override is not None:
        s_inputs = s_init_override  # 已含梯度的 STE 特征
    else:
        s_inputs = model.input_embedder(input_feature_dict, inplace_safe=False)

    # ── 2. 初始化 s_init / z_init（与 get_pairformer_output 完全一致）─────────
    s_init = model.linear_no_bias_sinit(s_inputs)  # [N_token, c_s]
    z_init = (
        model.linear_no_bias_zinit1(s_init)[..., None, :]
        + model.linear_no_bias_zinit2(s_init)[..., None, :, :]
    )  # [N_token, N_token, c_z]
    z_init = z_init + model.relative_position_encoding(input_feature_dict["relp"])
    z_init = z_init + model.linear_no_bias_token_bond(
        input_feature_dict["token_bonds"].unsqueeze(dim=-1)
    )

    # ── 3. 初始化循环变量 ─────────────────────────────────────────────────────
    z = torch.zeros_like(z_init)
    s = torch.zeros_like(s_init)

    # ── 4. N_cycle 循环（仅最后一个 cycle 保留梯度）──────────────────────────
    for cycle_no in range(N_cycle):
        keep_grad = (cycle_no == N_cycle - 1)

        with torch.set_grad_enabled(keep_grad):
            # 循环更新 z（recycling）
            z = z_init + model.linear_no_bias_z_cycle(model.layernorm_z_cycle(z))

            # Template Embedder（若有 template block）
            if model.template_embedder.n_blocks > 0:
                z = z + model.template_embedder(
                    input_feature_dict,
                    z,
                    triangle_multiplicative="torch",
                    triangle_attention="torch",
                    inplace_safe=False,
                    chunk_size=None,
                )

            # MSA Module：手动逐块调用，绕过 checkpoint_blocks（blocks_per_ckpt 默认 1）
            z = _msa_module_forward_no_ckpt(
                msa_module=model.msa_module,
                input_feature_dict=input_feature_dict,
                z=z,
                s_inputs=s_inputs,
            )

            # 单体特征更新（recycling）
            s = s_init + model.linear_no_bias_s(model.layernorm_s(s))

            # ── 5. PairformerStack：手动逐块调用，绕过 checkpoint_blocks ──────
            for block in model.pairformer_stack.blocks:
                s, z = block(
                    s=s,
                    z=z,
                    pair_mask=None,
                    triangle_multiplicative="torch",
                    triangle_attention="torch",
                    inplace_safe=False,  # 关键：禁止 inplace
                    chunk_size=None,
                )

    return s_inputs, s, z
