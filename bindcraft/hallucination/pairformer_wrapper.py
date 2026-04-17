"""
BindCraft v2 - Pairformer 前向包装（阶段 3）

保留历史接口名 ``pairformer_forward_no_ckpt``，但内部已恢复使用
Protenix 自带的 ``checkpoint_blocks()`` 以降低一阶优化的显存占用。

关键约束：
  - 继续保持 inplace_safe=False，避免 checkpoint 重算阶段遇到 inplace 冲突。
  - 暂时保留现有 "torch" triangle kernel，不在此文件中切换 kernel 配置。
"""

import torch
import torch.nn.functional as F
from functools import partial
from typing import Dict, Optional, Any
import logging

from protenix.model.utils import (
    checkpoint_blocks,
    sample_msa_feature_dict_random_without_replacement,
)

logger = logging.getLogger(__name__)


def _run_checkpointed_blocks(
    blocks,
    args,
    blocks_per_ckpt: Optional[int],
):
    """在 grad 开启时使用 activation checkpoint，其余情况保持直跑。"""
    return checkpoint_blocks(
        blocks=blocks,
        args=args,
        blocks_per_ckpt=blocks_per_ckpt,
    )


def _msa_module_forward_ckpt_safe(
    msa_module,
    input_feature_dict: Dict,
    z: torch.Tensor,
    s_inputs: torch.Tensor,
    chunk_size: int = None,
) -> torch.Tensor:
    """
    手动执行 MSAModule 前向，同时保留 checkpoint_blocks。

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

    blocks = [
        partial(
            block,
            pair_mask=None,
            triangle_multiplicative="torch",
            triangle_attention="torch",
            inplace_safe=False,
            chunk_size=chunk_size,
        )
        for block in msa_module.blocks
    ]

    msa_sample, z = _run_checkpointed_blocks(
        blocks=blocks,
        args=(msa_sample, z),
        blocks_per_ckpt=msa_module.blocks_per_ckpt,
    )

    return z


def pairformer_forward_no_ckpt(
    model,
    input_feature_dict: Dict,
    s_init_override: Optional[torch.Tensor] = None,
    N_cycle: int = 4,
    chunk_size: int = None,
) -> tuple:
    """
    Pairformer 完整前向。

    保留历史函数名以兼容现有调用，但内部已恢复使用 checkpoint_blocks，
    同时继续保持 inplace_safe=False，保证梯度正常反传。

    Args:
        model: 已加载权重的 Protenix 模型（权重已冻结）
        input_feature_dict: 输入特征字典（含 relp、token_bonds 等）
        s_init_override: 若不为 None，用此张量替换 model.input_embedder 的输出
                         作为 s_inputs（即 STE 桥接后的 s̃_init 拼接 s_target）。
                         shape: [N_token, c_s]
        N_cycle: 循环次数，默认 4
        chunk_size: Attention 分块大小。114+ aa 建议 256，避免 OOM。

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
                    chunk_size=chunk_size,
                )

            # 对 use_msa=False 的模型，直接跳过 MSA 路径。
            if getattr(model.configs, "use_msa", True):
                z = _msa_module_forward_ckpt_safe(
                    msa_module=model.msa_module,
                    input_feature_dict=input_feature_dict,
                    z=z,
                    s_inputs=s_inputs,
                    chunk_size=chunk_size,
                )

            # 单体特征更新（recycling）
            s = s_init + model.linear_no_bias_s(model.layernorm_s(s))

            # PairformerStack：恢复 checkpoint_blocks，但继续禁止 inplace。
            blocks = [
                partial(
                    block,
                    pair_mask=None,
                    triangle_multiplicative="torch",
                    triangle_attention="torch",
                    inplace_safe=False,
                    chunk_size=chunk_size,
                )
                for block in model.pairformer_stack.blocks
            ]
            s, z = _run_checkpointed_blocks(
                blocks=blocks,
                args=(s, z),
                blocks_per_ckpt=model.pairformer_stack.blocks_per_ckpt,
            )

    return s_inputs, s, z
