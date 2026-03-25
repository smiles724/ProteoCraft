"""
阶段 3 调试脚本：用 anomaly detection 精确定位 inplace 错误位置
"""
import sys
import os
sys.path.insert(0, "/hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft")

import torch
torch.autograd.set_detect_anomaly(True)

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from bindcraft.utils import build_complex_features, prepare_input_features, freeze_model
from bindcraft.hallucination.ste_bridge import precompute_waa_embeddings, ste_bridge
from bindcraft.hallucination.pairformer_wrapper import pairformer_forward_no_ckpt
from bindcraft.hallucination.diffusion_one_step import get_coarse_structure, one_step_denoise
from bindcraft.losses.geometry import compute_geometry_loss

def main():
    device = torch.device("cuda")
    
    # 加载模型
    logger.info("加载模型...")
    from scripts.run_hallucination import load_model
    model = load_model(device)
    freeze_model(model)
    
    target_seq = "LKDFDREVEIRIKQIESDRQNLLKEVDNLYNIEILRLPKALREMNWLDYFALGGN"
    L_binder = 40
    
    # 预计算 W_AAE
    W_AAE = precompute_waa_embeddings(model, device)
    
    # 初始化 theta
    theta = torch.randn(L_binder, 20, device=device) * 0.01
    theta.requires_grad_(True)
    
    optimizer = torch.optim.Adam([theta], lr=0.05)
    optimizer.zero_grad()
    
    # 获取 target 特征
    import numpy as np
    init_binder = "A" * L_binder
    with torch.no_grad():
        feat_init, _, _ = build_complex_features(init_binder, target_seq, device)
        feat_init = prepare_input_features(model, feat_init)
        s_inputs_full = model.input_embedder(feat_init)
        s_target = s_inputs_full[L_binder:, :].clone()
    
    # 单步前向
    logger.info("执行 STE 桥接...")
    s_binder, current_seq, input_feature_dict = ste_bridge(
        theta=theta, W_AAE=W_AAE, model=model,
        target_seq=target_seq, device=device, tau_temp=1.0
    )
    
    logger.info("执行 Pairformer 前向（无 checkpoint）...")
    s_inputs_ste = torch.cat([s_binder, s_target], dim=0)
    _, s_trunk, z_trunk = pairformer_forward_no_ckpt(
        model=model,
        input_feature_dict=input_feature_dict,
        s_init_override=s_inputs_ste,
        N_cycle=4,
    )
    
    logger.info("执行单步外推...")
    x_coarse = get_coarse_structure(model, input_feature_dict, s_inputs_ste, device)
    x0_hat = one_step_denoise(
        model=model, x_coarse=x_coarse, s_inputs=s_inputs_ste,
        s_trunk=s_trunk, z_trunk=z_trunk,
        input_feature_dict=input_feature_dict, tau=4.0, device=device,
    )
    
    logger.info("计算 Loss...")
    x0_flat = x0_hat.squeeze(0)
    atom_to_token = input_feature_dict["atom_to_token_idx"]
    binder_atom_mask = atom_to_token < L_binder
    target_atom_mask = atom_to_token >= L_binder
    losses = compute_geometry_loss(
        x0_hat=x0_flat,
        binder_atom_mask=binder_atom_mask,
        target_atom_mask=target_atom_mask,
    )
    total_loss = losses["total"]
    
    logger.info(f"Loss: {total_loss.item():.4f}")
    logger.info("执行 backward()...")
    total_loss.backward()
    
    logger.info("✅ backward() 成功！grad norm: {:.6f}".format(
        theta.grad.norm().item() if theta.grad is not None else 0.0
    ))

if __name__ == "__main__":
    main()
