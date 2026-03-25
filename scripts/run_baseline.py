#!/usr/bin/env python3
"""
BindCraft v2 - Baseline 评估脚本

计算 Poly-Ala 和 Random 序列的 loss，用于与 BindCraft v2 优化结果对比。

Baseline 逻辑：
  - 固定序列（不优化），重复 N_samples 次前向传播（每次 x_τ 随机噪声不同）
  - 取各项 loss 的均值 ± 标准差
  - 与 BindCraft v2 优化后的 best_loss 对比

使用方法：
    python scripts/run_baseline.py --target LKDFD... --binder_length 40
"""

import os
import sys
import argparse
import logging
import json
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)


# ============================================================
# 单次前向传播计算 loss（无梯度）
# ============================================================

def compute_loss_for_seq(
    model,
    binder_seq: str,
    target_seq: str,
    tau_noise: float,
    device: torch.device,
    loss_weights: dict,
) -> dict:
    """
    对给定的固定序列做一次前向传播，计算所有 loss 项。
    不做梯度优化，只用于 baseline 评估。

    Returns:
        dict: 各项 loss 的标量值
    """
    from bindcraft.utils import (
        build_complex_features,
        build_single_chain_features,
        prepare_input_features,
    )
    from bindcraft.hallucination.pairformer_wrapper import pairformer_forward_no_ckpt
    from bindcraft.hallucination.diffusion_one_step import one_step_denoise
    from bindcraft.losses.geometry import compute_geometry_loss
    from bindcraft.losses.confidence import compute_iptm_loss

    L_binder = len(binder_seq)

    with torch.no_grad():
        # 构建 binder+target 联合特征
        input_feature_dict, N_token, N_atom = build_complex_features(
            binder_seq, target_seq, device
        )
        input_feature_dict = prepare_input_features(model, input_feature_dict)

        # 获取 s_inputs（binder+target 联合）
        s_inputs_full = model.input_embedder(input_feature_dict, inplace_safe=False)

        # 获取 target-only 特征（s_target）
        feat_target_only, _, _ = build_single_chain_features(target_seq, device)
        feat_target_only = prepare_input_features(model, feat_target_only)
        s_target = model.input_embedder(feat_target_only, inplace_safe=False).clone()

        # 拼接 s_inputs_ste（binder 部分来自联合特征，target 来自单链）
        # 注意：baseline 不需要 STE，直接用 s_inputs_full 的 binder 部分
        s_binder_hard = s_inputs_full[:L_binder, :].clone()
        s_inputs_ste = torch.cat([s_binder_hard, s_target], dim=0)  # [N_token, c_s_inputs]

        # Pairformer 前向（无 checkpoint）
        _, s_trunk, z_trunk = pairformer_forward_no_ckpt(
            model=model,
            input_feature_dict=input_feature_dict,
            s_init_override=s_inputs_ste,
            N_cycle=4,
        )

        # 初始化 x_coarse
        x_coarse = input_feature_dict["ref_pos"].clone().unsqueeze(0)
        x_coarse = x_coarse + torch.randn_like(x_coarse) * 0.1

        # 单步外推
        x0_hat = one_step_denoise(
            model=model,
            x_coarse=x_coarse,
            s_inputs=s_inputs_ste,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            input_feature_dict=input_feature_dict,
            tau=tau_noise,
            device=device,
        )  # [1, N_atoms, 3]

        # 计算几何 loss
        x0_flat = x0_hat.squeeze(0)  # [N_atoms, 3]
        atom_to_token = input_feature_dict["atom_to_token_idx"]
        binder_atom_mask = atom_to_token < L_binder
        target_atom_mask = atom_to_token >= L_binder

        geo_losses = compute_geometry_loss(
            x0_hat=x0_flat,
            binder_atom_mask=binder_atom_mask,
            target_atom_mask=target_atom_mask,
            weights={k: v for k, v in loss_weights.items() if k != "iptm"},
        )

        total_loss = geo_losses["total"].clone()

        # iPTM loss
        if loss_weights.get("iptm", 0.0) > 0:
            loss_iptm = compute_iptm_loss(
                model=model,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs_ste,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                x0_hat=x0_hat,
                inplace_safe=False,
            )
            total_loss = total_loss + loss_weights["iptm"] * loss_iptm
            geo_losses["iptm"] = loss_iptm

        geo_losses["total"] = total_loss

    return {k: v.item() for k, v in geo_losses.items()}


# ============================================================
# 多次采样取均值
# ============================================================

def evaluate_baseline(
    model,
    binder_seq: str,
    target_seq: str,
    tau_noise: float,
    device: torch.device,
    loss_weights: dict,
    n_samples: int = 10,
    label: str = "baseline",
) -> dict:
    """
    重复 n_samples 次前向传播，统计 loss 均值和标准差。
    每次 x_τ 的随机噪声不同，反映结构预测的随机性。
    """
    all_losses = []
    logger.info(f"[{label}] 序列: {binder_seq}")
    logger.info(f"[{label}] 开始 {n_samples} 次采样...")

    for i in range(n_samples):
        losses = compute_loss_for_seq(
            model=model,
            binder_seq=binder_seq,
            target_seq=target_seq,
            tau_noise=tau_noise,
            device=device,
            loss_weights=loss_weights,
        )
        all_losses.append(losses)
        if (i + 1) % 5 == 0:
            logger.info(f"[{label}] 采样 {i+1}/{n_samples}: total={losses['total']:.4f}")

    # 统计
    keys = all_losses[0].keys()
    stats = {}
    for k in keys:
        vals = [d[k] for d in all_losses]
        stats[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    logger.info(f"[{label}] 结果：total = {stats['total']['mean']:.4f} ± {stats['total']['std']:.4f}")
    return stats


# ============================================================
# 主函数
# ============================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="BindCraft v2 - Baseline 评估")
    parser.add_argument("--target", type=str, required=True, help="Target 序列（单字母）")
    parser.add_argument("--binder_length", type=int, default=40, help="Binder 长度")
    parser.add_argument("--tau_noise", type=float, default=4.0, help="噪声时间步")
    parser.add_argument("--n_samples", type=int, default=10, help="每个 baseline 的采样次数")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    parser.add_argument("--model_name", type=str, default="protenix_base_default_v1.0.0")
    parser.add_argument("--output_dir", type=str, default="outputs/baseline")
    parser.add_argument(
        "--bindcraft_best_loss", type=float, default=None,
        help="BindCraft v2 优化后的 best_loss（用于对比）"
    )
    parser.add_argument(
        "--bindcraft_best_seq", type=str, default=None,
        help="BindCraft v2 优化后的最佳序列（用于对比）"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    sys.path.insert(0, PROJECT_ROOT)
    from scripts.run_hallucination import load_protenix_model
    model, _ = load_protenix_model(args.model_name, args.device)
    device = torch.device(args.device)

    loss_weights = {
        "attraction": 5.0,
        "clash": 2.0,
        "intra_clash": 1.0,
        "compactness": 0.5,
        "rg": 0.1,
        "iptm": 0.1,
    }

    L = args.binder_length

    # ── Baseline 1：Poly-Ala ─────────────────────────────────
    poly_ala_seq = "A" * L
    logger.info("=" * 60)
    logger.info("Baseline 1: Poly-Alanine")
    logger.info("=" * 60)
    stats_poly_ala = evaluate_baseline(
        model=model,
        binder_seq=poly_ala_seq,
        target_seq=args.target,
        tau_noise=args.tau_noise,
        device=device,
        loss_weights=loss_weights,
        n_samples=args.n_samples,
        label="Poly-Ala",
    )

    # ── Baseline 2：Random 序列（固定种子，可复现）────────────
    np.random.seed(0)
    aa_chars = list("ACDEFGHIKLMNPQRSTVWY")
    random_seq = "".join(np.random.choice(aa_chars, size=L))
    logger.info("=" * 60)
    logger.info(f"Baseline 2: Random ({random_seq})")
    logger.info("=" * 60)
    stats_random = evaluate_baseline(
        model=model,
        binder_seq=random_seq,
        target_seq=args.target,
        tau_noise=args.tau_noise,
        device=device,
        loss_weights=loss_weights,
        n_samples=args.n_samples,
        label="Random",
    )

    # ── BindCraft v2 对比（如果提供了最佳序列）────────────────
    stats_bindcraft = None
    if args.bindcraft_best_seq:
        logger.info("=" * 60)
        logger.info(f"BindCraft v2: {args.bindcraft_best_seq}")
        logger.info("=" * 60)
        stats_bindcraft = evaluate_baseline(
            model=model,
            binder_seq=args.bindcraft_best_seq,
            target_seq=args.target,
            tau_noise=args.tau_noise,
            device=device,
            loss_weights=loss_weights,
            n_samples=args.n_samples,
            label="BindCraft-v2",
        )

    # ── 汇总报告 ──────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("📊 Baseline 对比报告")
    logger.info("=" * 60)
    logger.info(f"Target: {args.target}")
    logger.info(f"Binder Length: {L}")
    logger.info(f"Tau noise: {args.tau_noise}")
    logger.info(f"N samples: {args.n_samples}")
    logger.info("")
    logger.info(f"{'Method':<20} {'total (mean±std)':<22} {'attraction':<14} {'iptm':<14}")
    logger.info("-" * 70)

    def fmt(stats, key):
        m, s = stats[key]["mean"], stats[key]["std"]
        return f"{m:+.4f}±{s:.4f}"

    logger.info(f"{'Poly-Ala':<20} {fmt(stats_poly_ala,'total'):<22} {fmt(stats_poly_ala,'attraction'):<14} {fmt(stats_poly_ala,'iptm'):<14}")
    logger.info(f"{'Random':<20} {fmt(stats_random,'total'):<22} {fmt(stats_random,'attraction'):<14} {fmt(stats_random,'iptm'):<14}")
    if stats_bindcraft:
        logger.info(f"{'BindCraft-v2':<20} {fmt(stats_bindcraft,'total'):<22} {fmt(stats_bindcraft,'attraction'):<14} {fmt(stats_bindcraft,'iptm'):<14}")
    if args.bindcraft_best_loss is not None:
        logger.info(f"{'BindCraft-v2 (opt)':<20} {args.bindcraft_best_loss:+.4f} (optimization best)")

    # 判断是否优于 baseline
    bc_total = stats_bindcraft["total"]["mean"] if stats_bindcraft else args.bindcraft_best_loss
    if bc_total is not None:
        poly_ala_total = stats_poly_ala["total"]["mean"]
        random_total = stats_random["total"]["mean"]
        better_than_poly_ala = bc_total < poly_ala_total
        better_than_random = bc_total < random_total
        logger.info("")
        logger.info(f"BindCraft v2 优于 Poly-Ala: {'✅' if better_than_poly_ala else '❌'} "
                    f"({bc_total:.4f} vs {poly_ala_total:.4f})")
        logger.info(f"BindCraft v2 优于 Random:   {'✅' if better_than_random else '❌'} "
                    f"({bc_total:.4f} vs {random_total:.4f})")

    # 保存 JSON
    result = {
        "target": args.target,
        "binder_length": L,
        "tau_noise": args.tau_noise,
        "n_samples": args.n_samples,
        "poly_ala_seq": poly_ala_seq,
        "random_seq": random_seq,
        "stats_poly_ala": stats_poly_ala,
        "stats_random": stats_random,
        "stats_bindcraft": stats_bindcraft,
        "bindcraft_best_loss_opt": args.bindcraft_best_loss,
        "bindcraft_best_seq": args.bindcraft_best_seq,
    }
    out_path = os.path.join(args.output_dir, "baseline_comparison.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"\n结果已保存到: {out_path}")


if __name__ == "__main__":
    main()
