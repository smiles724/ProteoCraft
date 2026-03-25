#!/usr/bin/env python3
"""
BindCraft v2 - 完整结构评估脚本

使用 Protenix 完整多步 ODE 推理（非单步近似），计算真正独立于优化目标的结构质量指标：
  - pLDDT：预测局部距离差异测试，衡量 Binder 结构置信度（0-100，≥70 为可信）
  - iPTM：接口预测 TM 分数，衡量 Binder-Target 复合物置信度（0-1，≥0.6 为可信）
  - pTM：整体预测 TM 分数
  - chain_pLDDT：各链的 pLDDT
  - chain_pair_iPTM：链间接口置信度

这些指标**不在优化目标里**，是真正独立的科学评估标准。

对比对象：
  1. Poly-Ala（全丙氨酸，无任何优化）
  2. Random（随机序列，无任何优化）
  3. BindCraft-v2 最佳序列（经过梯度优化）

使用方法：
    python scripts/run_full_eval.py --target LKDFD... --binder_length 40
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
# 完整推理：调用 Protenix main_inference_loop
# ============================================================

def run_full_inference(
    model,
    binder_seq: str,
    target_seq: str,
    device: torch.device,
    N_cycle: int = 4,
    N_sample: int = 1,
) -> dict:
    """
    对给定序列运行 Protenix 完整推理（多步 ODE 扩散采样），
    返回 summary_confidence 中的所有独立评估指标。

    这里的 iPTM/pLDDT 是真实的结构预测置信度，
    与 BindCraft 优化目标（单步近似）完全独立。

    Returns:
        dict: {
            'plddt': float,           # 全体原子平均 pLDDT（0-100）
            'iptm': float,            # 接口 pTM（0-1）
            'ptm': float,             # 整体 pTM（0-1）
            'chain_plddt_binder': float,  # Binder 链 pLDDT
            'chain_plddt_target': float,  # Target 链 pLDDT
            'chain_pair_iptm': float,     # Binder-Target 链对 iPTM
            'ranking_score': float,   # Protenix 综合排名分数（0.8*iptm + 0.2*ptm）
        }
    """
    from bindcraft.utils import build_complex_features, prepare_input_features

    with torch.no_grad():
        input_feature_dict, N_token, N_atom = build_complex_features(
            binder_seq, target_seq, device
        )
        input_feature_dict = prepare_input_features(model, input_feature_dict)

        # 调用 Protenix 完整推理（_main_inference_loop，单 seed）
        pred_dict, log_dict, time_tracker = model._main_inference_loop(
            input_feature_dict=input_feature_dict,
            label_dict=None,
            N_cycle=N_cycle,
            mode="inference",
            inplace_safe=True,
            chunk_size=None,
        )

    # summary_confidence 是 list[dict]，每个 dict 对应一个 diffusion sample
    # _main_inference_loop 默认 N_sample=1，取第一个
    sc_list = pred_dict["summary_confidence"]
    sc = sc_list[0] if isinstance(sc_list, list) else sc_list

    def _scalar(t):
        if isinstance(t, torch.Tensor):
            return float(t.flatten()[0].item())
        return float(t)

    result = {
        "plddt":         _scalar(sc.get("plddt", 0.0)),
        "iptm":          _scalar(sc.get("iptm", 0.0)),
        "ptm":           _scalar(sc.get("ptm", 0.0)),
        "ranking_score": _scalar(sc.get("ranking_score", 0.0)),
    }

    # 各链 pLDDT：chain_plddt 是 [N_chain] tensor
    if "chain_plddt" in sc:
        chain_plddt = sc["chain_plddt"]
        if isinstance(chain_plddt, torch.Tensor):
            chain_plddt = chain_plddt.flatten()
            if len(chain_plddt) >= 2:
                result["chain_plddt_binder"] = float(chain_plddt[0].item())
                result["chain_plddt_target"] = float(chain_plddt[1].item())
            elif len(chain_plddt) == 1:
                result["chain_plddt_binder"] = float(chain_plddt[0].item())

    # 链对 iPTM：chain_pair_iptm 是 [N_chain, N_chain] tensor
    if "chain_pair_iptm" in sc:
        cpi = sc["chain_pair_iptm"]
        if isinstance(cpi, torch.Tensor):
            cpi_flat = cpi.reshape(-1)
            if cpi.dim() == 2 and cpi.shape[0] >= 2 and cpi.shape[1] >= 2:
                result["chain_pair_iptm_binder_target"] = float(cpi[0, 1].item())
                result["chain_pair_iptm_target_binder"] = float(cpi[1, 0].item())

    return result


# ============================================================
# 多次采样统计
# ============================================================

def evaluate_seq_full(
    model,
    binder_seq: str,
    target_seq: str,
    device: torch.device,
    n_repeats: int = 3,
    label: str = "eval",
) -> dict:
    """
    重复 n_repeats 次完整推理，统计均值±标准差。
    每次扩散采样的随机种子不同，反映结构预测的随机性。
    """
    all_results = []
    logger.info(f"[{label}] 序列: {binder_seq[:30]}...")
    logger.info(f"[{label}] 开始 {n_repeats} 次完整推理...")

    for i in range(n_repeats):
        res = run_full_inference(
            model=model,
            binder_seq=binder_seq,
            target_seq=target_seq,
            device=device,
        )
        all_results.append(res)
        logger.info(
            f"[{label}] 推理 {i+1}/{n_repeats}: "
            f"pLDDT={res['plddt']:.1f}  iPTM={res['iptm']:.3f}  "
            f"pTM={res['ptm']:.3f}  ranking={res.get('ranking_score', 0):.3f}"
        )

    keys = all_results[0].keys()
    stats = {}
    for k in keys:
        vals = [d[k] for d in all_results]
        stats[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    logger.info(
        f"[{label}] 汇总: pLDDT={stats['plddt']['mean']:.1f}±{stats['plddt']['std']:.1f}  "
        f"iPTM={stats['iptm']['mean']:.3f}±{stats['iptm']['std']:.3f}  "
        f"ranking={stats.get('ranking_score', {}).get('mean', 0):.3f}"
    )
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

    parser = argparse.ArgumentParser(description="BindCraft v2 - 完整结构评估")
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--binder_length", type=int, default=40)
    parser.add_argument("--n_repeats", type=int, default=3,
                        help="每个序列的完整推理次数（每次扩散随机种子不同）")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_name", type=str, default="protenix_base_default_v1.0.0")
    parser.add_argument("--output_dir", type=str, default="outputs/full_eval")
    parser.add_argument("--bindcraft_best_seq", type=str, default=None,
                        help="BindCraft v2 优化后的最佳序列")
    parser.add_argument("--bindcraft_best_loss", type=float, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from scripts.run_hallucination import load_protenix_model
    model, _ = load_protenix_model(args.model_name, args.device)
    device = torch.device(args.device)

    L = args.binder_length
    np.random.seed(0)
    aa_chars = list("ACDEFGHIKLMNPQRSTVWY")

    seqs = {
        "Poly-Ala": "A" * L,
        "Random":   "".join(np.random.choice(aa_chars, size=L)),
    }
    if args.bindcraft_best_seq:
        seqs["BindCraft-v2"] = args.bindcraft_best_seq

    all_stats = {}
    for label, seq in seqs.items():
        logger.info("\n" + "=" * 60)
        logger.info(f"{label}: {seq}")
        logger.info("=" * 60)
        all_stats[label] = evaluate_seq_full(
            model=model,
            binder_seq=seq,
            target_seq=args.target,
            device=device,
            n_repeats=args.n_repeats,
            label=label,
        )

    # ── 汇总报告 ──────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("📊 完整结构评估报告（独立于优化目标的真实指标）")
    logger.info("=" * 70)
    logger.info(f"Target: {args.target}")
    logger.info(f"Binder Length: {L}  |  N repeats: {args.n_repeats}")
    logger.info("")
    logger.info(
        f"{'Method':<18} {'pLDDT↑':<14} {'iPTM↑':<12} {'pTM↑':<12} "
        f"{'ranking↑':<12} {'chain_iPTM↑':<14}"
    )
    logger.info("-" * 80)

    def fmt(stats, key, scale=1.0):
        if key not in stats:
            return "  N/A"
        m = stats[key]["mean"] * scale
        s = stats[key]["std"] * scale
        return f"{m:.3f}±{s:.3f}"

    for label, stats in all_stats.items():
        logger.info(
            f"{label:<18} {fmt(stats,'plddt'):<14} {fmt(stats,'iptm'):<12} "
            f"{fmt(stats,'ptm'):<12} {fmt(stats,'ranking_score'):<12} "
            f"{fmt(stats,'chain_pair_iptm_binder_target'):<14}"
        )

    # 判断 BindCraft v2 是否优于 baseline
    if "BindCraft-v2" in all_stats:
        bc = all_stats["BindCraft-v2"]
        pa = all_stats["Poly-Ala"]
        rd = all_stats["Random"]
        logger.info("")
        for metric, direction in [("plddt", "↑"), ("iptm", "↑"), ("ranking_score", "↑")]:
            if metric not in bc:
                continue
            bc_val = bc[metric]["mean"]
            pa_val = pa[metric]["mean"]
            rd_val = rd[metric]["mean"]
            better_pa = bc_val > pa_val if direction == "↑" else bc_val < pa_val
            better_rd = bc_val > rd_val if direction == "↑" else bc_val < rd_val
            logger.info(
                f"BindCraft v2 {metric} 优于 Poly-Ala: {'✅' if better_pa else '❌'} "
                f"({bc_val:.3f} vs {pa_val:.3f})"
            )
            logger.info(
                f"BindCraft v2 {metric} 优于 Random:   {'✅' if better_rd else '❌'} "
                f"({bc_val:.3f} vs {rd_val:.3f})"
            )

    # 指标含义说明
    logger.info("")
    logger.info("指标含义：")
    logger.info("  pLDDT  (0-100)：预测局部距离差异测试，衡量结构置信度，≥70 为可信")
    logger.info("  iPTM   (0-1)  ：接口预测TM分数，衡量Binder-Target复合物置信度，≥0.6 为可信")
    logger.info("  pTM    (0-1)  ：整体预测TM分数")
    logger.info("  ranking(0-1)  ：Protenix综合排名分数 = 0.8*iPTM + 0.2*pTM")
    logger.info("  chain_iPTM    ：Binder→Target 链对接口置信度（最直接的结合质量指标）")
    logger.info("")
    logger.info("注：以上指标均来自 Protenix 完整多步ODE推理，独立于 BindCraft 的优化目标")

    # 保存结果
    result = {
        "target": args.target,
        "binder_length": L,
        "n_repeats": args.n_repeats,
        "sequences": {k: v for k, v in seqs.items()},
        "stats": all_stats,
        "bindcraft_best_loss_opt": args.bindcraft_best_loss,
    }
    out_path = os.path.join(args.output_dir, "full_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"\n结果已保存到: {out_path}")


if __name__ == "__main__":
    main()
