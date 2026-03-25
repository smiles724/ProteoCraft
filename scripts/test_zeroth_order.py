#!/usr/bin/env python3
"""
阶段一验证脚本：测试零阶全链路推断模块

验证内容：
1. zeroth_order_eval 能否跑通完整 AF3 推断
2. 输出的 8 项 loss 是否合理（非 NaN/Inf，数量级正常）
3. summary_confidence 是否正常（pLDDT/iPTM/pTM）
4. pipeline.py 的重构是否保持了正确性（import 无报错）
"""

import os
import sys
import logging
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    device = torch.device("cuda:0")

    # ── 1. 验证 import 无报错 ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 1: 验证 import")
    logger.info("=" * 60)

    from bindcraft.losses.total_loss import compute_all_losses
    from bindcraft.hallucination.zeroth_order import zeroth_order_eval, ZerothOrderResult
    from bindcraft.hallucination.pipeline import run_hallucination, DEFAULT_LOSS_WEIGHTS

    logger.info("  compute_all_losses: OK")
    logger.info("  zeroth_order_eval:  OK")
    logger.info("  run_hallucination:  OK (import only, not running)")
    logger.info("  DEFAULT_LOSS_WEIGHTS: %s", DEFAULT_LOSS_WEIGHTS)

    # ── 2. 加载模型 ──────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 2: 加载 Protenix 模型")
    logger.info("=" * 60)

    from scripts.run_hallucination import load_protenix_model
    model, configs = load_protenix_model("protenix_base_default_v1.0.0", str(device))
    logger.info("模型加载完成")

    # ── 3. 运行零阶推断 ──────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 3: 运行零阶推断")
    logger.info("=" * 60)

    target_seq = "GIVEQCCTSICSLYQLENYCN"
    binder_seq = "ACDEFGHIKLMNPQRSTVWY" * 2  # 40 aa

    logger.info(f"Target:  {target_seq} ({len(target_seq)} aa)")
    logger.info(f"Binder:  {binder_seq} ({len(binder_seq)} aa)")

    result = zeroth_order_eval(
        binder_seq=binder_seq,
        target_seq=target_seq,
        model=model,
        device=device,
    )

    # ── 4. 检查结果 ──────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 4: 检查结果")
    logger.info("=" * 60)

    assert isinstance(result, ZerothOrderResult)
    logger.info(f"binder_seq:  {result.binder_seq[:30]}...")
    logger.info(f"total_loss:  {result.total_loss:.4f}")

    logger.info("")
    logger.info("8 项 loss:")
    all_ok = True
    for k, v in result.loss_dict.items():
        is_finite = not (v != v or abs(v) == float("inf"))  # NaN or Inf check
        status = "OK" if is_finite else "FAIL"
        if not is_finite:
            all_ok = False
        logger.info(f"  {k:>10s}: {v:>10.4f}  [{status}]")

    logger.info("")
    logger.info("Summary Confidence (Protenix 原生):")
    for k in ["plddt", "iptm", "ptm", "ranking_score"]:
        v = result.summary_confidence.get(k, "N/A")
        logger.info(f"  {k:>15s}: {v}")

    logger.info("")
    logger.info(f"Coordinate shape: {result.coordinate.shape}")

    # ── 5. 最终判定 ──────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    if all_ok:
        logger.info("PASS: 零阶推断模块验证通过")
    else:
        logger.info("FAIL: 存在异常 loss 值，请检查")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
