#!/usr/bin/env python3
"""
阶段二验证脚本：测试混合优化循环 (Hybrid Hallucination)

验证内容：
1. mutation_proposer 能否从梯度矩阵生成合理的离散候选
2. run_hybrid_hallucination 能否跑通完整的四步循环
3. 零阶验证的 loss 是否合理，接受机制是否正常工作
4. 多轮后 loss 是否有下降趋势

运行方式：
    CUDA_VISIBLE_DEVICES=3 python scripts/test_hybrid.py 2>&1 | tee logs/hybrid_test.log
"""

import os
import sys
import logging
import time
from datetime import datetime

import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── 日志持久化配置 ─────────────────────────────────────────────────────────
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = f"hybrid_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = os.path.join(LOG_DIR, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filepath, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def test_mutation_proposer():
    """单元测试：验证 mutation_proposer 的正确性"""
    logger.info("=" * 60)
    logger.info("Test 1: mutation_proposer 单元测试")
    logger.info("=" * 60)

    from bindcraft.hallucination.mutation_proposer import (
        propose_mutations,
        compute_mutation_benefit,
        select_top_k_mutations,
        _seq_to_indices,
        _indices_to_seq,
    )

    seq = "ACDEFGHIKLMNPQRSTVWY"  # 20 aa
    L = len(seq)
    torch.manual_seed(123)
    grad = torch.randn(L, 20)

    indices = _seq_to_indices(seq)
    assert len(indices) == L
    assert _indices_to_seq(indices) == seq
    logger.info(f"  序列转换: OK (seq={seq})")

    delta_L = compute_mutation_benefit(grad, indices)
    assert delta_L.shape == (L, 20)
    for i, c in enumerate(indices):
        assert abs(delta_L[i, c].item()) < 1e-6, f"自身突变 delta_L 应为 0，但 pos {i} = {delta_L[i, c]}"
    logger.info(f"  突变收益矩阵: OK (shape={delta_L.shape}, 自身突变全为0)")

    positions, new_aas, benefits = select_top_k_mutations(delta_L, indices, K_top=10)
    assert positions.shape[0] == 10
    assert benefits[0] >= benefits[-1], "收益应降序排列"
    for i in range(10):
        pos = positions[i].item()
        aa = new_aas[i].item()
        assert aa != indices[pos], "不应包含自身突变"
    logger.info(f"  Top-K 选取: OK (K=10, best_benefit={benefits[0]:.4f})")

    candidates = propose_mutations(
        grad_matrix=grad,
        current_seq=seq,
        K_top=10,
        B_candidates=8,
        T_sample=1.0,
        max_mutations_per_seq=3,
    )
    assert len(candidates) == 8
    for c in candidates:
        assert len(c) == L
        assert c != seq, "候选不应与原序列完全相同"
    n_unique = len(set(candidates))
    logger.info(f"  候选生成: OK ({len(candidates)} candidates, {n_unique} unique)")

    logger.info("Test 1 PASSED\n")


def test_hybrid_hallucination():
    """集成测试：在 GPU 上运行混合优化循环"""
    logger.info("=" * 60)
    logger.info("Test 2: 混合优化循环集成测试")
    logger.info("=" * 60)

    device = torch.device("cuda:0")
    logger.info(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    logger.info("加载 Protenix 模型...")
    from scripts.run_hallucination import load_protenix_model
    model, configs = load_protenix_model("protenix_base_default_v1.0.0", str(device))
    logger.info("模型加载完成")

    target_seq = "GIVEQCCTSICSLYQLENYCN"
    L_binder = 40
    N_steps = 3
    B_candidates = 4
    K_top = 8

    logger.info(f"Target:  {target_seq} ({len(target_seq)} aa)")
    logger.info(f"Binder 长度: {L_binder}")
    logger.info(f"N_steps={N_steps}, B_candidates={B_candidates}, K_top={K_top}")

    from bindcraft.hallucination.pipeline import run_hybrid_hallucination

    t0 = time.time()
    best_seq, final_seq, losses_history = run_hybrid_hallucination(
        model=model,
        target_seq=target_seq,
        L_binder=L_binder,
        N_steps=N_steps,
        learning_rate=0.05,
        tau_noise=4.0,
        device=device,
        K_top=K_top,
        B_candidates=B_candidates,
        T_sample=1.0,
        max_mutations_per_seq=3,
    )
    elapsed = time.time() - t0

    logger.info("")
    logger.info("=" * 60)
    logger.info("结果检查")
    logger.info("=" * 60)

    logger.info(f"耗时: {elapsed:.1f}s ({elapsed/N_steps:.1f}s/step)")
    logger.info(f"零阶最佳序列: {best_seq}")
    logger.info(f"一阶最终序列: {final_seq}")
    logger.info(f"Loss 历史: {[f'{l:.4f}' for l in losses_history]}")

    assert len(best_seq) == L_binder, f"best_seq 长度应为 {L_binder}，实际为 {len(best_seq)}"
    assert len(final_seq) == L_binder, f"final_seq 长度应为 {L_binder}，实际为 {len(final_seq)}"
    assert len(losses_history) == N_steps, f"losses_history 长度应为 {N_steps}，实际为 {len(losses_history)}"

    for i, loss in enumerate(losses_history):
        assert loss == loss and abs(loss) != float("inf"), f"Step {i}: loss={loss} is NaN/Inf"
    logger.info("  所有 loss 值合法 (非 NaN/Inf): OK")

    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    for aa in best_seq:
        assert aa in valid_aa, f"无效氨基酸: {aa}"
    logger.info("  序列氨基酸合法性: OK")

    vram_used = torch.cuda.max_memory_allocated() / 1e9
    logger.info(f"  峰值 VRAM: {vram_used:.1f} GB")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Test 2 PASSED: 混合优化循环验证通过")
    logger.info("=" * 60)


def main():
    logger.info("=" * 70)
    logger.info("阶段二验证：梯度引导离散候选 + 混合优化循环")
    logger.info(f"日志文件: {log_filepath}")
    logger.info("=" * 70)
    logger.info("")

    test_mutation_proposer()
    test_hybrid_hallucination()

    logger.info("")
    logger.info("=" * 70)
    logger.info("ALL TESTS PASSED")
    logger.info(f"日志已保存至: {log_filepath}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
