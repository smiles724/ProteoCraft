#!/usr/bin/env python3
"""
阶段三验证脚本：测试模拟退火 + 时间步动态调度

验证内容：
1. AnnealingScheduler 温度衰减和概率接受逻辑
2. TauScheduler 三阶段 tau 调度、动态权重、抖动
3. run_hybrid_hallucination 带调度的集成测试

运行方式：
    CUDA_VISIBLE_DEVICES=3 python scripts/test_scheduling.py 2>&1 | tee logs/scheduling_test.log
"""

import os
import sys
import logging
import time
import math
import random
from datetime import datetime

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── 日志持久化配置 ─────────────────────────────────────────────────────────
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = f"scheduling_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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


def test_annealing_scheduler():
    """单元测试：AnnealingScheduler"""
    logger.info("=" * 60)
    logger.info("Test 1: AnnealingScheduler 单元测试")
    logger.info("=" * 60)

    from bindcraft.hallucination.scheduler import AnnealingScheduler

    sched = AnnealingScheduler(T_init=1.0, T_min=0.01, decay="exponential")

    T_start = sched.get_temperature(0, 100)
    T_mid = sched.get_temperature(50, 100)
    T_end = sched.get_temperature(99, 100)

    logger.info(f"  T(step=0):  {T_start:.4f}")
    logger.info(f"  T(step=50): {T_mid:.4f}")
    logger.info(f"  T(step=99): {T_end:.4f}")

    assert abs(T_start - 1.0) < 0.01, f"T_start should be ~1.0, got {T_start}"
    assert T_mid < T_start, "T should decrease over time"
    assert abs(T_end - 0.01) < 0.005, f"T_end should be ~0.01, got {T_end}"
    logger.info("  温度衰减: OK")

    assert sched.accept(5.0, 4.0, 0, 100) is True, "Should always accept better candidate"
    logger.info("  严格改进必定接受: OK")

    random.seed(42)
    n_accept = sum(sched.accept(4.0, 4.5, 0, 100) for _ in range(1000))
    accept_rate = n_accept / 1000
    logger.info(f"  早期概率接受率 (L_cur=4.0, L_cand=4.5, step=0): {accept_rate:.3f}")
    assert 0.3 < accept_rate < 0.9, f"Early accept rate should be moderate, got {accept_rate}"

    random.seed(42)
    n_accept_late = sum(sched.accept(4.0, 4.5, 99, 100) for _ in range(1000))
    accept_rate_late = n_accept_late / 1000
    logger.info(f"  晚期概率接受率 (L_cur=4.0, L_cand=4.5, step=99): {accept_rate_late:.3f}")
    assert accept_rate_late < accept_rate, "Late accept rate should be lower than early"
    logger.info("  退火接受逻辑: OK")

    logger.info("Test 1 PASSED\n")


def test_tau_scheduler():
    """单元测试：TauScheduler"""
    logger.info("=" * 60)
    logger.info("Test 2: TauScheduler 单元测试")
    logger.info("=" * 60)

    from bindcraft.hallucination.scheduler import TauScheduler

    sched = TauScheduler(jitter=False)

    total = 100
    taus = []
    phases = []
    for s in range(total):
        tau = sched.get_tau(s, total)
        phase = sched.get_phase_name(s, total)
        taus.append(tau)
        phases.append(phase)

    tau_0 = taus[0]
    tau_20 = taus[20]
    tau_50 = taus[50]
    tau_80 = taus[80]
    tau_99 = taus[99]

    logger.info(f"  tau(step=0):  {tau_0:.4f} phase={phases[0]}")
    logger.info(f"  tau(step=20): {tau_20:.4f} phase={phases[20]}")
    logger.info(f"  tau(step=50): {tau_50:.4f} phase={phases[50]}")
    logger.info(f"  tau(step=80): {tau_80:.4f} phase={phases[80]}")
    logger.info(f"  tau(step=99): {tau_99:.4f} phase={phases[99]}")

    assert 0.6 <= tau_0 <= 0.9, f"Phase 1 tau should be in [0.6,0.9], got {tau_0}"
    assert 0.2 <= tau_50 <= 0.6, f"Phase 2 tau should be in [0.2,0.6], got {tau_50}"
    assert 0.01 <= tau_99 <= 0.2, f"Phase 3 tau should be in [0.01,0.2], got {tau_99}"
    logger.info("  三阶段 tau 范围: OK")

    assert phases[0] == "macro_topology"
    assert phases[50] == "interface_shaping"
    assert phases[99] == "atomic_refinement"
    logger.info("  阶段名称: OK")

    w_early = sched.get_dynamic_weights(5, total)
    w_mid = sched.get_dynamic_weights(50, total)
    w_late = sched.get_dynamic_weights(85, total)

    logger.info(f"  Early weights (step=5):  con={w_early['con']}, plddt={w_early['plddt']}")
    logger.info(f"  Mid weights (step=50):   con={w_mid['con']}, plddt={w_mid['plddt']}")
    logger.info(f"  Late weights (step=85):  con={w_late['con']}, plddt={w_late['plddt']}")

    assert w_early["con"] > w_late["con"], "Early phase should emphasize contact loss"
    assert w_late["plddt"] > w_early["plddt"], "Late phase should emphasize pLDDT"
    logger.info("  动态权重随阶段变化: OK")

    sched_jitter = TauScheduler(jitter=True, jitter_scale=0.1)
    random.seed(0)
    tau_samples = [sched_jitter.get_tau(50, total) for _ in range(50)]
    assert len(set(f"{t:.6f}" for t in tau_samples)) > 1, "Jitter should produce variation"
    tau_range = max(tau_samples) - min(tau_samples)
    logger.info(f"  Jitter range at step=50: {tau_range:.4f} (from {min(tau_samples):.4f} to {max(tau_samples):.4f})")
    logger.info("  时间步抖动: OK")

    logger.info("Test 2 PASSED\n")


def test_hybrid_with_scheduling():
    """集成测试：带调度的混合优化循环"""
    logger.info("=" * 60)
    logger.info("Test 3: 带调度的混合优化循环集成测试")
    logger.info("=" * 60)

    device = torch.device("cuda:0")
    logger.info(f"Device: {device} ({torch.cuda.get_device_name(0)})")

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
    logger.info("use_annealing=True, use_tau_scheduling=True, tau_jitter=True")

    from bindcraft.hallucination.pipeline import run_hybrid_hallucination

    t0 = time.time()
    best_seq, final_seq, losses_history = run_hybrid_hallucination(
        model=model,
        target_seq=target_seq,
        L_binder=L_binder,
        N_steps=N_steps,
        learning_rate=0.05,
        device=device,
        K_top=K_top,
        B_candidates=B_candidates,
        T_sample=1.0,
        max_mutations_per_seq=3,
        use_annealing=True,
        T_init=1.0,
        T_min=0.01,
        use_tau_scheduling=True,
        tau_jitter=True,
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

    assert len(best_seq) == L_binder
    assert len(final_seq) == L_binder
    assert len(losses_history) == N_steps

    for i, loss in enumerate(losses_history):
        assert loss == loss and abs(loss) != float("inf"), f"Step {i}: loss={loss} is NaN/Inf"
    logger.info("  所有 loss 值合法 (非 NaN/Inf): OK")

    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    for aa in best_seq:
        assert aa in valid_aa
    logger.info("  序列氨基酸合法性: OK")

    vram_used = torch.cuda.max_memory_allocated() / 1e9
    logger.info(f"  峰值 VRAM: {vram_used:.1f} GB")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Test 3 PASSED: 带调度的混合优化循环验证通过")
    logger.info("=" * 60)


def main():
    logger.info("=" * 70)
    logger.info("阶段三验证：模拟退火 + 时间步动态调度")
    logger.info(f"日志文件: {log_filepath}")
    logger.info("=" * 70)
    logger.info("")

    test_annealing_scheduler()
    test_tau_scheduler()
    test_hybrid_with_scheduling()

    logger.info("")
    logger.info("=" * 70)
    logger.info("ALL TESTS PASSED")
    logger.info(f"日志已保存至: {log_filepath}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
