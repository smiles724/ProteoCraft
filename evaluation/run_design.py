#!/usr/bin/env python3
"""
Retrodiction 评估 Pipeline - Step 2: 批量设计 Binder

对 selected_cases.csv 中的每个 target 调用 run_hybrid_hallucination() 设计 binder，
同时生成等长随机序列作为负对照。

输出: evaluation/outputs/designs/<entry_id>.json

用法:
    CUDA_VISIBLE_DEVICES=3 python evaluation/run_design.py \
        [--cases evaluation/outputs/selected_cases.csv] \
        [--n_steps 50] [--b_candidates 4] [--device cuda:0]
"""

import os
import sys
import csv
import json
import time
import random
import argparse
import logging
import fcntl
from datetime import datetime

import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.configs import (
    OUTPUT_DIR, DESIGNS_DIR, LOG_DIR, MODEL_NAME,
    DESIGN_L_BINDER, DESIGN_N_STEPS, DESIGN_LEARNING_RATE,
    DESIGN_K_TOP, DESIGN_B_CANDIDATES, DESIGN_T_SAMPLE,
    DESIGN_MAX_MUTATIONS, DESIGN_USE_ANNEALING, DESIGN_USE_TAU_SCHEDULING,
    DESIGN_NUM_BEAMS,
)

os.makedirs(LOG_DIR, exist_ok=True)
log_filepath = os.path.join(
    LOG_DIR, f"retrodiction_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filepath, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")


def generate_random_seq(length: int) -> str:
    return "".join(random.choices(AA_ALPHABET, k=length))


def run_design_batch(cases_csv: str, n_steps: int, b_candidates: int, device_str: str, entry_ids: list = None, num_beams: int = 1, L_binder_override: int = None):
    from scripts.run_hallucination import load_protenix_model
    from bindcraft.hallucination.pipeline import run_hybrid_hallucination

    with open(cases_csv) as f:
        cases = list(csv.DictReader(f))
    if entry_ids:
        cases = [c for c in cases if c["entry_id"] in set(entry_ids)]
    logger.info(f"共 {len(cases)} 个 target 待设计")

    device = torch.device(device_str)
    logger.info(f"加载模型 {MODEL_NAME} -> {device}")
    model, configs = load_protenix_model(MODEL_NAME, str(device))

    os.makedirs(DESIGNS_DIR, exist_ok=True)

    for i, case in enumerate(cases):
        entry_id = case["entry_id"]
        target_seq = case["target_sequence"]
        real_binder = case.get("binder_sequence", "")
        real_heavy = case.get("binder_sequence_heavy", "")

        L_binder = L_binder_override if L_binder_override is not None else DESIGN_L_BINDER
        out_path = os.path.join(DESIGNS_DIR, f"{entry_id}.json")
        lock_path = os.path.join(DESIGNS_DIR, f"{entry_id}.lock")
        if os.path.exists(out_path):
            logger.info(f"[{i+1}/{len(cases)}] {entry_id} 已存在，跳过")
            continue
        # 原子锁：防止多实例同时启动同一 entry 的设计
        lock_file = open(lock_path, "w")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info(f"[{i+1}/{len(cases)}] {entry_id} 正在其他实例中设计，跳过")
            lock_file.close()
            continue

        logger.info("=" * 60)
        logger.info(
            f"[{i+1}/{len(cases)}] 设计 {entry_id} | "
            f"target={len(target_seq)}aa | L_binder={L_binder}"
        )

        random_seq = generate_random_seq(L_binder)

        t0 = time.time()
        try:
            best_seq, final_seq, losses_history = run_hybrid_hallucination(
                model=model,
                target_seq=target_seq,
                L_binder=L_binder,
                N_steps=n_steps,
                learning_rate=DESIGN_LEARNING_RATE,
                device=device,
                K_top=DESIGN_K_TOP,
                B_candidates=b_candidates,
                T_sample=DESIGN_T_SAMPLE,
                max_mutations_per_seq=DESIGN_MAX_MUTATIONS,
                use_annealing=DESIGN_USE_ANNEALING,
                use_tau_scheduling=DESIGN_USE_TAU_SCHEDULING,
                num_beams=num_beams,
            )
            elapsed = time.time() - t0

            # best_loss / best_step 均从零阶 loss 历史中推算（优化目标为零阶 loss）
            best_loss = float(min(losses_history))
            best_step = int(losses_history.index(best_loss))

            result = {
                "entry_id": entry_id,
                "target_seq": target_seq,
                "target_length": len(target_seq),
                "designed_seq": best_seq,
                "designed_final_seq": final_seq,
                "random_seq": random_seq,
                "real_binder_seq": real_binder,
                "real_binder_heavy": real_heavy,
                "design_loss_history": [float(l) for l in losses_history],
                "best_loss": best_loss,
                "best_step": best_step,
                "design_time_sec": round(elapsed, 1),
                "design_params": {
                    "L_binder": L_binder,
                    "N_steps": n_steps,
                    "num_beams": num_beams,
                    "B_candidates": b_candidates,
                    "use_annealing": DESIGN_USE_ANNEALING,
                    "use_tau_scheduling": DESIGN_USE_TAU_SCHEDULING,
                },
            }

            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info(
                f"  完成 | {elapsed:.1f}s | best_loss={best_loss:.4f}@{best_step} | designed={best_seq[:20]}... | "
                f"final_loss={losses_history[-1]:.4f} | saved: {out_path}"
            )
        except Exception as e:
            logger.error(f"  设计失败: {e}")
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
            os.remove(lock_path)

    logger.info("=" * 60)
    logger.info(f"批量设计完成。日志: {log_filepath}")


def main():
    parser = argparse.ArgumentParser(description="批量设计 Binder")
    parser.add_argument(
        "--cases", type=str,
        default=os.path.join(OUTPUT_DIR, "selected_cases.csv"),
    )
    parser.add_argument("--n_steps", type=int, default=DESIGN_N_STEPS)
    parser.add_argument("--b_candidates", type=int, default=DESIGN_B_CANDIDATES)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--num_beams", type=int, default=DESIGN_NUM_BEAMS,
        help="Beam Search 宽度（演化路径数量）；默认为 1；设为 2 则启用 Beam Search（显存压力大）"
    )
    parser.add_argument(
        "--entry_ids", type=str, default=None,
        help="逗号分隔的 entry_id 列表，如 '8trt,9ho4'。指定时只处理这些 case。"
    )
    parser.add_argument(
        "--l_binder", type=int, default=None,
        help="Binder 长度（aa）。不指定时使用 configs.py 中的 DESIGN_L_BINDER（默认 80aa）。"
        "用于跑不同长度的对照实验，如 '--l_binder 114'。"
    )
    args = parser.parse_args()

    entry_ids = None
    if args.entry_ids:
        entry_ids = [e.strip() for e in args.entry_ids.split(",") if e.strip()]
    L_override = args.l_binder if args.l_binder is not None else DESIGN_L_BINDER
    run_design_batch(args.cases, args.n_steps, args.b_candidates, args.device, entry_ids, args.num_beams, L_binder_override=L_override)


if __name__ == "__main__":
    main()
