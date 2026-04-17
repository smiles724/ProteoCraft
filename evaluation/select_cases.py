#!/usr/bin/env python3
"""
Retrodiction 评估 Pipeline - Step 1: 案例筛选

从 retrodiction_list.csv（450 条）中筛选适合评估的子集：
  - target 长度在 [80, 570] 之间（上限保证 target + 80aa binder ≤ 650aa 安全边界）
  - 以 ab-raw.csv 中 Protenix-v1.0.0 的 best DockQ 排序，优先选界面清晰的案例

输出: evaluation/outputs/selected_cases.csv

用法:
    python evaluation/select_cases.py [--top_n 10]
"""

import os
import sys
import csv
import argparse
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.configs import (
    RETRODICTION_CSV, AB_RAW_CSV, OUTPUT_DIR,
    SELECT_TARGET_LENGTH_MIN, SELECT_TARGET_LENGTH_MAX, SELECT_TOP_N,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_retrodiction_list(csv_path: str):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def load_best_dockq(ab_raw_path: str):
    """
    从 ab-raw.csv 提取每个 entry_id 对应 Protenix-v1.0.0 的最佳 DockQ。
    返回 {entry_id: best_dockq}。
    """
    best = {}
    with open(ab_raw_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "Protenix-v1" not in row.get("model_name", ""):
                continue
            eid = row["entry_id"]
            try:
                dq = float(row["dockq"])
            except (ValueError, KeyError):
                continue
            if eid not in best or dq > best[eid]:
                best[eid] = dq
    return best


def select_cases(top_n: int = SELECT_TOP_N):
    logger.info(f"加载 retrodiction_list: {RETRODICTION_CSV}")
    rows = load_retrodiction_list(RETRODICTION_CSV)
    logger.info(f"  总条目: {len(rows)}")

    filtered = [
        r for r in rows
        if SELECT_TARGET_LENGTH_MIN <= int(r["target_length"]) <= SELECT_TARGET_LENGTH_MAX
    ]
    logger.info(
        f"  筛选 target_length in [{SELECT_TARGET_LENGTH_MIN}, {SELECT_TARGET_LENGTH_MAX}]: "
        f"{len(filtered)} 条"
    )

    logger.info(f"加载 ab-raw.csv 提取 best DockQ ...")
    best_dockq = load_best_dockq(AB_RAW_CSV)
    logger.info(f"  有 Protenix-v1 DockQ 数据的 entry: {len(best_dockq)}")

    for r in filtered:
        r["best_dockq"] = best_dockq.get(r["entry_id"], 0.0)

    filtered.sort(key=lambda r: r["best_dockq"], reverse=True)
    selected = filtered[:top_n]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "selected_cases.csv")
    fieldnames = [
        "entry_id", "target_sequence", "binder_sequence_heavy",
        "binder_sequence_light", "binder_sequence", "target_length",
        "binder_length", "best_dockq",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)

    logger.info(f"已选 {len(selected)} 条，保存至: {out_path}")
    for r in selected:
        logger.info(
            f"  {r['entry_id']} | target={r['target_length']}aa | "
            f"binder={r['binder_length']}aa | DockQ={r['best_dockq']:.3f}"
        )

    return out_path


def main():
    parser = argparse.ArgumentParser(description="筛选 Retrodiction 测试案例")
    parser.add_argument("--top_n", type=int, default=SELECT_TOP_N)
    args = parser.parse_args()
    select_cases(top_n=args.top_n)


if __name__ == "__main__":
    main()