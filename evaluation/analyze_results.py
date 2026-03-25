#!/usr/bin/env python3
"""
Retrodiction 评估 Pipeline - Step 4: 结果分析

读取 eval_results.csv，生成：
1. 每个 entry_id 三组序列的指标对比表（统一 AF3 指标 + DockQ 参考）
2. 汇总统计（三组用同一把尺子：iPTM / pLDDT / total_loss）
3. 配对比较：designed vs random, designed vs real
4. evaluation_report.md

用法:
    python evaluation/analyze_results.py
"""

import os
import sys
import csv
import logging
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.configs import OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

METRIC_KEYS_LOSS = ["total_loss", "plddt", "pae", "i_pae", "con", "i_con", "iptm", "rg"]
METRIC_KEYS_CONF = ["mean_plddt_conf", "iptm_conf", "ptm_conf"]
METRIC_KEYS_REF = ["dockq_ref", "lddt_ref"]


def load_results(csv_path: str):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in METRIC_KEYS_LOSS + METRIC_KEYS_CONF + METRIC_KEYS_REF + ["eval_time_sec"]:
            val = r.get(k, "")
            if val == "" or val is None:
                r[k] = None
            else:
                try:
                    r[k] = float(val)
                except ValueError:
                    r[k] = None
    return rows


def group_by_entry(rows):
    grouped = defaultdict(dict)
    for r in rows:
        grouped[r["entry_id"]][r["seq_type"]] = r
    return grouped


def compute_stats(values):
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return None, None, 0
    mean = sum(vals) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        std = var ** 0.5
    else:
        std = 0.0
    return mean, std, n


def fmt(val, decimals=3):
    if val is None:
        return "-"
    if decimals == 1:
        return f"{val:.1f}"
    return f"{val:.{decimals}f}"


def generate_report(csv_path: str, out_path: str):
    rows = load_results(csv_path)
    grouped = group_by_entry(rows)

    logger.info(f"读取 {len(rows)} 行评估结果，{len(grouped)} 个 entry")

    lines = []
    lines.append("# Retrodiction 评估报告\n")
    lines.append(f"数据: {csv_path}\n")
    lines.append(f"Entry 数: {len(grouped)}\n")
    lines.append("三组序列全部通过 AF3 完整推断，用同一把尺子（iPTM / pLDDT / total_loss）打分。\n")

    # ── 逐 entry 对比表 ─────────────────────────────────────────────────
    lines.append("\n## 逐案例对比\n")
    lines.append(
        "| entry_id | seq_type | binder_len | total_loss | "
        "mean_pLDDT | iPTM | pTM | DockQ_ref |"
    )
    lines.append(
        "|----------|----------|-----------|------------|"
        "-----------|------|-----|-----------|"
    )
    for eid in sorted(grouped):
        for st in ["designed", "random", "real"]:
            r = grouped[eid].get(st)
            if r is None:
                continue
            lines.append(
                f"| {eid} | {st} | {r.get('binder_length', '-')} | "
                f"{fmt(r['total_loss'])} | "
                f"{fmt(r['mean_plddt_conf'], 1)} | "
                f"{fmt(r['iptm_conf'])} | "
                f"{fmt(r['ptm_conf'])} | "
                f"{fmt(r['dockq_ref'])} |"
            )

    # ── 汇总统计（三组统一指标）────────────────────────────────────────────
    lines.append("\n## 汇总统计（三组统一 AF3 指标）\n")

    type_metrics = defaultdict(lambda: defaultdict(list))
    for r in rows:
        st = r["seq_type"]
        for k in METRIC_KEYS_LOSS + METRIC_KEYS_CONF:
            type_metrics[st][k].append(r[k])

    lines.append(
        "| seq_type | n | total_loss | mean_pLDDT | iPTM | pTM |"
    )
    lines.append(
        "|----------|---|------------|-----------|------|-----|"
    )
    for st in ["designed", "random", "real"]:
        m = type_metrics.get(st)
        if not m:
            continue
        _, _, n = compute_stats(m["total_loss"])
        if n == 0:
            continue
        tl_mean, tl_std, _ = compute_stats(m["total_loss"])
        pl_mean, pl_std, _ = compute_stats(m["mean_plddt_conf"])
        ip_mean, ip_std, _ = compute_stats(m["iptm_conf"])
        pt_mean, pt_std, _ = compute_stats(m["ptm_conf"])
        lines.append(
            f"| {st} | {n} | "
            f"{fmt(tl_mean)} +/- {fmt(tl_std)} | "
            f"{fmt(pl_mean, 1)} +/- {fmt(pl_std, 1)} | "
            f"{fmt(ip_mean)} +/- {fmt(ip_std)} | "
            f"{fmt(pt_mean)} +/- {fmt(pt_std)} |"
        )

    # ── 配对比较 (designed vs random) ────────────────────────────────────
    lines.append("\n## 配对比较: designed vs random\n")

    paired_dr = [
        eid for eid in grouped
        if "designed" in grouped[eid] and "random" in grouped[eid]
    ]
    if paired_dr:
        wins = {"total_loss": 0, "iptm_conf": 0, "mean_plddt_conf": 0}
        diffs_iptm, diffs_plddt = [], []
        for eid in paired_dr:
            d = grouped[eid]["designed"]
            r = grouped[eid]["random"]
            if d["total_loss"] is not None and r["total_loss"] is not None:
                if d["total_loss"] < r["total_loss"]:
                    wins["total_loss"] += 1
            if d["iptm_conf"] is not None and r["iptm_conf"] is not None:
                if d["iptm_conf"] > r["iptm_conf"]:
                    wins["iptm_conf"] += 1
                diffs_iptm.append(d["iptm_conf"] - r["iptm_conf"])
            if d["mean_plddt_conf"] is not None and r["mean_plddt_conf"] is not None:
                if d["mean_plddt_conf"] > r["mean_plddt_conf"]:
                    wins["mean_plddt_conf"] += 1
                diffs_plddt.append(d["mean_plddt_conf"] - r["mean_plddt_conf"])

        n_pairs = len(paired_dr)
        lines.append(f"配对数: {n_pairs}\n")
        lines.append("| 指标 | designed 胜出次数 | 胜率 |")
        lines.append("|------|-----------------|------|")
        for k, w in wins.items():
            lines.append(f"| {k} | {w}/{n_pairs} | {w/n_pairs*100:.0f}% |")

        mean_diff_iptm, _, _ = compute_stats(diffs_iptm)
        mean_diff_plddt, _, _ = compute_stats(diffs_plddt)
        lines.append(f"\n平均 iPTM 差异 (designed - random): {fmt(mean_diff_iptm, 4)}")
        lines.append(f"平均 pLDDT 差异 (designed - random): {fmt(mean_diff_plddt, 1)}")
    else:
        lines.append("无可配对数据。\n")

    # ── 配对比较 (designed vs real) ──────────────────────────────────────
    lines.append("\n## 配对比较: designed vs real\n")

    paired_dreal = [
        eid for eid in grouped
        if "designed" in grouped[eid] and "real" in grouped[eid]
    ]
    if paired_dreal:
        lines.append(
            "三组均通过 AF3 推断，指标完全可比。real 使用完整重链序列（~120aa）。\n"
        )

        wins_real = {"total_loss": 0, "iptm_conf": 0, "mean_plddt_conf": 0}
        diffs_iptm_real, diffs_plddt_real = [], []

        lines.append(
            "| entry_id | designed_iPTM | designed_pLDDT | "
            "real_iPTM | real_pLDDT | DockQ_ref |"
        )
        lines.append(
            "|----------|--------------|----------------|"
            "----------|------------|-----------|"
        )
        for eid in sorted(paired_dreal):
            d = grouped[eid]["designed"]
            r = grouped[eid]["real"]
            lines.append(
                f"| {eid} | {fmt(d['iptm_conf'])} | {fmt(d['mean_plddt_conf'], 1)} | "
                f"{fmt(r['iptm_conf'])} | {fmt(r['mean_plddt_conf'], 1)} | "
                f"{fmt(r['dockq_ref'])} |"
            )
            if d["total_loss"] is not None and r["total_loss"] is not None:
                if d["total_loss"] < r["total_loss"]:
                    wins_real["total_loss"] += 1
            if d["iptm_conf"] is not None and r["iptm_conf"] is not None:
                if d["iptm_conf"] > r["iptm_conf"]:
                    wins_real["iptm_conf"] += 1
                diffs_iptm_real.append(d["iptm_conf"] - r["iptm_conf"])
            if d["mean_plddt_conf"] is not None and r["mean_plddt_conf"] is not None:
                if d["mean_plddt_conf"] > r["mean_plddt_conf"]:
                    wins_real["mean_plddt_conf"] += 1
                diffs_plddt_real.append(d["mean_plddt_conf"] - r["mean_plddt_conf"])

        n_pairs = len(paired_dreal)
        lines.append(f"\n配对数: {n_pairs}\n")
        lines.append("| 指标 | designed 胜出次数 | 胜率 |")
        lines.append("|------|-----------------|------|")
        for k, w in wins_real.items():
            lines.append(f"| {k} | {w}/{n_pairs} | {w/n_pairs*100:.0f}% |")

        mean_diff_iptm_r, _, _ = compute_stats(diffs_iptm_real)
        mean_diff_plddt_r, _, _ = compute_stats(diffs_plddt_real)
        lines.append(f"\n平均 iPTM 差异 (designed - real): {fmt(mean_diff_iptm_r, 4)}")
        lines.append(f"平均 pLDDT 差异 (designed - real): {fmt(mean_diff_plddt_r, 1)}")
    else:
        lines.append("无可配对数据。\n")

    # ── DockQ 参考信息 ───────────────────────────────────────────────────
    real_with_dockq = [r for r in rows if r["seq_type"] == "real" and r["dockq_ref"] is not None]
    if real_with_dockq:
        lines.append("\n## DockQ 参考（ab-raw.csv，Protenix-v1 vs 实验 PDB）\n")
        dockq_vals = [r["dockq_ref"] for r in real_with_dockq]
        dq_mean, dq_std, dq_n = compute_stats(dockq_vals)
        lines.append(f"共 {dq_n} 个案例有 DockQ 参考值。")
        lines.append(f"DockQ 均值: {fmt(dq_mean)} +/- {fmt(dq_std)}\n")
        lines.append("DockQ 是 Protenix-v1 在完整真实序列上的预测结构与实验 PDB 结构的对比分数，")
        lines.append("代表该案例中 AF3 对真实抗体-抗原复合物的预测质量上界。")

    # ── 写出 ─────────────────────────────────────────────────────────────
    report_text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"报告已生成: {out_path}")
    print(report_text)


def main():
    csv_path = os.path.join(OUTPUT_DIR, "eval_results.csv")
    out_path = os.path.join(OUTPUT_DIR, "evaluation_report.md")
    if not os.path.exists(csv_path):
        logger.error(f"未找到评估结果: {csv_path}")
        logger.error("请先运行 run_eval.py")
        return
    generate_report(csv_path, out_path)


if __name__ == "__main__":
    main()
