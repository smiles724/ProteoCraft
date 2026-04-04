#!/usr/bin/env python3
"""
BindCraft v2 — 评估结果分析

支持两种评估结果的分析：

  1. zeroth 评估结果 (eval_results.csv):
     逐 entry 对比 designed/real_vh/real_full 三组，
     输出汇总统计和配对比较。输出 evaluation_report.md

  2. full ODE 评估结果 (full_eval_results.csv):
     高精度指标汇总：pLDDT / iPTM / pTM / gpde / chain_pair_iPTM，
     逐 entry 对比 designed/real_vh/real_full 三组。
     输出 full_evaluation_report.md

用法:
    python evaluation/analyze_results.py                     # 分析 zeroth 结果
    python evaluation/analyze_results.py --mode full         # 分析 full ODE 结果
    python evaluation/analyze_results.py --mode both         # 同时分析两种
"""

import os
import sys
import csv
import logging
import argparse
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.configs import OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# 通用工具函数
# ─────────────────────────────────────────────────────────────────

def compute_stats(values):
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return None, None, 0
    mean = sum(vals) / n
    if n > 1:
        std = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5
    else:
        std = 0.0
    return mean, std, n


def fmt(val, decimals=3):
    if val is None:
        return "-"
    if decimals == 1:
        return f"{val:.1f}"
    return f"{val:.{decimals}f}"


def _parse_float(val):
    if val == "" or val is None:
        return None
    try:
        return float(val)
    except ValueError:
        return None


# ═══════════════════════════════════════════════════════════════════
# 模式 1: zeroth 评估结果分析
# ═══════════════════════════════════════════════════════════════════

Z_LOSS_KEYS = ["total_loss",
    "plddt", "pae", "i_pae", "con", "i_con", "iptm", "rg",
    "helix", "clash", "i_clash", "pde", "i_pde"]
Z_CONF_KEYS = ["mean_plddt_conf", "iptm_conf", "ptm_conf"]
Z_REF_KEYS  = ["dockq_ref", "lddt_ref"]


def load_zeroth_csv(csv_path: str):
    """返回 list[dict]，所有数值列转为 float。"""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    ALL_NUMERIC = Z_LOSS_KEYS + Z_CONF_KEYS + Z_REF_KEYS + ["eval_time_sec"]
    for r in rows:
        for k in ALL_NUMERIC:
            r[k] = _parse_float(r.get(k, ""))
    return rows


def group_zeroth(rows):
    grouped = defaultdict(dict)
    for r in rows:
        grouped[r["entry_id"]][r["seq_type"]] = r
    return grouped


def _zeroth_summary_table(type_metrics, seq_types):
    lines = []
    lines.append(
        "| seq_type | n | "
        + " | ".join(
            f"{k}_mean±std" for k in ["total_loss", "plddt", "iptm", "mean_pLDDT", "ptm"]
        )
        + " |"
    )
    lines.append("|----------|---|" + "|".join(["-" * 14] * 5) + "|")
    for st in seq_types:
        m = type_metrics.get(st)
        if not m:
            continue
        _, _, n = compute_stats(m.get("total_loss", []))
        if n == 0:
            continue
        cells = [f"| {st} | {n}"]
        for k in ["total_loss", "plddt", "iptm", "mean_plddt_conf", "ptm_conf"]:
            mk = "mean_pLDDT" if k == "mean_plddt_conf" else k
            mm, ms, _ = compute_stats(m.get(k, []))
            cells.append(f"{fmt(mm,4)} ± {fmt(ms,4)}")
        lines.append(" | ".join(cells) + " |")
    return lines


def analyze_zeroth(csv_path: str, out_path: str):
    rows = load_zeroth_csv(csv_path)
    grouped = group_zeroth(rows)
    logger.info(f"[zeroth] {len(rows)} 行，{len(grouped)} 个 entry")

    lines = []
    lines.append("# Retrodiction 评估报告（zeroth 模式）\n")
    lines.append(f"数据文件: `{csv_path}`")
    lines.append(f"Entry 数: {len(grouped)}")
    lines.append("四组序列全部通过 AF3 完整 ODE 推断，用同一把尺子打分。\n")

    # ── 逐 entry 对比表 ──────────────────────────────────────────
    lines.append("## 逐案例对比\n")
    lines.append(
        "| entry_id | seq_type | L_b | L_t | "
        "total_loss | plddt | iptm | pTM | DockQ |"
    )
    lines.append(
        "|----------|----------|-----|-----|"
        "------------|-------|------|-----|-------|"
    )
    for eid in sorted(grouped):
        for st in ["designed", "real_vh", "real_full"]:
            r = grouped[eid].get(st)
            if not r:
                continue
            lines.append(
                f"| {eid} | {st} | "
                f"{r.get('binder_length','-')} | "
                f"{r.get('target_length','-')} | "
                f"{fmt(r['total_loss'])} | "
                f"{fmt(r['plddt'],4)} | "
                f"{fmt(r['iptm'],4)} | "
                f"{fmt(r.get('ptm_conf') or r.get('ptm',''),4)} | "
                f"{fmt(r.get('dockq_ref',''))} |"
            )

    # ── 汇总统计 ────────────────────────────────────────────────
    lines.append("\n## 汇总统计（12 项 loss + 置信度指标）\n")
    type_metrics = defaultdict(lambda: defaultdict(list))
    for r in rows:
        st = r["seq_type"]
        for k in Z_LOSS_KEYS + Z_CONF_KEYS:
            type_metrics[st][k].append(r[k])

    all_keys = ["total_loss"] + [
        k for k in Z_LOSS_KEYS if k != "total_loss"
    ] + Z_CONF_KEYS

    lines.append(
        "| seq_type | n | " + " | ".join(f"{k}_mean±std" for k in all_keys[:5]) + " |"
    )
    lines.append("|----------|---|" + "|".join(["-" * 16] * 6) + "|")
    for st in ["designed", "real_vh", "real_full"]:
        m = type_metrics.get(st)
        if not m:
            continue
        _, _, n = compute_stats(m.get("total_loss", []))
        if n == 0:
            continue
        cells = [f"| {st} | {n}"]
        for k in all_keys[:5]:
            mm, ms, _ = compute_stats(m.get(k, []))
            cells.append(f"{fmt(mm,3)}±{fmt(ms,3)}")
        lines.append(" | ".join(cells) + " |")

    # ── 配对比较: designed vs real_vh ─────────────────────────
    lines.append("\n## 配对比较: designed vs real_vh\n")
    paired_vh = [e for e in grouped
                 if "designed" in grouped[e] and "real_vh" in grouped[e]]
    if paired_vh:
        lines.append(
            "real_vh 使用仅重链（~115aa），与 designed 长度最接近。\n"
        )
        lines.append(
            "| entry_id | des_iPTM | des_pLDDT | "
            "real_iPTM | real_pLDDT | DockQ_ref |"
        )
        lines.append(
            "|----------|----------|-----------|"
            "------------|-------------|-----------|"
        )
        wins_vh = {k: 0 for k in ["total_loss", "iptm", "mean_plddt_conf"]}
        diffs_vh = {k: [] for k in ["iptm", "mean_plddt_conf"]}
        for eid in sorted(paired_vh):
            d, r_ = grouped[eid]["designed"], grouped[eid]["real_vh"]
            lines.append(
                f"| {eid} | "
                f"{fmt(d.get('iptm'))} | "
                f"{fmt(d.get('mean_plddt_conf'),1)} | "
                f"{fmt(r_.get('iptm'))} | "
                f"{fmt(r_.get('mean_plddt_conf'),1)} | "
                f"{fmt(r_.get('dockq_ref'))} |"
            )
            for k in wins_vh:
                dk, rk = d.get(k), r_.get(k)
                if dk is None or rk is None:
                    continue
                if k == "total_loss":
                    if dk < rk:
                        wins_vh[k] += 1
                else:
                    if dk > rk:
                        wins_vh[k] += 1
                    diffs_vh[k].append(dk - rk)

        n_p = len(paired_vh)
        lines.append(f"\n配对数: {n_p}\n")
        lines.append("| 指标 | designed 胜出 | 胜率 |")
        lines.append("|------|-------------|------|")
        for k, w in wins_vh.items():
            lines.append(f"| {k} | {w}/{n_p} | {w/n_p*100:.0f}% |")
        for k in ["iptm", "mean_plddt_conf"]:
            dm, _, _ = compute_stats(diffs_vh[k])
            lines.append(f"\n平均差异 (designed - real_vh): {k} = {fmt(dm,4)}")
    else:
        lines.append("无可配对数据（可能 target+binder 超过 VRAM 安全边界）。")

    # ── DockQ 参考 ─────────────────────────────────────────────
    real_dockq = [r["dockq_ref"] for r in rows
                  if r["seq_type"] == "real_full" and r["dockq_ref"] is not None]
    if real_dockq:
        dq_m, dq_s, dq_n = compute_stats(real_dockq)
        lines.append(
            f"\n## DockQ 参考（ab-raw.csv, {dq_n} 个 real_full 案例）\n"
        )
        lines.append(
            f"Protenix-v1 在完整真实序列上预测结构的 DockQ 均值: "
            f"{fmt(dq_m,4)} ± {fmt(dq_s,4)}。"
            "代表 AF3 对真实抗体-抗原复合物的预测质量上界。"
        )

    text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info(f"报告已生成: {out_path}")
    print(text)


# ═══════════════════════════════════════════════════════════════════
# 模式 2: full ODE 评估结果分析
# ═══════════════════════════════════════════════════════════════════

F_METRIC_KEYS = [
    "plddt_mean", "plddt_std",
    "iptm_mean", "iptm_std",
    "ptm_mean", "ptm_std",
    "ranking_score_mean", "ranking_score_std",
    "chain_pair_iptm_bt_mean", "chain_pair_iptm_bt_std",
    "gpde_mean", "gpde_std",
]


def load_full_csv(csv_path: str):
    """返回 list[dict]，所有数值列转为 float。"""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in F_METRIC_KEYS + ["binder_length", "target_length"]:
            r[k] = _parse_float(r.get(k, ""))
    return rows


def group_full(rows):
    grouped = defaultdict(dict)
    for r in rows:
        label = r.get("label", r.get("entry_id", "unknown"))
        grouped[label][r["seq_type"]] = r
    return grouped


def analyze_full(csv_path: str, out_path: str):
    rows = load_full_csv(csv_path)
    grouped = group_full(rows)
    logger.info(f"[full ODE] {len(rows)} 行，{len(grouped)} 个 entry/label")

    lines = []
    lines.append("# Full ODE 高精度评估报告\n")
    lines.append(f"数据文件: `{csv_path}`")
    lines.append(f"Entry/标签数: {len(grouped)}")
    lines.append(
        "以下指标均来自 Protenix 完整多步 ODE 扩散采样，独立于 BindCraft 优化目标。\n"
    )

    # ── 逐 entry 对比表 ───────────────────────────────────────
    lines.append("## 逐案例对比\n")
    lines.append(
        "| label | seq_type | L_b | L_t | "
        "pLDDT | iPTM | pTM | ranking | chain_iPTM | gpde |"
    )
    lines.append(
        "|-------|----------|-----|-----|"
        "-------|------|-----|--------|------------|------|"
    )
    for label in sorted(grouped):
        g = grouped[label]
        for st in ["designed", "real_vh", "real_full", "custom"]:
            r = g.get(st)
            if not r:
                continue
            lines.append(
                f"| {label} | {st} | "
                f"{int(r.get('binder_length', 0))} | "
                f"{int(r.get('target_length', 0))} | "
                f"{fmt(r.get('plddt_mean'), 1)} | "
                f"{fmt(r.get('iptm_mean'), 3)} | "
                f"{fmt(r.get('ptm_mean'), 3)} | "
                f"{fmt(r.get('ranking_score_mean'), 3)} | "
                f"{fmt(r.get('chain_pair_iptm_bt_mean'), 3)} | "
                f"{fmt(r.get('gpde_mean'), 3)} |"
            )

    # ── 汇总统计 ──────────────────────────────────────────────
    lines.append("\n## 汇总统计（按 seq_type 分组）\n")
    type_metrics = defaultdict(list)
    for r in rows:
        type_metrics[r["seq_type"]].append(r)

    # 主指标表
    main_keys = [
        ("plddt_mean",        "pLDDT"),
        ("iptm_mean",         "iPTM"),
        ("ptm_mean",          "pTM"),
        ("ranking_score_mean","ranking"),
        ("chain_pair_iptm_bt_mean", "chain_iPTM"),
        ("gpde_mean",         "gpde"),
    ]

    for st in ["designed", "real_vh", "real_full", "custom"]:
        grp = type_metrics.get(st, [])
        if not grp:
            continue
        lines.append(f"### {st}  (n={len(grp)})\n")
        lines.append(
            "| 指标 | mean ± std (min ~ max) |"
        )
        lines.append("|------|-------------------------|")
        for key, name in main_keys:
            vals = [_parse_float(r.get(key, "")) for r in grp]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            m = sum(vals) / len(vals)
            s = (sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5 \
                if len(vals) > 1 else 0.0
            mn = min(vals)
            mx = max(vals)
            lines.append(
                f"| {name} | {fmt(m,3)} ± {fmt(s,3)} "
                f"({fmt(mn,3)} ~ {fmt(mx,3)}) |"
            )
        lines.append("")

    # ── 配对比较: designed vs real_vh ─────────────────────────
    lines.append("\n## 配对比较: designed vs real_vh\n")
    paired_vh = [
        lbl for lbl in grouped
        if "designed" in grouped[lbl] and "real_vh" in grouped[lbl]
    ]
    if paired_vh:
        wins_vh = {k: 0 for k, _ in main_keys}
        diffs_vh = {k: [] for k, _ in main_keys}
        lines.append(
            "| label | des_iPTM | des_chain_iPTM | "
            "real_iPTM | real_chain_iPTM |"
        )
        lines.append(
            "|-------|----------|----------------|"
            "------------|-----------------|"
        )
        for lbl in sorted(paired_vh):
            d, r_ = grouped[lbl]["designed"], grouped[lbl]["real_vh"]
            lines.append(
                f"| {lbl} | "
                f"{fmt(d.get('iptm_mean'),3)} | "
                f"{fmt(d.get('chain_pair_iptm_bt_mean'),3)} | "
                f"{fmt(r_.get('iptm_mean'),3)} | "
                f"{fmt(r_.get('chain_pair_iptm_bt_mean'),3)} |"
            )
            for key, name in main_keys:
                dm = _parse_float(d.get(key, ""))
                rm = _parse_float(r_.get(key, ""))
                if dm is None or rm is None:
                    continue
                if key == "gpde_mean":
                    if dm < rm:
                        wins_vh[key] += 1
                    diffs_vh[key].append(dm - rm)
                else:
                    if dm > rm:
                        wins_vh[key] += 1
                    diffs_vh[key].append(dm - rm)

        n_p = len(paired_vh)
        lines.append(f"\n配对数: {n_p}\n")
        lines.append("| 指标 | designed 胜出 | 胜率 | 平均差异 |")
        lines.append("|------|-------------|------|----------|")
        for key, name in main_keys:
            w = wins_vh[key]
            di = diffs_vh[key]
            dm, _, _ = compute_stats(di)
            delta = f"{fmt(dm, 4)}"
            if key == "gpde_mean":
                lines.append(f"| {name} | {w}/{n_p} | {w/n_p*100:.0f}% | ↓{delta} |")
            else:
                lines.append(f"| {name} | {w}/{n_p} | {w/n_p*100:.0f}% | ↑{delta} |")
    else:
        lines.append("无可配对数据。")

    # ── 指标含义 ───────────────────────────────────────────────
    lines.append("\n## 指标含义\n")
    lines.append(
        "| 指标 | 范围 | 含义 |\n"
        "|------|------|------|\n"
        "| pLDDT | 0-100 | 结构置信度，≥70 为可信结构 |\n"
        "| iPTM | 0-1 | 接口 TM 分数，≥0.6 为可信界面 |\n"
        "| pTM | 0-1 | 整体 TM 分数，≥0.5 为可信整体折叠 |\n"
        "| ranking | 0-1 | Protenix 综合排名 = 0.8*iPTM + 0.2*pTM |\n"
        "| chain_iPTM | 0-1 | Binder→Target 链对接口置信度（最直接结合指标）|\n"
        "| gpde | 0-? | 全局预测 DockQ，≤3 为可信（<3 excellent, <6 good）|\n"
    )
    lines.append(
        "\n注：以上指标均来自 Protenix 完整多步 ODE 推理，"
        "独立于 BindCraft 优化目标。"
    )

    text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info(f"报告已生成: {out_path}")
    print(text)


# ═══════════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="BindCraft v2 — 评估结果分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", type=str, default="zeroth",
        choices=["zeroth", "full", "both"],
        help="zeroth=分析 eval_results.csv | full=分析 full_eval_results.csv | both=两者都分析"
    )
    args = parser.parse_args()

    zeroth_csv = os.path.join(OUTPUT_DIR, "eval_results.csv")
    full_csv   = os.path.join(OUTPUT_DIR, "full_eval_results.csv")

    if args.mode in ("zeroth", "both"):
        if os.path.exists(zeroth_csv):
            out = os.path.join(OUTPUT_DIR, "evaluation_report.md")
            logger.info(f"分析 zeroth 结果: {zeroth_csv}")
            analyze_zeroth(zeroth_csv, out)
        else:
            logger.warning(f"未找到: {zeroth_csv}（跳过 zeroth 分析）")

    if args.mode in ("full", "both"):
        if os.path.exists(full_csv):
            out = os.path.join(OUTPUT_DIR, "full_evaluation_report.md")
            logger.info(f"分析 full ODE 结果: {full_csv}")
            analyze_full(full_csv, out)
        else:
            logger.warning(f"未找到: {full_csv}（跳过 full ODE 分析）")


if __name__ == "__main__":
    main()
