#!/usr/bin/env python3
"""Build the final speed/quality report for the ESM optimization campaign.

The timing trace contains nested timers: ``proposal_update`` includes the
proposal and ESMC-prior forward calls emitted immediately before it.  This
script assigns the final 30 (Stage A) or 20 (Stage B) forward calls in each
update window to that update, then reports the uninstrumented update residual.
This keeps the wall-time decomposition disjoint.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARMS = ("baseline", "exact", "fast", "big")
COLORS = {
    "baseline": "#4C78A8",
    "exact": "#F58518",
    "fast": "#54A24B",
    "big": "#E45756",
}
CASE_LABELS = {
    "denovo_PD1": "PD1 binder (115+120 aa)",
    "tcr_1qrn_no_hotspot": "TCR 1QRN (443+120 aa)",
}
STAGE_A_UPDATES = 70
STAGE_A_SOFT_STEPS = 30
STAGE_B_UPDATES = 40
STAGE_B_SOFT_STEPS = 20
TOTAL_SOFT_STEPS = STAGE_A_UPDATES * STAGE_A_SOFT_STEPS + STAGE_B_UPDATES * STAGE_B_SOFT_STEPS
GIB = 1024**3


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else math.nan


def f(value: float, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{value:.{digits}f}"


def pct(value: float, digits: int = 1, signed: bool = False) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    flag = "+" if signed else ""
    return f"{value * 100:{flag}.{digits}f}%"


def md_table(headers: list[str], rows: list[list[Any]], align: list[str] | None = None) -> str:
    align = align or ["---"] * len(headers)
    result = ["| " + " | ".join(headers) + " |", "| " + " | ".join(align) + " |"]
    result.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    return "\n".join(result)


def parse_trace(path: Path, wall_s: float) -> dict[str, Any]:
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    phase_times: dict[str, list[float]] = defaultdict(list)
    for event in events:
        phase_times[event["phase"]].append(float(event["elapsed_s"]))

    update_events: list[dict[str, float]] = []
    window: list[dict[str, Any]] = []
    update_index = 0
    inner_proposal_total = 0.0
    inner_prior_total = 0.0
    for event in events:
        if event["phase"] != "proposal_update":
            window.append(event)
            continue
        expected = STAGE_A_SOFT_STEPS if update_index < STAGE_A_UPDATES else STAGE_B_SOFT_STEPS
        proposal = [float(x["elapsed_s"]) for x in window if x["phase"] == "proposal_forward"][-expected:]
        prior = [float(x["elapsed_s"]) for x in window if x["phase"] == "esmc_prior_forward"][-expected:]
        if len(proposal) != expected or len(prior) != expected:
            raise ValueError(f"Could not identify inner calls for update {update_index} in {path}")
        proposal_s = sum(proposal)
        prior_s = sum(prior)
        update_s = float(event["elapsed_s"])
        inner_proposal_total += proposal_s
        inner_prior_total += prior_s
        update_events.append(
            {
                "elapsed_s": update_s,
                "inner_proposal_s": proposal_s,
                "inner_prior_s": prior_s,
                "residual_s": update_s - proposal_s - prior_s,
            }
        )
        window = []
        update_index += 1

    if len(update_events) != STAGE_A_UPDATES + STAGE_B_UPDATES:
        raise ValueError(f"Expected 110 updates, found {len(update_events)} in {path}")

    verifier = phase_times["verifier"]
    proposal = phase_times["proposal_forward"]
    update_total = sum(phase_times["proposal_update"])
    verifier_total = sum(verifier)
    outside_proposal_total = sum(proposal) - inner_proposal_total
    other_s = wall_s - update_total - verifier_total - outside_proposal_total
    verifier_steady = verifier[1:] if len(verifier) > 1 else verifier
    cold_excess = max(0.0, verifier[0] - mean(verifier_steady)) if verifier else 0.0
    stage_a = update_events[:STAGE_A_UPDATES]
    stage_b = update_events[STAGE_A_UPDATES:]
    max_alloc = max(float(x.get("peak_allocated_bytes", 0)) for x in events) / GIB
    max_reserved = max(float(x.get("peak_reserved_bytes", 0)) for x in events) / GIB

    return {
        "update_calls": len(update_events),
        "soft_steps": TOTAL_SOFT_STEPS,
        "proposal_calls": len(proposal),
        "prior_calls": len(phase_times["esmc_prior_forward"]),
        "verifier_timed_calls": len(verifier),
        "wall_s": wall_s,
        "cold_adjusted_wall_s": wall_s - cold_excess,
        "cold_excess_s": cold_excess,
        "update_total_s": update_total,
        "stage_a_total_s": sum(x["elapsed_s"] for x in stage_a),
        "stage_a_mean_s": mean(x["elapsed_s"] for x in stage_a),
        "stage_a_median_s": statistics.median(x["elapsed_s"] for x in stage_a),
        "stage_b_total_s": sum(x["elapsed_s"] for x in stage_b),
        "stage_b_mean_s": mean(x["elapsed_s"] for x in stage_b),
        "stage_b_median_s": statistics.median(x["elapsed_s"] for x in stage_b),
        "inner_proposal_total_s": inner_proposal_total,
        "inner_prior_total_s": inner_prior_total,
        "update_residual_s": sum(x["residual_s"] for x in update_events),
        "update_residual_per_soft_step_s": sum(x["residual_s"] for x in update_events) / TOTAL_SOFT_STEPS,
        "proposal_all_total_s": sum(proposal),
        "proposal_mean_s": mean(proposal),
        "proposal_median_s": statistics.median(proposal),
        "prior_all_total_s": sum(phase_times["esmc_prior_forward"]),
        "prior_mean_s": mean(phase_times["esmc_prior_forward"]),
        "prior_median_s": statistics.median(phase_times["esmc_prior_forward"]),
        "verifier_total_s": verifier_total,
        "verifier_mean_s": mean(verifier),
        "verifier_median_s": statistics.median(verifier),
        "verifier_first_s": verifier[0],
        "verifier_steady_mean_s": mean(verifier_steady),
        "verifier_p95_s": percentile(verifier_steady, 95),
        "outside_proposal_total_s": outside_proposal_total,
        "other_s": other_s,
        "peak_allocated_gib": max_alloc,
        "peak_reserved_gib": max_reserved,
    }


def collect_timing(campaign_root: Path, report_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    official = {(x["arm"], x["case_id"]): x for x in report_rows}
    rows: list[dict[str, Any]] = []
    for case in CASE_LABELS:
        for arm in ARMS:
            info = official[(arm, case)]
            seed = info["seed"]
            trace = campaign_root / arm / case / f"seed-{seed}" / "optimization_report.timing.jsonl"
            parsed = parse_trace(trace, float(info["wall_s"]))
            rows.append(
                {
                    "case_id": case,
                    "arm": arm,
                    "unique_candidates": int(info["unique_candidates"]),
                    "reported_verifier_calls": int(info["verifier_calls"]),
                    **parsed,
                }
            )
    by_key = {(x["case_id"], x["arm"]): x for x in rows}
    for row in rows:
        baseline = by_key[(row["case_id"], "baseline")]
        row["wall_speedup"] = baseline["wall_s"] / row["wall_s"]
        row["cold_adjusted_speedup"] = baseline["cold_adjusted_wall_s"] / row["cold_adjusted_wall_s"]
        row["update_speedup"] = baseline["update_total_s"] / row["update_total_s"]
        row["stage_a_speedup"] = baseline["stage_a_mean_s"] / row["stage_a_mean_s"]
        row["stage_b_speedup"] = baseline["stage_b_mean_s"] / row["stage_b_mean_s"]
        row["proposal_speedup"] = baseline["proposal_mean_s"] / row["proposal_mean_s"]
        row["verifier_steady_speedup"] = baseline["verifier_steady_mean_s"] / row["verifier_steady_mean_s"]
    return rows


def sequence_statistics(sequences: list[str]) -> dict[str, float]:
    counts = Counter("".join(sequences))
    length = sum(counts.values())
    frequencies = [count / length for count in counts.values()]
    identities = [
        sum(a == b for a, b in zip(left, right)) / min(len(left), len(right))
        for left, right in itertools.combinations(sequences, 2)
    ]
    return {
        "composition_entropy_bits": -sum(p * math.log2(p) for p in frequencies),
        "charged_fraction": sum(counts[x] for x in "DEKR") / length,
        "hydrophobic_fraction": sum(counts[x] for x in "AVILMFWY") / length,
        "top5_pairwise_identity": mean(identities),
    }


def collect_quality(campaign_root: Path, report_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    official = {(x["arm"], x["case_id"]): x for x in report_rows}
    rows: list[dict[str, Any]] = []
    for case in CASE_LABELS:
        for arm in ARMS:
            base = official[(arm, case)]
            paths = sorted((campaign_root / "rescore" / arm / case / "seed-43").glob("rank-*/eval-*/score.json"))
            scores = [load_json(path) for path in paths]
            metrics = [x["metrics"] for x in scores]
            seq_by_rank: dict[int, str] = {}
            for score in scores:
                seq_by_rank[int(score["rank"])] = score["binder_seq"]
            seq_stats = sequence_statistics([seq_by_rank[x] for x in sorted(seq_by_rank)])
            rows.append(
                {
                    "case_id": case,
                    "arm": arm,
                    "samples": len(scores),
                    "top1_pair_chain_iptm": float(base["top1_pair_chain_iptm"]),
                    "best5_pair_chain_iptm": float(base["bestk_pair_chain_iptm"]),
                    "success_at_0.8": base["bestk_success"] == "True",
                    "all10_mean_pair_chain_iptm": mean(float(x["pair_chain_iptm"]) for x in metrics),
                    "all10_mean_binder_plddt": mean(float(x["binder_plddt"]) for x in metrics),
                    "hotspot_8a_coverage": mean(float(x["hotspot_heavy_atom_coverage"]) for x in metrics)
                    if case == "denovo_PD1"
                    else math.nan,
                    "hotspot_5a_coverage": mean(float(x["hotspot_heavy_atom_coverage_5a"]) for x in metrics)
                    if case == "denovo_PD1"
                    else math.nan,
                    "hotspot_mean_distance_a": mean(float(x["hotspot_mean_min_heavy_atom_distance"]) for x in metrics)
                    if case == "denovo_PD1"
                    else math.nan,
                    **seq_stats,
                }
            )
    by_key = {(x["case_id"], x["arm"]): x for x in rows}
    for row in rows:
        baseline = by_key[(row["case_id"], "baseline")]
        row["top1_delta_vs_baseline"] = row["top1_pair_chain_iptm"] - baseline["top1_pair_chain_iptm"]
        row["best5_delta_vs_baseline"] = row["best5_pair_chain_iptm"] - baseline["best5_pair_chain_iptm"]
    return rows


def collect_microbench(formal_root: Path, diagnosis_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root, source in ((formal_root, "formal"), (diagnosis_root, "diagnosis")):
        for path in sorted(root.glob("*/*/result.json")):
            result = load_json(path)
            differences = result.get("differences", {})
            diff = next(iter(differences.values()), {})
            rows.append(
                {
                    "source": source,
                    "phase": result["phase"],
                    "arm": result["arm"],
                    "median_s": float(result["median_s"]),
                    "speedup": float(result.get("speedup", 1.0)),
                    "first_call_s": result.get("first_call_s", math.nan),
                    "peak_allocated_gib": float(result["peak_allocated_bytes"]) / GIB,
                    "difference_name": next(iter(differences), ""),
                    "bitwise_equal": diff.get("bitwise_equal", ""),
                    "relative_l2": diff.get("relative_l2", math.nan),
                    "cosine": diff.get("cosine", math.nan),
                }
            )
    return rows


def collect_preliminary(experiment_root: Path) -> list[dict[str, Any]]:
    """Inventory superseded smoke runs so they are not silently mixed with final data."""
    rows: list[dict[str, Any]] = []
    for name in ("smoke", "smoke_repeat", "smoke_det_b200", "smoke_verify_b200", "micro_h200_formal"):
        for path in sorted((experiment_root / name).glob("*/*/result.json")):
            result = load_json(path)
            rows.append(
                {
                    "experiment": name,
                    "phase": result.get("phase", ""),
                    "arm": result.get("arm", ""),
                    "gpu": result.get("gpu", ""),
                    "torch": result.get("torch", ""),
                    "deterministic": result.get("deterministic", ""),
                    "status": result.get("status", ""),
                    "median_s": result.get("median_s", ""),
                    "speedup": result.get("speedup", ""),
                    "in_final_comparison": False,
                    "reason": "preliminary smoke or superseded software stack",
                }
            )
    rows.extend(
        [
            {
                "experiment": "micro_h100_formal", "phase": "", "arm": "baseline", "gpu": "H100 allocation",
                "torch": "", "deterministic": "", "status": "cancelled", "median_s": "", "speedup": "",
                "in_final_comparison": False, "reason": "job 122630 terminated before producing a result",
            },
            {
                "experiment": "paired_smoke", "phase": "campaign", "arm": "", "gpu": "", "torch": "",
                "deterministic": "", "status": "manifest_only", "median_s": "", "speedup": "",
                "in_final_comparison": False, "reason": "campaign manifest exists but no run result was produced",
            },
        ]
    )
    return rows


def make_plots(output: Path, timing: list[dict[str, Any]], quality: list[dict[str, Any]], micro: list[dict[str, Any]]) -> None:
    t = {(x["case_id"], x["arm"]): x for x in timing}
    q = {(x["case_id"], x["arm"]): x for x in quality}
    x = np.arange(len(ARMS))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for ax, case in zip(axes, CASE_LABELS):
        hours = [t[(case, arm)]["wall_s"] / 3600 for arm in ARMS]
        bars = ax.bar(x, hours, color=[COLORS[a] for a in ARMS])
        for bar, arm in zip(bars, ARMS):
            speed = t[(case, arm)]["wall_speedup"]
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{speed:.2f}×", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x, ARMS)
        ax.set_ylabel("Search wall time (hours)")
        ax.set_title(CASE_LABELS[case])
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(output / "wall_time_speedup.png", dpi=180)
    fig.savefig(output / "wall_time_speedup.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    components = [
        ("update_total_s", "Differentiable updates", "#4C78A8"),
        ("verifier_total_s", "Verifier", "#F58518"),
        ("outside_proposal_total_s", "Proposal outside update", "#54A24B"),
        ("other_s", "Load / Python / I/O", "#B8B8B8"),
    ]
    for ax, case in zip(axes, CASE_LABELS):
        bottom = np.zeros(len(ARMS))
        for key, label, color in components:
            values = np.asarray([t[(case, arm)][key] / 60 for arm in ARMS])
            ax.bar(x, values, bottom=bottom, label=label, color=color)
            bottom += values
        ax.set_xticks(x, ARMS)
        ax.set_ylabel("Minutes")
        ax.set_title(CASE_LABELS[case])
        ax.grid(axis="y", alpha=0.2)
    axes[1].legend(loc="upper left", fontsize=8)
    fig.savefig(output / "timing_breakdown.png", dpi=180)
    fig.savefig(output / "timing_breakdown.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, case in zip(axes, CASE_LABELS):
        for arm in ARMS[1:]:
            row_t, row_q = t[(case, arm)], q[(case, arm)]
            ax.scatter(row_t["wall_speedup"], row_q["best5_delta_vs_baseline"], s=90, color=COLORS[arm], label=arm)
            ax.annotate(arm, (row_t["wall_speedup"], row_q["best5_delta_vs_baseline"]), xytext=(5, 5), textcoords="offset points")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(-0.02, color="#E45756", linewidth=1, linestyle="--", label="−0.02 exploratory margin")
        ax.axvline(1, color="black", linewidth=0.8, linestyle=":")
        ax.set_xlabel("End-to-end speedup vs baseline")
        ax.set_ylabel("Best-of-5 pair-chain iPTM delta")
        ax.set_title(CASE_LABELS[case])
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.savefig(output / "quality_speed_tradeoff.png", dpi=180)
    fig.savefig(output / "quality_speed_tradeoff.pdf")
    plt.close(fig)

    selected = [x for x in micro if (x["source"], x["phase"]) in {("formal", "proposal"), ("formal", "verifier"), ("diagnosis", "prior"), ("diagnosis", "update")}]
    groups = defaultdict(list)
    for row in selected:
        groups[row["phase"]].append(row)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, phase in zip(axes.flat, ("proposal", "verifier", "prior", "update")):
        rows = groups[phase]
        labels = [x["arm"] for x in rows]
        values = [x["speedup"] for x in rows]
        ax.bar(np.arange(len(rows)), values, color=[COLORS.get(label.replace("proposal_", "").replace("verifier_", "").replace("esmc_", ""), "#777777") for label in labels])
        ax.axhline(1, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(rows)), labels, rotation=25, ha="right")
        ax.set_ylabel("Speedup vs phase baseline")
        ax.set_title(phase)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(output / "microbenchmark_speedups.png", dpi=180)
    fig.savefig(output / "microbenchmark_speedups.pdf")
    plt.close(fig)


def build_report(
    campaign_root: Path,
    timing: list[dict[str, Any]],
    quality: list[dict[str, Any]],
    micro: list[dict[str, Any]],
    preliminary: list[dict[str, Any]],
) -> str:
    t = {(x["case_id"], x["arm"]): x for x in timing}
    q = {(x["case_id"], x["arm"]): x for x in quality}
    m = {(x["source"], x["phase"], x["arm"]): x for x in micro}
    lines: list[str] = []
    lines += [
        "# ProteoCraft ESM 路线优化：最终速度与质量报告",
        "",
        "生成日期：2026-09-23  ",
        f"实验目录：`{campaign_root}`",
        "",
        "## 结论",
        "",
        "八个端到端搜索、统一盲重打分和 exact 定位微基准均已完成，没有失败任务。当前两目标结果支持把 **fast** 作为下一轮扩大验证的首选：PD1 加速 1.24×、TCR 加速 3.94×，两例 best-of-5 pair-chain iPTM 相对 baseline 分别为 −0.0180 和 +0.0234。它在预先使用的 −0.02 探索性容差内，但样本只有两个目标、每个目标一个搜索 seed，因此还不足以把 fast 设为默认生产配置。",
        "",
        "**big** 在 PD1 最快（1.38×），在 TCR 为 3.79×，但 PD1 的 top-1/best-of-5 iPTM 分别下降 0.0277/0.0383，且没有降低峰值显存；综合收益不如 fast。**exact** 的独立 proposal/verifier forward 有小幅加速，但完整可微更新变慢，导致 PD1 总时间增加 18.3%，TCR 增加 4.1%；它适合作为等价性对照，不适合作为速度配置。",
        "",
        "所有方案的 best-of-5 pair-chain iPTM 都低于 0.8 成功阈值，所以这些结果比较的是相对搜索质量，不能解释为已经获得可实验验证的 binder。",
        "",
        "## 实验设计",
        "",
        "- 任务覆盖一个普通单靶点 binder（PD1：115 aa 靶蛋白 + 120 aa 新 binder，4 个热点）和一个双链 TCR binder（1QRN：200 aa α 链 + 243 aa β 链 + 120 aa 新 binder，无指定热点）。",
        "- 每个任务运行 baseline、exact、fast、big，搜索 seed 固定为 43。每次搜索含 70 个 Stage A 更新（每次 30 个 soft step）和 40 个 Stage B 更新（每次 20 个 soft step），共 110 次更新、2,900 个 soft step。",
        "- 每个 arm/任务取搜索排名前 5 的唯一序列，用共同的、未优化 ESMFold2 verifier，按 eval seed 1043/1044 各重打分一次，共 10 个盲评样本。这样质量比较不依赖各 arm 自己的优化 verifier。",
        "- baseline 为 proposal/verifier/ESMC 全关；exact 为三者 exact；fast 为 proposal/verifier fast + ESMC exact；big 为 proposal/verifier big + ESMC exact。运行环境为 H200、PyTorch 2.11.0+cu128、Triton 3.6.0。当前适配层不是 upstream 的完整 mode，报告中的 arm 指 ProteoCraft 实际组合。",
        "",
        "### 执行状态与纳入范围",
        "",
        md_table(
            ["实验", "Slurm job", "状态/耗时", "纳入用途"],
            [
                ["H200 当前栈独立微基准", "122639", "COMPLETED / 3:56", "proposal/verifier 正式微基准"],
                ["2任务×4 arm 搜索", "122640[0-7]", "8/8 COMPLETED / 1:07–9:29", "端到端速度、搜索轨迹"],
                ["共同 evaluator 盲重打分", "122641", "COMPLETED / 8:03", "正式质量比较"],
                ["exact 路径定位", "122911", "COMPLETED / 29:28", "prior/update 原因诊断"],
            ],
        ),
        "",
        "更早的 B200/PyTorch 2.7 smoke 仅用于验证接线：proposal_exact 曾测得 1.05×/1.15×、ESMC prior exact 1.78×，但 verifier_exact 只有 0.76×；早期 H200/PyTorch 2.7 的 proposal_exact 为 0.81×，proposal_fast 运行失败。H100 job 122630 在产出结果前被终止，`paired_smoke` 只有 manifest、没有运行结果。这些结果的 GPU、软件栈、确定性设置或完成状态不同，不能和 H200/PyTorch 2.11 正式数据合并。明细保存在 `preliminary_experiment_inventory.csv`。这也解释了为何早期看到的“ESMC exact 变快”不能外推到当前完整可微更新。",
        "",
        "## 1. 端到端速度",
        "",
    ]
    rows = []
    for case in CASE_LABELS:
        base = t[(case, "baseline")]
        for arm in ARMS:
            row = t[(case, arm)]
            delta_min = (row["wall_s"] - base["wall_s"]) / 60
            rows.append(
                [
                    CASE_LABELS[case],
                    arm,
                    f(row["wall_s"] / 60, 2),
                    f(row["wall_speedup"], 3) + "×",
                    ("+" if delta_min >= 0 else "") + f(delta_min, 2),
                    row["unique_candidates"],
                    row["reported_verifier_calls"],
                ]
            )
    lines += [
        md_table(
            ["任务", "arm", "墙钟 (min)", "相对 baseline", "时间差 (min)", "唯一候选", "报告 verifier 次数"],
            rows,
            ["---", "---", "---:", "---:", "---:", "---:", "---:"],
        ),
        "",
        "PD1 是较短输入，fast 节省 17.89 分钟，big 节省 25.23 分钟；TCR 总输入从 235 个残基增至 563 个残基，baseline 的二次/高阶结构模块代价急剧增加，fast 和 big 分别节省约 407.61 和 402.20 分钟。优化收益随复合物长度明显放大。",
        "",
        "### 冷启动敏感性",
        "",
    ]
    cold_rows = []
    for case in CASE_LABELS:
        for arm in ARMS:
            row = t[(case, arm)]
            cold_rows.append(
                [
                    CASE_LABELS[case],
                    arm,
                    f(row["verifier_first_s"], 2),
                    f(row["verifier_steady_mean_s"], 3),
                    f(row["cold_excess_s"], 2),
                    f(row["cold_adjusted_wall_s"] / 60, 2),
                    f(row["cold_adjusted_speedup"], 3) + "×",
                ]
            )
    lines += [
        md_table(
            ["任务", "arm", "首次 verifier (s)", "后续均值 (s)", "冷启动超额 (s)", "校正墙钟 (min)", "校正加速比"],
            cold_rows,
            ["---", "---", "---:", "---:", "---:", "---:", "---:"],
        ),
        "",
        "这里的“校正”只把第一次 verifier 替换为该运行其余 verifier 的均值，用于解释节点缓存差异，不替代原始墙钟。PD1 baseline/fast 首次调用约 311/313 秒，而 exact/big 约 15–16 秒，说明四个作业没有处于相同文件缓存状态。校正后 PD1 fast 为 1.259×、big 为 1.304×；big 仍快，但和 fast 的差距从原始 7.6 分钟缩小到约 2.4 分钟。TCR exact 的首次调用也有约 255 秒冷启动；校正后仍比 baseline 慢约 3.4%。",
        "",
        "![端到端墙钟](wall_time_speedup.png)",
        "",
        "## 2. 搜索内部时间分解",
        "",
        "`proposal_update` 是外层计时器，包含其内部的 proposal forward 和 ESMC prior forward。下表把三者拆开，并将其余 update 时间记为 residual；residual 包含结构/先验反向传播、one-step diffusion、梯度组合、同步和少量 Python 开销。`other` 为墙钟扣除 update、verifier 及 update 外 proposal 后的余量，主要含加载、编译、未单独计时的 Python/I/O。各列为互斥墙钟分解，避免把嵌套计时重复相加。",
        "",
    ]
    breakdown_rows = []
    for case in CASE_LABELS:
        for arm in ARMS:
            row = t[(case, arm)]
            breakdown_rows.append(
                [
                    CASE_LABELS[case], arm,
                    f(row["update_total_s"] / 60, 2),
                    f(row["verifier_total_s"] / 60, 2),
                    f(row["outside_proposal_total_s"] / 60, 2),
                    f(row["other_s"] / 60, 2),
                    pct(row["update_total_s"] / row["wall_s"]),
                    pct(row["verifier_total_s"] / row["wall_s"]),
                ]
            )
    lines += [
        md_table(
            ["任务", "arm", "更新 (min)", "verifier (min)", "更新外 proposal (min)", "其它 (min)", "更新占比", "verifier 占比"],
            breakdown_rows,
            ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:"],
        ),
        "",
        "![时间分解](timing_breakdown.png)",
        "",
        "### Stage A / Stage B 与模型调用",
        "",
    ]
    stage_rows = []
    for case in CASE_LABELS:
        for arm in ARMS:
            row = t[(case, arm)]
            stage_rows.append(
                [
                    CASE_LABELS[case], arm,
                    f(row["stage_a_mean_s"], 2), f(row["stage_a_speedup"], 3) + "×",
                    f(row["stage_b_mean_s"], 2), f(row["stage_b_speedup"], 3) + "×",
                    f(row["proposal_mean_s"], 3), f(row["proposal_speedup"], 3) + "×",
                    f(row["prior_mean_s"], 3),
                    f(row["verifier_steady_mean_s"], 3), f(row["verifier_steady_speedup"], 3) + "×",
                ]
            )
    lines += [
        md_table(
            ["任务", "arm", "Stage A/更新 (s)", "A 加速", "Stage B/更新 (s)", "B 加速", "proposal/call (s)", "proposal 加速", "prior/call (s)", "verifier 稳态 (s)", "verifier 加速"],
            stage_rows,
            ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
        ),
        "",
        "每个 run 固定有 3,340 次 proposal forward，其中 2,900 次在 update 内、440 次在 update 外；ESMC prior 固定 2,900 次。verifier 次数依赖 MCTS 路径，所以不同 arm 会在 756–843 次之间变化。PD1 fast/big 的完整 update 仅加速 1.15×/1.25×，TCR 则达到 3.99×/3.87×。这比单个短 PD1 forward 更能说明优化的主要适用场景是长复合物。",
        "",
        "更新 residual 同样显著下降：TCR baseline 每 soft step 为 " + f(t[("tcr_1qrn_no_hotspot", "baseline")]["update_residual_per_soft_step_s"], 3) + " 秒，fast/big 为 " + f(t[("tcr_1qrn_no_hotspot", "fast")]["update_residual_per_soft_step_s"], 3) + "/" + f(t[("tcr_1qrn_no_hotspot", "big")]["update_residual_per_soft_step_s"], 3) + " 秒。这表明收益不只是 forward，也来自 fast/big 的反向核、checkpoint/block 策略和 diffusion 路径。",
        "",
        "## 3. 独立微基准与 exact 变慢原因",
        "",
    ]
    micro_rows = []
    desired = [
        ("formal", "proposal", "baseline"), ("formal", "proposal", "proposal_exact"), ("formal", "proposal", "proposal_fast"),
        ("formal", "verifier", "baseline"), ("formal", "verifier", "verifier_exact"), ("formal", "verifier", "verifier_fast"), ("formal", "verifier", "big"),
        ("diagnosis", "prior", "baseline"), ("diagnosis", "prior", "esmc_exact"),
        ("diagnosis", "update", "baseline"), ("diagnosis", "update", "proposal_exact"), ("diagnosis", "update", "esmc_exact"), ("diagnosis", "update", "exact"), ("diagnosis", "update", "proposal_fast"), ("diagnosis", "update", "fast"), ("diagnosis", "update", "big"),
    ]
    for key in desired:
        row = m[key]
        parity = ""
        if row["bitwise_equal"] != "":
            parity = "bitwise" if row["bitwise_equal"] else f"rel-L2 {f(row['relative_l2'], 4)}"
        micro_rows.append([row["phase"], row["arm"], f(row["median_s"], 4), f(row["speedup"], 3) + "×", f(row["peak_allocated_gib"], 2), parity or "—"])
    lines += [
        md_table(
            ["phase", "arm", "中位时间 (s)", "加速比", "峰值分配 (GiB)", "相对 baseline"],
            micro_rows,
            ["---", "---", "---:", "---:", "---:", "---"],
        ),
        "",
        "正式 proposal 微基准中，proposal_exact 的 distogram 和梯度逐位一致，但比 baseline 慢 19.5%；proposal_fast 加速 1.74×，distogram/梯度相对 L2 差异为 0.0234/0.0506，余弦相似度为 0.99973/0.99893。verifier_exact 坐标逐位一致但慢 2.8%；verifier_fast 和 big 分别加速 1.91×/1.51×，fast 坐标相对 L2 为 0.0135、余弦相似度 0.99991。",
        "",
        "exact 定位实验给出关键原因：带梯度的 ESMC prior 从 0.0815 秒增至 0.2882 秒，即慢 3.54×；30-step 完整 update 中，仅启用 proposal_exact、仅启用 esmc_exact 和全部 exact 都约慢 21%。当前 exact 的无梯度 QK 路径可以融合，但梯度路径仍走 stock attention，并增加 exact differentiable rotary，反向传播抵消了 forward 的小幅收益。多个 exact 开关的代价不简单相加，因为它们改变的是同一条关键路径，并受 kernel 编译、调度与同步影响。",
        "",
        "定位 update 使用 one-step diffusion interval=1，是强调反向路径的压力测试；正式 campaign 使用 interval=5，不能把 57.18/38.66 秒直接当作 campaign 每次 update 时间。定位 JSON 中 logits 差异为 NaN，是比较器对 120 个约束位置的 `-inf - -inf` 所致，并非模型输出出现 NaN；因此该文件不能证明 update 张量 exact parity，但 proposal/verifier 的独立逐位 parity 结论有效。",
        "",
        "![微基准加速](microbenchmark_speedups.png)",
        "",
        "## 4. 统一盲评质量",
        "",
    ]
    quality_rows = []
    for case in CASE_LABELS:
        for arm in ARMS:
            row = q[(case, arm)]
            quality_rows.append(
                [
                    CASE_LABELS[case], arm,
                    f(row["top1_pair_chain_iptm"], 4), f(row["top1_delta_vs_baseline"], 4),
                    f(row["best5_pair_chain_iptm"], 4), f(row["best5_delta_vs_baseline"], 4),
                    f(row["all10_mean_pair_chain_iptm"], 4), f(row["all10_mean_binder_plddt"], 4),
                    "是" if row["success_at_0.8"] else "否",
                ]
            )
    lines += [
        md_table(
            ["任务", "arm", "top-1 iPTM", "Δ", "best-of-5 iPTM", "Δ", "10样本均值 iPTM", "binder pLDDT", "≥0.8"],
            quality_rows,
            ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---"],
        ),
        "",
        "PD1 上，fast 的 top-1/best-of-5 分别下降 0.0080/0.0180，仍在 −0.02 探索性容差内；big 下降 0.0277/0.0383，超出容差。TCR 上三个优化 arm 都高于 baseline，big 的 best-of-5 最高（+0.0356），fast 次之（+0.0234）。exact 在 PD1 略降而在 TCR 上升，说明即使 kernel 在局部逐位一致，完整 MCTS 在 GPU 非确定性、候选排序边界和分支选择作用下仍可能走向不同轨迹；不能把这次 exact 的 TCR 提升解释为模型本身提升了精度。",
        "",
        "由于只有两个 source target，现有汇总所谓 95% bootstrap 区间实质上只是两个观测端点。例如 fast 的 best-of-5 Δ 跨 [−0.0180, +0.0234]，无法做正式非劣效检验。",
        "",
        "![速度质量权衡](quality_speed_tradeoff.png)",
        "",
        "### PD1 热点几何与序列性质",
        "",
    ]
    seq_rows = []
    for case in CASE_LABELS:
        for arm in ARMS:
            row = q[(case, arm)]
            seq_rows.append(
                [
                    CASE_LABELS[case], arm,
                    f(row["hotspot_8a_coverage"], 3), f(row["hotspot_5a_coverage"], 3), f(row["hotspot_mean_distance_a"], 3),
                    f(row["composition_entropy_bits"], 3), pct(row["charged_fraction"]), pct(row["hydrophobic_fraction"]), pct(row["top5_pairwise_identity"]),
                ]
            )
    lines += [
        md_table(
            ["任务", "arm", "热点≤8Å", "热点≤5Å", "热点距离Å", "组成熵 bit", "带电比例", "疏水比例", "top5 两两同一性"],
            seq_rows,
            ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
        ),
        "",
        "PD1 fast 的 5 Å 热点覆盖为 1.00、平均最近距离 3.75 Å，是四组里最好，但其组成熵最低且带电残基达到 31%，提示候选偏向较窄的序列组成。PD1 big 的 binder pLDDT 很高，却有较弱的界面 iPTM，说明“单体折叠得自信”不能替代界面质量。",
        "",
        "除 PD1 exact 外，各 arm 的 top-5 两两序列同一性都在 98%–99%以上，因此 best-of-5 大多是同一局部解附近的近克隆，不等于五个独立设计。TCR exact 的 top-5 疏水比例为 54%，rank-1 序列出现长串疏水残基，具有明显膜蛋白样/低可溶性风险；其 iPTM 较高不能单独作为可开发性证据。",
        "",
        "## 5. 显存",
        "",
    ]
    memory_rows = []
    for case in CASE_LABELS:
        for arm in ARMS:
            row = t[(case, arm)]
            memory_rows.append([CASE_LABELS[case], arm, f(row["peak_allocated_gib"], 2), f(row["peak_reserved_gib"], 2)])
    lines += [
        md_table(["任务", "arm", "峰值 allocated GiB", "峰值 reserved GiB"], memory_rows, ["---", "---", "---:", "---:"]),
        "",
        "big 没有实现预期的端到端显存降低：PD1 allocated 为 67.35 GiB（baseline 67.09），TCR 为 75.94 GiB（baseline 75.55）；fast 也略高。该指标是进程累计峰值，且同一进程常驻 proposal、prior、verifier 多个模型，所以不能隔离某个 kernel 的瞬时显存，但足以说明在当前 ProteoCraft 组合中 big 没有扩展可运行长度的实证。fast/big 的 reserved 显存明显更高，反映缓存分配策略而非实际同时使用的张量。",
        "",
        "## 6. 原因归纳",
        "",
        "1. **长度决定优化收益。** PD1 总长 235 aa 时，kernel 启动、Python 和固定开销占比较大；TCR 总长 563 aa 时，pair/triangle 和 attention 的计算量放大，fast/big 的融合反向核与块化策略把 proposal 和 verifier 稳态分别提升到约 6.6×和 4.3–4.5×。",
        "2. **单 forward 加速不等于完整 update 加速。** update 还含 2,900 个 prior、结构反向、diffusion、梯度组合和同步。exact 的 proposal/verifier forward 分别只有约 1.13×/1.03–1.12×实跑收益，而带梯度 ESMC prior 和完整反向更慢，最终墙钟没有收益。",
        "3. **fast/big 会改变搜索轨迹。** 近似 reduction 和反向核虽保持很高余弦相似度，但 MCTS 是离散分支系统；很小的 score/gradient 差异可改变候选排序，之后的序列完全分叉。因此质量必须用统一外部 evaluator 比较，不能只看张量误差。",
        "4. **big 的 chunk=32/图策略不是普遍更快。** big 在短 PD1 略胜 fast，在长 TCR 反而略慢，也未降低进程峰值显存；它的块化、CUDA graph 取舍与当前长度/形状不完全匹配。",
        "5. **质量方向具有任务依赖。** PD1 优化 arm 下降、TCR 优化 arm 上升，且只有一个 seed。当前结果更符合随机/轨迹差异，而不是某个近似模式系统性提高或降低生成质量。",
        "",
        "## 7. 建议",
        "",
        "- 保留 baseline 为生产默认；把 fast 作为候选配置扩大到至少 5–10 个普通 binder、5–10 个 TCR/多链目标，每目标至少 3 个 search seed。用目标等权方式报告速度和质量，避免长 TCR 因运行时间大而主导均值。",
        "- 下一轮为所有 arm 统一做模型预热或预先填充文件缓存，并记录 model load、compile/warm-up、Stage A、Stage B、search verifier、最终 rescore 六段墙钟。当前 trace 已能解释主体，但 `other` 仍把加载、编译与 Python/I/O 混在一起。",
        "- 质量门槛除 pair-chain iPTM 外，加入界面 PAE、clash、热点几何、疏水/电荷/低复杂度过滤和聚类后的有效多样性；再用独立 AF3/AtlasFold-M 复核。当前只有共同 ESMFold2 evaluator，尚无实验结合数据。",
        "- 修正 exact ESMC 的梯度路径：避免 exact differentiable rotary + stock attention 的慢反向，或只在无梯度场景启用当前 fused QK。修复 `-inf` 掩码感知的 logits comparator 后，再做完整 update parity。",
        "- 暂不推广 big。若目标是超长复合物或显存边界，应对 chunk size、CUDA graph 和 checkpoint block 单独扫参，并以进程峰值和最大可运行长度验证，而不是从 arm 名称推断它更省显存。",
        "",
        "## 8. 可复现文件",
        "",
        "- `run_timing_breakdown.csv`：8 个搜索的完整时间分解、调用次数、冷启动与显存。",
        "- `quality_summary.csv`：统一盲评、热点几何与序列组成。",
        "- `microbenchmark_summary.csv`：proposal/verifier/prior/update 微基准与数值差异。",
        "- `preliminary_experiment_inventory.csv`：早期 smoke、失败/取消任务及其排除原因。",
        "- `wall_time_speedup.*`、`timing_breakdown.*`、`quality_speed_tradeoff.*`、`microbenchmark_speedups.*`：PNG/PDF 图。",
        "",
        "分析边界：这是 2 个目标 × 1 个搜索 seed 的探索性实验；没有独立结构模型复核和湿实验验证。原始墙钟来自每个搜索自身记录，作业所在节点和缓存状态不完全一致。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, default=Path("outputs/esm_optimization/pilot_2case_43"))
    parser.add_argument("--formal-micro-root", type=Path, default=Path("outputs/esm_optimization/micro_h200_torch211"))
    parser.add_argument("--diagnosis-root", type=Path, default=Path("outputs/esm_optimization/micro_h200_exact_diagnosis"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    campaign_root = args.campaign_root.resolve()
    output = (args.output or campaign_root / "complete_analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_rows = read_csv(campaign_root / "pilot_report.csv")
    timing = collect_timing(campaign_root, report_rows)
    quality = collect_quality(campaign_root, report_rows)
    micro = collect_microbench(args.formal_micro_root.resolve(), args.diagnosis_root.resolve())
    preliminary = collect_preliminary(campaign_root.parent)
    write_csv(output / "run_timing_breakdown.csv", timing)
    write_csv(output / "quality_summary.csv", quality)
    write_csv(output / "microbenchmark_summary.csv", micro)
    write_csv(output / "preliminary_experiment_inventory.csv", preliminary)
    make_plots(output, timing, quality, micro)
    report = build_report(campaign_root, timing, quality, micro, preliminary)
    (output / "full_report_zh.md").write_text(report)
    print(json.dumps({"output": str(output), "timing_rows": len(timing), "quality_rows": len(quality), "micro_rows": len(micro), "preliminary_rows": len(preliminary)}, indent=2))


if __name__ == "__main__":
    main()
