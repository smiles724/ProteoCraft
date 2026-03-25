#!/usr/bin/env python3
"""
BindCraft v2 - tau_noise 扫描实验可视化

解析三组实验日志（tau=4/20/80），画出各 loss 分量随优化步骤的变化曲线。

用法：
    python scripts/plot_tau_sweep.py
    python scripts/plot_tau_sweep.py --log_dir logs --output_dir outputs/plots
"""

import re
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── 样式配置 ──────────────────────────────────────────────────────────────────
TAU_CONFIGS = {
    "tau=4.0":  {"tau": "4.0",  "color": "#2196F3", "ls": "-",  "label": "tau=4.0  (low noise)"},
    "tau=20.0": {"tau": "20.0", "color": "#FF9800", "ls": "--", "label": "tau=20.0 (mid noise)"},
    "tau=80.0": {"tau": "80.0", "color": "#F44336", "ls": "-.", "label": "tau=80.0 (high noise)"},
}

METRIC_LABELS = {
    "total":       "Total Loss",
    "attraction":  "Attraction",
    "clash":       "Clash",
    "intra_clash": "Intra-Clash",
    "compactness": "Compactness",
    "rg":          "Radius of Gyration (A)",
    "iptm":        "iPTM (negative = maximize)",
    "grad":        "Grad Norm",
}

# Step 行正则：匹配 "Step  XX/YYY | total: V | ... | grad: V | best@N"
STEP_RE = re.compile(
    r"Step\s+(\d+)/\d+\s+\|"
    r"\s+total:\s*([\d.\-]+)\s*\|"
    r"\s+attraction:\s*([\d.\-]+)\s*\|"
    r"\s+clash:\s*([\d.\-]+)\s*\|"
    r"\s+intra_clash:\s*([\d.\-]+)\s*\|"
    r"\s+compactness:\s*([\d.\-]+)\s*\|"
    r"\s+rg:\s*([\d.\-]+)\s*\|"
    r"\s+iptm:\s*([\d.\-]+)\s*\|"
    r".*?grad:\s*([\d.\-]+)"
)
BEST_RE = re.compile(r"loss=([\d.]+)\s+@step\s+(\d+)")
TAU_HEADER_RE = re.compile(r"tau_noise\s*=\s*([\d.]+)")


# ── 解析函数 ──────────────────────────────────────────────────────────────────

def parse_sweep_out(out_path: Path) -> dict:
    """
    从 tau_sweep_*.out 文件中解析三组实验数据。
    三组数据以 '>>> tau_noise = X' 分隔。
    """
    results = {}
    current_tau = None
    current_data = defaultdict(list)
    current_steps = []
    current_best_loss = None
    current_best_step = None

    def flush():
        if current_tau is not None:
            key = f"tau={current_tau}"
            results[key] = {
                "steps": np.array(current_steps),
                "data":  {k: np.array(v) for k, v in current_data.items()},
                "best_loss": current_best_loss,
                "best_step": current_best_step,
            }

    with open(out_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # 检测新的 tau 段开始
            th = TAU_HEADER_RE.search(line)
            if th:
                flush()
                current_tau = th.group(1)
                current_data = defaultdict(list)
                current_steps = []
                current_best_loss = None
                current_best_step = None
                continue

            # 解析 Step 行
            m = STEP_RE.search(line)
            if m:
                current_steps.append(int(m.group(1)))
                current_data["total"].append(float(m.group(2)))
                current_data["attraction"].append(float(m.group(3)))
                current_data["clash"].append(float(m.group(4)))
                current_data["intra_clash"].append(float(m.group(5)))
                current_data["compactness"].append(float(m.group(6)))
                current_data["rg"].append(float(m.group(7)))
                current_data["iptm"].append(float(m.group(8)))
                current_data["grad"].append(float(m.group(9)))
                continue

            # 解析 best 行
            bm = BEST_RE.search(line)
            if bm and current_tau is not None:
                current_best_loss = float(bm.group(1))
                current_best_step = int(bm.group(2))

    flush()
    return results


# ── 绘图函数 ──────────────────────────────────────────────────────────────────

def _draw_metric(ax, parsed, metric, tau_configs):
    """在单个 ax 上画一个 metric 的三条曲线。"""
    for key, cfg in tau_configs.items():
        p = parsed.get(key)
        if p is None:
            continue
        steps = p["steps"]
        vals  = p["data"].get(metric)
        if vals is None or len(vals) == 0:
            continue

        ax.plot(steps, vals,
                color=cfg["color"], ls=cfg["ls"],
                linewidth=1.5, alpha=0.85,
                label=cfg["label"])


    ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10, fontweight="bold")
    ax.set_xlabel("Optimization Step", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(fontsize=7, loc="best")


def plot_all(parsed: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 图 1：2×3 子图，6 个 metric ───────────────────────────────────────────
    plot_metrics = ["total", "attraction", "compactness", "rg", "iptm", "grad"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("BindCraft v2 - tau_noise Sweep: All Metrics vs Optimization Steps",
                 fontsize=13, fontweight="bold")

    for ax, metric in zip(axes.flat, plot_metrics):
        _draw_metric(ax, parsed, metric, TAU_CONFIGS)
        if metric == "grad":
            ax.set_yscale("log")
            ax.set_title("Grad Norm (log scale)", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out1 = output_dir / "tau_sweep_metrics.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 1] saved: {out1}")

    # ── 图 2：total loss 单图 + 滑动平均 ──────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.set_title("Total Loss Comparison Across tau_noise Values",
                  fontsize=12, fontweight="bold")

    for key, cfg in TAU_CONFIGS.items():
        p = parsed.get(key)
        if p is None:
            continue
        steps = p["steps"]
        vals  = p["data"].get("total")
        if vals is None or len(vals) == 0:
            continue

        # 原始曲线（半透明）
        ax2.plot(steps, vals, color=cfg["color"], ls=cfg["ls"],
                 linewidth=1.0, alpha=0.30)

        # 滑动平均（窗口=5，mode="same" 从第一步就有输出）
        w = min(5, len(vals))
        smoothed     = np.convolve(vals, np.ones(w)/w, mode="same")
        smooth_steps = steps
        ax2.plot(smooth_steps, smoothed,
                 color=cfg["color"], ls=cfg["ls"], linewidth=2.2, alpha=0.95,
                 label=f"{cfg['label']}  best={p['best_loss']:.2f} @step {p['best_step']}")


    ax2.set_xlabel("Optimization Step", fontsize=10)
    ax2.set_ylabel("Total Loss", fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle=":")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.text(0.02, 0.97, "thin=raw  thick=5-step moving avg (mode=same)",
             transform=ax2.transAxes, fontsize=7, va="top", color="gray")

    plt.tight_layout()
    out2 = output_dir / "tau_sweep_total_loss.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[Fig 2] saved: {out2}")

    # ── 图 3：iPTM + attraction 关键指标 ──────────────────────────────────────
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
    fig3.suptitle("Key Metrics: iPTM & Attraction vs Steps", fontsize=11, fontweight="bold")

    ylabels = {
        "iptm":       "iPTM loss (negative, higher=better)",
        "attraction": "Attraction loss (lower=closer to target)",
    }
    for ax, metric in zip(axes3, ["iptm", "attraction"]):
        _draw_metric(ax, parsed, metric, TAU_CONFIGS)
        ax.set_ylabel(ylabels[metric], fontsize=8)

    plt.tight_layout()
    out3 = output_dir / "tau_sweep_key_metrics.png"
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"[Fig 3] saved: {out3}")

    # ── 图 4：tau=4.0 单独全指标图 ────────────────────────────────────────────
    plot_tau4_single(parsed, output_dir)

    # ── 图 5：tau=4.0 简洁 loss 图 ────────────────────────────────────────────
    plot_tau4_loss_simple(parsed, output_dir)

    # ── 图 6：所有 tau 各自的简洁 loss 图 ────────────────────────────────────
    plot_all_tau_loss_simple(parsed, output_dir)



def plot_tau4_single(parsed: dict, output_dir: Path):
    """单独画 tau=4.0 的所有 8 个指标，方便详细分析。"""
    key = "tau=4.0"
    p = parsed.get(key)
    if p is None:
        print(f"[Fig 4] 未找到 {key} 数据，跳过。")
        return

    steps = p["steps"]
    color = TAU_CONFIGS[key]["color"]
    w = min(5, len(steps))

    all_metrics = ["total", "attraction", "clash", "intra_clash",
                   "compactness", "rg", "iptm", "grad"]
    titles = {
        "total":       "Total Loss",
        "attraction":  "Attraction Loss",
        "clash":       "Clash Loss",
        "intra_clash": "Intra-Chain Clash Loss",
        "compactness": "Compactness Loss",
        "rg":          "Radius of Gyration (Å)",
        "iptm":        "iPTM Loss  (negative → maximize iPTM)",
        "grad":        "Gradient Norm (log scale)",
    }

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("BindCraft v2  —  tau=4.0 (low noise): All Metrics vs Optimization Steps",
                 fontsize=13, fontweight="bold")

    for ax, metric in zip(axes.flat, all_metrics):
        vals = p["data"].get(metric)
        if vals is None or len(vals) == 0:
            ax.set_visible(False)
            continue

        # 原始曲线
        ax.plot(steps, vals, color=color, linewidth=1.0, alpha=0.30)

        # 平滑曲线
        smoothed = np.convolve(vals, np.ones(w) / w, mode="same")
        ax.plot(steps, smoothed, color=color, linewidth=2.2, alpha=0.95,
                label="5-step avg")

        # 标注 best 点（仅 total）
        if metric == "total" and p["best_step"] is not None:
            bs  = p["best_step"]
            idx = np.argmin(np.abs(steps - bs))
            ax.scatter(steps[idx], vals[idx],
                       color="gold", s=120, zorder=6,
                       marker="*", edgecolors="black", linewidths=0.8)
            ax.annotate(f"best@{bs}\nloss={p['best_loss']:.3f}",
                        xy=(steps[idx], vals[idx]),
                        xytext=(6, 4), textcoords="offset points",
                        fontsize=8, color="black")

        ax.set_title(titles[metric], fontsize=9, fontweight="bold")
        ax.set_xlabel("Optimization Step", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, linestyle=":")

        if metric == "grad":
            ax.set_yscale("log")

    # 统一图例说明
    fig.text(0.5, 0.01,
             "thin=raw data   thick=5-step moving avg (mode=same)   ★=best sequence",
             ha="center", fontsize=8, color="gray")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out4 = output_dir / "tau4_single_all_metrics.png"
    fig.savefig(out4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 4] saved: {out4}")


def plot_tau_loss_combined(parsed: dict, output_dir: Path):
    """三个 tau 的 total loss 画在同一张图里，纯曲线无标记点。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("BindCraft v2  —  Total Loss Comparison (tau sweep)",
                 fontsize=12, fontweight="bold")

    for key, cfg in TAU_CONFIGS.items():
        p = parsed.get(key)
        if p is None:
            continue
        steps = p["steps"]
        vals  = np.array(p["data"].get("total", []))
        if len(vals) == 0:
            continue
        ax.plot(steps, vals,
                color=cfg["color"], ls=cfg["ls"],
                linewidth=1.5, label=cfg["label"])

    ax.set_xlabel("Optimization Step", fontsize=11)
    ax.set_ylabel("Total Loss", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(fontsize=10, loc="upper right")
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    out = output_dir / "tau_sweep_loss_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 7] saved: {out}")


def plot_all_tau_loss_simple(parsed: dict, output_dir: Path):
    """为每个 tau 值各画一张简洁 total loss 图，风格与 tau4_loss_simple 一致。"""
    for key, cfg in TAU_CONFIGS.items():
        p = parsed.get(key)
        if p is None:
            print(f"[Fig 6] 未找到 {key} 数据，跳过。")
            continue

        steps = p["steps"]
        vals  = np.array(p["data"].get("total", []))
        if len(vals) == 0:
            continue

        color = cfg["color"]
        tau_str = cfg["tau"]

        best_idx  = int(np.argmin(vals))
        best_step = steps[best_idx]
        best_val  = vals[best_idx]

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.suptitle(f"BindCraft v2  —  tau={tau_str}: Total Loss vs Optimization Steps",
                     fontsize=12, fontweight="bold")

        ax.plot(steps, vals, color=color, linewidth=1.5)

        ax.scatter(best_step, best_val, color="gold", s=180, zorder=6,
                   marker="*", edgecolors="black", linewidths=0.8)
        ax.annotate(f"best loss = {best_val:.3f}\n@ step {best_step}",
                    xy=(best_step, best_val),
                    xytext=(12, 8), textcoords="offset points",
                    fontsize=9, color="black",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

        ax.set_xlabel("Optimization Step", fontsize=11)
        ax.set_ylabel("Total Loss", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.tick_params(labelsize=9)

        plt.tight_layout()
        tau_tag = tau_str.replace(".", "_")
        out_path = output_dir / f"tau{tau_tag}_loss_simple.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Fig 6] saved: {out_path}")


def plot_tau4_loss_simple(parsed: dict, output_dir: Path):
    """tau=4.0 的简洁 total loss 曲线图，原始数据 + 平滑线 + 最低点标注。"""
    key = "tau=4.0"
    p = parsed.get(key)
    if p is None:
        print(f"[Fig 5] 未找到 {key} 数据，跳过。")
        return

    steps = p["steps"]
    vals  = np.array(p["data"].get("total", []))
    if len(vals) == 0:
        return

    color = "#2196F3"

    # 真实最低点
    best_idx  = int(np.argmin(vals))
    best_step = steps[best_idx]
    best_val  = vals[best_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("BindCraft v2  —  tau=4.0: Total Loss vs Optimization Steps",
                 fontsize=12, fontweight="bold")

    # 原始数据
    ax.plot(steps, vals, color=color, linewidth=1.5)

    # 最低点标注
    ax.scatter(best_step, best_val, color="gold", s=180, zorder=6,
               marker="*", edgecolors="black", linewidths=0.8)
    ax.annotate(f"best loss = {best_val:.3f}\n@ step {best_step}",
                xy=(best_step, best_val),
                xytext=(12, 8), textcoords="offset points",
                fontsize=9, color="black",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    ax.set_xlabel("Optimization Step", fontsize=11)
    ax.set_ylabel("Total Loss", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    out5 = output_dir / "tau4_loss_simple.png"
    fig.savefig(out5, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 5] saved: {out5}")


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot tau_noise sweep results")
    parser.add_argument("--out_file", type=str,
                        default="logs/tau_sweep_9562909.out",
                        help="tau_sweep_*.out 文件路径")
    parser.add_argument("--output_dir", type=str, default="outputs/plots",
                        help="图片输出目录")
    args = parser.parse_args()

    out_path   = Path(args.out_file)
    output_dir = Path(args.output_dir)

    if not out_path.exists():
        print(f"找不到文件：{out_path}")
        return

    print(f"解析：{out_path}")
    parsed = parse_sweep_out(out_path)

    for key, p in parsed.items():
        print(f"  {key}: {len(p['steps'])} steps, "
              f"best_loss={p['best_loss']}, best_step={p['best_step']}")

    if not parsed:
        print("未解析到任何数据，退出。")
        return

    plot_all(parsed, output_dir)
    print(f"\nAll figures saved to: {output_dir}/")


if __name__ == "__main__":
    main()
