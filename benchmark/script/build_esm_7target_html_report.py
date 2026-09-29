#!/usr/bin/env python3
"""Build a self-contained HTML report for a completed ESM target panel."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/esm_optimization/target_panel_13_reporting_fixed"
TARGETS = (
    "BetV1",
    "DerF21",
    "DerF7",
    "HER2",
    "IFNAR2",
    "PD-L1",
    "PD1",
    "SpCas9",
    "1QRN",
    "3QDJ",
    "5BS0",
    "6BJ8",
    "9C3E",
)
ARMS = ("baseline", "exact", "fast", "big")
ARM_LABELS = {
    "baseline": "Baseline",
    "exact": "Exact",
    "fast": "Fast",
    "big": "Big",
}
COLORS = {
    "baseline": "#64748b",
    "exact": "#2563eb",
    "fast": "#059669",
    "big": "#ea580c",
}
LONG_TARGET_THRESHOLD = 250


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def fmt(value: Any, digits: int = 3) -> str:
    if value in (None, ""):
        return "—"
    return f"{float(value):.{digits}f}"


def fmt_minutes(seconds: Any, digits: int = 2) -> str:
    if seconds in (None, ""):
        return "—"
    return f"{float(seconds) / 60:.{digits}f}"


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.mean(math.log(value) for value in values))


def svg_wall_chart(timing: dict[tuple[str, str], dict[str, str]]) -> str:
    width, height = max(1160, 140 + len(TARGETS) * 118), 455
    left, right, top, bottom = 72, 24, 46, 92
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [
        float(timing[target, "baseline"]["search_wall_s"])
        / float(timing[target, arm]["search_wall_s"])
        for target in TARGETS for arm in ARMS
    ]
    maximum = math.ceil(max(values) * 2) / 2
    group_w = plot_w / len(TARGETS)
    bar_w = min(25, group_w / 5.2)
    parts = [
        f'<svg class="chart" style="min-width:{width}px" viewBox="0 0 {width} {height}" role="img" aria-label="Speedup versus baseline by target and arm">',
        '<rect width="100%" height="100%" fill="white" rx="12"/>',
    ]
    for tick in range(0, 6):
        value = maximum * tick / 5
        y = top + plot_h - plot_h * value / maximum
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="axis">{value:.1f}×</text>')
    baseline_y = top + plot_h - plot_h / maximum
    parts.append(f'<line x1="{left}" x2="{width-right}" y1="{baseline_y:.1f}" y2="{baseline_y:.1f}" stroke="#64748b" stroke-dasharray="5 5"/>')
    parts.append(f'<text x="{width-right-2}" y="{baseline_y-7:.1f}" text-anchor="end" class="axis">baseline = 1×</text>')
    for target_index, target in enumerate(TARGETS):
        center = left + group_w * (target_index + 0.5)
        for arm_index, arm in enumerate(ARMS):
            wall_minutes = float(timing[target, arm]["search_wall_s"]) / 60
            value = float(timing[target, "baseline"]["search_wall_s"]) / float(timing[target, arm]["search_wall_s"])
            x = center + (arm_index - 1.5) * bar_w - bar_w * 0.42
            y = top + plot_h - plot_h * value / maximum
            h = top + plot_h - y
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.84:.1f}" height="{h:.1f}" '
                f'fill="{COLORS[arm]}" rx="2"><title>{html.escape(target)} · {ARM_LABELS[arm]}: {value:.3f}×; {wall_minutes:.2f} min</title></rect>'
            )
        target_tokens = int(float(timing[target, "baseline"]["target_tokens"]))
        marker = "★ " if target_tokens >= LONG_TARGET_THRESHOLD else ""
        target_class = "axis target long-target" if marker else "axis target"
        parts.append(f'<text x="{center:.1f}" y="{height-bottom+25}" text-anchor="middle" class="{target_class}">{marker}{html.escape(target)} ({target_tokens} aa)</text>')
    parts.append(f'<text x="18" y="{top+plot_h/2:.1f}" transform="rotate(-90 18 {top+plot_h/2:.1f})" text-anchor="middle" class="label">Speedup vs baseline (×)</text>')
    legend_x = left
    for arm in ARMS:
        parts.append(f'<rect x="{legend_x}" y="15" width="12" height="12" fill="{COLORS[arm]}" rx="2"/>')
        parts.append(f'<text x="{legend_x+18}" y="26" class="legend">{ARM_LABELS[arm]}</text>')
        legend_x += 118
    parts.append("</svg>")
    return "".join(parts)


def svg_quality_chart(quality: dict[tuple[str, str], dict[str, str]]) -> str:
    width, height = 1160, 455
    left, right, top, bottom = 72, 24, 46, 92
    plot_w, plot_h = width - left - right, height - top - bottom
    all_means = [float(quality[target, arm]["best5_pair_chain_iptm_mean"]) for target in TARGETS for arm in ARMS]
    lower = max(0.0, math.floor((min(all_means) - 0.04) * 20) / 20)
    upper = min(1.0, math.ceil((max(all_means) + 0.04) * 20) / 20)
    group_w = plot_w / len(TARGETS)
    offset = min(26, group_w / 5)

    def y_of(value: float) -> float:
        return top + plot_h - plot_h * (value - lower) / (upper - lower)

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Five-seed mean pair-chain iPTM by target and arm">',
        '<rect width="100%" height="100%" fill="white" rx="12"/>',
    ]
    for tick in range(0, 6):
        value = lower + (upper - lower) * tick / 5
        y = y_of(value)
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="axis">{value:.2f}</text>')
    if lower <= 0.8 <= upper:
        y = y_of(0.8)
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#dc2626" stroke-dasharray="6 5"/>')
        parts.append(f'<text x="{width-right-2}" y="{y-7:.1f}" text-anchor="end" class="threshold">0.80 success threshold</text>')
    for target_index, target in enumerate(TARGETS):
        center = left + group_w * (target_index + 0.5)
        for arm_index, arm in enumerate(ARMS):
            row = quality[target, arm]
            mean = float(row["best5_pair_chain_iptm_mean"])
            std = float(row["best5_pair_chain_iptm_std"])
            x = center + (arm_index - 1.5) * offset
            y1, y2, y = y_of(min(upper, mean + std)), y_of(max(lower, mean - std)), y_of(mean)
            parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y1:.1f}" y2="{y2:.1f}" stroke="{COLORS[arm]}" stroke-width="2"/>')
            parts.append(f'<line x1="{x-4:.1f}" x2="{x+4:.1f}" y1="{y1:.1f}" y2="{y1:.1f}" stroke="{COLORS[arm]}"/>')
            parts.append(f'<line x1="{x-4:.1f}" x2="{x+4:.1f}" y1="{y2:.1f}" y2="{y2:.1f}" stroke="{COLORS[arm]}"/>')
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{COLORS[arm]}" stroke="white" stroke-width="1.5">'
                f'<title>{html.escape(target)} · {ARM_LABELS[arm]}: {mean:.4f} ± {std:.4f}</title></circle>'
            )
        parts.append(f'<text x="{center:.1f}" y="{height-bottom+25}" text-anchor="middle" class="axis target">{html.escape(target)}</text>')
    parts.append(f'<text x="18" y="{top+plot_h/2:.1f}" transform="rotate(-90 18 {top+plot_h/2:.1f})" text-anchor="middle" class="label">Best-of-5 pair-chain iPTM</text>')
    legend_x = left
    for arm in ARMS:
        parts.append(f'<circle cx="{legend_x+6}" cy="21" r="6" fill="{COLORS[arm]}"/>')
        parts.append(f'<text x="{legend_x+18}" y="26" class="legend">{ARM_LABELS[arm]}</text>')
        legend_x += 118
    parts.append("</svg>")
    return "".join(parts)


def svg_wall_breakdown(timing: dict[tuple[str, str], dict[str, str]]) -> str:
    width, height = 900, 380
    left, right, top, bottom = 75, 24, 45, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    components: dict[str, list[float]] = {}
    for arm in ARMS:
        update = sum(float(timing[target, arm]["proposal_update_total_s"]) for target in TARGETS) / 60
        verifier = sum(float(timing[target, arm]["verifier_total_s"]) for target in TARGETS) / 60
        wall = sum(float(timing[target, arm]["search_wall_s"]) for target in TARGETS) / 60
        components[arm] = [update, verifier, wall - update - verifier]
    maximum = math.ceil(max(sum(parts) for parts in components.values()) / 100) * 100
    segment_colors = ("#0f766e", "#7c3aed", "#cbd5e1")
    segment_labels = ("Proposal update", "Verifier", "Other")
    group_w = plot_w / len(ARMS)
    bar_w = 85
    parts = [
        f'<svg class="chart compact" viewBox="0 0 {width} {height}" role="img" aria-label="Top-level wall-time breakdown">',
        '<rect width="100%" height="100%" fill="white" rx="12"/>',
    ]
    for tick in range(0, 5):
        value = maximum * tick / 4
        y = top + plot_h - plot_h * value / maximum
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="axis">{value:.0f}</text>')
    for arm_index, arm in enumerate(ARMS):
        center = left + group_w * (arm_index + 0.5)
        cursor = top + plot_h
        for value, color, label in zip(components[arm], segment_colors, segment_labels):
            segment_h = plot_h * value / maximum
            cursor -= segment_h
            parts.append(
                f'<rect x="{center-bar_w/2:.1f}" y="{cursor:.1f}" width="{bar_w}" height="{segment_h:.1f}" fill="{color}">'
                f'<title>{ARM_LABELS[arm]} · {label}: {value:.2f} min</title></rect>'
            )
        total = sum(components[arm])
        parts.append(f'<text x="{center:.1f}" y="{cursor-8:.1f}" text-anchor="middle" class="value">{total:.1f}</text>')
        parts.append(f'<text x="{center:.1f}" y="{height-bottom+28}" text-anchor="middle" class="axis target">{ARM_LABELS[arm]}</text>')
    legend_x = left
    for color, label in zip(segment_colors, segment_labels):
        parts.append(f'<rect x="{legend_x}" y="14" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{legend_x+18}" y="25" class="legend">{label}</text>')
        legend_x += 170
    parts.append(f'<text x="18" y="{top+plot_h/2:.1f}" transform="rotate(-90 18 {top+plot_h/2:.1f})" text-anchor="middle" class="label">{len(TARGETS)}-target total (minutes)</text>')
    parts.append("</svg>")
    return "".join(parts)


def build(args: argparse.Namespace) -> Path:
    output = Path(args.output).resolve()
    timing_dir = output / "analysis"
    quality_root = output / "rescore_5seed"
    quality_dir = quality_root / "analysis"
    report_dir = quality_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    timing_rows: list[dict[str, str]] = []
    long_rows: list[dict[str, str]] = []
    for target_file in sorted(timing_dir.glob("run_timing.*.csv")):
        timing_rows.extend(read_csv(target_file))
    for target_file in sorted(timing_dir.glob("timing_long.*.csv")):
        long_rows.extend(read_csv(target_file))
    quality_rows = read_csv(quality_dir / "run_summary.csv")
    arm_rows = read_csv(quality_dir / "arm_summary.csv")
    campaign = json.loads((output / "campaign.json").read_text())
    reused_baseline_count = sum(
        record["arm"] == "baseline" and record.get("reuse_existing", False)
        for record in campaign["records"]
    )
    rerun_baseline_count = sum(
        record["arm"] == "baseline" and not record.get("reuse_existing", False)
        for record in campaign["records"]
    )
    target_chain_lengths: dict[str, list[int]] = {}
    for record in campaign["records"]:
        target_name = record["case"]["target_name"]
        if target_name in TARGETS and target_name not in target_chain_lengths:
            target_chain_lengths[target_name] = [
                len(chain) for chain in record["case"]["target_sequence"].split("|")
            ]
    timing = {
        (row["target_name"], row["arm"]): row
        for row in timing_rows if row["target_name"] in TARGETS
    }
    quality = {
        (row["target_name"], row["arm"]): row
        for row in quality_rows if row["target_name"] in TARGETS
    }
    long = {
        (row["target_name"], row["arm"], row["stage"], row["phase"]): row
        for row in long_rows if row["target_name"] in TARGETS
    }
    for target in TARGETS:
        for arm in ARMS:
            for name, mapping in (("timing", timing), ("quality", quality)):
                if (target, arm) not in mapping:
                    raise RuntimeError(f"Missing {name} data for {target}/{arm}")

    target_count = len(TARGETS)
    expected_runs = target_count * len(ARMS)
    expected_scores = expected_runs * 5 * 5
    evaluated_scores = sum(int(row["rescored_samples"]) for row in quality_rows if row["target_name"] in TARGETS)
    if len(quality) != expected_runs or evaluated_scores != expected_scores:
        raise RuntimeError(
            f"Incomplete five-seed evaluation: {len(quality)}/{expected_runs} runs and "
            f"{evaluated_scores}/{expected_scores} scores"
        )

    arm_summary = {row["arm"]: row for row in arm_rows}
    aggregate: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        walls = [float(timing[target, arm]["search_wall_s"]) for target in TARGETS]
        baseline_walls = [float(timing[target, "baseline"]["search_wall_s"]) for target in TARGETS]
        speedups = [base / current for base, current in zip(baseline_walls, walls)]
        iptm_deltas = [
            float(quality[target, arm]["best5_iptm_mean"])
            - float(quality[target, "baseline"]["best5_iptm_mean"])
            for target in TARGETS
        ]
        selected_std = [float(quality[target, arm]["best5_pair_chain_iptm_std"]) for target in TARGETS]
        aggregate[arm] = {
            "wall_total_min": sum(walls) / 60,
            "wall_mean_min": statistics.mean(walls) / 60,
            "wall_median_min": statistics.median(walls) / 60,
            "speed_median": statistics.median(speedups),
            "speed_geo": geometric_mean(speedups),
            "speed_mean": statistics.mean(speedups),
            "iptm_delta_mean": statistics.mean(iptm_deltas),
            "iptm_delta_std": statistics.stdev(iptm_deltas),
            "stage_a_total_min": sum(float(long[target, arm, "stage_a", "proposal_update"]["total_s"]) for target in TARGETS) / 60,
            "stage_b_total_min": sum(float(long[target, arm, "stage_b", "proposal_update"]["total_s"]) for target in TARGETS) / 60,
            "update_total_min": sum(float(timing[target, arm]["proposal_update_total_s"]) for target in TARGETS) / 60,
            "verifier_total_min": sum(float(timing[target, arm]["verifier_total_s"]) for target in TARGETS) / 60,
            "selected_seed_std": statistics.mean(selected_std),
        }
        aggregate[arm]["other_total_min"] = (
            aggregate[arm]["wall_total_min"]
            - aggregate[arm]["update_total_min"]
            - aggregate[arm]["verifier_total_min"]
        )

    winner_counts = {arm: 0 for arm in ARMS}
    target_winners: dict[str, str] = {}
    for target in TARGETS:
        winner = max(ARMS, key=lambda arm: float(quality[target, arm]["best5_pair_chain_iptm_mean"]))
        target_winners[target] = winner
        winner_counts[winner] += 1

    best_quality_arm = max(ARMS, key=lambda arm: float(arm_summary[arm]["best5_mean_across_targets"]))
    best_speed_arm = max(ARMS, key=lambda arm: aggregate[arm]["speed_mean"])
    largest_speedup = aggregate[best_speed_arm]["speed_mean"]
    success_counts = " / ".join(
        str(int(float(arm_summary[arm]["best5_successes_mean_ge_0_8"]))) for arm in ARMS
    )
    optimized_arms = ARMS[1:]
    ci_crossing_arms = [
        arm
        for arm in optimized_arms
        if float(arm_summary[arm]["paired_best5_delta_ci95_low"]) <= 0
        <= float(arm_summary[arm]["paired_best5_delta_ci95_high"])
    ]
    if len(ci_crossing_arms) == len(optimized_arms):
        bootstrap_conclusion = (
            f"三个优化 arm 的目标级 bootstrap 95% CI 均跨零；{target_count} 个目标下，"
            "没有证据表明质量相对 baseline 有一致的提升或下降。"
        )
    else:
        bootstrap_conclusion = (
            "目标级 bootstrap 95% CI 跨零的优化 arm："
            + ", ".join(ARM_LABELS[arm] for arm in ci_crossing_arms)
            + "。"
        )
    long_targets = [
        target for target in TARGETS
        if sum(target_chain_lengths[target]) >= LONG_TARGET_THRESHOLD
    ]
    long_target_description = "；".join(
        f"{target}（{sum(target_chain_lengths[target])} aa；"
        f"{' + '.join(map(str, target_chain_lengths[target]))}）"
        for target in long_targets
    )
    generated = __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds")

    css = """
    :root { --ink:#0f172a; --muted:#475569; --line:#e2e8f0; --panel:#ffffff; --soft:#f8fafc; --accent:#0f766e; }
    * { box-sizing:border-box; }
    html { scroll-behavior:smooth; }
    body { margin:0; color:var(--ink); background:#eef2f7; font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    header { color:white; background:linear-gradient(125deg,#0f172a 0%,#164e63 56%,#0f766e 100%); padding:54px max(28px,calc((100vw - 1220px)/2)); }
    header h1 { margin:0 0 10px; font-size:34px; letter-spacing:-.03em; }
    header p { margin:4px 0; color:#dbeafe; max-width:900px; }
    nav { position:sticky; top:0; z-index:4; padding:11px max(20px,calc((100vw - 1220px)/2)); background:rgba(255,255,255,.96); border-bottom:1px solid var(--line); backdrop-filter:blur(10px); }
    nav a { color:#334155; text-decoration:none; font-weight:650; margin-right:20px; font-size:13px; }
    main { max-width:1220px; margin:0 auto; padding:28px 22px 64px; }
    section { scroll-margin-top:58px; margin:0 0 32px; }
    h2 { margin:0 0 14px; font-size:24px; letter-spacing:-.02em; }
    h3 { margin:22px 0 10px; font-size:18px; }
    .subtle { color:var(--muted); }
    .grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; }
    .card,.panel,details { background:var(--panel); border:1px solid var(--line); border-radius:13px; box-shadow:0 1px 2px rgba(15,23,42,.04); }
    .card { padding:18px; }
    .card .eyebrow { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }
    .card .big { display:block; margin-top:5px; font-size:25px; font-weight:780; }
    .panel { padding:18px; margin:13px 0; overflow:hidden; }
    .charts { display:grid; grid-template-columns:1fr; gap:16px; }
    .chart { width:100%; min-width:760px; display:block; }
    .chart.compact { min-width:620px; }
    .chart-wrap { overflow-x:auto; }
    .axis,.legend,.label,.value,.threshold { font-family:Inter,ui-sans-serif,system-ui,sans-serif; fill:#475569; font-size:11px; }
    .target { font-weight:650; fill:#334155; }
    .long-target { fill:#c2410c; font-weight:800; }
    .label { font-size:12px; font-weight:650; }
    .value { fill:#0f172a; font-weight:700; }
    .threshold { fill:#b91c1c; font-weight:650; }
    .table-wrap { width:100%; overflow-x:auto; border:1px solid var(--line); border-radius:11px; background:white; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th { position:sticky; top:0; color:#334155; background:#f1f5f9; text-align:right; white-space:nowrap; }
    th:first-child,td:first-child { text-align:left; }
    th,td { padding:9px 10px; border-bottom:1px solid var(--line); text-align:right; }
    tr:last-child td { border-bottom:0; }
    tbody tr:hover { background:#f8fafc; }
    td.good { color:#047857; font-weight:700; }
    td.warn { color:#b45309; font-weight:700; }
    .arm { display:inline-flex; align-items:center; gap:7px; font-weight:700; }
    .dot { width:9px; height:9px; border-radius:50%; display:inline-block; }
    .note { padding:13px 15px; background:#f0fdfa; border-left:4px solid #0f766e; border-radius:7px; color:#134e4a; }
    .warning { background:#fff7ed; border-left-color:#ea580c; color:#7c2d12; }
    details { margin:12px 0; overflow:hidden; }
    summary { cursor:pointer; padding:15px 18px; font-weight:750; background:#fff; }
    details[open] summary { border-bottom:1px solid var(--line); background:#f8fafc; }
    .detail-body { padding:16px; }
    code.path { display:block; min-width:470px; max-width:780px; overflow-wrap:anywhere; white-space:normal; text-align:left; color:#334155; font-size:11.5px; }
    .pill { display:inline-block; padding:2px 8px; border-radius:999px; background:#e2e8f0; color:#334155; font-size:11px; font-weight:700; }
    .pill.long { background:#ffedd5; color:#9a3412; border:1px solid #fdba74; }
    .winner { background:#dcfce7; color:#166534; }
    .two-col { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
    ul.tight { margin:8px 0; padding-left:21px; }
    footer { color:#64748b; border-top:1px solid var(--line); padding-top:20px; font-size:12px; }
    a { color:#0369a1; }
    @media (max-width:900px) { .grid,.two-col { grid-template-columns:1fr 1fr; } header h1{font-size:28px;} }
    @media (max-width:600px) { .grid,.two-col { grid-template-columns:1fr; } nav{display:none;} main{padding:20px 12px;} }
    @media print { body{background:white;} nav{display:none;} .panel,.card,details{box-shadow:none; break-inside:avoid;} details{display:block;} details:not([open]) .detail-body{display:block;} }
    """

    aggregate_table_rows = []
    for arm in ARMS:
        a = aggregate[arm]
        aggregate_table_rows.append(f"""
        <tr>
          <td><span class="arm"><span class="dot" style="background:{COLORS[arm]}"></span>{ARM_LABELS[arm]}</span></td>
          <td>{a['speed_mean']:.3f}×</td>
          <td>{a['iptm_delta_mean']:+.4f}</td>
          <td>{a['iptm_delta_std']:.4f}</td>
        </tr>""")

    speed_target_rows = []
    quality_target_rows = []
    for target in TARGETS:
        baseline_wall = float(timing[target, "baseline"]["search_wall_s"])
        speed_winner = min(ARMS, key=lambda arm: float(timing[target, arm]["search_wall_s"]))
        speed_cells = []
        quality_cells = []
        for arm in ARMS:
            wall_minutes = float(timing[target, arm]["search_wall_s"]) / 60
            speedup = baseline_wall / float(timing[target, arm]["search_wall_s"])
            speed_class = "good" if arm == speed_winner else ""
            speed_cells.append(
                f'<td class="{speed_class}"><strong>{speedup:.3f}×</strong><br><span class="subtle">{wall_minutes:.2f} min</span></td>'
            )
            qrow = quality[target, arm]
            winner_class = "good" if arm == target_winners[target] else ""
            success = float(qrow["best5_pair_chain_iptm_mean"]) >= 0.8
            quality_cells.append(
                f'<td class="{winner_class}"><strong>{float(qrow["best5_pair_chain_iptm_mean"]):.4f}</strong> '
                f'± {float(qrow["best5_pair_chain_iptm_std"]):.4f}<br>'
                f'<span class="subtle">rank {qrow["best5_rank"]}{" · ≥0.8" if success else ""}</span></td>'
            )
        target_tokens = sum(target_chain_lengths[target])
        if target_tokens >= LONG_TARGET_THRESHOLD:
            chain_breakdown = " + ".join(map(str, target_chain_lengths[target]))
            length_label = (
                f'<br><span class="pill long">Long · {target_tokens} aa '
                f'({chain_breakdown})</span>'
            )
        else:
            length_label = f'<br><span class="subtle">{target_tokens} aa</span>'
        speed_target_rows.append(
            f'<tr><td><strong>{html.escape(target)}</strong>{length_label}</td>'
            f'{"".join(speed_cells)}</tr>'
        )
        quality_target_rows.append(
            f'<tr><td><strong>{html.escape(target)}</strong></td>{"".join(quality_cells)}</tr>'
        )

    target_sections = []
    for target in TARGETS:
        baseline_wall = float(timing[target, "baseline"]["search_wall_s"])
        timing_rows_html = []
        quality_rows_html = []
        for arm in ARMS:
            trow = timing[target, arm]
            qrow = quality[target, arm]
            wall = float(trow["search_wall_s"])
            update = float(trow["proposal_update_total_s"])
            verifier = float(trow["verifier_total_s"])
            other = wall - update - verifier
            stage_a = float(long[target, arm, "stage_a", "proposal_update"]["total_s"])
            stage_b = float(long[target, arm, "stage_b", "proposal_update"]["total_s"])
            nested_proposal = sum(float(long[target, arm, stage, "proposal_forward"]["total_s"]) for stage in ("stage_a", "stage_b"))
            nested_esmc = sum(float(long[target, arm, stage, "esmc_prior_forward"]["total_s"]) for stage in ("stage_a", "stage_b"))
            timing_rows_html.append(f"""
            <tr><td><span class="arm"><span class="dot" style="background:{COLORS[arm]}"></span>{ARM_LABELS[arm]}</span></td>
            <td>{wall/60:.2f}</td><td>{baseline_wall/wall:.3f}×</td><td>{update/60:.2f}</td>
            <td>{stage_a/60:.2f}</td><td>{stage_b/60:.2f}</td><td>{verifier/60:.2f}</td><td>{other/60:.2f}</td>
            <td>{nested_proposal/60:.2f}</td><td>{nested_esmc/60:.2f}</td><td>{float(trow['update_residual_s'])/60:.2f}</td>
            <td>{float(long[target, arm, 'all', 'proposal_update']['median_s']):.3f}</td>
            <td>{float(long[target, arm, 'all', 'verifier']['median_s']):.3f}</td></tr>""")
            winner_class = "good" if arm == target_winners[target] else ""
            quality_rows_html.append(f"""
            <tr><td><span class="arm"><span class="dot" style="background:{COLORS[arm]}"></span>{ARM_LABELS[arm]}</span></td>
            <td>{qrow['best5_rank']}</td><td class="{winner_class}">{float(qrow['best5_pair_chain_iptm_mean']):.4f} ± {float(qrow['best5_pair_chain_iptm_std']):.4f}</td>
            <td>[{float(qrow['best5_pair_chain_iptm_min']):.4f}, {float(qrow['best5_pair_chain_iptm_max']):.4f}]</td>
            <td>{float(qrow['top1_pair_chain_iptm_mean']):.4f} ± {float(qrow['top1_pair_chain_iptm_std']):.4f}</td>
            <td>{float(qrow['top5_pool_pair_chain_iptm_mean']):.4f}</td><td>{float(qrow['best5_iptm_mean']):.4f}</td>
            <td>{float(qrow['best5_ptm_mean']):.4f}</td><td>{float(qrow['best5_plddt_mean']):.4f}</td>
            <td>{float(qrow['best5_binder_plddt_mean']):.4f}</td><td>{float(qrow['best5_total_loss_mean']):.4f}</td>
            <td>{'Yes' if qrow['best5_success_mean_ge_0_8'] == 'True' else 'No'}</td></tr>""")
        target_sections.append(f"""
        <details>
          <summary>{html.escape(target)} <span class="pill winner">Quality winner: {ARM_LABELS[target_winners[target]]}</span></summary>
          <div class="detail-body">
            <h3>Wall time and detailed timing</h3>
            <div class="table-wrap"><table><thead><tr>
              <th>Arm</th><th>Wall min</th><th>Speedup</th><th>Update min</th><th>Stage A min</th><th>Stage B min</th>
              <th>Verifier min</th><th>Other min</th><th>Nested proposal min</th><th>Nested ESMC min</th><th>Update residual min</th>
              <th>Median update s</th><th>Median verifier s</th>
            </tr></thead><tbody>{''.join(timing_rows_html)}</tbody></table></div>
            <p class="subtle">Stage A/B are proposal-update totals. Nested proposal, ESMC, and update residual sum to proposal update; they must not be added again to wall time.</p>
            <h3>Five-seed ESMFold2 quality</h3>
            <div class="table-wrap"><table><thead><tr>
              <th>Arm</th><th>Best rank</th><th>Best-of-5 pair iPTM mean ± SD</th><th>Seed range</th><th>Top-1 mean ± SD</th>
              <th>Top-5 pool</th><th>iPTM</th><th>pTM</th><th>pLDDT</th><th>Binder pLDDT</th><th>Total loss↓</th><th>≥0.8</th>
            </tr></thead><tbody>{''.join(quality_rows_html)}</tbody></table></div>
          </div>
        </details>""")

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>ESMFold2 Optimization — {target_count}-target speed and quality report</title>
  <style>{css}</style>
</head>
<body>
<header>
  <h1>ESMFold2 优化：{target_count}-target 速度与性能报告</h1>
  <p>Baseline / Exact / Fast / Big 的单 GPU 搜索速度、阶段细分时间，以及未优化 ESMFold2 五-seed盲重评分比较。</p>
  <p>Targets: {', '.join(TARGETS)} · Generated: {generated}</p>
</header>
<nav><a href="#summary">摘要</a><a href="#speed">速度</a><a href="#quality">性能</a><a href="#targets">逐目标</a></nav>
<main>
  <section id="summary">
    <h2>执行摘要</h2>
    <div class="grid">
      <div class="card"><span class="eyebrow">Highest mean speedup</span><span class="big">{ARM_LABELS[best_speed_arm]} · {largest_speedup:.2f}×</span><span class="subtle">{target_count} 个 target 等权算术平均，相对各自 baseline</span></div>
      <div class="card"><span class="eyebrow">Highest mean quality</span><span class="big">{ARM_LABELS[best_quality_arm]} · {float(arm_summary[best_quality_arm]['best5_mean_across_targets']):.4f}</span><span class="subtle">Best-of-5 五-seed平均 pair-chain iPTM</span></div>
      <div class="card"><span class="eyebrow">Evaluation matrix</span><span class="big">{evaluated_scores} / {expected_scores}</span><span class="subtle">{target_count} targets × {len(ARMS)} arms × 5 candidates × 5 seeds</span></div>
      <div class="card"><span class="eyebrow">Successes ≥ 0.8</span><span class="big">{success_counts}</span><span class="subtle">Baseline / Exact / Fast / Big，按候选五-seed均值判断</span></div>
    </div>
    <div class="panel">
      <p><strong>主要结论：</strong>Fast 与 Big 的端到端平均加速分别为 {aggregate['fast']['speed_mean']:.2f}× 和 {aggregate['big']['speed_mean']:.2f}×。{ARM_LABELS[best_quality_arm]} 的 {target_count}-target 平均 pair-chain iPTM 最高（{float(arm_summary[best_quality_arm]['best5_mean_across_targets']):.4f}）。Exact 的平均速度为 baseline 的 {aggregate['exact']['speed_mean']:.2f}×，应解释为速度基本持平，且此前 provenance 热路径造成的异常额外耗时已移除。</p>
      <p class="subtle">五个 ESMFold2 eval seed 用于量化同一候选的推理波动；每个 arm 仍只有一个 search seed，因此当前结果不能替代多搜索-seed验证。</p>
    </div>
    <div class="table-wrap"><table><thead><tr><th>Arm</th><th>Mean speedup vs baseline</th><th>Mean ESMFold2 iPTM Δ vs baseline</th><th>SD of paired iPTM Δ across targets</th></tr></thead>
      <tbody>{''.join(aggregate_table_rows)}</tbody></table></div>
    <p class="subtle">平均加速为 {target_count} 个 target 配对 speedup 的算术平均。候选仍按五-seed平均 pair-chain iPTM 选择；摘要中的 iPTM Δ 使用该候选的 ESMFold2 iPTM，先在每个 target 内与 baseline 配对相减，再报告跨 {target_count} 个 target 的均值和样本标准差。</p>
  </section>

  <section id="speed">
    <h2>速度比较</h2>
    <p class="subtle">图中全部数值均为相对同一 target baseline 的加速倍数，baseline 固定为 1×。鼠标悬停可同时查看倍数和实际 wall time。</p>
    <p class="note"><strong>公平性说明：</strong>{reused_baseline_count} 个已完成 baseline 按预先规则复用，{rerun_baseline_count} 个 baseline 在本轮运行；复用前已核对相同 profile SHA256，且每个 target 的四组使用相同输入、search seed 和模型配置。复用 baseline 不会进入此前高耗时的模块路径解析分支，但它们运行于计时探针修复前，仍可能保留很小的轻量探针开销；加之每组只有一次 search run 且并非同期运行，因此 Exact 的 {aggregate['exact']['speed_mean']:.3f}× 应视为与 baseline 基本持平，而不是可确认的算法加速。</p>
    <p class="note warning"><strong>长 target 标记：</strong>以目标链总长度 ≥ {LONG_TARGET_THRESHOLD} aa 为统一标准，图中用 ★、表中用 Long 标签标出。本组包括：{html.escape(long_target_description)}。括号内为总长度及各链长度，不包含 120 aa binder。</p>
    <div class="panel chart-wrap">{svg_wall_chart(timing)}</div>
    <h3>逐 target 速度表现</h3>
    <p class="subtle">每个单元格第一行是 baseline-relative speedup，第二行是该 run 的实际 wall time。完整 Stage A/B 和 phase 细分仍保留在后面的逐目标展开表中。</p>
    <div class="table-wrap"><table><thead><tr><th>Target</th><th>Baseline</th><th>Exact</th><th>Fast</th><th>Big</th></tr></thead><tbody>{''.join(speed_target_rows)}</tbody></table></div>
  </section>

  <section id="quality">
    <h2>五-seed ESMFold2 性能比较</h2>
    <p class="subtle">表中主数值是 Best-of-5 候选的五-seed平均 pair-chain iPTM，± 后为五个 eval seed（1042–1046）的样本标准差。每个候选先跨 seed 求均值，再从搜索 top-5 中选择最佳候选；绿色为该 target 的最高均值。</p>
    <div class="table-wrap"><table><thead><tr><th>Target</th><th>Baseline mean ± SD</th><th>Exact mean ± SD</th><th>Fast mean ± SD</th><th>Big mean ± SD</th></tr></thead><tbody>{''.join(quality_target_rows)}</tbody></table></div>
    <div class="two-col">
      <div class="panel"><h3>跨目标表现</h3><ul class="tight">
        <li>{ARM_LABELS[best_quality_arm]} 平均 Best-of-5 = <strong>{float(arm_summary[best_quality_arm]['best5_mean_across_targets']):.4f}</strong>，相对 baseline 平均 Δ = {float(arm_summary[best_quality_arm]['paired_best5_delta_mean']):+.4f}。</li>
        <li>Fast 在 <strong>{arm_summary['fast']['best5_wins_vs_baseline']}/{target_count}</strong> 个目标胜过 baseline，配对中位差为 {float(arm_summary['fast']['paired_best5_delta_median']):+.4f}。</li>
        <li>{bootstrap_conclusion}</li>
      </ul></div>
      <div class="panel"><h3>每个 target 的质量赢家</h3><ul class="tight">{''.join(f'<li>{html.escape(target)}: <strong>{ARM_LABELS[target_winners[target]]}</strong></li>' for target in TARGETS)}</ul></div>
    </div>
  </section>

  <section id="targets">
    <h2>逐目标详细结果</h2>
    <p class="subtle">展开每个 target 查看完整 wall time、Stage A/B、嵌套 phase，以及五-seed均值、标准差和范围。</p>
    {''.join(target_sections)}
  </section>
</main>
</body>
</html>
"""
    destination = Path(args.destination).resolve() if args.destination else report_dir / f"full_{len(TARGETS)}target_speed_quality_report.html"
    destination.write_text(document)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--destination")
    args = parser.parse_args()
    destination = build(args)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
