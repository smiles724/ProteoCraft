#!/usr/bin/env python3
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/data/huangyanjie/binder_design/BindCraft-v2")
BENCHMARK_ROOT = PROJECT_ROOT / "benchmark"
RESULTS_ROOT = BENCHMARK_ROOT / "results"
PICTURE_DIR = BENCHMARK_ROOT / "outputs" / "picture"
OUTPUT_DIR = RESULTS_ROOT / "analysis" / "target_hotspot_confidence"


TARGETS = {
    "PD1": {
        "baseline_label": "No hotspot",
        "baseline": RESULTS_ROOT
        / "bc_v1_random_avglen_hybrid_template"
        / "bc_v1_random_avglen_hybrid_template"
        / "PD1"
        / "result.json",
        "baseline_note": "nearest_available_control: same template and binder length, but init sequence differs and loss weights are implicit default",
        "hotspots": [
            (
                "B64,B126,B129,B133",
                RESULTS_ROOT
                / "pd1_default_loss_hotspot_B64_B126_B129_B133"
                / "pd1_default_loss_hotspot_B64_B126_B129_B133"
                / "PD1"
                / "result.json",
            )
        ],
    },
    "PD-L1": {
        "baseline_label": "No hotspot",
        "baseline": RESULTS_ROOT
        / "custom_loss_no_hotspot_random_avglen"
        / "custom_loss_no_hotspot_random_avglen"
        / "PD-L1"
        / "result.json",
        "baseline_note": "matched_control: same custom loss, no template, binder length, and initial sequence",
        "hotspots": [
            (
                "B54,B56,B66,B115",
                RESULTS_ROOT
                / "pd_l1_custom_loss_hotspot_B54_B56_B66_B115"
                / "pd_l1_custom_loss_hotspot_B54_B56_B66_B115"
                / "PD-L1"
                / "result.json",
            )
        ],
    },
    "IFNAR2": {
        "baseline_label": "No hotspot",
        "baseline": RESULTS_ROOT
        / "custom_loss_no_hotspot_random_avglen"
        / "custom_loss_no_hotspot_random_avglen"
        / "IFNAR2"
        / "result.json",
        "baseline_note": "matched_control: same custom loss, no template, binder length, and initial sequence",
        "hotspots": [
            (
                "B52,B80,B82,B84,B96,B98",
                RESULTS_ROOT
                / "ifnar2_custom_loss_hotspot_B52_B80_B82_B84_B96_B98"
                / "ifnar2_custom_loss_hotspot_B52_B80_B82_B84_B96_B98"
                / "IFNAR2"
                / "result.json",
            )
        ],
    },
    "DerF21": {
        "baseline_label": "No hotspot",
        "baseline": RESULTS_ROOT
        / "custom_loss_no_hotspot_random_avglen"
        / "custom_loss_no_hotspot_random_avglen"
        / "DerF21"
        / "result.json",
        "baseline_note": "matched_control: same custom loss, no template, binder length, and initial sequence",
        "hotspots": [
            (
                "B34",
                RESULTS_ROOT / "derf21_custom_loss_hotspot_B34" / "derf21_custom_loss_hotspot_B34" / "DerF21" / "result.json",
            ),
            (
                "B52",
                RESULTS_ROOT / "derf21_custom_loss_hotspot_B52" / "derf21_custom_loss_hotspot_B52" / "DerF21" / "result.json",
            ),
            (
                "B87",
                RESULTS_ROOT / "derf21_custom_loss_hotspot_B87" / "derf21_custom_loss_hotspot_B87" / "DerF21" / "result.json",
            ),
            (
                "B105",
                RESULTS_ROOT
                / "derf21_custom_loss_hotspot_B105"
                / "derf21_custom_loss_hotspot_B105"
                / "DerF21"
                / "result.json",
            ),
            (
                "B116",
                RESULTS_ROOT
                / "derf21_custom_loss_hotspot_B116"
                / "derf21_custom_loss_hotspot_B116"
                / "DerF21"
                / "result.json",
            ),
        ],
    },
    "DerF7": {
        "baseline_label": "No hotspot",
        "baseline": RESULTS_ROOT
        / "custom_loss_no_hotspot_random_avglen"
        / "custom_loss_no_hotspot_random_avglen"
        / "DerF7"
        / "result.json",
        "baseline_note": "matched_control: same custom loss, no template, binder length, and initial sequence",
        "hotspots": [
            (
                "B132",
                RESULTS_ROOT / "derf7_custom_loss_hotspot_B132" / "derf7_custom_loss_hotspot_B132" / "DerF7" / "result.json",
            ),
            (
                "B178",
                RESULTS_ROOT / "derf7_custom_loss_hotspot_B178" / "derf7_custom_loss_hotspot_B178" / "DerF7" / "result.json",
            ),
        ],
    },
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def find_best_meta(result_path: Path, result: dict) -> tuple[dict, str]:
    best_step = int(result["best_step"])
    step_dir = result_path.parent / "checkpoints" / f"step_{best_step:04d}"
    meta_paths = sorted(step_dir.glob("*_meta.json"))
    if not meta_paths:
        return {}, ""

    best_seq = result.get("best_seq")
    metas = [(path, load_json(path)) for path in meta_paths]
    exact = [(path, meta) for path, meta in metas if meta.get("binder_seq") == best_seq]
    candidates = exact or metas
    path, meta = min(candidates, key=lambda item: float(item[1].get("loss_dict", {}).get("total", math.inf)))
    return meta, str(path.relative_to(PROJECT_ROOT))


def normalize_loss_weights(result: dict) -> str:
    weights = result.get("params", {}).get("loss_weights_override")
    if weights is None:
        return "implicit_default"
    return json.dumps(weights, ensure_ascii=False, sort_keys=True)


def run_record(target: str, condition: str, result_path: Path, control_note: str) -> dict:
    result = load_json(result_path)
    meta, meta_path = find_best_meta(result_path, result)
    confidence = meta.get("summary_confidence", {})
    params = result.get("params", {})
    return {
        "target": target,
        "condition": condition,
        "hotspot_residues": result.get("target_hotspot_residues") or "None",
        "best_step": int(result.get("best_step", -1)),
        "best_loss": float(result.get("best_loss", math.nan)),
        "binder_length": int(result.get("binder_length", 0)),
        "plddt": float(confidence.get("plddt", math.nan)),
        "confidence_score": float(confidence.get("ranking_score", math.nan)),
        "iptm": float(confidence.get("iptm", math.nan)),
        "target_template_path": params.get("target_template_path") or "None",
        "loss_weights_key": normalize_loss_weights(result),
        "init_binder_seq": result.get("init_binder_seq", ""),
        "best_binder_seq": result.get("best_seq", ""),
        "control_note": control_note,
        "result_path": str(result_path.relative_to(PROJECT_ROOT)),
        "meta_path": meta_path,
    }


def compare(hotspot: dict, baseline: dict, baseline_note: str) -> dict:
    return {
        "target": hotspot["target"],
        "hotspot": hotspot["condition"],
        "baseline": baseline["condition"],
        "baseline_plddt": baseline["plddt"],
        "hotspot_plddt": hotspot["plddt"],
        "delta_plddt": hotspot["plddt"] - baseline["plddt"],
        "baseline_confidence_score": baseline["confidence_score"],
        "hotspot_confidence_score": hotspot["confidence_score"],
        "delta_confidence_score": hotspot["confidence_score"] - baseline["confidence_score"],
        "baseline_iptm": baseline["iptm"],
        "hotspot_iptm": hotspot["iptm"],
        "delta_iptm": hotspot["iptm"] - baseline["iptm"],
        "baseline_best_loss": baseline["best_loss"],
        "hotspot_best_loss": hotspot["best_loss"],
        "delta_best_loss": hotspot["best_loss"] - baseline["best_loss"],
        "same_loss_weights": hotspot["loss_weights_key"] == baseline["loss_weights_key"],
        "same_template": hotspot["target_template_path"] == baseline["target_template_path"],
        "same_binder_length": hotspot["binder_length"] == baseline["binder_length"],
        "same_init_sequence": hotspot["init_binder_seq"] == baseline["init_binder_seq"],
        "control_note": baseline_note,
        "baseline_result_path": baseline["result_path"],
        "hotspot_result_path": hotspot["result_path"],
        "baseline_meta_path": baseline["meta_path"],
        "hotspot_meta_path": hotspot["meta_path"],
    }


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fields = sorted({field for row in rows for field in row})
    preferred = [
        "target",
        "condition",
        "hotspot",
        "baseline",
        "hotspot_residues",
        "plddt",
        "baseline_plddt",
        "hotspot_plddt",
        "delta_plddt",
        "confidence_score",
        "baseline_confidence_score",
        "hotspot_confidence_score",
        "delta_confidence_score",
        "iptm",
        "baseline_iptm",
        "hotspot_iptm",
        "delta_iptm",
        "best_loss",
        "baseline_best_loss",
        "hotspot_best_loss",
        "delta_best_loss",
        "best_step",
        "binder_length",
        "same_loss_weights",
        "same_template",
        "same_binder_length",
        "same_init_sequence",
        "control_note",
        "result_path",
        "meta_path",
    ]
    ordered = [field for field in preferred if field in fields]
    ordered.extend(field for field in fields if field not in set(ordered) and field not in {"init_binder_seq", "best_binder_seq"})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in ordered})


def fmt(value: float, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def delta_fmt(value: float, digits: int = 3) -> str:
    text = fmt(value, digits)
    if text == "NA":
        return text
    return f"+{text}" if float(value) > 0 else text


def plot_absolute(records: list[dict], output: Path) -> None:
    rows = records
    labels = [f"{row['target']}\n{row['condition']}" for row in rows]
    x = list(range(len(rows)))
    width = 0.36

    fig, ax1 = plt.subplots(figsize=(13, 5.5))
    ax2 = ax1.twinx()
    plddt = [row["plddt"] for row in rows]
    confidence = [row["confidence_score"] for row in rows]

    bars1 = ax1.bar([v - width / 2 for v in x], plddt, width, label="pLDDT", color="#2a9d8f")
    bars2 = ax2.bar([v + width / 2 for v in x], confidence, width, label="confidence score", color="#e76f51")
    ax1.set_ylabel("pLDDT")
    ax2.set_ylabel("Confidence score (ranking_score)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right")
    ax1.set_title("No-hotspot and hotspot conditions: pLDDT and confidence score")
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    handles = [bars1, bars2]
    labels_legend = [handle.get_label() for handle in handles]
    ax1.legend(handles, labels_legend, loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_deltas(comparisons: list[dict], output: Path) -> None:
    labels = [f"{row['target']}\n{row['hotspot']}" for row in comparisons]
    x = list(range(len(comparisons)))
    width = 0.36

    fig, ax1 = plt.subplots(figsize=(13, 5.5))
    ax2 = ax1.twinx()
    delta_plddt = [row["delta_plddt"] for row in comparisons]
    delta_confidence = [row["delta_confidence_score"] for row in comparisons]

    colors_plddt = ["#c0392b" if value >= 0 else "#457b9d" for value in delta_plddt]
    colors_conf = ["#f4a261" if value >= 0 else "#90caf9" for value in delta_confidence]
    bars1 = ax1.bar([v - width / 2 for v in x], delta_plddt, width, label="delta pLDDT", color=colors_plddt)
    bars2 = ax2.bar(
        [v + width / 2 for v in x],
        delta_confidence,
        width,
        label="delta confidence score",
        color=colors_conf,
    )
    ax1.axhline(0, color="#333333", linewidth=0.8)
    ax2.axhline(0, color="#333333", linewidth=0.8)
    ax1.set_ylabel("Delta pLDDT")
    ax2.set_ylabel("Delta confidence score")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right")
    ax1.set_title("Hotspot - no-hotspot metric deltas")
    ax1.grid(axis="y", alpha=0.25)

    handles = [bars1, bars2]
    labels_legend = [handle.get_label() for handle in handles]
    ax1.legend(handles, labels_legend, loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def write_report(comparisons: list[dict], path: Path) -> None:
    lines = [
        "# Hotspot 对 pLDDT 和 confidence score 的影响",
        "",
        "这里使用 best checkpoint 的 `summary_confidence`，其中 `confidence score` 对应 `ranking_score`。差值定义为 `hotspot - no hotspot`，因此正值表示添加 hotspot 后指标更高。",
        "",
        "## 对照说明",
        "",
        "- PD-L1、IFNAR2、DerF21、DerF7 使用 `custom_loss_no_hotspot_random_avglen` 作为 no-hotspot 对照；这些对照与 hotspot run 在 loss weights、template 状态、binder length 和初始序列上匹配。",
        "- PD1 当前没有完全只差 hotspot 的 no-hotspot run；这里使用同 template、同 binder length 的 default/template no-hotspot run 作为 nearest available control。因此 PD1 只能作为参考，不应写成严格 only-hotspot ablation。",
        "",
        "## 指标差值",
        "",
        "| Target | Hotspot | delta pLDDT | delta confidence score | delta ipTM | Control quality |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in comparisons:
        control_quality = (
            "matched"
            if row["same_loss_weights"] and row["same_template"] and row["same_binder_length"] and row["same_init_sequence"]
            else "nearest available"
        )
        lines.append(
            "| {target} | {hotspot} | {plddt} | {conf} | {iptm} | {quality} |".format(
                target=row["target"],
                hotspot=row["hotspot"],
                plddt=delta_fmt(row["delta_plddt"]),
                conf=delta_fmt(row["delta_confidence_score"]),
                iptm=delta_fmt(row["delta_iptm"]),
                quality=control_quality,
            )
        )

    lines.extend(
        [
            "",
            "## 结论",
            "",
        ]
    )
    positive = [
        row
        for row in comparisons
        if row["delta_plddt"] > 0 and row["delta_confidence_score"] > 0
    ]
    negative = [
        row
        for row in comparisons
        if row["delta_plddt"] < 0 and row["delta_confidence_score"] < 0
    ]
    lines.append(
        f"- 同时提升 pLDDT 和 confidence score 的条件数：{len(positive)} / {len(comparisons)}。"
    )
    lines.append(
        f"- 同时降低 pLDDT 和 confidence score 的条件数：{len(negative)} / {len(comparisons)}。"
    )
    lines.append(
        "- 在严格匹配的条件中，PD-L1 和 IFNAR2 添加 hotspot 后 pLDDT 与 confidence score 都下降；DerF7 两个 hotspot 条件两个指标都上升。"
    )
    lines.append(
        "- DerF21 的 B34 同时提升 pLDDT 与 confidence score；B52、B87、B105、B116 的 pLDDT 上升，但 confidence score 小幅下降。"
    )
    lines.append(
        "- PD1 的指标看起来上升，但该比较不是严格消融，因为 baseline 与 hotspot run 的初始序列不同，且 loss 配置记录方式不同。"
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    comparisons = []
    for target, config in TARGETS.items():
        baseline = run_record(target, config["baseline_label"], config["baseline"], config["baseline_note"])
        records.append(baseline)
        for hotspot_label, hotspot_path in config["hotspots"]:
            hotspot = run_record(target, hotspot_label, hotspot_path, config["baseline_note"])
            records.append(hotspot)
            comparisons.append(compare(hotspot, baseline, config["baseline_note"]))

    write_csv(records, OUTPUT_DIR / "target_hotspot_confidence_metrics.csv")
    write_csv(comparisons, OUTPUT_DIR / "target_hotspot_confidence_deltas.csv")
    write_report(comparisons, OUTPUT_DIR / "target_hotspot_confidence_report_zh.md")

    plot_absolute(records, PICTURE_DIR / "target_hotspot_confidence_metrics.png")
    plot_deltas(comparisons, PICTURE_DIR / "target_hotspot_confidence_deltas.png")
    print(f"Wrote {OUTPUT_DIR}")
    print(f"Wrote {PICTURE_DIR / 'target_hotspot_confidence_metrics.png'}")
    print(f"Wrote {PICTURE_DIR / 'target_hotspot_confidence_deltas.png'}")


if __name__ == "__main__":
    main()
