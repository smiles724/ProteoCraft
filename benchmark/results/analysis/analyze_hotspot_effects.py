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
RESULTS_ROOT = PROJECT_ROOT / "benchmark" / "results"
OUTPUT_DIR = RESULTS_ROOT / "analysis" / "hotspot_effects"

HOTSPOT_RESULT = (
    RESULTS_ROOT
    / "pd1_default_loss_hotspot_B64_B126_B129_B133"
    / "pd1_default_loss_hotspot_B64_B126_B129_B133"
    / "PD1"
    / "result.json"
)

BASELINE_RESULTS = {
    "same_template_seed43_default_no_hotspot": (
        RESULTS_ROOT / "bc_v1_random_avglen_hybrid_template" / "bc_v1_random_avglen_hybrid_template" / "PD1" / "result.json"
    ),
    "same_template_seed43_v1loss_no_hotspot": (
        RESULTS_ROOT
        / "bc_v1_random_avglen_hybrid_v1loss_template"
        / "bc_v1_random_avglen_hybrid_v1loss_template"
        / "PD1"
        / "result.json"
    ),
    "custom_loss_template_seed42_no_hotspot": (
        RESULTS_ROOT
        / "custom_loss_with_template_no_hotspot_random_avglen"
        / "custom_loss_with_template_no_hotspot_random_avglen"
        / "PD1"
        / "result.json"
    ),
    "custom_loss_no_template_seed42_no_hotspot": (
        RESULTS_ROOT / "custom_loss_no_hotspot_random_avglen" / "custom_loss_no_hotspot_random_avglen" / "PD1" / "result.json"
    ),
    "bc_v1_random_seed43_no_template_no_hotspot": (
        RESULTS_ROOT / "bc_v1_random_avglen_hybrid" / "bc_v1_random_avglen_hybrid" / "PD1" / "result.json"
    ),
}

PRIMARY_BASELINE = "same_template_seed43_default_no_hotspot"
LOSS_METRICS = ["total", "plddt", "pae", "i_pae", "con", "i_con", "iptm", "rg", "helix", "clash", "i_clash", "pde", "i_pde"]
CONF_METRICS = ["ranking_score", "plddt", "iptm", "ptm", "gpde", "chain_iptm", "chain_ptm"]


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def hamming_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(left != right for left, right in zip(a, b))


def find_best_meta(result_path: Path, result: dict) -> dict:
    case_dir = result_path.parent
    best_step = int(result["best_step"])
    paths = sorted((case_dir / "checkpoints" / f"step_{best_step:04d}").glob("*_meta.json"))
    if not paths:
        return {}

    best_seq = result.get("best_seq")
    metas = [load_json(path) for path in paths]
    exact = [meta for meta in metas if meta.get("binder_seq") == best_seq]
    candidates = exact or metas
    return min(candidates, key=lambda meta: float(meta.get("loss_dict", {}).get("total", math.inf)))


def as_float_map(values: dict) -> dict:
    output = {}
    for key, value in values.items():
        try:
            output[key] = float(value)
        except (TypeError, ValueError):
            pass
    return output


def load_run(label: str, result_path: Path) -> dict:
    result = load_json(result_path)
    meta = find_best_meta(result_path, result)
    params = result.get("params", {})
    return {
        "label": label,
        "result_path": str(result_path.relative_to(PROJECT_ROOT)),
        "entry_id": result.get("entry_id"),
        "binder_length": int(result.get("binder_length", 0)),
        "init_seq": result.get("init_binder_seq", ""),
        "best_seq": result.get("best_seq", ""),
        "best_loss": float(result.get("best_loss", math.nan)),
        "final_loss": float(result.get("final_loss", math.nan)),
        "best_step": int(result.get("best_step", -1)),
        "elapsed_sec": float(result.get("elapsed_sec", math.nan)),
        "target_hotspot_residues": result.get("target_hotspot_residues"),
        "target_hotspot_labels": ";".join(result.get("target_hotspot_labels", [])),
        "target_hotspot_concat_indices": ";".join(str(v) for v in result.get("target_hotspot_concat_indices", [])),
        "target_template_path": params.get("target_template_path"),
        "loss_weights_override": json.dumps(params.get("loss_weights_override"), ensure_ascii=False, sort_keys=True),
        "mutation_count": hamming_distance(result.get("init_binder_seq", ""), result.get("best_seq", "")),
        "loss": as_float_map(meta.get("loss_dict", {})),
        "conf": as_float_map(meta.get("summary_confidence", {})),
    }


def flatten_run(run: dict) -> dict:
    row = {key: value for key, value in run.items() if key not in {"loss", "conf", "init_seq", "best_seq"}}
    row["init_binder_seq"] = run["init_seq"]
    row["best_seq"] = run["best_seq"]
    for metric in LOSS_METRICS:
        if metric in run["loss"]:
            row[f"loss_{metric}"] = run["loss"][metric]
    for metric in CONF_METRICS:
        if metric in run["conf"]:
            row[f"conf_{metric}"] = run["conf"][metric]
    return row


def compare_runs(hotspot: dict, baseline: dict) -> dict:
    row = {
        "baseline": baseline["label"],
        "baseline_result_path": baseline["result_path"],
        "baseline_best_loss": baseline["best_loss"],
        "hotspot_best_loss": hotspot["best_loss"],
        "delta_best_loss_hotspot_minus_baseline": hotspot["best_loss"] - baseline["best_loss"],
        "baseline_final_loss": baseline["final_loss"],
        "hotspot_final_loss": hotspot["final_loss"],
        "delta_final_loss_hotspot_minus_baseline": hotspot["final_loss"] - baseline["final_loss"],
        "baseline_best_step": baseline["best_step"],
        "hotspot_best_step": hotspot["best_step"],
        "baseline_mutation_count": baseline["mutation_count"],
        "hotspot_mutation_count": hotspot["mutation_count"],
        "delta_mutation_count": hotspot["mutation_count"] - baseline["mutation_count"],
        "same_binder_length": hotspot["binder_length"] == baseline["binder_length"],
        "same_init_sequence": hotspot["init_seq"] == baseline["init_seq"],
        "baseline_has_template": bool(baseline.get("target_template_path")),
    }
    for metric in LOSS_METRICS:
        if metric in hotspot["loss"] and metric in baseline["loss"]:
            row[f"baseline_loss_{metric}"] = baseline["loss"][metric]
            row[f"hotspot_loss_{metric}"] = hotspot["loss"][metric]
            row[f"delta_loss_{metric}"] = hotspot["loss"][metric] - baseline["loss"][metric]
    for metric in CONF_METRICS:
        if metric in hotspot["conf"] and metric in baseline["conf"]:
            row[f"baseline_conf_{metric}"] = baseline["conf"][metric]
            row[f"hotspot_conf_{metric}"] = hotspot["conf"][metric]
            row[f"delta_conf_{metric}"] = hotspot["conf"][metric] - baseline["conf"][metric]
    return row


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "label",
        "baseline",
        "entry_id",
        "binder_length",
        "target_hotspot_residues",
        "target_hotspot_labels",
        "target_hotspot_concat_indices",
        "best_loss",
        "final_loss",
        "best_step",
        "conf_ranking_score",
        "conf_plddt",
        "conf_iptm",
        "conf_gpde",
        "loss_i_con",
        "loss_i_pae",
        "loss_con",
        "mutation_count",
        "baseline_best_loss",
        "hotspot_best_loss",
        "delta_best_loss_hotspot_minus_baseline",
        "baseline_conf_ranking_score",
        "hotspot_conf_ranking_score",
        "delta_conf_ranking_score",
        "baseline_conf_plddt",
        "hotspot_conf_plddt",
        "delta_conf_plddt",
        "baseline_conf_iptm",
        "hotspot_conf_iptm",
        "delta_conf_iptm",
        "baseline_loss_i_con",
        "hotspot_loss_i_con",
        "delta_loss_i_con",
        "same_binder_length",
        "same_init_sequence",
        "baseline_has_template",
        "result_path",
        "baseline_result_path",
    ]
    ordered = [name for name in preferred if name in fieldnames]
    ordered.extend(name for name in fieldnames if name not in set(ordered))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def delta_text(row: dict, key: str, digits: int = 3) -> str:
    if key not in row:
        return "NA"
    value = float(row[key])
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{digits}f}"


def write_report(hotspot: dict, comparisons: list[dict], path: Path) -> None:
    primary = next(row for row in comparisons if row["baseline"] == PRIMARY_BASELINE)
    lines = [
        "# PD1 hotspot B64/B126/B129/B133 影响分析",
        "",
        "## 输入结果",
        "",
        f"- hotspot result：`{hotspot['result_path']}`",
        "- 配置：`benchmark/yaml/pd1_default_loss_hotspot_B64_B126_B129_B133.yaml`",
        f"- hotspot：`{hotspot['target_hotspot_residues']}`，解析为 `{hotspot['target_hotspot_labels']}`，concat indices `{hotspot['target_hotspot_concat_indices']}`。",
        f"- template：`{hotspot['target_template_path']}`",
        f"- binder length：{hotspot['binder_length']}；best step：{hotspot['best_step']}；运行时间：{fmt(hotspot['elapsed_sec'] / 60, 1)} min。",
        "",
        "## Hotspot run 指标",
        "",
        "| best_loss | final_loss | ranking_score | pLDDT | ipTM | gpDE | i_con loss | i_pae loss | mutations |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| {best} | {final} | {rank} | {plddt} | {iptm} | {gpde} | {icon} | {ipae} | {mut} |".format(
            best=fmt(hotspot["best_loss"]),
            final=fmt(hotspot["final_loss"]),
            rank=fmt(hotspot["conf"].get("ranking_score")),
            plddt=fmt(hotspot["conf"].get("plddt")),
            iptm=fmt(hotspot["conf"].get("iptm")),
            gpde=fmt(hotspot["conf"].get("gpde")),
            icon=fmt(hotspot["loss"].get("i_con")),
            ipae=fmt(hotspot["loss"].get("i_pae")),
            mut=fmt(hotspot["mutation_count"], 0),
        ),
        "",
        "## 与 no-hotspot 结果的参考对比",
        "",
        "这里列出的是已有 PD1 no-hotspot runs。注意这次 hotspot run 与这些 baseline 不一定只差 hotspot：部分 run 的 loss weights、template、random seed 或初始序列不同，所以差值应作为参考，不应直接解释为纯 hotspot 因果效应。",
        "",
        "| Baseline | same init | template | delta best_loss | delta ranking_score | delta pLDDT | delta ipTM | delta i_con loss | delta mutations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            "| {baseline} | {same_init} | {template} | {best} | {rank} | {plddt} | {iptm} | {icon} | {mut} |".format(
                baseline=row["baseline"],
                same_init="yes" if row["same_init_sequence"] else "no",
                template="yes" if row["baseline_has_template"] else "no",
                best=delta_text(row, "delta_best_loss_hotspot_minus_baseline"),
                rank=delta_text(row, "delta_conf_ranking_score"),
                plddt=delta_text(row, "delta_conf_plddt"),
                iptm=delta_text(row, "delta_conf_iptm"),
                icon=delta_text(row, "delta_loss_i_con"),
                mut=delta_text(row, "delta_mutation_count", 0),
            )
        )
    lines.extend(
        [
            "",
            "## 主要观察",
            "",
            f"- 相对最接近的 no-hotspot 参考 `{PRIMARY_BASELINE}`，hotspot run 的 best_loss 变化为 {delta_text(primary, 'delta_best_loss_hotspot_minus_baseline')}，ranking_score 变化为 {delta_text(primary, 'delta_conf_ranking_score')}，pLDDT 变化为 {delta_text(primary, 'delta_conf_plddt')}，ipTM 变化为 {delta_text(primary, 'delta_conf_iptm')}。",
            f"- hotspot run 的接口接触项 i_con loss 为 {fmt(hotspot['loss'].get('i_con'))}；相对 `{PRIMARY_BASELINE}` 的 i_con loss 变化为 {delta_text(primary, 'delta_loss_i_con')}。在这个 loss 定义下，i_con loss 越低通常表示接口接触目标更好。",
            f"- best step 为 {hotspot['best_step']}，之后 loss 没有继续刷新；最终序列与 best 序列一致。",
            "- 现有数据不能严格隔离 hotspot 的单独贡献，因为缺少一组与该 hotspot run 完全相同但 `target_hotspot_residues: null` 的 PD1 对照。更严谨的 hotspot effect 需要补跑这一组 only-hotspot-ablation。",
            "",
            "## 文件",
            "",
            "- `pd1_hotspot_run.csv`：hotspot run 的完整指标。",
            "- `pd1_hotspot_vs_no_hotspot.csv`：hotspot run 与各 no-hotspot PD1 结果的差值。",
            "- `pd1_hotspot_vs_no_hotspot_delta_conf.png`：confidence 指标差值图。",
            "- `pd1_hotspot_vs_no_hotspot_delta_loss.png`：loss 指标差值图。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_comparison(comparisons: list[dict], keys: list[str], labels: list[str], title: str, ylabel: str, output: Path) -> None:
    x = list(range(len(comparisons)))
    width = 0.8 / len(keys)
    plt.figure(figsize=(12, 5))
    for i, (key, label) in enumerate(zip(keys, labels)):
        values = [float(row[key]) if key in row else math.nan for row in comparisons]
        positions = [value + (i - (len(keys) - 1) / 2) * width for value in x]
        plt.bar(positions, values, width=width, label=label)
    plt.axhline(0, color="#333333", linewidth=0.8)
    plt.xticks(x, [row["baseline"] for row in comparisons], rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hotspot = load_run("pd1_hotspot_B64_B126_B129_B133", HOTSPOT_RESULT)
    baselines = [load_run(label, path) for label, path in BASELINE_RESULTS.items() if path.exists()]
    comparisons = [compare_runs(hotspot, baseline) for baseline in baselines]

    write_csv([flatten_run(hotspot)], OUTPUT_DIR / "pd1_hotspot_run.csv")
    write_csv(comparisons, OUTPUT_DIR / "pd1_hotspot_vs_no_hotspot.csv")
    write_report(hotspot, comparisons, OUTPUT_DIR / "hotspot_report_zh.md")
    plot_comparison(
        comparisons,
        ["delta_conf_ranking_score", "delta_conf_plddt", "delta_conf_iptm"],
        ["ranking_score", "pLDDT", "ipTM"],
        "PD1 hotspot - no-hotspot confidence deltas",
        "Delta",
        OUTPUT_DIR / "pd1_hotspot_vs_no_hotspot_delta_conf.png",
    )
    plot_comparison(
        comparisons,
        ["delta_best_loss_hotspot_minus_baseline", "delta_loss_i_con", "delta_loss_i_pae"],
        ["best_loss", "i_con loss", "i_pae loss"],
        "PD1 hotspot - no-hotspot loss deltas",
        "Delta",
        OUTPUT_DIR / "pd1_hotspot_vs_no_hotspot_delta_loss.png",
    )
    print(f"Wrote PD1 hotspot analysis to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
