#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/data/huangyanjie/binder_design/BindCraft-v2")
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "benchmark" / "results"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "benchmark" / "results" / "analysis" / "loss_history"
PREFERRED_CASE_ORDER = ["BetV1", "DerF21", "DerF7", "HER2", "IFNAR2", "PD-L1", "PD1", "SpCas9"]
METRICS = [
    "total",
    "plddt",
    "pae",
    "i_pae",
    "con",
    "i_con",
    "iptm",
    "rg",
    "helix",
    "clash",
    "i_clash",
    "pde",
    "i_pde",
]
STEP_RE = re.compile(r"step_(\d+)")
EXCLUDED_DISCOVERY_PARTS = {"analysis", "__pycache__"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-metric loss histories from benchmark checkpoint meta files. "
            "By default the script auto-discovers experiment directories under benchmark/results."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory used for experiment auto-discovery and output naming",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where plots/CSV files are written",
    )
    parser.add_argument(
        "--experiment-dir",
        dest="experiment_dirs",
        type=Path,
        action="append",
        default=[],
        help=(
            "Leaf experiment directory that directly contains per-case subdirectories with "
            "checkpoints/. Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--case-order",
        type=str,
        default=None,
        help=(
            "Optional comma-separated case order. Cases not listed here are appended alphabetically. "
            "If omitted, bc_v1 cases use the preferred order and other experiments use alphabetical order."
        ),
    )
    return parser.parse_args()


def parse_step(meta_path: Path, meta: dict) -> int:
    if "step" in meta:
        return int(meta["step"])
    match = STEP_RE.search(str(meta_path.parent.name))
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not parse step from {meta_path}")


def load_case_history(case_dir: Path) -> dict[str, list[tuple[int, float]]]:
    checkpoint_dir = case_dir / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints directory: {checkpoint_dir}")

    histories = {metric: [] for metric in METRICS}
    meta_paths = sorted(checkpoint_dir.glob("step_*/*_meta.json"))
    if not meta_paths:
        raise FileNotFoundError(f"No checkpoint meta files found in {checkpoint_dir}")

    for meta_path in meta_paths:
        with meta_path.open() as handle:
            meta = json.load(handle)
        step = parse_step(meta_path, meta)
        loss_dict = meta.get("loss_dict", {})
        for metric in METRICS:
            if metric not in loss_dict:
                print(f"Warning: {case_dir.name} step={step} missing metric={metric}")
                continue
            histories[metric].append((step, float(loss_dict[metric])))

    for metric in METRICS:
        histories[metric].sort(key=lambda item: item[0])
    return histories


def resolve_case_order(case_names: list[str], case_order_arg: str | None) -> list[str]:
    if case_order_arg:
        requested = [part.strip() for part in case_order_arg.split(",") if part.strip()]
        ordered = [name for name in requested if name in case_names]
        remaining = sorted(name for name in case_names if name not in set(ordered))
        return ordered + remaining

    preferred = [name for name in PREFERRED_CASE_ORDER if name in case_names]
    remaining = sorted(name for name in case_names if name not in set(preferred))
    return preferred + remaining


def plot_metric(
    experiment_name: str,
    experiment_data: dict[str, dict[str, list[tuple[int, float]]]],
    case_order: list[str],
    metric: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(12, 7))
    plotted = 0
    for case_name in case_order:
        case_history = experiment_data.get(case_name, {})
        points = case_history.get(metric, [])
        if not points:
            print(f"Warning: no points for {experiment_name} {case_name} {metric}")
            continue
        steps = [step for step, _ in points]
        values = [value for _, value in points]
        plt.plot(steps, values, marker="o", markersize=2.5, linewidth=1.7, label=case_name)
        plotted += 1

    plt.title(f"{experiment_name} - {metric}")
    plt.xlabel("Step")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)
    if plotted:
        plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_long_csv(
    experiment_name: str,
    experiment_data: dict[str, dict[str, list[tuple[int, float]]]],
    case_order: list[str],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment", "case", "step", "metric", "value"])
        writer.writeheader()
        for case_name in case_order:
            case_history = experiment_data.get(case_name, {})
            for metric in METRICS:
                for step, value in case_history.get(metric, []):
                    writer.writerow(
                        {
                            "experiment": experiment_name,
                            "case": case_name,
                            "step": step,
                            "metric": metric,
                            "value": value,
                        }
                    )


def directory_has_case_checkpoints(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        children = list(path.iterdir())
    except OSError:
        return False
    return any(child.is_dir() and (child / "checkpoints").is_dir() for child in children)


def should_skip_discovery(path: Path, results_root: Path) -> bool:
    try:
        relative_parts = path.relative_to(results_root).parts
    except ValueError:
        relative_parts = path.parts
    return any(part in EXCLUDED_DISCOVERY_PARTS for part in relative_parts)


def discover_experiment_dirs(results_root: Path) -> list[Path]:
    candidates = [results_root]
    candidates.extend(sorted(path for path in results_root.rglob("*") if path.is_dir()))

    discovered: list[Path] = []
    for path in candidates:
        if should_skip_discovery(path, results_root):
            continue
        if directory_has_case_checkpoints(path):
            discovered.append(path)
    return discovered


def build_experiment_name(experiment_dir: Path, results_root: Path) -> str:
    try:
        relative = experiment_dir.resolve().relative_to(results_root.resolve())
    except ValueError:
        return experiment_dir.name
    collapsed_parts: list[str] = []
    for part in relative.parts:
        if collapsed_parts and part == collapsed_parts[-1]:
            continue
        collapsed_parts.append(part)
    if len(collapsed_parts) == 1:
        return collapsed_parts[0]
    return "__".join(collapsed_parts)


def process_experiment(
    experiment_dir: Path,
    experiment_name: str,
    output_root: Path,
    case_order_arg: str | None,
) -> None:
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")

    case_dirs = sorted(
        child for child in experiment_dir.iterdir() if child.is_dir() and (child / "checkpoints").is_dir()
    )
    if not case_dirs:
        raise FileNotFoundError(
            f"No case directories with checkpoints/ found in experiment dir: {experiment_dir}"
        )

    experiment_data = {case_dir.name: load_case_history(case_dir) for case_dir in case_dirs}
    case_order = resolve_case_order(sorted(experiment_data), case_order_arg)

    output_dir = output_root / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in METRICS:
        plot_metric(experiment_name, experiment_data, case_order, metric, output_dir / f"loss_{metric}.png")
    write_long_csv(experiment_name, experiment_data, case_order, output_dir / "loss_history_long.csv")

    png_count = len(list(output_dir.glob("loss_*.png")))
    csv_rows = sum(
        len(experiment_data.get(case_name, {}).get(metric, []))
        for case_name in case_order
        for metric in METRICS
    )
    print(
        f"{experiment_name}: cases={len(experiment_data)} png={png_count} "
        f"csv_rows={csv_rows} source={experiment_dir} output={output_dir}"
    )


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    output_root = args.output_root.resolve()

    if args.experiment_dirs:
        experiment_dirs = [path.resolve() for path in args.experiment_dirs]
    else:
        experiment_dirs = discover_experiment_dirs(results_root)

    if not experiment_dirs:
        raise FileNotFoundError(
            f"No experiment directories were found under {results_root}. "
            "Use --experiment-dir to point at a specific result directory."
        )

    print(f"Found {len(experiment_dirs)} experiment(s)")
    for experiment_dir in experiment_dirs:
        experiment_name = build_experiment_name(experiment_dir, results_root)
        process_experiment(experiment_dir, experiment_name, output_root, args.case_order)


if __name__ == "__main__":
    main()
