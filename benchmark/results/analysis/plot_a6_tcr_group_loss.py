#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS_DIR = Path("/data/huangyanjie/binder_design/BindCraft-v2/benchmark/results/A6_TCR")
DEFAULT_OUTPUT = Path(
    "/data/huangyanjie/binder_design/BindCraft-v2/benchmark/results/analysis/A6_TCR/group_mean_loss_history.png"
)
EXPECTED_GROUPS = [
    "alpha_patch_19_23",
    "alpha_patch_53_56",
    "alpha_patch_63_66",
    "alpha_patch_72_76",
    "beta_patch_21_24",
    "beta_patch_5_7",
    "beta_patch_66_69",
    "beta_patch_74_77",
]
RUN_DIR_PATTERN = re.compile(r"^(?P<group>.+)_batch_(?P<batch>\d+)$")
EXPECTED_RUNS_PER_GROUP = 5
EXPECTED_HISTORY_LENGTH = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mean loss_history curves for A6_TCR benchmark groups"
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--title",
        type=str,
        default="A6_TCR mean loss history by patch group",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def load_loss_history(result_path: Path) -> list[float]:
    with result_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if "loss_history" not in payload:
        raise ValueError(f"Missing loss_history in {result_path}")

    loss_history = payload["loss_history"]
    if not isinstance(loss_history, list):
        raise ValueError(f"loss_history is not a list in {result_path}")
    if len(loss_history) != EXPECTED_HISTORY_LENGTH:
        raise ValueError(
            f"loss_history length is {len(loss_history)} in {result_path}, expected {EXPECTED_HISTORY_LENGTH}"
        )

    try:
        return [float(value) for value in loss_history]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"loss_history contains non-numeric values in {result_path}") from exc


def collect_group_histories(results_dir: Path) -> dict[str, list[list[float]]]:
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    grouped_runs: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        match = RUN_DIR_PATTERN.match(child.name)
        if not match:
            continue
        group_name = match.group("group")
        batch_index = int(match.group("batch"))
        grouped_runs[group_name].append((batch_index, child))

    found_groups = sorted(grouped_runs)
    if found_groups != sorted(EXPECTED_GROUPS):
        raise ValueError(
            f"Found groups {found_groups}, expected {sorted(EXPECTED_GROUPS)}"
        )

    grouped_histories: dict[str, list[list[float]]] = {}
    for group_name in sorted(EXPECTED_GROUPS):
        runs = sorted(grouped_runs[group_name], key=lambda item: item[0])
        if len(runs) != EXPECTED_RUNS_PER_GROUP:
            raise ValueError(
                f"Group {group_name} has {len(runs)} runs, expected {EXPECTED_RUNS_PER_GROUP}"
            )

        histories: list[list[float]] = []
        for batch_index, run_dir in runs:
            result_path = run_dir / "result.json"
            if not result_path.is_file():
                raise FileNotFoundError(f"Missing result.json: {result_path}")
            history = load_loss_history(result_path)
            histories.append(history)
        grouped_histories[group_name] = histories

    return grouped_histories


def compute_group_means(grouped_histories: dict[str, list[list[float]]]) -> dict[str, np.ndarray]:
    group_means: dict[str, np.ndarray] = {}
    for group_name, histories in grouped_histories.items():
        array = np.asarray(histories, dtype=float)
        if array.shape != (EXPECTED_RUNS_PER_GROUP, EXPECTED_HISTORY_LENGTH):
            raise ValueError(
                f"Unexpected shape for {group_name}: {array.shape}, expected ({EXPECTED_RUNS_PER_GROUP}, {EXPECTED_HISTORY_LENGTH})"
            )
        group_means[group_name] = array.mean(axis=0)
    return group_means


def plot_group_means(group_means: dict[str, np.ndarray], output_path: Path, title: str, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    steps = np.arange(1, EXPECTED_HISTORY_LENGTH + 1)
    plt.figure(figsize=(12, 7))
    for group_name in sorted(group_means):
        plt.plot(steps, group_means[group_name], linewidth=2, label=group_name)

    plt.xlabel("Optimization step")
    plt.ylabel("Mean loss")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def print_summary(grouped_histories: dict[str, list[list[float]]], group_means: dict[str, np.ndarray]) -> None:
    print(f"Detected {len(grouped_histories)} groups")
    for group_name in sorted(grouped_histories):
        print(
            f"- {group_name}: {len(grouped_histories[group_name])} runs, mean curve length {len(group_means[group_name])}"
        )


def main() -> None:
    args = parse_args()
    grouped_histories = collect_group_histories(args.results_dir)
    group_means = compute_group_means(grouped_histories)
    plot_group_means(group_means, args.output, args.title, args.dpi)
    print_summary(grouped_histories, group_means)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
