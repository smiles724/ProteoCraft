#!/usr/bin/env python3
"""Five-seed, unoptimized-ESMFold2 rescoring for the fixed 13-target panel.

The search seed is deliberately not reused as an evaluation seed.  Every
target/arm candidate is evaluated with the same five seeds, and candidate
means are computed before choosing the best of the five search candidates.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import run_esm_optimization as experiments
import run_esm_target_panel as panel
import run_esm_target_panel_13_fixed as fixed_panel


DEFAULT_OUTPUT = ROOT / "outputs/esm_optimization/target_panel_13_reporting_fixed"
EVAL_SEEDS = (1042, 1043, 1044, 1045, 1046)
ARMS = ("baseline", "exact", "fast", "big")
TOP_K = 5
LOOPS = 4
SAMPLING_STEPS = 50
SUCCESS_THRESHOLD = 0.8
PRIMARY_METRIC = "pair_chain_iptm"
REPORT_METRICS = (
    "pair_chain_iptm",
    "iptm",
    "ptm",
    "plddt",
    "binder_plddt",
    "binder_radius_gyration",
    "binder_rg_excess",
    "hotspot_mean_min_heavy_atom_distance",
    "hotspot_heavy_atom_coverage",
    "total_loss",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def campaign_records(output: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    campaign = json.loads((output / "campaign.json").read_text())
    by_case: dict[str, list[dict[str, Any]]] = {}
    for record in campaign["records"]:
        by_case.setdefault(record["case"]["case_id"], []).append(record)
    expected_cases = {case_id for _, case_id, _, _ in fixed_panel.TARGET_SPECS}
    if set(by_case) != expected_cases:
        raise RuntimeError(
            f"Campaign cases differ from the 13-target specification: "
            f"missing={sorted(expected_cases-set(by_case))}, "
            f"extra={sorted(set(by_case)-expected_cases)}"
        )
    for case_id, records in by_case.items():
        observed = {record["arm"] for record in records}
        if observed != set(ARMS) or len(records) != len(ARMS):
            raise RuntimeError(f"{case_id}: expected one record per arm, found {observed}")
    return campaign, by_case


def rescore_root(output: Path) -> Path:
    return output / "rescore_5seed"


def prepare(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    campaign, by_case = campaign_records(output)
    root = rescore_root(output)
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    arm_order = {arm: index for index, arm in enumerate(ARMS)}
    for _, case_id, _, _ in fixed_panel.TARGET_SPECS:
        records = sorted(by_case[case_id], key=lambda record: arm_order[record["arm"]])
        write_json(
            manifests / f"{case_id}.json",
            panel.sliced_campaign(campaign, records),
        )
    spec = {
        "schema_version": 1,
        "campaign": str((output / "campaign.json").resolve()),
        "evaluator": "unoptimized_esmfold2",
        "model": campaign["models"]["verifier"],
        "selection": "top_5_unique_sequences_by_search_total_loss",
        "top_k": TOP_K,
        "eval_seeds": list(EVAL_SEEDS),
        "search_seeds_are_not_eval_seeds": True,
        "num_loops": LOOPS,
        "num_sampling_steps": SAMPLING_STEPS,
        "primary_metric": PRIMARY_METRIC,
        "aggregation": {
            "candidate": "arithmetic_mean_over_5_eval_seeds",
            "run_top1": "search_rank_1_candidate_seed_mean",
            "run_best5": "maximum_candidate_seed_mean_after_seed_averaging",
            "run_top5_pool": "arithmetic_mean_over_all_25_candidate_seed_scores",
            "arm": "equal_weight_per_target",
        },
        "expected": {
            "targets": len(fixed_panel.TARGET_SPECS),
            "arms_per_target": len(ARMS),
            "candidates_per_run": TOP_K,
            "eval_seeds_per_candidate": len(EVAL_SEEDS),
            "score_files": len(fixed_panel.TARGET_SPECS) * len(ARMS) * TOP_K * len(EVAL_SEEDS),
        },
    }
    spec_path = root / "spec.json"
    if spec_path.is_file() and json.loads(spec_path.read_text()) != spec:
        raise RuntimeError(f"Five-seed rescore contract changed: {spec_path}")
    write_json(spec_path, spec)
    print(f"Prepared five-seed rescore contract at {spec_path}")


def target_environment(root: Path, case_id: str) -> dict[str, str]:
    environment = os.environ.copy()
    cache = root / "runtime_cache" / case_id
    (cache / "triton").mkdir(parents=True, exist_ok=True)
    (cache / "inductor").mkdir(parents=True, exist_ok=True)
    environment["TRITON_CACHE_DIR"] = str(cache / "triton")
    environment["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
    return environment


def load_scores(path: Path) -> list[dict[str, Any]]:
    return [json.loads(score.read_text()) for score in sorted(path.rglob("score.json"))]


def validate_target_scores(
    records: list[dict[str, Any]], scores: list[dict[str, Any]], *, case_id: str
) -> None:
    expected = {
        (record["arm"], record["seed"], rank, eval_seed)
        for record in records
        for rank in range(1, TOP_K + 1)
        for eval_seed in EVAL_SEEDS
    }
    observed = {
        (score["arm"], score["search_seed"], score["rank"], score["eval_seed"])
        for score in scores
        if score.get("case_id") == case_id
    }
    if observed != expected:
        raise RuntimeError(
            f"{case_id}: incomplete five-seed score matrix; "
            f"missing={len(expected-observed)}, extra={len(observed-expected)}"
        )
    for score in scores:
        if score.get("case_id") != case_id:
            raise RuntimeError(f"Unexpected case in {case_id} score directory: {score.get('case_id')}")
        if score.get("evaluator") != "unoptimized_esmfold2":
            raise RuntimeError(f"Unexpected evaluator in score: {score.get('evaluator')}")
        if score.get("loops") != LOOPS or score.get("sampling_steps") != SAMPLING_STEPS:
            raise RuntimeError("ESMFold2 scoring contract mismatch")


def rescore_target(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    root = rescore_root(output)
    campaign, by_case = campaign_records(output)
    del campaign
    index = int(args.index)
    if not 0 <= index < len(fixed_panel.TARGET_SPECS):
        raise ValueError(f"index must be in [0, {len(fixed_panel.TARGET_SPECS)-1}]")
    target_name, case_id, _, _ = fixed_panel.TARGET_SPECS[index]
    records = by_case[case_id]
    for record in records:
        receipt_path = Path(record["directory"]) / "experiment_run.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
        if receipt.get("status") != "complete" or receipt.get("returncode") != 0:
            raise RuntimeError(f"Search is not complete: {record['arm']} {case_id}")

    status_path = root / "status" / f"{case_id}.json"
    started = time.time()
    write_json(status_path, {
        "target_name": target_name,
        "case_id": case_id,
        "status": "running",
        "started_unix": started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    })
    score_dir = root / "scores" / case_id
    log_path = root / "logs" / f"{case_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(panel.RUNNER),
        "rescore",
        "--manifest",
        str(root / "manifests" / f"{case_id}.json"),
        "--output",
        str(score_dir),
        "--top-k",
        str(TOP_K),
        "--seeds",
        *(str(seed) for seed in EVAL_SEEDS),
        "--loops",
        str(LOOPS),
        "--sampling-steps",
        str(SAMPLING_STEPS),
        "--proposal-model",
        str(fixed_panel.DEFAULT_PROPOSAL_MODEL),
        "--verifier-model",
        str(fixed_panel.DEFAULT_VERIFIER_MODEL),
        "--kit-root",
        str(fixed_panel.DEFAULT_UPLIFT_ROOT),
        "--device",
        "cuda:0",
    ]
    try:
        with log_path.open("a", buffering=1) as handle:
            subprocess.run(
                command,
                cwd=ROOT,
                env=target_environment(root, case_id),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        scores = load_scores(score_dir)
        validate_target_scores(records, scores, case_id=case_id)
    except Exception as exc:
        write_json(status_path, {
            "target_name": target_name,
            "case_id": case_id,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "started_unix": started,
            "finished_unix": time.time(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        })
        raise
    finished = time.time()
    write_json(status_path, {
        "target_name": target_name,
        "case_id": case_id,
        "status": "complete",
        "started_unix": started,
        "finished_unix": finished,
        "wall_s": finished - started,
        "score_files": len(scores),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    })
    print(f"Completed five-seed rescore for {target_name}: {len(scores)} scores")


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def metric_summary(values: list[float], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": statistics.mean(values),
        f"{prefix}_std": sample_std(values),
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_interval(values: list[float], *, seed: int = 2026, iterations: int = 10000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    samples = sorted(
        statistics.mean(rng.choices(values, k=len(values)))
        for _ in range(iterations)
    )
    return [percentile(samples, 0.025), percentile(samples, 0.975)]


def geometric_mean(values: list[float]) -> float | None:
    if not values or any(value <= 0 for value in values):
        return None
    return math.exp(statistics.mean(math.log(value) for value in values))


def summarize(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    root = rescore_root(output)
    spec = json.loads((root / "spec.json").read_text())
    campaign, by_case = campaign_records(output)
    scores = load_scores(root / "scores")
    expected_score_count = int(spec["expected"]["score_files"])
    if len(scores) != expected_score_count and not args.allow_partial:
        raise RuntimeError(f"Expected {expected_score_count} scores, found {len(scores)}")

    target_order = {
        case_id: index for index, (_, case_id, _, _) in enumerate(fixed_panel.TARGET_SPECS)
    }
    arm_order = {arm: index for index, arm in enumerate(ARMS)}
    record_map = {
        (record["case"]["case_id"], record["arm"], record["seed"]): record
        for record in campaign["records"]
    }
    raw_rows = []
    for score in scores:
        raw_rows.append({
            "target_name": score["target_name"],
            "case_id": score["case_id"],
            "arm": score["arm"],
            "search_seed": score["search_seed"],
            "rank": score["rank"],
            "eval_seed": score["eval_seed"],
            "binder_seq": score["binder_seq"],
            "elapsed_s": score["elapsed_s"],
            **{metric: score["metrics"].get(metric) for metric in REPORT_METRICS},
        })
    raw_rows.sort(key=lambda row: (
        target_order.get(row["case_id"], 10**9),
        arm_order.get(row["arm"], 10**9),
        row["rank"],
        row["eval_seed"],
    ))
    raw_fields = [
        "target_name", "case_id", "arm", "search_seed", "rank", "eval_seed",
        "binder_seq", "elapsed_s", *REPORT_METRICS,
    ]
    analysis = root / "analysis"
    write_csv(analysis / "seed_scores.csv", raw_rows, raw_fields)

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in raw_rows:
        key = (
            row["target_name"], row["case_id"], row["arm"],
            row["search_seed"], row["rank"], row["binder_seq"],
        )
        grouped.setdefault(key, []).append(row)
    candidate_rows = []
    for key, group in grouped.items():
        row = dict(zip(
            ("target_name", "case_id", "arm", "search_seed", "rank", "binder_seq"),
            key,
        ))
        row["eval_seed_count"] = len(group)
        row["eval_seeds"] = ",".join(str(item["eval_seed"]) for item in group)
        row["elapsed_total_s"] = sum(float(item["elapsed_s"]) for item in group)
        for metric in REPORT_METRICS:
            values = [float(item[metric]) for item in group if item.get(metric) is not None]
            if values:
                row.update(metric_summary(values, metric))
        candidate_rows.append(row)
    candidate_rows.sort(key=lambda row: (
        target_order.get(row["case_id"], 10**9),
        arm_order.get(row["arm"], 10**9),
        row["rank"],
    ))
    candidate_fields = [
        "target_name", "case_id", "arm", "search_seed", "rank", "binder_seq",
        "eval_seed_count", "eval_seeds", "elapsed_total_s",
        *(f"{metric}_{stat}" for metric in REPORT_METRICS for stat in ("mean", "std", "min", "max")),
    ]
    write_csv(analysis / "candidate_summary.csv", candidate_rows, candidate_fields)

    run_groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for row in candidate_rows:
        key = (row["target_name"], row["case_id"], row["arm"], row["search_seed"])
        run_groups.setdefault(key, []).append(row)
    run_rows = []
    for key, candidates in run_groups.items():
        target_name, case_id, arm, search_seed = key
        candidates.sort(key=lambda row: row["rank"])
        top1 = next((row for row in candidates if row["rank"] == 1), None)
        eligible = [row for row in candidates if row.get(f"{PRIMARY_METRIC}_mean") is not None]
        if top1 is None or not eligible:
            continue
        best = max(eligible, key=lambda row: (row[f"{PRIMARY_METRIC}_mean"], -row["rank"]))
        raw_group = [
            row for row in raw_rows
            if (row["case_id"], row["arm"], row["search_seed"])
            == (case_id, arm, search_seed)
        ]
        best_raw = [row for row in raw_group if row["rank"] == best["rank"]]
        receipt_path = Path(record_map[case_id, arm, search_seed]["directory"]) / "experiment_run.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
        row = {
            "target_name": target_name,
            "case_id": case_id,
            "arm": arm,
            "search_seed": search_seed,
            "candidate_count": len(candidates),
            "eval_seed_count_per_candidate": min(item["eval_seed_count"] for item in candidates),
            "rescored_samples": len(raw_group),
            "search_wall_s": receipt.get("wall_s"),
            "top1_pair_chain_iptm_mean": top1[f"{PRIMARY_METRIC}_mean"],
            "top1_pair_chain_iptm_std": top1[f"{PRIMARY_METRIC}_std"],
            "top1_pair_chain_iptm_min": top1[f"{PRIMARY_METRIC}_min"],
            "top1_pair_chain_iptm_max": top1[f"{PRIMARY_METRIC}_max"],
            "best5_rank": best["rank"],
            "best5_binder_seq": best["binder_seq"],
            "best5_pair_chain_iptm_mean": best[f"{PRIMARY_METRIC}_mean"],
            "best5_pair_chain_iptm_std": best[f"{PRIMARY_METRIC}_std"],
            "best5_pair_chain_iptm_min": best[f"{PRIMARY_METRIC}_min"],
            "best5_pair_chain_iptm_max": best[f"{PRIMARY_METRIC}_max"],
            "best5_success_mean_ge_0_8": best[f"{PRIMARY_METRIC}_mean"] >= SUCCESS_THRESHOLD,
            "best5_seed_success_rate": statistics.mean(
                float(item[PRIMARY_METRIC]) >= SUCCESS_THRESHOLD for item in best_raw
            ),
            "top5_pool_pair_chain_iptm_mean": statistics.mean(
                float(item[PRIMARY_METRIC]) for item in raw_group
            ),
            "top5_pool_pair_chain_iptm_std": sample_std([
                float(item[PRIMARY_METRIC]) for item in raw_group
            ]),
            "rescore_elapsed_total_s": sum(float(item["elapsed_s"]) for item in raw_group),
        }
        for metric in REPORT_METRICS:
            for stat in ("mean", "std", "min", "max"):
                value = best.get(f"{metric}_{stat}")
                if value is not None:
                    row[f"best5_{metric}_{stat}"] = value
        run_rows.append(row)
    run_rows.sort(key=lambda row: (
        target_order.get(row["case_id"], 10**9),
        arm_order.get(row["arm"], 10**9),
    ))
    if not run_rows:
        raise RuntimeError("No complete five-seed run summaries are available")
    run_fields = list(dict.fromkeys(key for row in run_rows for key in row))
    write_csv(analysis / "run_summary.csv", run_rows, run_fields)

    run_map = {(row["case_id"], row["arm"]): row for row in run_rows}
    paired_rows = []
    for row in run_rows:
        if row["arm"] == "baseline":
            continue
        baseline = run_map.get((row["case_id"], "baseline"))
        if baseline is None:
            continue
        paired_rows.append({
            "target_name": row["target_name"],
            "case_id": row["case_id"],
            "arm": row["arm"],
            "top1_delta_vs_baseline": row["top1_pair_chain_iptm_mean"] - baseline["top1_pair_chain_iptm_mean"],
            "best5_delta_vs_baseline": row["best5_pair_chain_iptm_mean"] - baseline["best5_pair_chain_iptm_mean"],
            "top5_pool_delta_vs_baseline": row["top5_pool_pair_chain_iptm_mean"] - baseline["top5_pool_pair_chain_iptm_mean"],
            "best5_win_vs_baseline": row["best5_pair_chain_iptm_mean"] > baseline["best5_pair_chain_iptm_mean"],
            "search_speedup_vs_baseline": (
                baseline["search_wall_s"] / row["search_wall_s"]
                if baseline.get("search_wall_s") and row.get("search_wall_s") else None
            ),
        })
    paired_fields = list(paired_rows[0]) if paired_rows else ["arm"]
    write_csv(analysis / "paired_comparison.csv", paired_rows, paired_fields)

    arm_rows = []
    arm_json: dict[str, Any] = {}
    for arm in ARMS:
        selected = [row for row in run_rows if row["arm"] == arm]
        pairs = [row for row in paired_rows if row["arm"] == arm]
        best_values = [row["best5_pair_chain_iptm_mean"] for row in selected]
        top1_values = [row["top1_pair_chain_iptm_mean"] for row in selected]
        pooled_values = [row["top5_pool_pair_chain_iptm_mean"] for row in selected]
        deltas = [row["best5_delta_vs_baseline"] for row in pairs]
        speedups = [row["search_speedup_vs_baseline"] for row in pairs if row["search_speedup_vs_baseline"]]
        summary = {
            "arm": arm,
            "runs": len(selected),
            "top1_mean_across_targets": statistics.mean(top1_values) if top1_values else None,
            "top1_median_across_targets": statistics.median(top1_values) if top1_values else None,
            "best5_mean_across_targets": statistics.mean(best_values) if best_values else None,
            "best5_median_across_targets": statistics.median(best_values) if best_values else None,
            "top5_pool_mean_across_targets": statistics.mean(pooled_values) if pooled_values else None,
            "best5_successes_mean_ge_0_8": sum(row["best5_success_mean_ge_0_8"] for row in selected),
            "mean_candidate_seed_std": statistics.mean(
                row[f"{PRIMARY_METRIC}_std"]
                for row in candidate_rows if row["arm"] == arm
            ) if selected else None,
            "paired_best5_delta_mean": statistics.mean(deltas) if deltas else 0.0 if arm == "baseline" else None,
            "paired_best5_delta_median": statistics.median(deltas) if deltas else 0.0 if arm == "baseline" else None,
            "paired_best5_delta_ci95_low": None,
            "paired_best5_delta_ci95_high": None,
            "best5_wins_vs_baseline": sum(row["best5_win_vs_baseline"] for row in pairs),
            "paired_search_speedup_median": statistics.median(speedups) if speedups else 1.0 if arm == "baseline" else None,
            "paired_search_speedup_geomean": geometric_mean(speedups) if speedups else 1.0 if arm == "baseline" else None,
        }
        for metric in REPORT_METRICS:
            values = [
                row[f"best5_{metric}_mean"]
                for row in selected
                if row.get(f"best5_{metric}_mean") is not None
            ]
            summary[f"best5_{metric}_mean_across_targets"] = (
                statistics.mean(values) if values else None
            )
        interval = bootstrap_interval(deltas)
        if interval:
            summary["paired_best5_delta_ci95_low"], summary["paired_best5_delta_ci95_high"] = interval
        arm_rows.append(summary)
        arm_json[arm] = summary
    arm_fields = list(arm_rows[0])
    write_csv(analysis / "arm_summary.csv", arm_rows, arm_fields)

    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    write_json(reports / "summary.json", {
        "contract": spec,
        "score_files_found": len(scores),
        "run_summaries": len(run_rows),
        "complete": len(scores) == expected_score_count and len(run_rows) == 52,
        "arms": arm_json,
    })

    def f(value: Any, digits: int = 4) -> str:
        return "—" if value is None else f"{float(value):.{digits}f}"

    lines = [
        "# 13-target ESMFold2 five-seed rescore report",
        "",
        "## Contract",
        "",
        f"- Evaluator: unoptimized ESMFold2 ({spec['model']})",
        f"- Evaluation seeds: {', '.join(map(str, EVAL_SEEDS))}",
        f"- Search candidates per run: top {TOP_K} unique sequences by search total loss",
        f"- ESMFold2 settings: loops={LOOPS}, sampling_steps={SAMPLING_STEPS}",
        "- Candidate score: arithmetic mean over five evaluation seeds",
        "- Best-of-5: choose the largest five-seed mean, not the best individual seed",
        f"- Completeness: {len(scores)}/{expected_score_count} seed scores; {len(run_rows)}/52 runs",
        "",
        "## Arm summary",
        "",
        "| Arm | Runs | Top-1 mean | Best-of-5 mean | Top-5 pool mean | Successes ≥0.8 | Δ best-of-5 vs baseline | 95% bootstrap CI | Wins vs baseline | Search speedup (geo) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in arm_rows:
        ci = (
            "—" if row["paired_best5_delta_ci95_low"] is None else
            f"[{f(row['paired_best5_delta_ci95_low'])}, {f(row['paired_best5_delta_ci95_high'])}]"
        )
        lines.append(
            f"| {row['arm']} | {row['runs']} | {f(row['top1_mean_across_targets'])} | "
            f"{f(row['best5_mean_across_targets'])} | {f(row['top5_pool_mean_across_targets'])} | "
            f"{row['best5_successes_mean_ge_0_8']} | {f(row['paired_best5_delta_mean'])} | "
            f"{ci} | {row['best5_wins_vs_baseline']} | {f(row['paired_search_speedup_geomean'], 3)}× |"
        )
    lines.extend([
        "",
        "## Metrics for the selected best-of-5 candidate",
        "",
        "Each entry is the equal-target-weight mean. Candidates are selected by five-seed mean pair-chain iPTM; the remaining metrics describe those same selected candidates.",
        "",
        "| Arm | Pair-chain iPTM | iPTM | pTM | pLDDT | Binder pLDDT | Total loss | Mean seed SD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in arm_rows:
        lines.append(
            f"| {row['arm']} | {f(row['best5_pair_chain_iptm_mean_across_targets'])} | "
            f"{f(row['best5_iptm_mean_across_targets'])} | {f(row['best5_ptm_mean_across_targets'])} | "
            f"{f(row['best5_plddt_mean_across_targets'])} | {f(row['best5_binder_plddt_mean_across_targets'])} | "
            f"{f(row['best5_total_loss_mean_across_targets'])} | {f(row['mean_candidate_seed_std'])} |"
        )
    lines.extend([
        "",
        "## Best-of-5 five-seed means by target",
        "",
        "Values are mean ± sample standard deviation over the five ESMFold2 seeds; `rN` is the selected search rank.",
        "",
        "| Target | Baseline | Exact | Fast | Big |",
        "|---|---:|---:|---:|---:|",
    ])
    for target_name, case_id, _, _ in fixed_panel.TARGET_SPECS:
        values = []
        for arm in ARMS:
            row = run_map.get((case_id, arm))
            values.append(
                "—" if row is None else
                f"{f(row['best5_pair_chain_iptm_mean'])} ± {f(row['best5_pair_chain_iptm_std'])} (r{row['best5_rank']})"
            )
        lines.append(f"| {target_name} | " + " | ".join(values) + " |")
    lines.extend([
        "",
        "## Interpretation notes",
        "",
        "- Each target has equal weight in the arm summary; long proteins do not receive extra quality weight.",
        "- Seed averaging happens before best-of-5 selection, preventing a single lucky evaluation seed from determining the winner.",
        "- Top-1 measures search ranking quality; best-of-5 measures whether the run generated at least one strong candidate; top-5 pool measures overall candidate-set quality.",
        "- The 95% interval resamples targets, not individual seed scores, because targets are the independent comparison units.",
        "- These are ESMFold2 confidence scores, not experimental binding measurements.",
    ])
    (reports / "report.md").write_text("\n".join(lines) + "\n")

    if run_rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(13, 6))
        width = 0.19
        x_values = list(range(len(fixed_panel.TARGET_SPECS)))
        for arm_index, arm in enumerate(ARMS):
            values = [
                run_map.get((case_id, arm), {}).get("best5_pair_chain_iptm_mean", float("nan"))
                for _, case_id, _, _ in fixed_panel.TARGET_SPECS
            ]
            errors = [
                run_map.get((case_id, arm), {}).get("best5_pair_chain_iptm_std", 0.0)
                for _, case_id, _, _ in fixed_panel.TARGET_SPECS
            ]
            axis.bar(
                [x + (arm_index - 1.5) * width for x in x_values],
                values,
                width=width,
                yerr=errors,
                capsize=1.5,
                label=arm,
            )
        axis.axhline(SUCCESS_THRESHOLD, color="black", linestyle="--", linewidth=1, label="0.8 threshold")
        axis.set_ylabel("Best-of-5 pair-chain iPTM (mean over 5 seeds)")
        axis.set_xticks(x_values)
        axis.set_xticklabels([name for name, _, _, _ in fixed_panel.TARGET_SPECS], rotation=35, ha="right")
        axis.set_ylim(0, 1)
        axis.legend(ncol=5, fontsize=8)
        figure.tight_layout()
        figure.savefig(reports / "best5_by_target.png", dpi=180)
        figure.savefig(reports / "best5_by_target.pdf")
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(8, 6))
        for arm in ARMS:
            selected = [
                row for row in run_rows
                if row["arm"] == arm and row.get("search_wall_s") is not None
            ]
            axis.scatter(
                [row["search_wall_s"] / 3600 for row in selected],
                [row["best5_pair_chain_iptm_mean"] for row in selected],
                label=arm,
                alpha=0.8,
            )
        axis.axhline(SUCCESS_THRESHOLD, color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("Single-GPU search wall time (hours, including startup)")
        axis.set_ylabel("Best-of-5 pair-chain iPTM (mean over 5 seeds)")
        axis.set_ylim(0, 1)
        axis.legend()
        figure.tight_layout()
        figure.savefig(reports / "quality_vs_time.png", dpi=180)
        figure.savefig(reports / "quality_vs_time.pdf")
        plt.close(figure)

    print(f"Wrote five-seed comparison report to {reports / 'report.md'}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    target_parser = commands.add_parser("rescore-target")
    target_parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    target_parser.add_argument("--index", type=int, required=True)
    summary_parser = commands.add_parser("summarize")
    summary_parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    summary_parser.add_argument("--allow-partial", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "rescore-target":
        rescore_target(args)
    else:
        summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
