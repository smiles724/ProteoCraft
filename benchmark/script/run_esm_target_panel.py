#!/usr/bin/env python3
"""Run and summarize the 12-target ESM optimization panel.

Each target is one Slurm array task.  Two isolated CUDA lanes run
baseline/fast and exact/big, then a common unoptimized ESMFold2 verifier
rescoring pass evaluates the top five candidates with seed 42.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmark/script/run_esm_optimization.py"
DATASET_CONFIG = ROOT / "benchmark/datasets/proteocraft_59_v1/benchmark.yaml"
REFERENCE_CAMPAIGN = ROOT / "outputs/esm_optimization/pilot_2case_43/campaign.json"
DEFAULT_OUTPUT = ROOT / "outputs/esm_optimization/target_panel_12_seed42"
DEFAULT_UPLIFT_ROOT = os.environ.get("PROTEOCRAFT_UPLIFT_ROOT")
_hf_home = os.environ.get("HF_HOME")
DEFAULT_MODEL_ROOT = Path(_hf_home).expanduser() / "hub" if _hf_home else None
DEFAULT_PROPOSAL_MODEL = (
    DEFAULT_MODEL_ROOT
    / "models--biohub--ESMFold2-Experimental-Fast/snapshots"
    / "04dec820ede9283c9893e318e5ca5a9ac2ab93bc"
    if DEFAULT_MODEL_ROOT is not None
    else None
)
DEFAULT_VERIFIER_MODEL = (
    DEFAULT_MODEL_ROOT
    / "models--biohub--ESMFold2/snapshots"
    / "1ebf0e3481a5184eb6171d40615c79e384b48796"
    if DEFAULT_MODEL_ROOT is not None
    else None
)

# One representative design case per requested target.  Multi-case de novo
# targets use the first manifest case; TCR targets use the no-hotspot case, as
# in the completed 1QRN pilot.
TARGET_CASES = (
    ("BetV1", "denovo_BetV1"),
    ("DerF21", "denovo_DerF21_B10"),
    ("DerF7", "denovo_DerF7_B132"),
    ("HER2", "denovo_HER2"),
    ("IFNAR2", "denovo_IFNAR2"),
    ("PD-L1", "denovo_PD-L1"),
    ("PD1", "denovo_PD1"),
    ("SpCas9", "denovo_SpCas9"),
    ("3QDJ", "tcr_3qdj_no_hotspot"),
    ("5BS0", "tcr_5bs0_no_hotspot"),
    ("6BJ8", "tcr_6bj8_no_hotspot"),
    ("9C3E", "tcr_9c3e_no_hotspot"),
)
ARMS = ("baseline", "exact", "fast", "big")
LANES = (("baseline", "fast"), ("exact", "big"))
ALL_TIMING_PHASES = (
    "proposal_update",
    "proposal_forward",
    "esmc_prior_forward",
    "verifier",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    temporary.replace(path)


def run_checked(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def prepare(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference = json.loads(Path(args.reference_campaign).read_text())
    command = [
        sys.executable,
        str(RUNNER),
        "prepare",
        "--output",
        str(output),
        "--config",
        str(Path(args.config).resolve()),
        "--kit-root",
        str(Path(args.kit_root).resolve()),
        "--proposal-model",
        str(Path(args.proposal_model).resolve()),
        "--verifier-model",
        str(Path(args.verifier_model).resolve()),
        "--cases",
        *(case_id for _, case_id in TARGET_CASES),
        "--arms",
        *ARMS,
        "--seeds",
        "42",
    ]
    run_checked(command, cwd=ROOT)
    campaign_path = output / "campaign.json"
    campaign = json.loads(campaign_path.read_text())
    if campaign["profile_sha256"] != reference["profile_sha256"]:
        raise RuntimeError("esmfold2_d profile differs from the completed 1QRN pilot")
    expected = {(case_id, arm) for _, case_id in TARGET_CASES for arm in ARMS}
    observed = {
        (record["case"]["case_id"], record["arm"])
        for record in campaign["records"]
    }
    if observed != expected:
        raise RuntimeError(
            f"Campaign matrix mismatch: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )
    panel = {
        "schema_version": 1,
        "campaign": str(campaign_path),
        "reference_campaign": str(Path(args.reference_campaign).resolve()),
        "reference_profile_sha256": reference["profile_sha256"],
        "search_seed": 42,
        "arms": list(ARMS),
        "lanes": [list(lane) for lane in LANES],
        "targets": [
            {"array_index": index, "target_name": target, "case_id": case_id}
            for index, (target, case_id) in enumerate(TARGET_CASES)
        ],
        "blind_rescore": {
            "evaluator": "unoptimized ESMFold2",
            "top_k": 5,
            "seeds": [42],
            "num_loops": 4,
            "num_sampling_steps": 50,
        },
        "timing_phases": list(ALL_TIMING_PHASES),
        "fine_grained_timing_probes": False,
    }
    write_json(output / "panel_spec.json", panel)
    print(f"Prepared {len(campaign['records'])} searches in {campaign_path}")


def sliced_campaign(campaign: dict[str, Any], records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    result = dict(campaign)
    result["records"] = list(records)
    return result


def lane_environment(output: Path, case_id: str, lane: int) -> dict[str, str]:
    environment = os.environ.copy()
    visible = [
        item.strip()
        for item in environment.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
        if item.strip()
    ]
    if lane >= len(visible):
        raise RuntimeError(
            f"Target worker needs two visible GPUs, found CUDA_VISIBLE_DEVICES={visible}"
        )
    environment["CUDA_VISIBLE_DEVICES"] = visible[lane]
    cache = output / "runtime_cache" / case_id / f"lane-{lane}"
    triton = cache / "triton"
    inductor = cache / "inductor"
    triton.mkdir(parents=True, exist_ok=True)
    inductor.mkdir(parents=True, exist_ok=True)
    environment["TRITON_CACHE_DIR"] = str(triton)
    environment["TORCHINDUCTOR_CACHE_DIR"] = str(inductor)
    return environment


def run_target(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    campaign = json.loads((output / "campaign.json").read_text())
    index = int(args.index)
    if not 0 <= index < len(TARGET_CASES):
        raise ValueError(f"index must be in [0, {len(TARGET_CASES) - 1}]")
    target_name, case_id = TARGET_CASES[index]
    records = [r for r in campaign["records"] if r["case"]["case_id"] == case_id]
    by_arm = {record["arm"]: record for record in records}
    if set(by_arm) != set(ARMS):
        raise RuntimeError(f"{case_id}: expected arms {ARMS}, found {sorted(by_arm)}")

    manifest_dir = output / "target_manifests"
    target_manifest = manifest_dir / f"{case_id}.json"
    write_json(target_manifest, sliced_campaign(campaign, [by_arm[a] for a in ARMS]))
    status_path = output / "target_status" / f"{case_id}.json"
    started = time.time()
    write_json(status_path, {
        "target_name": target_name,
        "case_id": case_id,
        "array_index": index,
        "status": "running",
        "started_unix": started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    })

    target_lanes = []
    for lane, lane_arms in enumerate(LANES):
        active_arms = tuple(
            arm for arm in lane_arms
            if not by_arm[arm].get("reuse_existing", False)
        )
        if active_arms:
            target_lanes.append((lane, active_arms))

    processes = []
    handles = []
    lane_started = {}
    try:
        for lane, lane_arms in target_lanes:
            lane_manifest = manifest_dir / f"{case_id}.lane-{lane}.json"
            write_json(
                lane_manifest,
                sliced_campaign(campaign, [by_arm[arm] for arm in lane_arms]),
            )
            log_path = output / "lane_logs" / f"{case_id}.lane-{lane}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", buffering=1)
            handles.append(handle)
            command = [
                sys.executable,
                str(RUNNER),
                "run",
                "--manifest",
                str(lane_manifest),
                "--skip-existing",
            ]
            lane_started[lane] = time.time()
            print(
                f"Starting {target_name} lane {lane} ({', '.join(lane_arms)}) on "
                f"CUDA_VISIBLE_DEVICES={lane}",
                flush=True,
            )
            processes.append((
                lane,
                lane_arms,
                subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=lane_environment(output, case_id, lane),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                ),
            ))
        lane_results = []
        failed = False
        for lane, lane_arms, process in processes:
            returncode = process.wait()
            lane_results.append({
                "lane": lane,
                "arms": list(lane_arms),
                "returncode": returncode,
                "wall_s": time.time() - lane_started[lane],
            })
            failed |= returncode != 0
        if failed:
            raise RuntimeError(f"One or more search lanes failed: {lane_results}")
    except Exception as exc:
        for _, _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, _, process in processes:
            if process.poll() is None:
                process.wait()
        write_json(status_path, {
            "target_name": target_name,
            "case_id": case_id,
            "array_index": index,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "started_unix": started,
            "finished_unix": time.time(),
        })
        raise
    finally:
        for handle in handles:
            handle.close()

    rescore_started = time.time()
    rescore_dir = output / "rescore" / case_id
    try:
        rescore_log = output / "rescore_logs" / f"{case_id}.log"
        rescore_log.parent.mkdir(parents=True, exist_ok=True)
        rescore_command = [
            sys.executable,
            str(RUNNER),
            "rescore",
            "--manifest",
            str(target_manifest),
            "--output",
            str(rescore_dir),
            "--top-k",
            "5",
            "--seeds",
            "42",
            "--loops",
            "4",
            "--sampling-steps",
            "50",
        ]
        with rescore_log.open("a", buffering=1) as handle:
            run_checked(
                rescore_command,
                cwd=ROOT,
                env=lane_environment(output, case_id, 0),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        report_dir = output / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        run_checked([
            sys.executable,
            str(RUNNER),
            "report",
            "--manifest",
            str(target_manifest),
            "--scores",
            str(rescore_dir),
            "--output",
            str(report_dir / f"{case_id}.csv"),
        ], cwd=ROOT)
        write_timing_tables(output, case_filter={case_id}, suffix=f".{case_id}")
    except Exception as exc:
        write_json(status_path, {
            "target_name": target_name,
            "case_id": case_id,
            "array_index": index,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "started_unix": started,
            "finished_unix": time.time(),
            "search_lanes": lane_results,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        })
        raise
    finished = time.time()
    write_json(status_path, {
        "target_name": target_name,
        "case_id": case_id,
        "array_index": index,
        "status": "complete",
        "started_unix": started,
        "finished_unix": finished,
        "wall_s": finished - started,
        "search_lanes": lane_results,
        "rescore_wall_s": finished - rescore_started,
        "target_manifest": str(target_manifest),
        "rescore_dir": str(rescore_dir),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    })
    print(f"Completed {target_name} ({case_id}) in {(finished-started)/3600:.2f} h")


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_timing_tables(
    output: Path,
    *,
    case_filter: set[str] | None = None,
    suffix: str = "",
) -> None:
    campaign = json.loads((output / "campaign.json").read_text())
    long_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for record in campaign["records"]:
        case = record["case"]
        case_id = case["case_id"]
        if case_filter is not None and case_id not in case_filter:
            continue
        directory = Path(record["directory"])
        receipt_path = directory / "experiment_run.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
        events = read_events(directory / "optimization_report.timing.jsonl")
        groups: dict[tuple[str, str], list[float]] = {}
        for event in events:
            if event.get("status") != "ok" or event.get("phase") not in ALL_TIMING_PHASES:
                continue
            groups.setdefault(("all", event["phase"]), []).append(float(event["elapsed_s"]))
        update_details = []
        window = []
        update_index = 0
        for event in events:
            if event.get("phase") != "proposal_update":
                window.append(event)
                continue
            expected = 30 if update_index < 70 else 20
            stage = "stage_a" if update_index < 70 else "stage_b"
            proposal = [
                float(item["elapsed_s"])
                for item in window
                if item.get("phase") == "proposal_forward"
            ][-expected:]
            prior = [
                float(item["elapsed_s"])
                for item in window
                if item.get("phase") == "esmc_prior_forward"
            ][-expected:]
            if len(proposal) == expected and len(prior) == expected:
                update_s = float(event["elapsed_s"])
                groups.setdefault((stage, "proposal_update"), []).append(update_s)
                groups.setdefault((stage, "proposal_forward"), []).extend(proposal)
                groups.setdefault((stage, "esmc_prior_forward"), []).extend(prior)
                update_details.append({
                    "elapsed_s": update_s,
                    "proposal_s": sum(proposal),
                    "prior_s": sum(prior),
                })
            window = []
            update_index += 1
        for (stage, phase), values in sorted(groups.items()):
            long_rows.append({
                "target_name": case["target_name"],
                "case_id": case_id,
                "task_type": case["task_type"],
                "target_tokens": len(case["target_sequence"].replace("|", "")),
                "binder_tokens": case["binder_length"],
                "arm": record["arm"],
                "stage": stage,
                "phase": phase,
                "calls": len(values),
                "total_s": sum(values),
                "mean_s": statistics.mean(values),
                "median_s": statistics.median(values),
                "p95_s": percentile(values, 0.95),
                "min_s": min(values),
                "max_s": max(values),
            })
        totals = {
            phase: sum(event["elapsed_s"] for event in events
                       if event.get("status") == "ok" and event.get("phase") == phase)
            for phase in ALL_TIMING_PHASES
        }
        counts = {
            phase: sum(1 for event in events
                       if event.get("status") == "ok" and event.get("phase") == phase)
            for phase in ALL_TIMING_PHASES
        }
        update_residual = sum(
            item["elapsed_s"] - item["proposal_s"] - item["prior_s"]
            for item in update_details
        )
        row = {
            "target_name": case["target_name"],
            "case_id": case_id,
            "task_type": case["task_type"],
            "target_tokens": len(case["target_sequence"].replace("|", "")),
            "binder_tokens": case["binder_length"],
            "arm": record["arm"],
            "status": receipt.get("status", "missing"),
            "returncode": receipt.get("returncode"),
            "search_wall_s": receipt.get("wall_s"),
            "update_total_s": totals["proposal_update"],
            "update_residual_s": update_residual,
        }
        for phase in ALL_TIMING_PHASES:
            row[f"{phase}_calls"] = counts[phase]
            row[f"{phase}_total_s"] = totals[phase]
            row[f"{phase}_mean_s"] = (
                totals[phase] / counts[phase] if counts[phase] else None
            )
        run_rows.append(row)

    analysis = output / "analysis"
    long_fields = [
        "target_name", "case_id", "task_type", "target_tokens", "binder_tokens",
        "arm", "stage", "phase", "calls", "total_s", "mean_s", "median_s",
        "p95_s", "min_s", "max_s",
    ]
    run_fields = [
        "target_name", "case_id", "task_type", "target_tokens", "binder_tokens",
        "arm", "status", "returncode", "search_wall_s", "update_total_s",
        "update_residual_s",
    ]
    for phase in ALL_TIMING_PHASES:
        run_fields.extend((f"{phase}_calls", f"{phase}_total_s", f"{phase}_mean_s"))
    write_csv(analysis / f"timing_long{suffix}.csv", long_rows, long_fields)
    write_csv(analysis / f"run_timing{suffix}.csv", run_rows, run_fields)


def summarize(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    write_timing_tables(output)
    rescore = output / "rescore"
    if rescore.exists():
        run_checked([
            sys.executable,
            str(RUNNER),
            "report",
            "--manifest",
            str(output / "campaign.json"),
            "--scores",
            str(rescore),
            "--output",
            str(output / "analysis/quality_report.csv"),
        ], cwd=ROOT)
    statuses = []
    for target_name, case_id in TARGET_CASES:
        path = output / "target_status" / f"{case_id}.json"
        status = json.loads(path.read_text()) if path.is_file() else {"status": "missing"}
        statuses.append({
            "target_name": target_name,
            "case_id": case_id,
            "status": status.get("status", "missing"),
            "wall_s": status.get("wall_s"),
            "rescore_wall_s": status.get("rescore_wall_s"),
            "slurm_job_id": status.get("slurm_job_id"),
        })
    write_csv(
        output / "analysis/panel_status.csv",
        statuses,
        ["target_name", "case_id", "status", "wall_s", "rescore_wall_s", "slurm_job_id"],
    )
    print(f"Wrote panel analysis to {output / 'analysis'}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    prepare_parser.add_argument("--config", default=str(DATASET_CONFIG))
    prepare_parser.add_argument("--reference-campaign", default=str(REFERENCE_CAMPAIGN))
    prepare_parser.add_argument(
        "--kit-root", default=DEFAULT_UPLIFT_ROOT,
        required=DEFAULT_UPLIFT_ROOT is None,
    )
    prepare_parser.add_argument(
        "--proposal-model",
        default=None if DEFAULT_PROPOSAL_MODEL is None else str(DEFAULT_PROPOSAL_MODEL),
        required=DEFAULT_PROPOSAL_MODEL is None,
    )
    prepare_parser.add_argument(
        "--verifier-model",
        default=None if DEFAULT_VERIFIER_MODEL is None else str(DEFAULT_VERIFIER_MODEL),
        required=DEFAULT_VERIFIER_MODEL is None,
    )
    run_parser = subparsers.add_parser("run-target")
    run_parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    run_parser.add_argument("--index", type=int, required=True)
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run-target":
        run_target(args)
    else:
        summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
