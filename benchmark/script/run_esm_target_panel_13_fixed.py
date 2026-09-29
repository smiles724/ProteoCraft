#!/usr/bin/env python3
"""Run the 13-target panel after separating timing from provenance reporting.

Completed baseline searches are referenced in place.  Incomplete baselines and
all optimized arms are written under a fresh output root.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import run_esm_optimization as experiments
import run_esm_target_panel as panel
from run_esmfold2_59 import build_hallucination_command, load_benchmark_config, load_prepared_cases


DEFAULT_OUTPUT = ROOT / "outputs/esm_optimization/target_panel_13_reporting_fixed"
DATASET_CONFIG = ROOT / "benchmark/datasets/proteocraft_59_v1/benchmark.yaml"
DEFAULT_UPLIFT_ROOT = os.environ.get("PROTEOCRAFT_UPLIFT_ROOT")
DEFAULT_PROPOSAL_MODEL = panel.DEFAULT_PROPOSAL_MODEL
DEFAULT_VERIFIER_MODEL = panel.DEFAULT_VERIFIER_MODEL

OLD_PANEL = ROOT / "outputs/esm_optimization/target_panel_12_seed42"
OLD_PILOT = ROOT / "outputs/esm_optimization/pilot_2case_43"

# name, case id, search seed, existing baseline directory
TARGET_SPECS = (
    ("BetV1", "denovo_BetV1", 42, OLD_PANEL / "baseline/denovo_BetV1/seed-42"),
    ("DerF21", "denovo_DerF21_B10", 42, OLD_PANEL / "baseline/denovo_DerF21_B10/seed-42"),
    ("DerF7", "denovo_DerF7_B132", 42, OLD_PANEL / "baseline/denovo_DerF7_B132/seed-42"),
    ("HER2", "denovo_HER2", 42, OLD_PANEL / "baseline/denovo_HER2/seed-42"),
    ("IFNAR2", "denovo_IFNAR2", 42, OLD_PANEL / "baseline/denovo_IFNAR2/seed-42"),
    ("PD-L1", "denovo_PD-L1", 42, OLD_PANEL / "baseline/denovo_PD-L1/seed-42"),
    ("PD1", "denovo_PD1", 42, OLD_PANEL / "baseline/denovo_PD1/seed-42"),
    ("SpCas9", "denovo_SpCas9", 42, OLD_PANEL / "baseline/denovo_SpCas9/seed-42"),
    ("1QRN", "tcr_1qrn_no_hotspot", 43, OLD_PILOT / "baseline/tcr_1qrn_no_hotspot/seed-43"),
    ("3QDJ", "tcr_3qdj_no_hotspot", 42, OLD_PANEL / "baseline/tcr_3qdj_no_hotspot/seed-42"),
    ("5BS0", "tcr_5bs0_no_hotspot", 42, OLD_PANEL / "baseline/tcr_5bs0_no_hotspot/seed-42"),
    ("6BJ8", "tcr_6bj8_no_hotspot", 42, OLD_PANEL / "baseline/tcr_6bj8_no_hotspot/seed-42"),
    ("9C3E", "tcr_9c3e_no_hotspot", 42, OLD_PANEL / "baseline/tcr_9c3e_no_hotspot/seed-42"),
)

panel.TARGET_CASES = tuple((name, case_id) for name, case_id, _, _ in TARGET_SPECS)


def completed_baseline(directory: Path, *, case_id: str, seed: int) -> dict[str, Any] | None:
    receipt_path = directory / "experiment_run.json"
    required = (
        directory / "hallucination_result.txt",
        directory / "optimization_report.json",
        directory / "esmfold2_artifacts/run_summary.json",
    )
    if not receipt_path.is_file() or not all(path.is_file() for path in required):
        return None
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("arm") != "baseline"
        or receipt.get("seed") != seed
        or receipt.get("case", {}).get("case_id") != case_id
        or receipt.get("status") != "complete"
        or receipt.get("returncode") != 0
    ):
        return None
    return receipt


def new_record(
    *, output: Path, arm: str, seed: int, case: Any, kit_root: Path,
    proposal_model: Path, verifier_model: Path,
) -> dict[str, Any]:
    proposal, verifier, esmc = experiments.ARMS[arm]
    directory = output / arm / case.case_id / f"seed-{seed}"
    settings = [
        ("seed", str(seed)),
        ("esmfold2_optimization_root", str(kit_root)),
        ("esmfold2_proposal_optimization", proposal),
        ("esmfold2_verifier_optimization", verifier),
        ("esmfold2_esmc_optimization", esmc),
        ("esmfold2_verifier_model", str(verifier_model)),
        ("esmfold2_experimental_model", str(proposal_model)),
    ]
    command = build_hallucination_command(
        case, profile="esmfold2_d", device="cuda:0", case_dir=directory,
        overrides=settings,
    )
    command.append("--esmfold2_optimization_timing")
    command.extend(("--esmfold2_save_verified_candidate_cif", "true"))
    return {
        "arm": arm,
        "seed": seed,
        "case": asdict(case),
        "directory": str(directory),
        "command": command,
        "reuse_existing": False,
    }


def prepare(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    if (output / "campaign.json").exists():
        raise FileExistsError(f"Campaign already exists: {output / 'campaign.json'}")
    output.mkdir(parents=True, exist_ok=True)
    config = Path(args.config).resolve()
    kit_root = Path(args.kit_root).resolve()
    proposal_model = Path(args.proposal_model).resolve()
    verifier_model = Path(args.verifier_model).resolve()
    cases = {
        case.case_id: case
        for case in load_prepared_cases(load_benchmark_config(config))
    }
    current_profile = ROOT / "configs/proteocraft_profiles/esmfold2_d.yaml"
    profile_hash = hashlib.sha256(current_profile.read_bytes()).hexdigest()
    for old_campaign in (OLD_PANEL / "campaign.json", OLD_PILOT / "campaign.json"):
        old = json.loads(old_campaign.read_text())
        if old["profile_sha256"] != profile_hash:
            raise RuntimeError(f"Profile mismatch with reusable baselines in {old_campaign}")

    records = []
    reused = []
    rerun_baselines = []
    for target_name, case_id, seed, baseline_dir in TARGET_SPECS:
        case = cases[case_id]
        baseline = completed_baseline(baseline_dir, case_id=case_id, seed=seed)
        if baseline is not None:
            records.append({
                "arm": "baseline",
                "seed": seed,
                "case": baseline["case"],
                "directory": baseline["directory"],
                "command": baseline["command"],
                "reuse_existing": True,
                "source_campaign": str(baseline_dir.parents[2]),
            })
            reused.append(target_name)
        else:
            records.append(new_record(
                output=output, arm="baseline", seed=seed, case=case,
                kit_root=kit_root, proposal_model=proposal_model,
                verifier_model=verifier_model,
            ))
            rerun_baselines.append(target_name)
        for arm in ("exact", "fast", "big"):
            records.append(new_record(
                output=output, arm=arm, seed=seed, case=case,
                kit_root=kit_root, proposal_model=proposal_model,
                verifier_model=verifier_model,
            ))

    # Preserve reproducible arm ordering within each target manifest.
    order = {arm: index for index, arm in enumerate(panel.ARMS)}
    random.Random(42).shuffle(records)
    records.sort(key=lambda r: (
        next(i for i, spec in enumerate(TARGET_SPECS) if spec[1] == r["case"]["case_id"]),
        order[r["arm"]],
    ))
    campaign = {
        "schema_version": 1,
        "composition": "proteocraft-uplifting-v1",
        "records": records,
        "repository": experiments.git_state(ROOT),
        "optimization_source": experiments.git_state(kit_root),
        "models": {"proposal": str(proposal_model), "verifier": str(verifier_model)},
        "profile": str(current_profile),
        "profile_sha256": profile_hash,
        "timing_provenance_fix": {
            "measure_writes_full_report": False,
            "final_report_method": "OptimizationManager.finalize",
            "resolved_module_paths_cached": True,
        },
    }
    panel.write_json(output / "campaign.json", campaign)
    panel.write_json(output / "panel_spec.json", {
        "schema_version": 1,
        "campaign": str(output / "campaign.json"),
        "targets": [
            {"array_index": index, "target_name": name, "case_id": case_id, "seed": seed}
            for index, (name, case_id, seed, _) in enumerate(TARGET_SPECS)
        ],
        "arms": list(panel.ARMS),
        "lanes": [list(lane) for lane in panel.LANES],
        "reused_baselines": reused,
        "rerun_baselines": rerun_baselines,
        "blind_rescore": {
            "evaluator": "unoptimized ESMFold2", "top_k": 5,
            "seeds": [42], "num_loops": 4, "num_sampling_steps": 50,
        },
        "timing_phases": list(panel.ALL_TIMING_PHASES),
        "timing_provenance_fix": campaign["timing_provenance_fix"],
    })
    print(
        f"Prepared {len(records)} records: reused {len(reused)} baselines; "
        f"rerunning {len(rerun_baselines)} baselines and 39 optimized searches"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    prepare_parser.add_argument("--config", default=str(DATASET_CONFIG))
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
        panel.run_target(args)
    else:
        panel.summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
