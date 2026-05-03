#!/usr/bin/env python3
"""Launch TCR cases from a YAML config across one or more GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOSS_WEIGHT_KEYS = (
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-ids", default=None, help="Comma-separated GPU IDs; overrides YAML gpu_ids.")
    parser.add_argument("--python-bin", default=None, help="Python executable; overrides YAML python_bin.")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("YAML config must define a non-empty cases list")
    return config


def case_name(case: Dict[str, Any]) -> str:
    return f"{case['pdb_id'].upper()}_alpha_{case['hotspot_start']}_{case['hotspot_end']}"


def build_command(config: Dict[str, Any], case: Dict[str, Any], python_bin: str, gpu_id: str) -> tuple[List[str], Path]:
    name = case_name(case)
    output_root = Path(config["output_root"])
    log_dir = Path(config.get("log_dir") or output_root / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    case_log = log_dir / f"{name}.log"

    protenix_root = Path(config["protenix_root_dir"]) / name
    cmd = [
        python_bin,
        str(PROJECT_ROOT / "scripts" / "run_bindcraft_tcr_case.py"),
        "--pdb-id",
        str(case["pdb_id"]),
        "--hotspot-start",
        str(case["hotspot_start"]),
        "--hotspot-end",
        str(case["hotspot_end"]),
        "--alpha-chain",
        str(case["alpha_chain"]),
        "--beta-chain",
        str(case["beta_chain"]),
        "--n-steps",
        str(config.get("n_steps", 50)),
        "--learning-rate",
        str(config.get("learning_rate", 0.05)),
        "--tau-noise",
        str(config.get("tau_noise", 4.0)),
        "--k-top",
        str(config.get("k_top", 8)),
        "--b-candidates",
        str(config.get("b_candidates", 4)),
        "--max-mutations-per-seq",
        str(config.get("max_mutations_per_seq", 3)),
        "--output-root",
        str(output_root),
        "--protenix-root-dir",
        str(protenix_root),
        "--model-checkpoint-dir",
        str(config["model_checkpoint_dir"]),
        "--components-cache-dir",
        str(config["components_cache_dir"]),
        "--model-name",
        str(config.get("model_name", "protenix_base_20250630_v1.0.0")),
        "--device",
        "cuda:0",
        "--log-file",
        str(case_log),
    ]
    if case.get("pdb_path"):
        cmd.extend(["--pdb-path", str(case["pdb_path"])])

    weights = config.get("loss_weights") or {}
    for key in LOSS_WEIGHT_KEYS:
        if key in weights:
            cmd.extend([f"--w-{key.replace('_', '-')}", str(weights[key])])

    stdout_log = log_dir / f"{name}.stdout.log"
    return cmd, stdout_log


def run_case(config: Dict[str, Any], case: Dict[str, Any], gpu_id: str, python_bin: str) -> int:
    name = case_name(case)
    cmd, stdout_log = build_command(config, case, python_bin, gpu_id)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print(f"[{time.strftime('%F %T')}] START {name} on GPU {gpu_id}; stdout={stdout_log}", flush=True)
    with stdout_log.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        return_code = proc.wait()

    status = "DONE" if return_code == 0 else "FAILED"
    print(f"[{time.strftime('%F %T')}] {status} {name} on GPU {gpu_id}; exit={return_code}", flush=True)
    return return_code


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    python_bin = args.python_bin or config.get("python_bin") or sys.executable
    gpu_ids = (
        [part.strip() for part in args.gpu_ids.split(",") if part.strip()]
        if args.gpu_ids
        else [str(gpu_id) for gpu_id in config.get("gpu_ids", [0])]
    )
    if not gpu_ids:
        raise ValueError("No GPU IDs configured")

    cases = list(config["cases"])
    print(f"[{time.strftime('%F %T')}] Config: {args.config}", flush=True)
    print(f"[{time.strftime('%F %T')}] GPUs: {', '.join(gpu_ids)}", flush=True)
    print(f"[{time.strftime('%F %T')}] Cases: {', '.join(case_name(case) for case in cases)}", flush=True)

    status = 0
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = {
            executor.submit(run_case, config, case, gpu_ids[index % len(gpu_ids)], python_bin): case
            for index, case in enumerate(cases)
        }
        for future in as_completed(futures):
            status = max(status, int(future.result() != 0))
    print(f"[{time.strftime('%F %T')}] YAML TCR queue finished; status={status}", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
