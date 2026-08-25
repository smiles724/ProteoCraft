#!/usr/bin/env python
"""A/B/C/D smoke for the TCR sequence prior x contact loss 2x2.

Single case (ALGRRIPPV_beta), seed 42, CDR3-only, n_steps=3. All four arms
share scaffold, seed, search budget and verifier settings; only the two TCR
channels differ.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path("/root/mingxuan/ProteoCraft")
PYTHON = "/root/miniforge3/envs/proteocraft_esmfold2_py312/bin/python"
CASE_SPEC = Path(
    "/root/Bingchen/experiments/20260819_liubo_cdr6_native_20x200"
    "/inputs/case_specs/ALGRRIPPV_beta.json"
)
TCRLM_REPO = "/root/mingxuan/tcr_prior_backends/tcrLM"
TCRLM_CKPT = f"{TCRLM_REPO}/pretrained_model/pretrained_model_tcr.bin"

PRIOR_FLAGS = {
    "off": ["--use_tcr_prior_gradient", "false", "--use_tcr_prior_verifier", "false"],
    "tcrlm": [
        "--use_tcr_prior_gradient", "true",
        "--use_tcr_prior_verifier", "true",
        "--tcr_prior_gradient_weight", "0.1",
        "--tcr_prior_verifier_weight", "0.1",
        "--tcr_prior_region", "cdr",
        "--tcr_prior_model_name", f"tcrlm:{TCRLM_REPO}",
        "--tcr_prior_checkpoint", TCRLM_CKPT,
        "--require_real_tcr_prior", "true",
    ],
    "esmc": [
        "--use_tcr_prior_gradient", "true",
        "--use_tcr_prior_verifier", "true",
        "--tcr_prior_gradient_weight", "0.1",
        "--tcr_prior_verifier_weight", "0.1",
        "--tcr_prior_region", "cdr",
        "--tcr_prior_model_name", "esmc:biohub/ESMC-6B",
        "--require_real_tcr_prior", "true",
    ],
}

CONTACT_FLAGS = {
    "off": ["--use_tcr_contact_loss", "false"],
    "on": [
        "--use_tcr_contact_loss", "true",
        "--tcr_contact_lambda", "0.1",
        "--tcr_contact_lambda_fw", "1.0",
        "--tcr_contact_fw_hinge", "6.0",
        "--tcr_contact_verifier_weight", "1.0",
        "--tcr_contact_peptide_chain_index", "0",
        "--tcr_contact_mhc_chain_indices", "1",
    ],
}

ARMS = {
    "A": ("off", "off"),
    "B": ("tcrlm", "off"),
    "C": ("off", "on"),
    "D": ("tcrlm", "on"),
    "E": ("esmc", "off"),
    "F": ("esmc", "on"),
}


def build_command(arm: str, case: dict, out_root: Path, args) -> list[str]:
    prior_key, contact_key = ARMS[arm]
    case_dir = out_root / f"arm_{arm}"
    case_dir.mkdir(parents=True, exist_ok=True)
    mutable = "".join("1" if v else "0" for v in case["mutable_mask"])
    command = [
        PYTHON, str(REPO / "scripts" / "run_hallucination.py"),
        "--profile", "esmfold2_d",
        "--model_backend", "esmfold2",
        "--mode", "mcts",
        "--esmfold2_task_type", "cdr_redesign",
        "--target", "|".join(case["partner_sequences"]),
        "--esmfold2_target_chain_roles", ",".join(case["target_chain_roles"]),
        "--binder_length", str(len(case["binder_sequence"])),
        "--init_binder_seq", case["binder_sequence"],
        "--esmfold2_mutable_mask", mutable,
        "--seed", str(args.seed),
        "--n_steps", str(args.n_steps),
        "--device", "cuda",
        # Matched, deliberately small verifier budget for the smoke.
        "--esmfold2_verifier_num_loops", str(args.verifier_loops),
        "--esmfold2_verifier_num_sampling_steps", str(args.verifier_steps),
        "--esmfold2_stagea_soft_steps", str(args.soft_steps),
        "--esmfold2_stageb_soft_steps", str(args.soft_steps),
        "--mcts_bootstrap_steps", str(args.n_steps),
        "--mcts_bootstrap_max_steps", str(args.n_steps),
        "--output_dir", str(case_dir),
        "--esmfold2_artifact_dir", str(case_dir / "artifacts"),
        "--log_file", str(case_dir / "run.log"),
    ]
    command += PRIOR_FLAGS[prior_key]
    command += CONTACT_FLAGS[contact_key]
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="A,B,C,D")
    parser.add_argument("--out", default="/root/mingxuan/smoke_tcr_abcd_20260820")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-steps", dest="n_steps", type=int, default=3)
    parser.add_argument("--verifier-loops", dest="verifier_loops", type=int, default=2)
    parser.add_argument("--verifier-steps", dest="verifier_steps", type=int, default=20)
    parser.add_argument("--soft-steps", dest="soft_steps", type=int, default=4)
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    case = json.loads(CASE_SPEC.read_text())
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    import os

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["HF_HOME"] = "/root/Bingchen/resources/hf_cache"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONPATH"] = str(REPO)

    results = {}
    for arm in args.arms.split(","):
        arm = arm.strip()
        command = build_command(arm, case, out_root, args)
        print(f"\n=== ARM {arm} ({ARMS[arm][0]} prior / contact {ARMS[arm][1]}) ===", flush=True)
        if args.dry_run:
            print(" ".join(command[:14]), "...")
            continue
        proc = subprocess.run(command, cwd=str(REPO), env=env)
        results[arm] = proc.returncode
        print(f"ARM {arm} exit={proc.returncode}", flush=True)

    if results:
        (out_root / "arm_exit_codes.json").write_text(json.dumps(results, indent=2))
        print("\nexit codes:", results)
    return 0 if all(v == 0 for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
