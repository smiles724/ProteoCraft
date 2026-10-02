#!/usr/bin/env python
"""First formal matched run: 3 peptides x alpha/beta x arms, seed 42, CDR3-only.

Every arm shares scaffold, seed, backbone, search budget and verifier budget.
Only the two TCR channels differ, so compute stays matched across arms.
Jobs are pulled from one queue by per-GPU workers; a job that dies (usually a
neighbour taking the card) is retried once on a different GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

REPO = Path("/root/mingxuan/ProteoCraft")
PYTHON = "/root/miniforge3/envs/proteocraft_esmfold2_py312/bin/python"
SPEC_DIR = Path(
    "/root/Bingchen/experiments/20260819_liubo_cdr6_native_20x200/inputs/case_specs"
)
TCRLM_REPO = "/root/mingxuan/tcr_prior_backends/tcrLM"
TCRLM_CKPT = f"{TCRLM_REPO}/pretrained_model/pretrained_model_tcr.bin"

CASES = [
    "ALGRRIPPV_alpha", "ALGRRIPPV_beta",
    "YLNPRVARI_alpha", "YLNPRVARI_beta",
    "KLNPRIPRI_alpha", "KLNPRIPRI_beta",
]

PRIOR_FLAGS = {
    "off": ["--use_tcr_prior_gradient", "false", "--use_tcr_prior_verifier", "false"],
    "tcrlm": [
        "--use_tcr_prior_gradient", "true", "--use_tcr_prior_verifier", "true",
        "--tcr_prior_gradient_weight", "0.1", "--tcr_prior_verifier_weight", "0.1",
        "--tcr_prior_region", "cdr",
        "--tcr_prior_model_name", f"tcrlm:{TCRLM_REPO}",
        "--tcr_prior_checkpoint", TCRLM_CKPT,
        "--require_real_tcr_prior", "true",
    ],
    "esmc": [
        "--use_tcr_prior_gradient", "true", "--use_tcr_prior_verifier", "true",
        "--tcr_prior_gradient_weight", "0.1", "--tcr_prior_verifier_weight", "0.1",
        "--tcr_prior_region", "cdr",
        "--tcr_prior_model_name", "esmc:biohub/ESMC-6B",
        "--require_real_tcr_prior", "true",
    ],
}
CONTACT_FLAGS = {
    "off": ["--use_tcr_contact_loss", "false"],
    "on": [
        "--use_tcr_contact_loss", "true",
        "--tcr_contact_lambda", "0.1", "--tcr_contact_lambda_fw", "1.0",
        "--tcr_contact_fw_hinge", "7.0", "--tcr_contact_fw_hinge_verifier", "12.5",
        "--tcr_contact_verifier_weight", "1.0",
        "--tcr_contact_peptide_chain_index", "0",
        "--tcr_contact_mhc_chain_indices", "1",
        "--tcr_contact_partner_tcr_chain_index", "2",
    ],
}
ARMS = {
    "A": ("off", "off"), "B": ("tcrlm", "off"),
    "C": ("off", "on"), "D": ("tcrlm", "on"),
    "E": ("esmc", "off"), "F": ("esmc", "on"),
}


def build_command(case_id: str, arm: str, out_root: Path, args) -> tuple[list[str], Path]:
    case = json.loads((SPEC_DIR / f"{case_id}.json").read_text())
    case_dir = out_root / case_id / f"arm_{arm}"
    case_dir.mkdir(parents=True, exist_ok=True)
    prior_key, contact_key = ARMS[arm]
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
        "--esmfold2_verifier_num_loops", str(args.verifier_loops),
        "--esmfold2_verifier_num_sampling_steps", str(args.verifier_steps),
        # Deep-and-narrow, matching 20260814_liubo_cdr6_fast_search: fewer
        # candidates per step, many more steps.
        "--b_candidates", "4",
        "--esmfold2_stageb_rerank_pool_size", "4",
        "--esmfold2_stageb_verifier_candidates", "4",
        "--mcts_handoff_top_k", "2",
        "--esmfold2_stagea_mutation_fraction", "0.5",
        "--esmfold2_stagea_candidate_strategy", "positive_score",
        "--esmfold2_stagea_include_argmax_candidate", "true",
        "--esmfold2_stagea_soft_steps", str(args.stagea_soft),
        "--esmfold2_stageb_soft_steps", str(args.stageb_soft),
        # Bingchen splits 110 as Stage A 70 + Stage B 40.
        "--mcts_bootstrap_steps", "70",
        "--mcts_bootstrap_max_steps", "70",
        "--output_dir", str(case_dir),
        "--esmfold2_artifact_dir", str(case_dir / "artifacts"),
        "--log_file", str(case_dir / "run.log"),
    ]
    command += PRIOR_FLAGS[prior_key] + CONTACT_FLAGS[contact_key]
    return command, case_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="A,B,C,D")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--gpus", default="4,6,7")
    parser.add_argument("--out", default="/root/mingxuan/matched_tcr6_20260826")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--n-steps", dest="n_steps", type=int, default=110)
    parser.add_argument("--verifier-loops", dest="verifier_loops", type=int, default=4)
    parser.add_argument("--verifier-steps", dest="verifier_steps", type=int, default=50)
    parser.add_argument("--stagea-soft", dest="stagea_soft", type=int, default=30)
    parser.add_argument("--stageb-soft", dest="stageb_soft", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]

    jobs: queue.Queue = queue.Queue()
    # Interleave by arm so partial results still cover every arm evenly.
    # Arm A last: it is the arm most likely to be replaceable by Bingchen's
    # existing 20260814 run, so if the queue is cut short it is the cheapest
    # one to lose.
    ordered_arms = [a for a in arms if a != "A"] + [a for a in arms if a == "A"]

    def already_done(case_id: str, arm: str) -> bool:
        # A finished job leaves hallucination_result.txt; anything else (a
        # half-written artifacts dir from an OOM kill) is treated as unfinished
        # and re-run.
        return (out_root / case_id / f"arm_{arm}" / "hallucination_result.txt").exists()

    skipped = 0
    for arm in ordered_arms:
        for case_id in cases:
            if already_done(case_id, arm):
                skipped += 1
                continue
            jobs.put((case_id, arm, 0))

    manifest = {
        "seed": args.seed, "n_steps": args.n_steps,
        "verifier": f"{args.verifier_loops}x{args.verifier_steps}",
        "stagea_soft": args.stagea_soft, "stageb_soft": args.stageb_soft,
        "arms": arms, "cases": cases, "gpus": gpus, "resumed_skipped": skipped,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        return 0

    results: dict[str, int] = {}
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                case_id, arm, attempt = jobs.get_nowait()
            except queue.Empty:
                return
            key = f"{case_id}/arm_{arm}"
            command, case_dir = build_command(case_id, arm, out_root, args)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["HF_HOME"] = "/root/Bingchen/resources/hf_cache"
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            env["PYTHONPATH"] = str(REPO)
            started = time.time()
            print(f"[gpu{gpu}] START {key} (attempt {attempt})", flush=True)
            with open(case_dir / "stdout.log", "w") as handle:
                proc = subprocess.run(
                    command, cwd=str(REPO), env=env, stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            elapsed = time.time() - started
            print(f"[gpu{gpu}] END   {key} rc={proc.returncode} "
                  f"({elapsed/60:.1f} min)", flush=True)
            if proc.returncode != 0 and attempt == 0:
                print(f"[gpu{gpu}] requeue {key}", flush=True)
                jobs.put((case_id, arm, 1))
            else:
                with lock:
                    results[key] = proc.returncode
                    (out_root / "job_status.json").write_text(
                        json.dumps(results, indent=2, sort_keys=True)
                    )
            jobs.task_done()

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    failed = {k: v for k, v in results.items() if v != 0}
    print(f"\ndone: {len(results)} jobs, {len(failed)} failed", flush=True)
    if failed:
        print("failed:", json.dumps(failed, indent=2), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
