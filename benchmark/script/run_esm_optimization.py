#!/usr/bin/env python3
"""Paired ESM optimization experiments: microbench, campaign, rescore, report.

Each optimization arm runs in a fresh process. Final rescoring always uses
unoptimized ESMFold2; search confidence is never substituted for final quality.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

ARMS = {
    "baseline": ("off", "off", "off"),
    "esmc_exact": ("off", "off", "exact"),
    "proposal_exact": ("exact", "off", "off"),
    "verifier_exact": ("off", "exact", "off"),
    "exact": ("exact", "exact", "exact"),
    "proposal_fast": ("fast", "exact", "off"),
    "verifier_fast": ("exact", "fast", "off"),
    "fast": ("fast", "fast", "exact"),
    "big": ("big", "big", "exact"),
}
CAMPAIGN_ARMS = ("baseline", "exact", "fast", "big")
MODEL_REVISIONS = {
    "biohub/ESMFold2": "1ebf0e3481a5184eb6171d40615c79e384b48796",
    "biohub/ESMFold2-Experimental-Fast": "04dec820ede9283c9893e318e5ca5a9ac2ab93bc",
}


def pinned_model(value):
    if value not in MODEL_REVISIONS:
        path = Path(value).expanduser().resolve()
        if not (path / "config.json").is_file():
            raise ValueError(f"Use a pinned local model snapshot: {value}")
        return str(path)
    from huggingface_hub import snapshot_download
    return snapshot_download(value, revision=MODEL_REVISIONS[value], local_files_only=True)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n")
    temporary.replace(path)


def git_state(path):
    path = Path(path).resolve()
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=path, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def selected_candidates(directory, top_k):
    path = Path(directory) / "esmfold2_artifacts/candidates.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    unique = {}
    for row in rows:
        seq = row["sequence"]
        if seq not in unique or row["total_loss"] < unique[seq]["total_loss"]:
            unique[seq] = row
    return sorted(unique.values(), key=lambda row: (row["total_loss"], row["sequence"]))[:top_k], len(unique), len(rows)



def compare_tensors(actual, expected):
    """Compare finite values separately from constrained infinities; NaN fails."""
    import numpy as np
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape:
        raise ValueError("Tensor shape mismatch")
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("NaN in tensor comparison")
    af, bf = np.isfinite(a), np.isfinite(b)
    matching = bool(np.array_equal(af, bf)
                    and np.array_equal(np.isposinf(a), np.isposinf(b))
                    and np.array_equal(np.isneginf(a), np.isneginf(b)))
    result = {"bitwise_equal": bool(np.array_equal(a, b)),
              "nonfinite_pattern_equal": matching,
              "finite_count": int((af & bf).sum()),
              "max_abs": None, "relative_l2": None, "cosine": None}
    if not matching:
        return result
    x, y = a[af].astype(np.float64), b[bf].astype(np.float64)
    result["max_abs"] = float(np.max(np.abs(x - y))) if x.size else 0.0
    result["relative_l2"] = float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-30))
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    result["cosine"] = float(x @ y / norm) if norm else None
    return result


def verifier_counts(directory):
    path = Path(directory) / "optimization_report.json"
    counters = json.loads(path.read_text()).get("counters", {}) if path.exists() else {}
    return {"verifier_calls": counters.get("verifier.calls"),
            "verifier_successes": counters.get("verifier.successes"),
            "verifier_errors": counters.get("verifier.errors")}


def score_key(row):
    return (row["arm"], row["case_id"], row["search_seed"], row["rank"], row["eval_seed"])


def freeze_rescore_spec(args, campaign):
    expected = []
    failed_searches = []
    for record in campaign["records"]:
        directory = Path(record["directory"])
        receipt = directory / "experiment_run.json"
        returncode = json.loads(receipt.read_text()).get("returncode") if receipt.exists() else None
        if returncode is None:
            raise ValueError("Freeze rescoring only after all planned searches finish")
        if returncode != 0:
            failed_searches.append({"arm": record["arm"], "case_id": record["case"]["case_id"],
                                    "search_seed": record["seed"], "returncode": returncode})
            continue
        candidates, _, _ = selected_candidates(directory, args.top_k)
        if not candidates:
            raise ValueError(f"No candidates at {directory}")
        if record["seed"] in args.seeds:
            raise ValueError("Evaluation seeds must differ from search seeds")
        for rank, candidate in enumerate(candidates, 1):
            for seed in args.seeds:
                expected.append({"arm": record["arm"], "case_id": record["case"]["case_id"],
                                 "search_seed": record["seed"], "rank": rank,
                                 "eval_seed": seed, "binder_seq": candidate["sequence"]})
    spec = {"schema_version": 1,
            "campaign_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
            "top_k": args.top_k, "seeds": args.seeds, "loops": args.loops,
            "sampling_steps": args.sampling_steps, "model": args.verifier_model,
            "evaluator": "unoptimized_esmfold2", "expected": expected,
            "failed_searches": failed_searches}
    spec["contract_hash"] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return spec


def validate_score(score, expected, spec, score_path):
    if score_key(score) != score_key(expected) or score["binder_seq"] != expected["binder_seq"]:
        raise ValueError("Rescore candidate identity mismatch")
    for key in ("contract_hash", "loops", "sampling_steps", "model", "evaluator"):
        if score.get(key) != spec[key]:
            raise ValueError(f"Rescore contract mismatch: {key}")
    value = score.get("metrics", {}).get("pair_chain_iptm")
    if value is None or not math.isfinite(float(value)):
        raise ValueError("Rescore pair_chain_iptm is missing or nonfinite")
    structure = Path(score_path).with_name("complex.cif")
    structure_hash = score.get("structure_sha256")
    if not structure_hash:
        raise ValueError("Rescore structure hash missing; rerun legacy scores in a new output directory")
    if not structure.is_file() or hashlib.sha256(structure.read_bytes()).hexdigest() != structure_hash:
        raise ValueError(f"Rescore structure is missing or changed: {structure}")


def validated_scores(scores_root, manifest):
    root = Path(scores_root)
    score_paths = list(root.rglob("score.json"))
    spec_path = root / "rescore_spec.json"
    if not spec_path.exists():
        if score_paths:
            raise ValueError("Missing frozen rescore_spec.json; cannot verify score completeness")
        return [], {}, {}
    spec = json.loads(spec_path.read_text())
    payload = {key: value for key, value in spec.items() if key != "contract_hash"}
    if spec["contract_hash"] != hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest():
        raise ValueError("Rescore specification hash mismatch")
    if spec["campaign_sha256"] != hashlib.sha256(Path(manifest).read_bytes()).hexdigest():
        raise ValueError("Rescore campaign mismatch")
    expected = {score_key(row): row for row in spec["expected"]}
    if len(expected) != len(spec["expected"]):
        raise ValueError("Duplicate expected rescore keys")
    observed = {}
    for path in score_paths:
        score = json.loads(path.read_text())
        key = score_key(score)
        if key in observed or key not in expected:
            raise ValueError("Duplicate or unexpected rescore key")
        validate_score(score, expected[key], spec, path)
        observed[key] = score
    required, counts = {}, {}
    for key in expected:
        required[key[:3]] = required.get(key[:3], 0) + 1
    for key in observed:
        counts[key[:3]] = counts.get(key[:3], 0) + 1
    complete = {key for key, count in required.items() if counts.get(key, 0) == count}
    return [row for key, row in observed.items() if key[:3] in complete], required, counts


def cases(config, names):
    from run_esmfold2_59 import load_benchmark_config, load_prepared_cases
    all_cases = load_prepared_cases(load_benchmark_config(Path(config).resolve()))
    unknown = set(names) - {case.case_id for case in all_cases}
    if unknown:
        raise ValueError(f"Unknown cases: {sorted(unknown)}")
    return [case for case in all_cases if not names or case.case_id in names]


def backend_config(args, case, *, reference=False):
    from bindcraft.hallucination.esmfold2_backend import ESMFold2ProposalConfig
    p, v, e = ARMS["baseline" if reference else args.arm]
    return ESMFold2ProposalConfig(
        experimental_model_name=args.proposal_model,
        verifier_model_name=args.verifier_model,
        optimization_root=args.kit_root, proposal_optimization=p,
        verifier_optimization=v, esmc_optimization=e,
        optimization_report_path=str(Path(args.output) / "optimization_report.json"),
        optimization_timing=True, verifier_loss_mode="aligned_common",
        verifier_num_loops=args.loops, verifier_num_sampling_steps=args.sampling_steps,
        target_chain_roles=case.target_chain_roles or None,
        binder_mutable_mask=tuple(c == "1" for c in case.mutable_mask) if case.mutable_mask else None,
        soft_steps=args.soft_steps if hasattr(args, "soft_steps") else 1,
        one_step_diffusion_interval=1,
    )


def scalar_metrics(result):
    values = {"total_loss": float(result.total_loss)}
    for key, value in {**result.summary_confidence, **result.loss_dict}.items():
        if isinstance(value, (int, float)):
            values[key] = float(value)
    return values


def apply_micro_determinism(kit_root):
    """Apply ef2inv's stock-vs-exact parity recipe to every micro arm."""
    if not kit_root:
        raise ValueError("--kit-root is required with --deterministic")
    source = Path(kit_root).expanduser().resolve() / "ef2inv/opt"
    if not source.is_dir():
        raise FileNotFoundError(source)
    sys.path.insert(0, str(source))
    os.environ.setdefault("TORCHINDUCTOR_SHAPE_PADDING", "0")
    os.environ.setdefault("TORCHINDUCTOR_DETERMINISTIC", "1")
    from ef2inv_opt import det
    return det.apply(1)


def micro(args):
    import numpy as np
    import torch
    from bindcraft.hallucination.esmfold2_runtime import ESMFold2RuntimeBackend, _seed_context, _structure_losses

    if not torch.cuda.is_available():
        raise RuntimeError("micro requires CUDA")
    determinism = apply_micro_determinism(args.kit_root) if args.deterministic else {}
    case = cases(args.config, [args.case])[0]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    sequence = case.initial_binder_sequence or "".join(rng.choice("ARNDQEGHILKMFPSTWYV") for _ in range(case.binder_length))
    backend = ESMFold2RuntimeBackend(config=backend_config(args, case), device=args.device, task_type=case.task_type)
    torch.cuda.set_device(backend.device)
    info = {"arm": args.arm, "case_id": case.case_id, "phase": args.phase,
            "seed": args.seed, "gpu": torch.cuda.get_device_name(backend.device),
            "torch": torch.__version__, "loops": args.loops, "sampling_steps": args.sampling_steps,
            "target_tokens": len(case.target_sequence.replace("|", "")), "binder_tokens": len(sequence),
            "composition": "proteocraft-uplifting-v2", "deterministic": args.deterministic,
            "determinism_evidence": determinism, "config": asdict(backend.config)}
    tensor_output = {}
    started = time.perf_counter()
    try:
        if args.phase == "verifier":
            backend._ensure_verifier_loaded()
        else:
            backend._ensure_proposal_models_loaded()
        torch.cuda.synchronize(backend.device)
        info["load_s"] = time.perf_counter() - started
        info["model_allocated_bytes"] = torch.cuda.memory_allocated(backend.device)
        info["model_reserved_bytes"] = torch.cuda.memory_reserved(backend.device)
        torch.cuda.reset_peak_memory_stats(backend.device)

        def call():
            nonlocal tensor_output
            tensor_output = {}
            with _seed_context(args.seed):
                if args.phase == "verifier":
                    result = backend.verify(binder_seq=sequence, target_seq=case.target_sequence, seed=args.seed)
                    tensor_output = {"coordinates": result.coordinate.detach()} if result.coordinate is not None else {}
                    return scalar_metrics(result)
                if args.phase == "update":
                    result = backend.continuous_design(
                        current_seq=sequence, target_seq=case.target_sequence,
                        global_step=0, seed=args.seed,
                        mutable_mask=backend.config.binder_mutable_mask,
                        soft_steps_override=args.soft_steps,
                    )
                    tensor_output = {"logits": result.logits_after.detach()}
                    return {"sequences": list(result.sequences), "soft_steps": list(result.soft_step_metrics)}
                design = backend._sequence_to_design_one_hot(sequence).requires_grad_(True)
                if args.phase == "proposal":
                    result = backend._fold_distogram(target_seq=case.target_sequence, design=design, seed=args.seed)
                    losses = _structure_losses(result["distogram_logits"], len(sequence))
                    loss = sum(losses[k] for k in ("intra_contact_loss", "inter_contact_loss", "glob_loss"))
                    tensor_output = {"distogram": result["distogram_logits"].detach()}
                else:
                    mask = torch.tensor(case.mutable_mask and [c == "1" for c in case.mutable_mask] or [True] * len(sequence), device=backend.device)
                    loss = backend._compute_esmc_pseudoperplexity(design, score_mask=mask).mean()
                    tensor_output = {}
                grad = torch.autograd.grad(loss, design)[0]
                if not torch.isfinite(grad).all() or grad.norm() == 0:
                    raise RuntimeError("Input gradient is zero or non-finite")
                tensor_output["gradient"] = grad.detach()
                return {"loss": float(loss.detach()), "gradient_norm": float(grad.norm())}

        times = []
        for i in range(args.warmup + args.repeats):
            torch.cuda.synchronize(backend.device)
            start = time.perf_counter()
            metrics = call()
            torch.cuda.synchronize(backend.device)
            elapsed = time.perf_counter() - start
            if i == 0:
                info["first_call_s"] = elapsed
            if i >= args.warmup:
                times.append(elapsed)
        info.update(status="ok", times_s=times, median_s=statistics.median(times),
                    p95_s=float(np.percentile(times, 95)), metrics=metrics,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(backend.device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(backend.device))
        tensor_output = {key: value.float().cpu().numpy() for key, value in tensor_output.items()}
        np.savez_compressed(out / "tensors.npz", **tensor_output)
        if args.reference:
            reference = Path(args.reference)
            meta = json.loads((reference / "result.json").read_text())
            for key in ("case_id", "phase", "seed", "gpu", "torch", "loops",
                        "sampling_steps", "target_tokens", "binder_tokens", "deterministic"):
                if info[key] != meta[key]:
                    raise ValueError(f"Reference mismatch: {key}")
            for key in ("experimental_model_name", "verifier_model_name", "soft_steps",
                        "binder_mutable_mask", "target_chain_roles"):
                # Normalize tuples through JSON as the saved reference uses lists.
                current = json.loads(json.dumps(info["config"][key]))
                if current != meta["config"][key]:
                    raise ValueError(f"Reference configuration mismatch: {key}")
            with np.load(reference / "tensors.npz") as ref:
                diffs = {}
                for key, value in tensor_output.items():
                    expected = ref[key]
                    diffs[key] = compare_tensors(value, expected)
            info["differences"] = diffs
            info["speedup"] = meta["median_s"] / info["median_s"]
    except Exception as exc:
        info.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        info["process_work_s"] = time.perf_counter() - started
        write_json(out / "result.json", info)
        backend.optimizations.finalize()
    print(json.dumps({k: info[k] for k in ("arm", "phase", "median_s", "peak_allocated_bytes")}, indent=2))


def prepare(args):
    from run_esmfold2_59 import build_hallucination_command, parse_overrides
    selected = cases(args.config, args.cases)
    records = []
    overrides = parse_overrides(args.set)
    reserved = {"seed", "esmfold2_proposal_optimization", "esmfold2_verifier_optimization", "esmfold2_esmc_optimization", "esmfold2_optimization_root", "esmfold2_optimization_timing"}
    if reserved.intersection(k for k, _ in overrides):
        raise ValueError("Use campaign arm/seed options instead of overriding experimental factors")
    for case in selected:
        for seed in args.seeds:
            # Reproducible rotation avoids always benchmarking baseline on a cold node.
            arms = list(args.arms)
            random.Random(f"{seed}:{case.case_id}").shuffle(arms)
            for arm in arms:
                p, v, e = ARMS[arm]
                directory = Path(args.output).resolve() / arm / case.case_id / f"seed-{seed}"
                settings = [*overrides, ("seed", str(seed)),
                            ("esmfold2_optimization_root", str(Path(args.kit_root).resolve())),
                            ("esmfold2_proposal_optimization", p), ("esmfold2_verifier_optimization", v),
                            ("esmfold2_esmc_optimization", e),
                            ("esmfold2_verifier_model", args.verifier_model),
                            ("esmfold2_experimental_model", args.proposal_model)]
                command = build_hallucination_command(case, profile="esmfold2_d", device=args.device, case_dir=directory, overrides=settings)
                command.append("--esmfold2_optimization_timing")
                command.extend(("--esmfold2_save_verified_candidate_cif", "true"))
                records.append({"arm": arm, "seed": seed, "case": asdict(case), "directory": str(directory), "command": command})
    manifest = {"schema_version": 1, "composition": "proteocraft-uplifting-v2", "records": records,
                "repository": git_state(ROOT), "optimization_source": git_state(args.kit_root),
                "models": {"proposal": args.proposal_model, "verifier": args.verifier_model},
                "profile": str(ROOT / "configs/proteocraft_profiles/esmfold2_d.yaml"),
                "profile_sha256": hashlib.sha256((ROOT / "configs/proteocraft_profiles/esmfold2_d.yaml").read_bytes()).hexdigest()}
    write_json(Path(args.output) / "campaign.json", manifest)
    print(f"Prepared {len(records)} searches: {Path(args.output) / 'campaign.json'}")


def run(args):
    campaign = json.loads(Path(args.manifest).read_text())
    if hashlib.sha256(Path(campaign["profile"]).read_bytes()).hexdigest() != campaign["profile_sha256"]:
        raise ValueError("Launch profile changed since campaign preparation")
    failed = False
    for index, rec in enumerate(campaign["records"]):
        if index % args.num_shards != args.shard_index:
            continue
        directory = Path(rec["directory"])
        receipt = directory / "experiment_run.json"
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if args.skip_existing and old.get("returncode") == 0 and old["command"] == rec["command"]:
                continue
            raise FileExistsError(f"Run already exists; choose a new output root: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "case_spec.json", rec["case"])
        write_json(receipt, {**rec, "status": "running", "returncode": None})
        begin = time.perf_counter()
        with (directory / "process.log").open("w") as handle:
            result = subprocess.run(rec["command"], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
        complete = (directory / "esmfold2_artifacts/run_summary.json").exists()
        returncode = result.returncode if result.returncode or complete else 1
        write_json(receipt, {**rec, "status": "complete" if returncode == 0 else "failed",
                            "returncode": returncode, "wall_s": time.perf_counter() - begin})
        print(f"{rec['arm']} {rec['case']['case_id']} seed={rec['seed']}: exit {result.returncode}", flush=True)
        failed |= returncode != 0
    return int(failed)


def rescore(args):
    """Use a clean baseline process and common seeds for all arms' selected sequences."""
    from types import SimpleNamespace
    from bindcraft.hallucination.esmfold2_runtime import ESMFold2RuntimeBackend
    campaign = json.loads(Path(args.manifest).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    spec = freeze_rescore_spec(args, campaign)
    spec_path = output / "rescore_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError("Rescore contract changed; choose a fresh output root")
    write_json(spec_path, spec)
    expected = {score_key(row): row for row in spec["expected"]}
    shared_model, shared_builder = None, None
    for rec in campaign["records"]:
        directory = Path(rec["directory"])
        receipt = directory / "experiment_run.json"
        if not receipt.exists() or json.loads(receipt.read_text()).get("returncode") != 0:
            continue
        selected, _, _ = selected_candidates(directory, args.top_k)
        case = SimpleNamespace(**rec["case"])
        config = backend_config(args, case, reference=True)
        config = replace(config, optimization_report_path=str(
            output / rec["arm"] / case.case_id / f"seed-{rec['seed']}" / "optimization_report.json"
        ))
        backend = ESMFold2RuntimeBackend(config=config, device=args.device, task_type=case.task_type,
                                        verifier_model=shared_model, input_builder=shared_builder)
        try:
            for rank, row in enumerate(selected, 1):
                for eval_seed in args.seeds:
                    key = f"{rec['arm']}/{case.case_id}/seed-{rec['seed']}/rank-{rank}/eval-{eval_seed}"
                    destination = output / key
                    if (destination / "score.json").exists():
                        old = json.loads((destination / "score.json").read_text())
                        identity = (rec["arm"], case.case_id, rec["seed"], rank, eval_seed)
                        validate_score(old, expected[identity], spec, destination / "score.json")
                        continue
                    start = time.perf_counter()
                    hotspots = {"target_hotspot_concat_indices": [int(i) - 1 for i in case.target_hotspots_1based]}
                    result = backend.verify(binder_seq=row["sequence"], target_seq=case.target_sequence, seed=eval_seed, hotspot_params=hotspots)
                    cif = result.summary_confidence.get("mmcif")
                    if not isinstance(cif, str) or not cif.strip():
                        raise ValueError("Rescore did not produce a non-empty mmCIF structure")
                    destination.mkdir(parents=True, exist_ok=True)
                    structure = destination / "complex.cif"
                    temporary = structure.with_suffix(".cif.tmp")
                    temporary.write_text(cif)
                    temporary.replace(structure)
                    score = {
                        "arm": rec["arm"], "case_id": case.case_id, "target_name": case.target_name,
                        "search_seed": rec["seed"], "eval_seed": eval_seed, "rank": rank,
                        "binder_seq": row["sequence"], "metrics": scalar_metrics(result),
                        "loops": args.loops, "sampling_steps": args.sampling_steps,
                        "model": args.verifier_model, "contract_hash": spec["contract_hash"],
                        "evaluator": "unoptimized_esmfold2", "elapsed_s": time.perf_counter() - start,
                        "structure_sha256": hashlib.sha256(structure.read_bytes()).hexdigest()}
                    identity = (rec["arm"], case.case_id, rec["seed"], rank, eval_seed)
                    validate_score(score, expected[identity], spec, destination / "score.json")
                    # The score is the commit marker, written only after the CIF.
                    write_json(destination / "score.json", score)
        finally:
            backend.optimizations.finalize()
            shared_model, shared_builder = backend.verifier_model, backend.input_builder
            del backend


def export(args):
    """Build the existing metric pipeline's manifest for AF3 / AtlasFold-M."""
    import gemmi
    import string
    sys.path.insert(0, str(ROOT / "evaluation/metric_pipeline"))
    from eval_pipeline.manifest import ManifestRow, write_manifest
    campaign = json.loads(Path(args.manifest).read_text())
    rows = []
    artifact_dir = Path(args.output).resolve().parent / "reference_structures"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for rec in campaign["records"]:
        directory = Path(rec["directory"])
        receipt_path = directory / "experiment_run.json"
        if not receipt_path.exists():
            continue
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("returncode") != 0:
            continue
        selected, n_unique, candidate_records = selected_candidates(directory, args.top_k)
        counts = verifier_counts(directory)
        case = rec["case"]
        targets = case["target_sequence"].split("|")
        roles = case["target_chain_roles"] or ["target"] * len(targets)
        # target_seq contains every context chain, including CDR scaffolds.  The
        # metric pipeline checks its residue count against these chain IDs and
        # uses target_chain_roles separately to exclude scaffolds from metrics.
        target_ids = "+".join(string.ascii_uppercase[i] for i in range(len(targets)))
        for rank, candidate in enumerate(selected, 1):
            if not candidate.get("structure_cif"):
                raise ValueError(f"Missing search structure for {directory}; enable --esmfold2_save_verified_candidate_cif")
            source = directory / "esmfold2_artifacts" / candidate["structure_cif"]
            stem = f"{rec['arm']}__{case['case_id']}__s{rec['seed']}__r{rank}"
            pdb = artifact_dir / f"{stem}.pdb"
            gemmi.read_structure(str(source)).write_pdb(str(pdb))
            rows.append(ManifestRow(
                method=rec["arm"], case=case["case_id"], design_id=stem,
                target_pdb_path=str(pdb), target_chain=target_ids,
                target_seq=":".join(targets), binder_seq=candidate["sequence"],
                designed_complex_pdb_path=str(pdb), target_chain_in_complex=target_ids,
                binder_chain_in_complex=string.ascii_uppercase[len(targets)],
                n_generated_trajectories=-1, n_accepted_by_method=len(selected),
                candidate_rank=rank, source_filter_status="top_k_search_loss",
                verifier_calls=counts["verifier_calls"] if counts["verifier_calls"] is not None else -1, design_gpu_seconds=float(receipt["wall_s"]),
                target_chain_roles=",".join(roles),
                target_hotspot_concat_indices=",".join(str(int(i)-1) for i in case["target_hotspots_1based"]),
                notes=("Reference is the search prediction, not experimental ground truth. "
                       "Single-GPU wall seconds include startup. Candidate records are not model calls. "
                       f"{candidate_records} candidate records; {n_unique} unique sequences. "
                       "Generated trajectory count is unavailable (-1)."),
            ))
    write_manifest(rows, args.output)
    print(f"Exported {len(rows)} candidates to {args.output}")


def clustered_interval(values, seed=2026, iterations=2000):
    """Equal weight per independent source target; resample target clusters."""
    means = [statistics.mean(group) for group in values.values() if group]
    if not means:
        return None
    rng = random.Random(seed)
    samples = sorted(statistics.mean(rng.choices(means, k=len(means))) for _ in range(iterations))
    return {"mean": statistics.mean(means), "ci95": [samples[int(iterations * .025)], samples[int(iterations * .975)]],
            "source_targets": len(means), "interpretation": "exploratory" if len(means) < 5 else "target_cluster_bootstrap"}


def report(args):
    campaign = json.loads(Path(args.manifest).read_text())
    scores, expected_counts, observed_counts = validated_scores(args.scores, args.manifest)
    success_threshold = float(getattr(args, "success_threshold", 0.8))
    if not 0.0 <= success_threshold <= 1.0:
        raise ValueError("success-threshold must be between zero and one")
    rows = []
    for rec in campaign["records"]:
        receipt = Path(rec["directory"]) / "experiment_run.json"
        receipt = json.loads(receipt.read_text()) if receipt.exists() else {}
        selected = [s for s in scores if (s["arm"], s["case_id"], s["search_seed"]) == (rec["arm"], rec["case"]["case_id"], rec["seed"])]
        by_rank = {}
        for score in selected:
            value = score["metrics"].get("pair_chain_iptm")
            if value is not None:
                by_rank.setdefault(score["rank"], []).append(value)
        rank_means = {rank: statistics.mean(values) for rank, values in by_rank.items()}
        primary = rank_means.get(1)
        best_k = max(rank_means.values()) if rank_means else None
        candidate_file = Path(rec["directory"]) / "esmfold2_artifacts/candidates.jsonl"
        if candidate_file.exists():
            _, unique_candidates, candidate_records = selected_candidates(rec["directory"], 10**9)
        else:
            unique_candidates, candidate_records = None, None
        identity = (rec["arm"], rec["case"]["case_id"], rec["seed"])
        expected_count = expected_counts.get(identity, 0)
        observed_count = observed_counts.get(identity, 0)
        complete = expected_count > 0 and observed_count == expected_count
        if receipt.get("returncode") != 0:
            primary = best_k = None
        rows.append({"arm": rec["arm"], "case_id": rec["case"]["case_id"], "target_name": rec["case"]["target_name"],
                     "seed": rec["seed"], "returncode": receipt.get("returncode"), "wall_s": receipt.get("wall_s"),
                     "unique_candidates": unique_candidates, "candidate_records": candidate_records,
                     **verifier_counts(rec["directory"]),
                     "rescore_complete": complete, "expected_scores": expected_count,
                     "observed_scores": observed_count,
                     "rescored_samples": len(selected), "rescored_ranks": len(rank_means),
                     "top1_pair_chain_iptm": primary, "bestk_pair_chain_iptm": best_k,
                     "bestk_success": (best_k >= success_threshold) if best_k is not None else None})
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["arm"])
        writer.writeheader()
        writer.writerows(rows)
    baseline = {(r["case_id"], r["seed"]): r for r in rows if r["arm"] == "baseline"}
    summary = {}
    for arm in sorted({r["arm"] for r in rows}):
        arm_rows = [r for r in rows if r["arm"] == arm]
        quality, bestk_quality, speed = {}, {}, {}
        for row in arm_rows:
            ref = baseline.get((row["case_id"], row["seed"]))
            if ref is None:
                continue
            if row["top1_pair_chain_iptm"] is not None and ref["top1_pair_chain_iptm"] is not None:
                quality.setdefault(row["target_name"], []).append(row["top1_pair_chain_iptm"] - ref["top1_pair_chain_iptm"])
            if row["bestk_pair_chain_iptm"] is not None and ref["bestk_pair_chain_iptm"] is not None:
                bestk_quality.setdefault(row["target_name"], []).append(row["bestk_pair_chain_iptm"] - ref["bestk_pair_chain_iptm"])
            if row["returncode"] == ref["returncode"] == 0 and row["wall_s"] and ref["wall_s"]:
                speed.setdefault(row["target_name"], []).append(ref["wall_s"] / row["wall_s"])
        summary[arm] = {"planned": len(arm_rows), "completed": sum(r["returncode"] == 0 for r in arm_rows),
                        "failed": sum(r["returncode"] not in (None, 0) for r in arm_rows),
                        "missing": sum(r["returncode"] is None for r in arm_rows),
                        "quality_threshold": success_threshold,
                        "rescore_complete_runs": sum(r["rescore_complete"] for r in arm_rows),
                        "bestk_successes": sum(r["bestk_success"] is True for r in arm_rows),
                        "paired_quality_delta": clustered_interval(quality),
                        "paired_bestk_quality_delta": clustered_interval(bestk_quality),
                        "paired_speedup": clustered_interval(speed)}
    write_json(path.with_suffix(".summary.json"), summary)
    plot_rows = [r for r in rows if r["returncode"] == 0 and r["top1_pair_chain_iptm"] is not None]
    if plot_rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for arm in sorted({r["arm"] for r in plot_rows}):
            selected = [r for r in plot_rows if r["arm"] == arm]
            ax.scatter([r["wall_s"] / 3600 for r in selected], [r["top1_pair_chain_iptm"] for r in selected], label=arm)
        ax.set(xlabel="Single-GPU search wall time (hours, including startup)", ylabel="Top-1 unoptimized ESMFold2 pair-chain iPTM")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path.with_suffix(".png"), dpi=180)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
    print(f"Wrote {len(rows)} paired rows to {path}; missing scores remain missing")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("micro", "prepare", "rescore"):
        q = sub.add_parser(name)
        q.add_argument("--output", required=True)
        q.add_argument("--device", default="cuda:0")
        q.add_argument("--kit-root", default=os.environ.get("PROTEOCRAFT_UPLIFT_ROOT"))
        q.add_argument("--proposal-model", default="biohub/ESMFold2-Experimental-Fast")
        q.add_argument("--verifier-model", default="biohub/ESMFold2")
        if name != "rescore":
            q.add_argument("--config", default=str(ROOT / "benchmark/datasets/proteocraft_59_v1/benchmark.yaml"))
        if name != "prepare":
            q.add_argument("--loops", type=int, default=20 if name == "rescore" else 4)
            q.add_argument("--sampling-steps", type=int, default=200 if name == "rescore" else 50)
        if name == "micro":
            q.add_argument("--arm", choices=ARMS, required=True)
            q.add_argument("--case", required=True)
            q.add_argument("--phase", choices=["proposal", "prior", "verifier", "update"], required=True)
            q.add_argument("--seed", type=int, default=43)
            q.add_argument("--warmup", type=int, default=1)
            q.add_argument("--repeats", type=int, default=3)
            q.add_argument("--soft-steps", type=int, default=1)
            q.add_argument("--reference")
            q.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
        if name == "prepare":
            q.add_argument("--cases", nargs="*", default=[])
            q.add_argument("--arms", nargs="+", choices=ARMS, default=list(CAMPAIGN_ARMS))
            q.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45])
            q.add_argument("--set", action="append", default=[])
        if name == "rescore":
            q.add_argument("--manifest", required=True)
            q.add_argument("--top-k", type=int, default=5)
            q.add_argument("--seeds", nargs="+", type=int, default=[1043, 1044, 1045])
    q = sub.add_parser("run")
    q.add_argument("--manifest", required=True)
    q.add_argument("--num-shards", type=int, default=1)
    q.add_argument("--shard-index", type=int, default=0)
    q.add_argument("--skip-existing", action="store_true")
    q = sub.add_parser("report")
    q.add_argument("--manifest", required=True)
    q.add_argument("--scores", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--success-threshold", type=float, default=0.8)
    q = sub.add_parser("export")
    q.add_argument("--manifest", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--top-k", type=int, default=5)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if hasattr(args, "repeats") and (args.repeats < 1 or args.warmup < 0):
        raise ValueError("repeats must be positive and warmup nonnegative")
    if args.command == "prepare" and not args.kit_root:
        raise ValueError("--kit-root is required")
    if args.command == "run" and not (args.num_shards > 0 and 0 <= args.shard_index < args.num_shards):
        raise ValueError("Invalid shard")
    if hasattr(args, "seeds") and len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique")
    if hasattr(args, "top_k") and args.top_k <= 0:
        raise ValueError("top-k must be positive")
    if hasattr(args, "verifier_model"):
        args.verifier_model = pinned_model(args.verifier_model)
        args.proposal_model = pinned_model(args.proposal_model)
    return globals()[args.command](args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
