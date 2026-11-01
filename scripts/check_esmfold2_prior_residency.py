#!/usr/bin/env python3
"""Capture an unchanged search with read-only gradient hooks, or compare arms.

Run each arm in a fresh process. Model inputs and budgets belong to the supplied
search CLI/config, not this utility. Strict differences close the admission gate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import random
import resource
import runpy
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_copy(item) for item in value)
    return value


def capture(args):
    from bindcraft.hallucination.esmfold2_runtime import ESMFold2RuntimeBackend

    directory = args.trace_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    command = args.search_args[1:] if args.search_args[:1] == ["--"] else args.search_args
    if "--output-dir" in command:
        raise ValueError("Output is derived from trace-dir; do not pass --output-dir")
    torch.manual_seed(args.rng_seed)
    random.seed(args.rng_seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Full-model residency comparison requires CUDA")
    torch.cuda.manual_seed_all(args.rng_seed)
    trace = {"chunks": [], "raw_gradients": [], "updates": [], "proposals": []}
    original_init = ESMFold2RuntimeBackend._init_current_sequence_logits
    original_propose = ESMFold2RuntimeBackend.propose
    original_step = torch.optim.SGD.step
    original_argv = sys.argv
    tracked = {}

    def init_logits(self, *values, **kwargs):
        logits = original_init(self, *values, **kwargs)
        chunk = len(trace["chunks"])
        tracked[id(logits)] = chunk
        trace["chunks"].append(cpu_copy(logits))
        if "runtime" not in trace:
            trace["runtime"] = asdict(self.config)
            trace["task_type"] = self.task_type
        def record(gradient):
            trace["raw_gradients"].append({"chunk": chunk, "value": cpu_copy(gradient)})
        logits.register_hook(record)
        return logits

    def optimizer_step(self, *values, **kwargs):
        updates = []
        for group in self.param_groups:
            for parameter in group["params"]:
                if id(parameter) not in tracked:
                    raise RuntimeError("Unexpected SGD parameter in residency comparison")
                if parameter.grad is None:
                    raise RuntimeError("Missing update gradient")
                updates.append((parameter, {"chunk": tracked[id(parameter)],
                    "lr": group["lr"], "before": cpu_copy(parameter),
                    "gradient": cpu_copy(parameter.grad)}))
        result = original_step(self, *values, **kwargs)
        for parameter, record in updates:
            record["after"] = cpu_copy(parameter)
            trace["updates"].append(record)
        return result

    def propose(self, *values, **kwargs):
        result = original_propose(self, *values, **kwargs)
        trace["proposals"].append(cpu_copy({
            "inputs": {key: kwargs[key] for key in
                ("current_seq", "target_seq", "global_step", "seed") if key in kwargs},
            "before": result.logits_before, "after": result.logits_after,
            "sequences": result.sequences, "soft_step_metrics": result.soft_step_metrics,
            "candidate_metadata": result.candidate_metadata,
            "proposal_diagnostics": result.proposal_diagnostics}))
        return result

    ESMFold2RuntimeBackend._init_current_sequence_logits = init_logits
    ESMFold2RuntimeBackend.propose = propose
    torch.optim.SGD.step = optimizer_step
    started = time.perf_counter()
    receipt = {"status": "failed", "hooks_change_gradients": False,
               "timing_includes_symmetric_trace_copy_overhead": True}
    try:
        sys.argv = [str(args.entrypoint), *command, "--output-dir", str(directory / "search")]
        runpy.run_path(str(args.entrypoint), run_name="__main__")
        manifest = json.loads((directory / "search/manifest.json").read_text())
        if manifest["status"] != "completed" or not trace["updates"]:
            raise RuntimeError("Capture did not complete its search and gradient updates")
        torch.cuda.synchronize()
        trace["cpu_rng_after"] = torch.get_rng_state()
        trace["cuda_rng_after"] = torch.cuda.get_rng_state_all()
        trace["python_rng_after"] = random.getstate()
        receipt["status"] = "completed"
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        ESMFold2RuntimeBackend._init_current_sequence_logits = original_init
        ESMFold2RuntimeBackend.propose = original_propose
        torch.optim.SGD.step = original_step
        sys.argv = original_argv
        receipt.update(elapsed_s=time.perf_counter()-started,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            chunks=len(trace["chunks"]), raw_gradients=len(trace["raw_gradients"]),
            updates=len(trace["updates"]), proposals=len(trace["proposals"]))
        torch.save(trace, directory / "trace.pt")
        (directory / "CAPTURE.json").write_text(json.dumps(receipt, indent=2) + "\n")


def differences(left, right, path="root"):
    """Strict numeric equality; diagnostics never relax the admission rule."""
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.dtype != right.dtype or left.shape != right.shape:
            return [path + ": tensor shape/dtype differs"]
        if torch.equal(left, right):
            return []
        finite = torch.isfinite(left) & torch.isfinite(right)
        maximum = float((left[finite].double()-right[finite].double()).abs().max()) if finite.any() else None
        return [f"{path}: tensor differs; finite_max_abs={maximum}"]
    if type(left) is not type(right):
        return [path + ": type differs"]
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return [path + ": keys differ"]
        return [item for key in left for item in differences(left[key], right[key], f"{path}.{key}")]
    if isinstance(left, (tuple, list)):
        if len(left) != len(right):
            return [path + ": length differs"]
        return [item for index, (a, b) in enumerate(zip(left, right, strict=True))
                for item in differences(a, b, f"{path}[{index}]")]
    return [] if left == right else [f"{path}: values differ"]


def verification_values(directory):
    records = [json.loads(line) for line in (directory / "search/verifications.jsonl").read_text().splitlines()]
    if not records or any(record["status"] != "ok" for record in records):
        raise RuntimeError("Incomplete or failed full verification")
    for record in records:
        del record["context"]  # Runtime identity includes the intended placement setting.
        summary = record["summary_confidence"]
        summary.pop("structure_path", None)  # Output location is not a prediction.
        # Retain the deterministic, mtime=0 coordinate-file digest. A structure
        # change must fail even when the reported scalar scores happen to match.
    return records


def optimization_identity(directory):
    receipt = json.loads((directory / "search/optimization_report.json").read_text())
    return {key: receipt[key] for key in
            ("source_root", "source_revision", "source_hashes", "versions", "requested")}


def compare(args):
    receipts = [json.loads((directory / "CAPTURE.json").read_text())
                for directory in (args.baseline, args.candidate)]
    if any(receipt["status"] != "completed" for receipt in receipts):
        raise RuntimeError("Both capture processes must have completed")
    traces = [torch.load(directory / "trace.pt", map_location="cpu", weights_only=True)
              for directory in (args.baseline, args.candidate)]
    policies = [trace["runtime"]["memory_prior_residency"] for trace in traces]
    if policies[0] != "per_step" or policies[1] == "per_step":
        raise ValueError("Compare historical per_step against an explicit retention policy")
    for trace in traces:
        for key in ("memory_prior_residency", "optimization_report_path"):
            del trace["runtime"][key]
    failed = differences(traces[0], traces[1])
    failed += differences(optimization_identity(args.baseline), optimization_identity(args.candidate), "implementation")
    failed += differences(verification_values(args.baseline), verification_values(args.candidate), "verifications")
    result = {"status": "passed" if not failed else "failed", "strict_equality": not failed,
              "policies": policies, "baseline": receipts[0], "candidate": receipts[1],
              "difference_count": len(failed), "first_differences": failed[:30]}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if failed:
        raise RuntimeError(f"Residency parity failed: {failed[0]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    run = commands.add_parser("capture")
    run.add_argument("--trace-dir", type=Path, required=True)
    run.add_argument("--rng-seed", type=int, required=True)
    run.add_argument("--entrypoint", type=Path, default=ROOT / "scripts/run_esmfold2_dual_target.py")
    run.add_argument("search_args", nargs=argparse.REMAINDER)
    check = commands.add_parser("compare")
    check.add_argument("--baseline", type=Path, required=True)
    check.add_argument("--candidate", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (capture if args.mode == "capture" else compare)(args)


if __name__ == "__main__":
    main()
