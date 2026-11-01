"""Independent entry point; existing launch profiles are never modified."""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read_fasta(path):
    records, name, chunks = {}, None, []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records[name] = "".join(chunks)
            name, chunks = line[1:].strip(), []
            if not name or name in records:
                raise ValueError("FASTA identifiers must be nonempty and unique")
        else:
            if name is None:
                raise ValueError("FASTA must start with a header")
            chunks.append(line)
    if name is not None:
        records[name] = "".join(chunks)
    if not records or any(not seq for seq in records.values()):
        raise ValueError("FASTA contains no sequences or an empty record")
    return records


def main(required_design_mode=None):
    parser = argparse.ArgumentParser(description="Independent two-stage dual-target binder search")
    parser.add_argument("--config", required=True, help="JSON with runtime and search settings")
    parser.add_argument("--human-fasta", required=True, help="Human target: one monomer record, or three explicit homotrimer records")
    parser.add_argument("--mouse-fasta", required=True, help="Mouse target: one monomer record, or three explicit homotrimer records")
    starts = parser.add_mutually_exclusive_group(required=True)
    starts.add_argument("--seeds-fasta", help="Unique nonempty canonical amino-acid starting sequences")
    starts.add_argument("--stagea-results", help="Completed Stage-A directory; run only the remaining Stage B")
    parser.add_argument("--hotspots-json", help="Optional human/mouse runtime hotspot_params dictionaries")
    parser.add_argument("--output-dir", required=True, help="New run directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--optimization-root", help="Pinned uplifting source tree; also PROTEOCRAFT_UPLIFT_ROOT")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs without loading models or running search")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from bindcraft.hallucination.esmfold2_backend import ESMFold2ProposalConfig
    from bindcraft.hallucination.esmfold2_dual_target import (DualTargetConfig, run_dual_target,
        validate_sequence, validate_target_context, prepare_seed_sequences)
    payload = json.loads(Path(args.config).read_text())
    if set(payload) != {"runtime", "search"}:
        raise ValueError("Config must contain exactly runtime and search")
    runtime_values = {"proposal_optimization": "fast", "verifier_optimization": "fast",
                      "esmc_optimization": "exact", **payload["runtime"]}
    root = args.optimization_root or runtime_values.get("optimization_root") or os.environ.get("PROTEOCRAFT_UPLIFT_ROOT")
    if root:
        runtime_values["optimization_root"] = str(Path(root).expanduser().resolve())
    runtime = ESMFold2ProposalConfig(**runtime_values)
    runtime = replace(runtime, optimization_report_path=str(Path(args.output_dir).resolve() / "optimization_report.json"))
    search = DualTargetConfig(**payload["search"])
    if required_design_mode is not None and search.design_mode != required_design_mode:
        raise ValueError(f"This entry point requires design_mode={required_design_mode}")
    expected_source_stagea_steps = search.stagea_steps if search.strategy == "two_stage" else None
    if args.stagea_results:
        remaining = (search.outer_steps - search.stagea_steps
                     if search.strategy == "two_stage" else search.outer_steps)
        search = replace(search, strategy="stage_b_only", outer_steps=remaining)
    elif search.strategy == "stage_b_only":
        raise ValueError("stage_b_only requires --stagea-results")
    targets = {}
    hotspots = json.loads(Path(args.hotspots_json).read_text()) if args.hotspots_json else {}
    if set(hotspots) - {"human", "mouse"}:
        raise ValueError("Hotspots must be keyed by human/mouse")
    for side, path in (("human", args.human_fasta), ("mouse", args.mouse_fasta)):
        records = read_fasta(path)
        if runtime.target_assembly == "homotrimer":
            if len(records) != 3:
                raise ValueError(f"{side} homotrimer FASTA must contain exactly three separate records")
            target = {"sequence": "|".join(validate_sequence(seq) for seq in records.values()),
                      "hotspot_params": hotspots.get(side), "assembly": "homotrimer",
                      "chain_ids": list(records)}
        else:
            if len(records) != 1:
                raise ValueError(f"{side} target FASTA must contain exactly one record")
            target = {"sequence": validate_sequence(next(iter(records.values()))),
                      "hotspot_params": hotspots.get(side)}
        targets[side] = validate_target_context(target, runtime, side=side)
    if targets["human"]["sequence"] == targets["mouse"]["sequence"]:
        raise ValueError("Human and mouse targets must be distinct")
    source = None
    if args.stagea_results:
        from bindcraft.hallucination.esmfold2_dual_target_stageb import load_stagea_source
        source = load_stagea_source(args.stagea_results, search, targets, runtime)
        if (expected_source_stagea_steps is not None
                and source["completed_stagea_steps"] != expected_source_stagea_steps):
            raise ValueError("Source Stage-A budget does not match configured stagea_steps")
        seeds = {f"handoff_{i}": record["sequence"] for i, record in enumerate(source["roots"])}
    else:
        seeds = read_fasta(args.seeds_fasta)
    for sequence in seeds.values():
        validate_sequence(sequence, binder=True)
    if len(set(seeds.values())) != len(seeds):
        raise ValueError("Duplicate seed sequences")
    from bindcraft.hallucination.esmfold2_dual_target_constraints import validate_design
    mask = validate_design(search, runtime, seeds)
    initialized, seed_changes = prepare_seed_sequences(seeds, search, mask)
    if runtime.verifier_loss_mode != "aligned_common":
        raise ValueError("Dual target requires aligned_common verifier")
    if Path(args.output_dir).exists() and any(Path(args.output_dir).iterdir()):
        raise FileExistsError("Output directory must be new or empty")
    if args.validate_only:
        print(json.dumps({"status": "inputs_valid", "seeds": len(seeds),
                          "target_lengths": {s: sum(map(len, t["sequence"].split("|"))) for s, t in targets.items()},
                          "target_chain_counts": {s: len(t["sequence"].split("|")) for s, t in targets.items()},
                          "target_chain_ids": {s: t.get("chain_ids") for s, t in targets.items()},
                          "target_assembly": runtime.target_assembly,
                          "target_interface_policy": {"mode": runtime.target_interface_mode,
                              "contact_cutoff_angstrom": runtime.target_interface_contact_cutoff,
                              "min_residue_contacts": runtime.target_interface_min_contacts},
                          "outer_steps": search.outer_steps, "models_loaded": False,
                          "strategy": search.strategy, "design_mode": search.design_mode,
                          "mutable_positions": sum(mask) if mask is not None else None,
                          "stage_budgets": {
                              "stage_a": search.stagea_steps if search.strategy == "two_stage"
                              else 0 if source else search.outer_steps,
                              "stage_b": search.outer_steps - search.stagea_steps
                              if search.strategy == "two_stage" else search.outer_steps if source else 0},
                          "handoff_state_origin": source["state_origin"] if source else None,
                          "optimization_requested": {"proposal": runtime.proposal_optimization,
                              "verifier": runtime.verifier_optimization, "esmc": runtime.esmc_optimization},
                          "optimization_root": runtime.optimization_root,
                          "seed_cysteine_policy": search.seed_cysteine_policy,
                          "cysteines_replaced": sum(len(r["replacements"]) for r in seed_changes.values()),
                          "runtime_environment_validated": False}, indent=2))
        return
    enabled = any(getattr(runtime, field) != "off" for field in
                  ("proposal_optimization", "verifier_optimization", "esmc_optimization"))
    if enabled:
        if not runtime.optimization_root:
            raise ValueError("Acceleration requires --optimization-root or PROTEOCRAFT_UPLIFT_ROOT; no silent stock fallback")
        for relative in ("ef2inv/opt/forward/ESMFOLD2_DESIGN_FAST_INFERENCE_KIT_v1/k",
                         "esmfold2/opt/forward/fast_inference/driver", "esmc/opt", "common/opt_core"):
            if not (Path(runtime.optimization_root) / relative).is_dir():
                raise FileNotFoundError(Path(runtime.optimization_root) / relative)
    from bindcraft.hallucination.esmfold2_runtime import ESMFold2RuntimeBackend
    backend = ESMFold2RuntimeBackend(config=runtime, device=args.device,
        task_type="antibody" if search.design_mode == "masked_antibody" else "denovo")
    ranked = run_dual_target(backend=backend, targets=targets, seeds={} if source else seeds,
                             config=search, output_dir=args.output_dir,
                             stagea_results=args.stagea_results)
    print(json.dumps({"status": "completed", "best_sequence": ranked[0]["sequence"],
                      "worst_loss": ranked[0]["worst_loss"], "output_dir": args.output_dir}, indent=2))


if __name__ == "__main__":
    main()
