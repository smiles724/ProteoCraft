#!/usr/bin/env python3
"""Prepare PXDesign and RF3 inputs for the five TCR benchmark cases."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BENCHMARK_ROOT.parent
PXD_ROOT = Path("/data/huangyanjie/binder_design/PXDesign")
RF3_ROOT = Path("/data/huangyanjie/binder_design/rf3_binder_design")
TCR_INPUT_DIR = BENCHMARK_ROOT / "input" / "tcr_case"
YAML_DIR = BENCHMARK_ROOT / "yaml"
TCR_YAML_DIR = YAML_DIR / "tcr"
OUT_ROOT = BENCHMARK_ROOT / "outputs"

LENGTHS = (60, 80, 100, 120, 140)
SAMPLES_PER_LENGTH = 10
CASE_ORDER = ("1QRN", "3QDJ", "5BS0", "6BJ8", "9C3E")

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


@dataclass(frozen=True)
class TcrCase:
    pdb_id: str
    pdb_path: Path
    alpha_chain: str
    beta_chain: str


@dataclass(frozen=True)
class Condition:
    case: TcrCase
    name: str
    hotspot_chain_role: str | None
    hotspot_chain_id: str | None
    hotspot_start: int | None
    hotspot_end: int | None
    source_yaml: Path | None

    @property
    def is_hotspot(self) -> bool:
        return self.hotspot_chain_role is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUT_ROOT,
        help="Benchmark output root containing pxd_tcr/rf3_tcr.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite generated configs even if they already exist.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def discover_tcr_yaml_paths() -> list[Path]:
    search_root = TCR_YAML_DIR if TCR_YAML_DIR.is_dir() else YAML_DIR
    paths = sorted(search_root.rglob("*_custom_loss_hotspot.yaml"))
    paths.extend(sorted(search_root.rglob("1qrn_*_template*.yaml")))
    return paths


def discover_hotspot_cases() -> dict[str, TcrCase]:
    cases: dict[str, TcrCase] = {}
    for path in discover_tcr_yaml_paths():
        data = load_yaml(path)
        pdb_id = str(data.get("pdb_id", "")).upper()
        if pdb_id not in CASE_ORDER:
            continue
        required = ("pdb_path", "alpha_chain", "beta_chain")
        missing = [key for key in required if not data.get(key)]
        if missing:
            raise ValueError(f"{path} missing required fields: {missing}")
        candidate = TcrCase(
            pdb_id=pdb_id,
            pdb_path=Path(data["pdb_path"]).resolve(),
            alpha_chain=str(data["alpha_chain"]),
            beta_chain=str(data["beta_chain"]),
        )
        existing = cases.get(pdb_id)
        if existing is not None and existing != candidate:
            raise ValueError(
                f"Conflicting chain mapping for {pdb_id}: {existing} vs {candidate} from {path}"
            )
        cases[pdb_id] = candidate
    missing_cases = [case for case in CASE_ORDER if case not in cases]
    if missing_cases:
        raise ValueError(f"Missing hotspot YAML-derived cases: {missing_cases}")
    return {case: cases[case] for case in CASE_ORDER}


def discover_conditions(cases: dict[str, TcrCase]) -> list[Condition]:
    conditions: list[Condition] = []
    seen = set()
    for path in discover_tcr_yaml_paths():
        data = load_yaml(path)
        pdb_id = str(data.get("pdb_id", "")).upper()
        if pdb_id not in CASE_ORDER:
            continue
        hotspot_chain = data.get("hotspot_chain")
        hotspot_start = data.get("hotspot_start")
        hotspot_end = data.get("hotspot_end")
        if not hotspot_chain or hotspot_start is None or hotspot_end is None:
            continue
        case = cases[pdb_id]
        if hotspot_chain == "alpha":
            chain_id = case.alpha_chain
        elif hotspot_chain == "beta":
            chain_id = case.beta_chain
        else:
            raise ValueError(f"{path} has unsupported hotspot_chain={hotspot_chain!r}")
        name = f"{hotspot_chain}_{int(hotspot_start)}_{int(hotspot_end)}"
        key = (pdb_id, name)
        if key in seen:
            continue
        seen.add(key)
        conditions.append(
            Condition(
                case=case,
                name=name,
                hotspot_chain_role=str(hotspot_chain),
                hotspot_chain_id=chain_id,
                hotspot_start=int(hotspot_start),
                hotspot_end=int(hotspot_end),
                source_yaml=path.resolve(),
            )
        )

    for case in cases.values():
        conditions.append(
            Condition(
                case=case,
                name="no_hotspot",
                hotspot_chain_role=None,
                hotspot_chain_id=None,
                hotspot_start=None,
                hotspot_end=None,
                source_yaml=None,
            )
        )

    case_rank = {case: idx for idx, case in enumerate(CASE_ORDER)}
    role_rank = {"alpha": 0, "beta": 1, None: 2}
    conditions.sort(
        key=lambda c: (
            case_rank[c.case.pdb_id],
            role_rank[c.hotspot_chain_role],
            c.hotspot_start or 10**9,
            c.hotspot_end or 10**9,
        )
    )
    return conditions


def iter_pdb_atoms(pdb_path: Path) -> Iterable[str]:
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "TER   ", "END")):
                yield line


def extract_tcr_pdb(case: TcrCase, out_path: Path, *, skip_insertion_codes: bool) -> None:
    keep = {case.alpha_chain, case.beta_chain}
    lines: list[str] = []
    last_kept_chain = None
    for line in iter_pdb_atoms(case.pdb_path):
        record = line[:6]
        if record == "ATOM  " and line[21] in keep:
            if skip_insertion_codes and line[26].strip():
                continue
            lines.append(line)
            last_kept_chain = line[21]
        elif record == "TER   " and last_kept_chain in keep:
            lines.append(line)
            last_kept_chain = None
    if not lines:
        raise ValueError(f"No TCR atoms extracted for {case.pdb_id} from {case.pdb_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.writelines(lines)
        handle.write("END\n")


def residue_ids(pdb_path: Path, chain_id: str) -> list[int]:
    ids: list[int] = []
    seen = set()
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
                continue
            if line[21].strip() != chain_id:
                continue
            icode = line[26].strip()
            if icode:
                continue
            resid = int(line[22:26])
            if resid not in seen:
                ids.append(resid)
                seen.add(resid)
    if not ids:
        raise ValueError(f"No CA residues found for chain {chain_id} in {pdb_path}")
    return ids


def residue_names(pdb_path: Path, chain_id: str) -> dict[int, str]:
    names: dict[int, str] = {}
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM  ") or line[21].strip() != chain_id:
                continue
            resid = int(line[22:26])
            names.setdefault(resid, line[17:20].strip())
    return names


def ranges_from_ids(ids: list[int], chain_id: str) -> list[str]:
    ranges: list[str] = []
    start = prev = ids[0]
    for resid in ids[1:]:
        if resid == prev + 1:
            prev = resid
            continue
        ranges.append(f"{chain_id}{start}" if start == prev else f"{chain_id}{start}-{prev}")
        start = prev = resid
    ranges.append(f"{chain_id}{start}" if start == prev else f"{chain_id}{start}-{prev}")
    return ranges


def hotspot_residues(condition: Condition) -> list[int]:
    if not condition.is_hotspot:
        return []
    assert condition.hotspot_start is not None and condition.hotspot_end is not None
    return list(range(condition.hotspot_start, condition.hotspot_end + 1))


def validate_hotspot(condition: Condition, ids_by_chain: dict[str, list[int]]) -> None:
    if not condition.is_hotspot:
        return
    assert condition.hotspot_chain_id is not None
    available = set(ids_by_chain[condition.hotspot_chain_id])
    missing = [resid for resid in hotspot_residues(condition) if resid not in available]
    if missing:
        raise ValueError(
            f"{condition.case.pdb_id} {condition.name} missing hotspot residues "
            f"on chain {condition.hotspot_chain_id}: {missing}"
        )


def pxd_yaml(condition: Condition, length: int, target_pdb: Path, out_dir: Path) -> dict:
    chains: dict[str, dict] = {
        condition.case.alpha_chain: {},
        condition.case.beta_chain: {},
    }
    if condition.is_hotspot:
        assert condition.hotspot_chain_id is not None
        chains[condition.hotspot_chain_id]["hotspots"] = hotspot_residues(condition)
    return {
        "task_name": f"{condition.case.pdb_id.lower()}_{condition.name}_len_{length}",
        "target": {"file": str(target_pdb), "chains": chains},
        "binder_length": length,
    }


def rf3_json(
    condition: Condition,
    length: int,
    target_pdb: Path,
    alpha_ranges: list[str],
    beta_ranges: list[str],
) -> dict:
    task: dict = {
        "dialect": 2,
        "input": str(target_pdb),
        "contig": f"{length}-{length},/0,{','.join(alpha_ranges)},/0,{','.join(beta_ranges)}",
        "is_non_loopy": True,
    }
    if condition.is_hotspot:
        assert condition.hotspot_chain_id is not None
        task["infer_ori_strategy"] = "hotspots"
        task["select_hotspots"] = {
            f"{condition.hotspot_chain_id}{resid}": "CA"
            for resid in hotspot_residues(condition)
        }
    else:
        task["infer_ori_strategy"] = "com"
    return {condition.name: task}


def write_yaml(path: Path, data: dict, force: bool) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def write_json(path: Path, data: dict, force: bool) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def count_expected_conditions(conditions: list[Condition]) -> None:
    counts = {case: 0 for case in CASE_ORDER}
    for condition in conditions:
        counts[condition.case.pdb_id] += 1
    expected = {"1QRN": 9, "3QDJ": 8, "5BS0": 9, "6BJ8": 8, "9C3E": 8}
    if counts != expected:
        raise ValueError(f"Unexpected condition counts: got {counts}, expected {expected}")


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    pxd_root = output_root / "pxd_tcr"
    rf3_root = output_root / "rf3_tcr"
    config_root = output_root / "tcr_configs"

    cases = discover_hotspot_cases()
    conditions = discover_conditions(cases)
    count_expected_conditions(conditions)

    manifest_rows: list[dict[str, str | int]] = []
    task_commands: list[tuple[str, str]] = []
    case_inputs_pxd: dict[str, Path] = {}
    case_inputs_rf3: dict[str, Path] = {}
    case_ranges: dict[str, tuple[list[str], list[str], dict[str, list[int]]]] = {}

    for case in cases.values():
        pxd_target_pdb = (
            config_root
            / "inputs"
            / f"{case.pdb_id}_tcr_{case.alpha_chain}_{case.beta_chain}_pxd.pdb"
        )
        rf3_target_pdb = (
            config_root
            / "inputs"
            / f"{case.pdb_id}_tcr_{case.alpha_chain}_{case.beta_chain}_rf3.pdb"
        )
        extract_tcr_pdb(case, pxd_target_pdb, skip_insertion_codes=True)
        extract_tcr_pdb(case, rf3_target_pdb, skip_insertion_codes=True)
        case_inputs_pxd[case.pdb_id] = pxd_target_pdb
        case_inputs_rf3[case.pdb_id] = rf3_target_pdb
        alpha_ids = residue_ids(rf3_target_pdb, case.alpha_chain)
        beta_ids = residue_ids(rf3_target_pdb, case.beta_chain)
        case_ranges[case.pdb_id] = (
            ranges_from_ids(alpha_ids, case.alpha_chain),
            ranges_from_ids(beta_ids, case.beta_chain),
            {case.alpha_chain: alpha_ids, case.beta_chain: beta_ids},
        )

    for condition in conditions:
        alpha_ranges, beta_ranges, ids_by_chain = case_ranges[condition.case.pdb_id]
        validate_hotspot(condition, ids_by_chain)
        pxd_target_pdb = case_inputs_pxd[condition.case.pdb_id]
        rf3_target_pdb = case_inputs_rf3[condition.case.pdb_id]
        for length in LENGTHS:
            rel_base = Path(condition.case.pdb_id.lower()) / condition.name / f"len_{length}"
            pxd_cfg = config_root / "pxd" / rel_base / "input.yaml"
            rf3_cfg = config_root / "rf3" / rel_base / "input.json"
            pxd_out = pxd_root / rel_base
            rf3_out = rf3_root / rel_base

            write_yaml(pxd_cfg, pxd_yaml(condition, length, pxd_target_pdb, pxd_out), args.force)
            write_json(
                rf3_cfg,
                rf3_json(condition, length, rf3_target_pdb, alpha_ranges, beta_ranges),
                args.force,
            )

            common = {
                "case": condition.case.pdb_id,
                "condition": condition.name,
                "is_hotspot": int(condition.is_hotspot),
                "hotspot_chain_role": condition.hotspot_chain_role or "",
                "hotspot_chain_id": condition.hotspot_chain_id or "",
                "hotspot_start": condition.hotspot_start or "",
                "hotspot_end": condition.hotspot_end or "",
                "length": length,
                "samples": SAMPLES_PER_LENGTH,
                "source_yaml": str(condition.source_yaml or ""),
            }
            manifest_rows.append(
                {
                    "method": "pxdesign",
                    **common,
                    "config": str(pxd_cfg),
                    "out_dir": str(pxd_out),
                    "command": (
                        "PX_GPU_ID=${PX_GPU_ID:-0} bash "
                        f"{shell_quote(str(PXD_ROOT / 'scripts' / 'run_pxdesign_gpu.sh'))} "
                        "pxdesign infer "
                        f"-i {shell_quote(str(pxd_cfg))} "
                        f"-o {shell_quote(str(pxd_out))} "
                        f"--N_sample {SAMPLES_PER_LENGTH} --dtype bf16"
                    ),
                }
            )
            manifest_rows.append(
                {
                    "method": "rf3",
                    **common,
                    "config": str(rf3_cfg),
                    "out_dir": str(rf3_out),
                    "command": (
                        "CUDA_VISIBLE_DEVICES=${RF3_GPU_ID:-0} "
                        f"{shell_quote(str(RF3_ROOT / 'run_rf3.sh'))} "
                        f"{shell_quote(str(rf3_cfg))} "
                        f"{shell_quote(str(rf3_out))} "
                        f"n_batches={SAMPLES_PER_LENGTH} diffusion_batch_size=1"
                    ),
                }
            )

    manifest = output_root / "tcr_design_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "method",
            "case",
            "condition",
            "is_hotspot",
            "hotspot_chain_role",
            "hotspot_chain_id",
            "hotspot_start",
            "hotspot_end",
            "length",
            "samples",
            "source_yaml",
            "config",
            "out_dir",
            "command",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "cases": list(CASE_ORDER),
        "lengths": list(LENGTHS),
        "samples_per_length": SAMPLES_PER_LENGTH,
        "conditions": len(conditions),
        "tasks_per_method": len(conditions) * len(LENGTHS),
        "binders_per_method": len(conditions) * len(LENGTHS) * SAMPLES_PER_LENGTH,
        "manifest": str(manifest),
        "pxdesign_output_root": str(pxd_root),
        "rf3_output_root": str(rf3_root),
        "config_root": str(config_root),
    }
    write_json(output_root / "tcr_design_summary.json", summary, force=True)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
