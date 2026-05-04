#!/usr/bin/env python3
"""Prepare PXDesign and RF3 inputs for the BC v1 benchmark cases."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from Bio.PDB import MMCIFParser, PDBIO, Select


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
PXD_ROOT = Path("/data/huangyanjie/binder_design/PXDesign")
RF3_ROOT = Path("/data/huangyanjie/binder_design/rf3_binder_design")
BCV1_YAML_DIR = BENCHMARK_ROOT / "yaml" / "bcv1"
OUT_ROOT = BENCHMARK_ROOT / "outputs" / "8_bcv1" / "pxd_rf3"

CASE_ORDER = ("BetV1", "DerF21", "DerF7", "HER2", "IFNAR2", "PD-L1", "PD1", "SpCas9")
DEFAULT_LENGTHS = (80, 100, 120)
DEFAULT_SAMPLES_PER_LENGTH = 5
SKIP_HOTSPOTS = {
    ("DerF7", "B201"),
}


@dataclass(frozen=True)
class Bcv1Case:
    entry_id: str
    fasta_path: Path
    cif_path: Path
    target_chain_id: str


@dataclass(frozen=True)
class Condition:
    case: Bcv1Case
    name: str
    hotspot_residues: tuple[int, ...]
    source_hotspot_residues: str
    source_yaml: Path


class TargetSelect(Select):
    def __init__(self, chain_id: str):
        self.chain_id = chain_id

    def accept_model(self, model):
        return int(model.id == 0)

    def accept_chain(self, chain):
        return int(chain.id == self.chain_id)

    def accept_residue(self, residue):
        return int(residue.id[0] == " ")

    def accept_atom(self, atom):
        element = (atom.element or "").upper()
        atom_name = atom.get_name().strip().upper()
        return int(element != "H" and not atom_name.startswith("H"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUT_ROOT,
        help="Output root containing bcv1_configs/pxd_bcv1/rf3_bcv1.",
    )
    parser.add_argument(
        "--yaml-dir",
        type=Path,
        default=BCV1_YAML_DIR,
        help="Directory containing *_custom_loss_hotspot*.yaml files.",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_LENGTHS),
        help="Binder lengths to generate.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES_PER_LENGTH,
        help="Samples per method/condition/length.",
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


def case_slug(entry_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", entry_id.lower()).strip("_")


def parse_fasta_chain(fasta_path: Path) -> str:
    header = fasta_path.read_text(encoding="utf-8").splitlines()[0]
    match = re.search(r"(?:^|[|])chain=([^|]+)", header)
    return match.group(1) if match else "A"


def parse_hotspot_residues(raw: str) -> tuple[int, ...]:
    residues: list[int] = []
    for token in re.split(r"\s*,\s*", raw.strip()):
        if not token:
            continue
        match = re.fullmatch(r"([A-Za-z]+)(\d+)(?:-(\d+))?", token)
        if not match:
            raise ValueError(f"Unsupported hotspot residue token: {token!r}")
        start = int(match.group(2))
        end = int(match.group(3) or start)
        if end < start:
            raise ValueError(f"Hotspot range has end before start: {token!r}")
        residues.extend(range(start, end + 1))

    seen = set()
    ordered_unique: list[int] = []
    for resid in residues:
        if resid in seen:
            continue
        ordered_unique.append(resid)
        seen.add(resid)
    if not ordered_unique:
        raise ValueError("No hotspot residues parsed")
    return tuple(ordered_unique)


def hotspot_name(raw: str) -> str:
    parts = [token for token in re.split(r"\s*,\s*", raw.strip()) if token]
    return re.sub(r"[^A-Za-z0-9]+", "_", "_".join(parts)).strip("_")


def discover_conditions(yaml_dir: Path) -> list[Condition]:
    paths = sorted(yaml_dir.glob("*_custom_loss_hotspot*.yaml"))
    if not paths:
        raise ValueError(f"No hotspot YAML files found in {yaml_dir}")

    cases: dict[str, Bcv1Case] = {}
    conditions: list[Condition] = []
    seen_conditions = set()
    for path in paths:
        data = load_yaml(path)
        entry_ids = data.get("entry_ids") or []
        if not isinstance(entry_ids, list) or len(entry_ids) != 1:
            raise ValueError(f"{path} must contain exactly one entry_id")
        entry_id = str(entry_ids[0])
        if entry_id not in CASE_ORDER:
            raise ValueError(f"{path} has unexpected entry_id={entry_id!r}")

        raw_hotspots = str(data.get("target_hotspot_residues") or "").strip()
        if not raw_hotspots:
            raise ValueError(f"{path} missing target_hotspot_residues")
        if (entry_id, raw_hotspots) in SKIP_HOTSPOTS:
            continue

        fasta_dir = Path(data["target_fasta_dir"])
        template_dir = Path(data["target_template_dir"])
        fasta_path = fasta_dir / f"{entry_id}.fasta"
        cif_path = template_dir / f"{entry_id}.cif"
        missing = [str(p) for p in (fasta_path, cif_path) if not p.exists()]
        if missing:
            raise ValueError(f"{path} references missing input files: {missing}")

        target_chain_id = parse_fasta_chain(fasta_path)
        candidate = Bcv1Case(
            entry_id=entry_id,
            fasta_path=fasta_path.resolve(),
            cif_path=cif_path.resolve(),
            target_chain_id=target_chain_id,
        )
        existing = cases.get(entry_id)
        if existing is not None and existing != candidate:
            raise ValueError(f"Conflicting case metadata for {entry_id}: {existing} vs {candidate}")
        cases[entry_id] = candidate

        name = hotspot_name(raw_hotspots)
        key = (entry_id, name)
        if key in seen_conditions:
            raise ValueError(f"Duplicate condition for {entry_id} {name}")
        seen_conditions.add(key)
        conditions.append(
            Condition(
                case=candidate,
                name=name,
                hotspot_residues=parse_hotspot_residues(raw_hotspots),
                source_hotspot_residues=raw_hotspots,
                source_yaml=path.resolve(),
            )
        )

    missing_cases = [case for case in CASE_ORDER if case not in cases]
    if missing_cases:
        raise ValueError(f"Missing BC v1 cases in hotspot YAMLs: {missing_cases}")

    case_rank = {case: idx for idx, case in enumerate(CASE_ORDER)}
    conditions.sort(key=lambda c: (case_rank[c.case.entry_id], c.name))
    return conditions


def export_target_pdb(case: Bcv1Case, out_path: Path, *, force: bool) -> None:
    if out_path.exists() and not force:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parser = MMCIFParser(auth_chains=True, auth_residues=True, QUIET=True)
    structure = parser.get_structure(case.entry_id, str(case.cif_path))
    if case.target_chain_id not in {chain.id for chain in structure.get_chains()}:
        raise ValueError(
            f"{case.cif_path} does not contain target chain {case.target_chain_id!r}"
        )
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path), select=TargetSelect(case.target_chain_id))


def residue_ids(pdb_path: Path, chain_id: str) -> list[int]:
    ids: list[int] = []
    seen = set()
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
                continue
            if line[21].strip() != chain_id:
                continue
            if line[26].strip():
                continue
            resid = int(line[22:26])
            if resid in seen:
                continue
            ids.append(resid)
            seen.add(resid)
    if not ids:
        raise ValueError(f"No CA residues found for chain {chain_id} in {pdb_path}")
    return ids


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


def validate_hotspots(condition: Condition, ids_by_case: dict[str, list[int]]) -> None:
    available = set(ids_by_case[condition.case.entry_id])
    missing = [resid for resid in condition.hotspot_residues if resid not in available]
    if missing:
        raise ValueError(
            f"{condition.case.entry_id} {condition.name} missing hotspot residues "
            f"on chain {condition.case.target_chain_id}: {missing}"
        )


def pxd_yaml(condition: Condition, length: int, target_pdb: Path) -> dict:
    return {
        "task_name": f"{case_slug(condition.case.entry_id)}_{condition.name}_len_{length}",
        "target": {
            "file": str(target_pdb),
            "chains": {
                condition.case.target_chain_id: {
                    "hotspots": list(condition.hotspot_residues),
                }
            },
        },
        "binder_length": length,
    }


def rf3_json(condition: Condition, length: int, target_pdb: Path, target_ranges: list[str]) -> dict:
    task = {
        "dialect": 2,
        "input": str(target_pdb),
        "contig": f"{length}-{length},/0,{','.join(target_ranges)}",
        "is_non_loopy": True,
        "infer_ori_strategy": "hotspots",
        "select_hotspots": {
            f"{condition.case.target_chain_id}{resid}": "CA"
            for resid in condition.hotspot_residues
        },
    }
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


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    config_root = output_root / "bcv1_configs"
    pxd_root = output_root / "pxd_bcv1"
    rf3_root = output_root / "rf3_bcv1"
    lengths = tuple(args.lengths)
    samples = int(args.samples)
    if not lengths:
        raise ValueError("At least one length is required")
    if samples < 1:
        raise ValueError("--samples must be positive")

    conditions = discover_conditions(args.yaml_dir)
    cases = {condition.case.entry_id: condition.case for condition in conditions}
    manifest_rows: list[dict[str, str | int]] = []
    target_pdbs: dict[str, Path] = {}
    ids_by_case: dict[str, list[int]] = {}
    ranges_by_case: dict[str, list[str]] = {}

    for entry_id in CASE_ORDER:
        case = cases[entry_id]
        target_pdb = config_root / "inputs" / f"{case_slug(entry_id)}_target_{case.target_chain_id}.pdb"
        export_target_pdb(case, target_pdb, force=args.force)
        target_pdbs[entry_id] = target_pdb
        ids = residue_ids(target_pdb, case.target_chain_id)
        ids_by_case[entry_id] = ids
        ranges_by_case[entry_id] = ranges_from_ids(ids, case.target_chain_id)

    for condition in conditions:
        validate_hotspots(condition, ids_by_case)
        target_pdb = target_pdbs[condition.case.entry_id]
        target_ranges = ranges_by_case[condition.case.entry_id]
        for length in lengths:
            rel_base = Path(case_slug(condition.case.entry_id)) / condition.name / f"len_{length}"
            pxd_cfg = config_root / "pxd" / rel_base / "input.yaml"
            rf3_cfg = config_root / "rf3" / rel_base / "input.json"
            pxd_out = pxd_root / rel_base
            rf3_out = rf3_root / rel_base

            write_yaml(pxd_cfg, pxd_yaml(condition, length, target_pdb), args.force)
            write_json(rf3_cfg, rf3_json(condition, length, target_pdb, target_ranges), args.force)

            common = {
                "case": condition.case.entry_id,
                "condition": condition.name,
                "source_hotspot_residues": condition.source_hotspot_residues,
                "target_chain_id": condition.case.target_chain_id,
                "target_hotspot_residues": ",".join(
                    f"{condition.case.target_chain_id}{resid}"
                    for resid in condition.hotspot_residues
                ),
                "length": length,
                "samples": samples,
                "source_yaml": str(condition.source_yaml),
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
                        f"--N_sample {samples} --dtype bf16"
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
                        f"n_batches={samples} diffusion_batch_size=1"
                    ),
                }
            )

    manifest = output_root / "bcv1_design_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "method",
            "case",
            "condition",
            "source_hotspot_residues",
            "target_chain_id",
            "target_hotspot_residues",
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
        "lengths": list(lengths),
        "samples_per_length": samples,
        "conditions": len(conditions),
        "tasks_per_method": len(conditions) * len(lengths),
        "binders_per_method": len(conditions) * len(lengths) * samples,
        "manifest": str(manifest),
        "pxdesign_output_root": str(pxd_root),
        "rf3_output_root": str(rf3_root),
        "config_root": str(config_root),
    }
    write_json(output_root / "bcv1_design_summary.json", summary, force=True)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
