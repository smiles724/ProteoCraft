#!/usr/bin/env python3
"""Calculate chain-aware Rosetta pass rates from structure manifests.

The thresholds follow the ProteoCraft paper:

Low:
  contact >= 3 and clash == 0
Medium:
  Low + interface_dG <= -12 + sc >= 0.54 + packstat >= 0.60
      + interface_hbonds >= 2
High:
  Medium + interface_dG <= -28 + sc >= 0.56 + packstat >= 0.62
      + interface_hbonds >= 5

Unlike the original helper in run_structure_scoring.py, this script never
guesses binder by shortest chain. It always uses binder_chain/target_chains
from the input manifest.
"""

from __future__ import annotations

import argparse
import csv
import math
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CONTACT_CUTOFF = 4.0
CLASH_CUTOFF = 2.5


@dataclass(frozen=True)
class Atom:
    chain: str
    residue_id: str
    atom_name: str
    element: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class ContactClash:
    contact: int
    clash: int


def split_chains(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        parts = value.replace(",", ";").split(";")
    else:
        parts = list(value)
    return [part.strip() for part in parts if part and part.strip()]


def is_hydrogen(atom: Atom) -> bool:
    element = atom.element.strip().upper()
    return element == "H" or atom.atom_name.strip().upper().startswith("H")


def infer_element(atom_name: str, explicit: str = "") -> str:
    explicit = explicit.strip()
    if explicit and explicit not in {".", "?"}:
        return explicit.upper()
    letters = "".join(ch for ch in atom_name.strip() if ch.isalpha())
    if not letters:
        return ""
    if len(letters) >= 2 and letters[:2].upper() in {"CL", "BR", "NA", "MG", "ZN", "FE", "CA"}:
        return letters[:2].upper()
    return letters[0].upper()


def parse_structure(path: Path) -> list[Atom]:
    suffix = path.suffix.lower()
    if suffix in {".pdb", ".ent"}:
        return parse_pdb(path)
    if suffix in {".cif", ".mmcif"}:
        return parse_mmcif(path)
    raise ValueError(f"Unsupported structure format: {path}")


def parse_pdb(path: Path) -> list[Atom]:
    atoms: list[Atom] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            chain = line[21].strip()
            resseq = line[22:26].strip()
            icode = line[26].strip()
            resname = line[17:20].strip()
            element = infer_element(atom_name, line[76:78] if len(line) >= 78 else "")
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            residue_id = f"{chain}:{resseq}{icode}:{resname}"
            atoms.append(Atom(chain, residue_id, atom_name, element, x, y, z))
    return atoms


def parse_mmcif(path: Path) -> list[Atom]:
    atoms: list[Atom] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        lines = list(handle)

    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if line != "loop_":
            idx += 1
            continue

        idx += 1
        columns: list[str] = []
        while idx < len(lines) and lines[idx].strip().startswith("_atom_site."):
            columns.append(lines[idx].strip().split(".", 1)[1])
            idx += 1
        if not columns:
            continue

        while idx < len(lines):
            row = lines[idx].strip()
            if not row or row.startswith("#") or row == "loop_" or row.startswith("_"):
                break
            if row.startswith(("ATOM", "HETATM")):
                values = shlex.split(row, comments=False, posix=True)
                if len(values) >= len(columns):
                    record = dict(zip(columns, values))
                    atom = atom_from_mmcif_record(record)
                    if atom is not None:
                        atoms.append(atom)
            idx += 1
    return atoms


def atom_from_mmcif_record(record: dict[str, str]) -> Atom | None:
    model = record.get("pdbx_PDB_model_num")
    if model not in {None, ".", "?", "1"}:
        return None
    chain = clean_cif_value(record.get("auth_asym_id") or record.get("label_asym_id"))
    atom_name = clean_cif_value(record.get("auth_atom_id") or record.get("label_atom_id"))
    resseq = clean_cif_value(record.get("auth_seq_id") or record.get("label_seq_id"))
    resname = clean_cif_value(record.get("auth_comp_id") or record.get("label_comp_id"))
    icode = clean_cif_value(record.get("pdbx_PDB_ins_code"))
    element = infer_element(atom_name, clean_cif_value(record.get("type_symbol")))
    try:
        x = float(record["Cartn_x"])
        y = float(record["Cartn_y"])
        z = float(record["Cartn_z"])
    except (KeyError, ValueError):
        return None
    residue_id = f"{chain}:{resseq}{'' if icode in {'', '.', '?'} else icode}:{resname}"
    return Atom(chain, residue_id, atom_name, element, x, y, z)


def clean_cif_value(value: str | None) -> str:
    if value is None or value in {".", "?"}:
        return ""
    return value


def compute_contact_clash(
    structure_path: str | Path,
    binder_chain: str,
    target_chains: Iterable[str],
    contact_cutoff: float = CONTACT_CUTOFF,
    clash_cutoff: float = CLASH_CUTOFF,
) -> ContactClash:
    atoms = parse_structure(Path(structure_path))
    target_chain_set = set(split_chains(target_chains))
    binder_chain = binder_chain.strip()

    binder_atoms = [atom for atom in atoms if atom.chain == binder_chain and not is_hydrogen(atom)]
    target_atoms = [atom for atom in atoms if atom.chain in target_chain_set and not is_hydrogen(atom)]
    if not binder_atoms:
        raise ValueError(f"No binder atoms found for chain {binder_chain} in {structure_path}")
    if not target_atoms:
        raise ValueError(f"No target atoms found for chains {sorted(target_chain_set)} in {structure_path}")

    contact = count_contacting_binder_residues(binder_atoms, target_atoms, contact_cutoff)
    clash = count_ca_clashes(binder_atoms, target_atoms, clash_cutoff)
    return ContactClash(contact=contact, clash=clash)


def count_contacting_binder_residues(
    binder_atoms: list[Atom],
    target_atoms: list[Atom],
    cutoff: float,
) -> int:
    cutoff2 = cutoff * cutoff
    grid = build_grid(target_atoms, cutoff)
    contacting: set[str] = set()
    for atom in binder_atoms:
        if atom.residue_id in contacting:
            continue
        for target_atom in nearby_atoms(atom, grid, cutoff):
            if squared_distance(atom, target_atom) <= cutoff2:
                contacting.add(atom.residue_id)
                break
    return len(contacting)


def count_ca_clashes(binder_atoms: list[Atom], target_atoms: list[Atom], cutoff: float) -> int:
    cutoff2 = cutoff * cutoff
    binder_ca = [atom for atom in binder_atoms if atom.atom_name.strip().upper() == "CA"]
    target_ca = [atom for atom in target_atoms if atom.atom_name.strip().upper() == "CA"]
    grid = build_grid(target_ca, cutoff)
    clashes = 0
    for atom in binder_ca:
        for target_atom in nearby_atoms(atom, grid, cutoff):
            if squared_distance(atom, target_atom) < cutoff2:
                clashes += 1
    return clashes


def build_grid(atoms: list[Atom], cell_size: float) -> dict[tuple[int, int, int], list[Atom]]:
    grid: dict[tuple[int, int, int], list[Atom]] = {}
    for atom in atoms:
        grid.setdefault(grid_key(atom, cell_size), []).append(atom)
    return grid


def grid_key(atom: Atom, cell_size: float) -> tuple[int, int, int]:
    return (
        math.floor(atom.x / cell_size),
        math.floor(atom.y / cell_size),
        math.floor(atom.z / cell_size),
    )


def nearby_atoms(
    atom: Atom,
    grid: dict[tuple[int, int, int], list[Atom]],
    cell_size: float,
) -> Iterable[Atom]:
    cx, cy, cz = grid_key(atom, cell_size)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                yield from grid.get((cx + dx, cy + dy, cz + dz), [])


def squared_distance(left: Atom, right: Atom) -> float:
    return (left.x - right.x) ** 2 + (left.y - right.y) ** 2 + (left.z - right.z) ** 2


def to_float(value: object) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def classify_pass(row: dict[str, object]) -> tuple[str, str, str]:
    if str(row.get("status", "")).lower() not in {"ok", ""}:
        return ("false", "false", "false")

    contact = to_float(row.get("contact"))
    clash = to_float(row.get("clash"))
    interface_dg = to_float(row.get("interface_dG"))
    sc = to_float(row.get("sc"))
    packstat = to_float(row.get("packstat"))
    hbonds = to_float(row.get("interface_hbonds"))

    low = contact >= 3 and clash == 0
    medium = low and interface_dg <= -12 and sc >= 0.54 and packstat >= 0.60 and hbonds >= 2
    high = medium and interface_dg <= -28 and sc >= 0.56 and packstat >= 0.62 and hbonds >= 5
    return (bool_str(low), bool_str(medium), bool_str(high))


def bool_str(value: bool) -> str:
    return "true" if value else "false"


def pass_level(row: dict[str, object]) -> str:
    if row.get("pass_high") == "true":
        return "high"
    if row.get("pass_medium") == "true":
        return "medium"
    if row.get("pass_low") == "true":
        return "low"
    return "fail"


def read_table(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def build_candidate_rows(
    manifest_path: str | Path,
    rosetta_csv: str | Path,
    structure_source: str,
) -> list[dict[str, object]]:
    manifest_rows = read_table(Path(manifest_path))
    rosetta_rows = read_table(Path(rosetta_csv))
    rosetta_by_filename = {row["filename"]: row for row in rosetta_rows}

    output_rows: list[dict[str, object]] = []
    for manifest in manifest_rows:
        structure_id = manifest.get("structure_id") or manifest.get("candidate_id")
        if not structure_id:
            raise ValueError(f"Manifest row missing structure_id/candidate_id: {manifest}")
        rosetta = rosetta_by_filename.get(structure_id)
        if rosetta is None:
            raise ValueError(f"No Rosetta row found for structure_id={structure_id}")

        cc = compute_contact_clash(
            manifest["structure_path"],
            binder_chain=manifest["binder_chain"],
            target_chains=split_chains(manifest["target_chains"]),
        )
        row: dict[str, object] = {
            "candidate_id": manifest.get("candidate_id", structure_id),
            "structure_id": structure_id,
            "structure_source": structure_source,
            "task_type": manifest.get("task_type", ""),
            "case_id": manifest.get("case_id", ""),
            "method": manifest.get("method", ""),
            "selected_by": manifest.get("selected_by", ""),
            "structure_path": manifest["structure_path"],
            "binder_chain": manifest["binder_chain"],
            "target_chains": manifest["target_chains"],
            "contact": cc.contact,
            "clash": cc.clash,
            "status": rosetta.get("status", ""),
            "interface_dG": rosetta.get("interface_dG", ""),
            "interface_dSASA": rosetta.get("interface_dSASA", ""),
            "sc": rosetta.get("sc", ""),
            "dG_dSASA_ratio": rosetta.get("dG_dSASA_ratio", ""),
            "interface_hbonds": rosetta.get("interface_hbonds", ""),
            "packstat": rosetta.get("packstat", ""),
            "error": rosetta.get("error", ""),
            "score_path": rosetta.get("score_path", ""),
        }
        row["pass_low"], row["pass_medium"], row["pass_high"] = classify_pass(row)
        row["pass_level"] = pass_level(row)
        output_rows.append(row)
    return output_rows


def summarize_rows(
    rows: list[dict[str, object]],
    group_columns: list[str],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(str(row.get(column, "")) for column in group_columns)
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        total = len(group)
        low = sum(row.get("pass_low") == "true" for row in group)
        medium = sum(row.get("pass_medium") == "true" for row in group)
        high = sum(row.get("pass_high") == "true" for row in group)
        out = {column: value for column, value in zip(group_columns, key)}
        out.update(
            {
                "n": total,
                "low_pass": low,
                "medium_pass": medium,
                "high_pass": high,
                "low_pass_rate": f"{low / total:.4f}" if total else "nan",
                "medium_pass_rate": f"{medium / total:.4f}" if total else "nan",
                "high_pass_rate": f"{high / total:.4f}" if total else "nan",
            }
        )
        summary.append(out)
    return summary


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, help="CSV/TSV manifest with structure_path, binder_chain, target_chains")
    parser.add_argument("--rosetta-csv", action="append", required=True, help="Rosetta ddg_scores.csv matching each manifest")
    parser.add_argument("--structure-source", action="append", required=True, help="Label for each manifest/Rosetta pair")
    parser.add_argument("--candidate-output", required=True, help="Output candidate-level TSV")
    parser.add_argument("--summary-output", required=True, help="Output grouped summary TSV")
    parser.add_argument(
        "--group-columns",
        default="structure_source,task_type,method",
        help="Comma-separated columns for summary grouping",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (len(args.manifest) == len(args.rosetta_csv) == len(args.structure_source)):
        raise SystemExit("--manifest, --rosetta-csv, and --structure-source must have the same count")

    rows: list[dict[str, object]] = []
    for manifest_path, rosetta_csv, structure_source in zip(args.manifest, args.rosetta_csv, args.structure_source):
        rows.extend(build_candidate_rows(manifest_path, rosetta_csv, structure_source))

    group_columns = [column.strip() for column in args.group_columns.split(",") if column.strip()]
    summary = summarize_rows(rows, group_columns)
    write_tsv(Path(args.candidate_output), rows)
    write_tsv(Path(args.summary_output), summary)
    print(f"Wrote candidate rows: {len(rows)} -> {args.candidate_output}")
    print(f"Wrote summary rows: {len(summary)} -> {args.summary_output}")


if __name__ == "__main__":
    main()
