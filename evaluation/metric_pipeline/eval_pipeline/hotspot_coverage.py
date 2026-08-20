"""Residue-level hotspot contact coverage for predicted protein complexes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def _load_first_model(structure_path: str | Path):
    from Bio.PDB import MMCIFParser, PDBParser

    path = Path(structure_path)
    suffix = path.suffix.lower()
    parser = MMCIFParser(QUIET=True) if suffix in {".cif", ".mmcif"} else PDBParser(QUIET=True)
    structure = parser.get_structure("hotspot", str(path))
    try:
        return next(structure.get_models())
    except StopIteration as exc:
        raise ValueError(f"structure has no model: {path}") from exc


def _polymer_residues(chain) -> list[Any]:
    return [residue for residue in chain if residue.id[0] == " " and len(residue) > 0]


def _chain_residues(model, chain_id: str) -> list[Any]:
    if chain_id not in model:
        available = ",".join(chain.id for chain in model)
        raise ValueError(f"chain {chain_id!r} not found; available chains: {available}")
    residues = _polymer_residues(model[chain_id])
    if not residues:
        raise ValueError(f"chain {chain_id!r} has no polymer residues")
    return residues


def _heavy_atom_coordinates(residues: Iterable[Any]) -> np.ndarray:
    coordinates: list[np.ndarray] = []
    for residue in residues:
        for atom in residue.get_atoms():
            element = str(getattr(atom, "element", "") or "").strip().upper()
            if element == "H" or (not element and atom.name.strip().upper().startswith("H")):
                continue
            coordinates.append(np.asarray(atom.coord, dtype=np.float64))
    if not coordinates:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack(coordinates)


def _ca_coordinates(residues: Iterable[Any]) -> np.ndarray:
    coordinates = [np.asarray(residue["CA"].coord, dtype=np.float64) for residue in residues if "CA" in residue]
    if not coordinates:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack(coordinates)


def _minimum_pair_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return float("nan")
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=-1)
    return float(np.min(distances))


def _normalize_indices(values: Sequence[int], *, target_length: int) -> list[int]:
    indices: list[int] = []
    seen: set[int] = set()
    for raw_value in values:
        if isinstance(raw_value, bool):
            raise ValueError("hotspot indices must be integers, not booleans")
        index = int(raw_value)
        if float(raw_value) != float(index):
            raise ValueError(f"hotspot index must be an integer: {raw_value!r}")
        if index < 0 or index >= target_length:
            raise ValueError(
                f"hotspot concat index {index} is outside target length {target_length}"
            )
        if index not in seen:
            seen.add(index)
            indices.append(index)
    if not indices:
        raise ValueError("at least one hotspot index is required")
    return indices


def calculate_hotspot_coverage(
    structure_path: str | Path,
    *,
    target_chain_ids: Sequence[str],
    binder_chain_id: str,
    hotspot_concat_indices_0based: Sequence[int],
    expected_target_chain_lengths: Sequence[int] | None = None,
    heavy_atom_cutoff: float = 5.0,
    ca_cutoff: float = 8.0,
) -> dict[str, Any]:
    """Calculate per-hotspot contact distances and aggregate coverage.

    Hotspot indices are zero-based positions in the target chains concatenated in
    ``target_chain_ids`` order. A hotspot is covered when any target/binder heavy
    atom pair is within ``heavy_atom_cutoff``. C-alpha coverage is reported as a
    secondary, less strict geometric view.
    """

    if not target_chain_ids:
        raise ValueError("target_chain_ids cannot be empty")
    if binder_chain_id in target_chain_ids:
        raise ValueError("binder_chain_id must not also be a target chain")
    if heavy_atom_cutoff <= 0.0 or ca_cutoff <= 0.0:
        raise ValueError("contact cutoffs must be positive")

    model = _load_first_model(structure_path)
    target_chains = [_chain_residues(model, chain_id) for chain_id in target_chain_ids]
    binder_residues = _chain_residues(model, binder_chain_id)
    observed_lengths = [len(residues) for residues in target_chains]
    if expected_target_chain_lengths is not None:
        expected_lengths = [int(length) for length in expected_target_chain_lengths]
        if expected_lengths != observed_lengths:
            raise ValueError(
                f"target chain lengths do not match: expected={expected_lengths}, "
                f"observed={observed_lengths}"
            )

    concatenated_target = [
        (chain_id, residue)
        for chain_id, residues in zip(target_chain_ids, target_chains)
        for residue in residues
    ]
    hotspot_indices = _normalize_indices(
        hotspot_concat_indices_0based,
        target_length=len(concatenated_target),
    )
    binder_heavy = _heavy_atom_coordinates(binder_residues)
    binder_ca = _ca_coordinates(binder_residues)
    if binder_heavy.size == 0:
        raise ValueError(f"binder chain {binder_chain_id!r} has no heavy atoms")

    per_hotspot: list[dict[str, Any]] = []
    for concat_index in hotspot_indices:
        chain_id, residue = concatenated_target[concat_index]
        heavy_distance = _minimum_pair_distance(
            _heavy_atom_coordinates([residue]),
            binder_heavy,
        )
        ca_distance = _minimum_pair_distance(_ca_coordinates([residue]), binder_ca)
        residue_number = int(residue.id[1])
        insertion_code = str(residue.id[2]).strip()
        per_hotspot.append(
            {
                "concat_index_0based": concat_index,
                "chain_id": chain_id,
                "residue_number": residue_number,
                "insertion_code": insertion_code,
                "residue_name": str(residue.resname),
                "min_heavy_atom_distance": heavy_distance,
                "min_ca_distance": ca_distance,
                "heavy_atom_covered": bool(heavy_distance <= heavy_atom_cutoff),
                "ca_covered": bool(np.isfinite(ca_distance) and ca_distance <= ca_cutoff),
            }
        )

    heavy_count = sum(int(item["heavy_atom_covered"]) for item in per_hotspot)
    ca_count = sum(int(item["ca_covered"]) for item in per_hotspot)
    total = len(per_hotspot)
    heavy_distances = [float(item["min_heavy_atom_distance"]) for item in per_hotspot]
    finite_ca_distances = [
        float(item["min_ca_distance"])
        for item in per_hotspot
        if np.isfinite(item["min_ca_distance"])
    ]
    return {
        "structure_path": str(Path(structure_path)),
        "target_chain_ids": list(target_chain_ids),
        "binder_chain_id": binder_chain_id,
        "target_chain_lengths": observed_lengths,
        "hotspot_indices_0based": hotspot_indices,
        "hotspot_count": total,
        "heavy_atom_cutoff": float(heavy_atom_cutoff),
        "heavy_atom_covered_count": heavy_count,
        "heavy_atom_coverage": heavy_count / total,
        "mean_min_heavy_atom_distance": float(np.mean(heavy_distances)),
        "ca_cutoff": float(ca_cutoff),
        "ca_covered_count": ca_count,
        "ca_coverage": ca_count / total,
        "mean_min_ca_distance": (
            float(np.mean(finite_ca_distances)) if finite_ca_distances else float("nan")
        ),
        "per_hotspot": per_hotspot,
        "per_hotspot_json": json.dumps(per_hotspot, ensure_ascii=False),
    }
