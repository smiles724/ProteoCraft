"""
PDB chain manipulation utilities.

Inspired by NVIDIA-Digital-Bio/proteina-complexa (Apache-2.0).
Clean reimplementation in BioPython — no Hydra / no upstream deps.
See ./LICENSE.txt and ./NOTICE.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import List

import numpy as np
from Bio.PDB import PDBIO, PDBParser, Select
from Bio.SeqUtils import seq1

warnings.filterwarnings("ignore", module="Bio")


class _ChainSelect(Select):
    def __init__(self, chains: List[str]):
        self.chains = set(chains)

    def accept_chain(self, chain):
        return chain.id in self.chains


def extract_chains_to_pdb(pdb_path: str, chain_ids: List[str], out_path: str) -> str:
    """Write a new PDB containing only the requested chain IDs. Returns out_path."""
    structure = PDBParser(QUIET=True).get_structure("s", pdb_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path, _ChainSelect(chain_ids))
    return out_path


def list_chain_ids(pdb_path: str) -> List[str]:
    """Return chain IDs in order of appearance in the first model."""
    structure = PDBParser(QUIET=True).get_structure("s", pdb_path)
    return [c.id for c in structure[0]]


def get_chain_sequence(pdb_path: str, chain_id: str) -> str:
    """1-letter amino acid sequence of standard residues only.

    Skips HETATM, water, and non-standard residues (replaced with 'X').
    """
    structure = PDBParser(QUIET=True).get_structure("s", pdb_path)
    aa = []
    for res in structure[0][chain_id]:
        if res.id[0] != " ":
            continue
        try:
            aa.append(seq1(res.resname))
        except Exception:
            aa.append("X")
    return "".join(aa)


def get_chain_ca_coords(pdb_path: str, chain_id: str) -> np.ndarray:
    """Cα coords for standard residues, shape (N, 3). Skips residues without CA atom."""
    structure = PDBParser(QUIET=True).get_structure("s", pdb_path)
    coords = []
    for res in structure[0][chain_id]:
        if res.id[0] != " " or "CA" not in res:
            continue
        coords.append(res["CA"].coord.copy())
    return np.array(coords, dtype=np.float64)
