"""pMHC construction, hotspot mapping, and PDB sequence extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from bindcraft.affibody.profiles import STANDARD_AAS


PDB_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U",
}


# The three non-binder chains are always ordered as HLA, beta2m, peptide.
# HLA and beta2m provide the fixed pMHC context; peptide is the specificity
# target used by target--binder interface losses.
PMHC_TARGET_CHAIN_ROLES = ("scaffold", "scaffold", "target")
PMHC_TARGET_CHAIN_ROLES_ARGUMENT = ",".join(PMHC_TARGET_CHAIN_ROLES)


def validate_protein_sequence(sequence: str, *, label: str) -> str:
    sequence = str(sequence).strip().upper()
    if not sequence:
        raise ValueError(f"{label} sequence cannot be empty")
    invalid = sorted(set(sequence) - STANDARD_AAS)
    if invalid:
        raise ValueError(f"{label} contains invalid amino acid(s): {invalid}")
    return sequence


def pdb_chain_sequences(path: str | Path) -> dict[str, str]:
    """Extract polymer sequences from ATOM records without Biopython."""

    sequences: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM  "):
                continue
            chain = line[21].strip() or "_"
            residue_id = line[22:27]
            residue_name = line[17:20].strip().upper()
            key = (chain, residue_id)
            if key in seen or residue_name not in PDB_THREE_TO_ONE:
                continue
            seen.add(key)
            aa = PDB_THREE_TO_ONE[residue_name]
            if aa in STANDARD_AAS:
                sequences.setdefault(chain, []).append(aa)
    return {chain: "".join(values) for chain, values in sequences.items()}


@dataclass(frozen=True)
class PMHCTarget:
    target_id: str
    hla_sequence: str
    b2m_sequence: str
    peptide: str
    register: str = "P4/P5"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "hla_sequence", validate_protein_sequence(self.hla_sequence, label="HLA")
        )
        object.__setattr__(
            self, "b2m_sequence", validate_protein_sequence(self.b2m_sequence, label="β2m")
        )
        peptide = validate_protein_sequence(self.peptide, label="peptide")
        if len(peptide) != 9:
            raise ValueError(f"pMHC peptide must be 9 aa, got {len(peptide)}")
        object.__setattr__(self, "peptide", peptide)
        register = str(self.register).strip().upper()
        if register not in {"P4/P5", "P5/P6"}:
            raise ValueError("register must be P4/P5 or P5/P6")
        object.__setattr__(self, "register", register)

    @property
    def target_argument(self) -> str:
        return "|".join((self.hla_sequence, self.b2m_sequence, self.peptide))

    @property
    def target_chain_roles(self) -> tuple[str, str, str]:
        return PMHC_TARGET_CHAIN_ROLES

    @property
    def target_chain_roles_argument(self) -> str:
        return PMHC_TARGET_CHAIN_ROLES_ARGUMENT

    @property
    def peptide_offset(self) -> int:
        return len(self.hla_sequence) + len(self.b2m_sequence)

    @property
    def hotspot_peptide_positions(self) -> tuple[int, int]:
        return (4, 5) if self.register == "P4/P5" else (5, 6)

    @property
    def hotspot_positions_1based(self) -> tuple[int, int]:
        return tuple(
            self.peptide_offset + position for position in self.hotspot_peptide_positions
        )

    @property
    def hotspot_argument(self) -> str:
        return ",".join(str(position) for position in self.hotspot_positions_1based)

    @property
    def up_facing_pair(self) -> str:
        left, right = self.hotspot_peptide_positions
        return self.peptide[left - 1] + self.peptide[right - 1]

    def as_dict(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "hla_sequence": self.hla_sequence,
            "b2m_sequence": self.b2m_sequence,
            "peptide": self.peptide,
            "register": self.register,
            "up_facing_pair": self.up_facing_pair,
            "target_argument": self.target_argument,
            "target_chain_roles": self.target_chain_roles_argument,
            "hotspot_positions_1based": list(self.hotspot_positions_1based),
            "hotspot_argument": self.hotspot_argument,
        }


def build_pmhc_target(
    *,
    target_id: str,
    peptide: str,
    hla_sequence: str,
    b2m_sequence: str,
    register: str = "P4/P5",
) -> PMHCTarget:
    return PMHCTarget(
        target_id=str(target_id),
        hla_sequence=hla_sequence,
        b2m_sequence=b2m_sequence,
        peptide=peptide,
        register=register,
    )


def chains_from_reference_pdb(
    path: str | Path,
    *,
    hla_chain: str = "A",
    b2m_chain: str = "B",
) -> tuple[str, str]:
    sequences: Mapping[str, str] = pdb_chain_sequences(path)
    missing = [chain for chain in (hla_chain, b2m_chain) if chain not in sequences]
    if missing:
        raise ValueError(f"missing reference PDB chain(s): {missing}")
    return sequences[hla_chain], sequences[b2m_chain]
