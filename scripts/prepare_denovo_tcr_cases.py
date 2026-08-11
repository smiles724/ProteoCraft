#!/usr/bin/env python3
"""Build de novo TCR design inputs from an HLA allele + peptide + generic scaffold.

The CDR redesign pipeline (`benchmark/script/run_cdr_tcr_domain_redesign.py`)
consumes a multichain FASTA plus a complex template CIF. Starting from only
"HLA-A*02:01 + 9-mer peptide" we are missing both, so this script assembles
them from a *non-cognate* TCR-pMHC scaffold complex:

  * chain A/B/C sequences (MHC heavy / beta-2m / peptide) come from the
    scaffold FASTA, with chain C replaced by the requested peptide;
  * chain D/E (TCR alpha / beta) keep the scaffold framework verbatim — they
    are the redesign substrate, not a cognate starting point;
  * the template CIF keeps the scaffold's docking geometry, with chain C
    threaded onto the new peptide (backbone + CB retained, side chains
    dropped) so the template never leaks the scaffold peptide's identity.

CDR boundaries are derived from the IMGT anchor residues (Cys23, Trp41,
Cys104, Phe/Trp118) rather than ANARCI, which is not installed here. The
CDR3 range this produces reproduces the hand-curated `cdr_motif` values in
`benchmark/yaml/cdr_tcr/cdr_tcr_domain_redesign.yaml` exactly.

Usage (needs gemmi, e.g. the protenix env):
    python scripts/prepare_denovo_tcr_cases.py \
        --scaffold-fasta benchmark/input/cdr_tcr_case/rcsb_pdb_1QRN_abtrunc.fasta \
        --scaffold-template benchmark/input/cdr_tcr_case/1QRN_fixed_template.cif \
        --peptide ALGRRIPPV --peptide YLNPRVARI --peptide KLNPRIPRI \
        --out-dir benchmark/input/denovo_tcr_case
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

AA_1_TO_3 = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}

# Atoms kept when threading a new residue onto scaffold backbone coordinates.
BACKBONE_ATOMS = ("N", "CA", "C", "O", "CB")

# IMGT CDR2 spans slots 56-65; for TRAV/TRBV V domains the realised loop is
# 6 residues, so we take a fixed 6-residue window from the FR2 anchor.
CDR2_LENGTH = 6


# ---------------------------------------------------------------------------
# FASTA
# ---------------------------------------------------------------------------

def read_fasta_records(fasta_path: Path) -> Dict[str, Dict[str, str]]:
    """Parse `>...|Chain X|description` records, keyed by chain id.

    Mirrors the reader in `benchmark/script/run_cdr_tcr_domain_redesign.py`;
    duplicated here so this prep step stays importable without torch/protenix.
    """
    records: Dict[str, Dict[str, str]] = {}
    header: str = ""
    seq_parts: List[str] = []

    def flush() -> None:
        if not header:
            return
        chain_id = ""
        for part in header.split("|"):
            part = part.strip()
            if part.startswith("Chain "):
                chain_id = part.replace("Chain ", "", 1).strip()
                break
        if not chain_id:
            raise ValueError(f"Could not parse chain id from FASTA header: {header}")
        records[chain_id] = {"header": header, "sequence": "".join(seq_parts).upper()}

    with fasta_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    flush()
    return records


# ---------------------------------------------------------------------------
# IMGT anchor-based CDR numbering (ANARCI-free)
# ---------------------------------------------------------------------------

def find_imgt_anchors(sequence: str) -> Dict[str, int]:
    """Locate the four conserved IMGT anchors in a TCR V domain (0-based).

    Cys23 is the first cysteine, Trp41 the first tryptophan after it, Phe/Trp118
    the aromatic of the J-region `[FW]GxG` motif, and Cys104 the last cysteine
    before that motif.
    """
    cys23 = sequence.find("C")
    if cys23 < 0:
        raise ValueError("no cysteine found; not a V domain?")
    trp41 = sequence.find("W", cys23 + 1)
    if trp41 < 0:
        raise ValueError("no FR2 tryptophan found after Cys23")
    jmotif = re.search(r"[FW]G.G", sequence[cys23:])
    if jmotif is None:
        raise ValueError("no J-region [FW]GxG motif found")
    f118 = cys23 + jmotif.start()
    cys104 = sequence.rfind("C", 0, f118)
    if cys104 <= trp41:
        raise ValueError("no Cys104 found between FR2 and the J motif")
    return {"C23": cys23, "W41": trp41, "C104": cys104, "F118": f118}


def imgt_cdr_ranges(sequence: str) -> Dict[str, Tuple[int, int]]:
    """Return inclusive 0-based (start, end) for CDR1/CDR2/CDR3.

    CDR3 is reported as the Cys104..Phe118 span, matching the `cdr_motif`
    convention already used by the 4-case config.
    """
    a = find_imgt_anchors(sequence)
    cdr1 = (a["C23"] + 4, a["W41"] - 3)          # IMGT 27-38
    cdr2 = (a["W41"] + 15, a["W41"] + 14 + CDR2_LENGTH)  # IMGT 56-65
    cdr3 = (a["C104"], a["F118"])                # IMGT 104-118 inclusive
    for name, (start, end) in (("CDR1", cdr1), ("CDR2", cdr2), ("CDR3", cdr3)):
        if not 0 <= start <= end < len(sequence):
            raise ValueError(f"{name} range {start}-{end} outside sequence")
    return {"CDR1": cdr1, "CDR2": cdr2, "CDR3": cdr3}


def describe_cdrs(sequence: str, chain_id: str, chain_kind: str) -> Dict[str, object]:
    ranges = imgt_cdr_ranges(sequence)
    anchors = find_imgt_anchors(sequence)
    regions = {}
    for name, (start, end) in ranges.items():
        motif = sequence[start : end + 1]
        if sequence.count(motif) != 1:
            raise ValueError(
                f"{chain_kind} {name} motif {motif!r} is not unique in the chain; "
                "the motif-based mask in the runner requires uniqueness"
            )
        regions[name] = {
            "motif": motif,
            "start_0based": start,
            "end_0based_inclusive": end,
            "start_1based": start + 1,
            "end_1based": end + 1,
            "length": end - start + 1,
        }
    return {
        "chain": chain_id,
        "kind": chain_kind,
        "length": len(sequence),
        "imgt_anchors_0based": anchors,
        "regions": regions,
    }


# ---------------------------------------------------------------------------
# Template peptide threading
# ---------------------------------------------------------------------------

class _CifLoop:
    """A parsed mmCIF `loop_` block: column names plus raw whitespace tokens."""

    def __init__(self, header_start: int, columns: List[str], rows: List[List[str]], end: int):
        self.header_start = header_start
        self.columns = columns
        self.rows = rows
        self.end = end

    def index(self, column: str) -> int:
        return self.columns.index(column)


def _find_loop(lines: List[str], category: str) -> _CifLoop:
    """Locate the `loop_` whose columns belong to ``category`` (e.g. `_atom_site`)."""
    prefix = category + "."
    for idx, line in enumerate(lines):
        if line.strip() != "loop_":
            continue
        columns: List[str] = []
        cursor = idx + 1
        while cursor < len(lines) and lines[cursor].strip().startswith("_"):
            columns.append(lines[cursor].strip())
            cursor += 1
        if not columns or not columns[0].startswith(prefix):
            continue
        rows: List[List[str]] = []
        while cursor < len(lines):
            stripped = lines[cursor].strip()
            if not stripped or stripped.startswith(("#", "loop_", "_", "data_")):
                break
            rows.append(stripped.split())
            cursor += 1
        return _CifLoop(idx, columns, rows, cursor)
    raise ValueError(f"mmCIF loop for {category} not found")


def _render_loop(loop: _CifLoop) -> List[str]:
    return ["loop_"] + loop.columns + [" ".join(row) for row in loop.rows]


def thread_peptide_into_template(
    template_path: Path,
    peptide: str,
    out_path: Path,
    peptide_chain: str = "C",
) -> None:
    """Rewrite ``peptide_chain`` of the template onto ``peptide``.

    Keeps backbone (+CB) coordinates so the template still encodes the pMHC
    groove geometry and the TCR docking pose, while the residue identities
    become the new peptide's; scaffold side-chain atoms are dropped rather
    than reused. This edits the mmCIF text in place (atom records plus the
    entity seqres) instead of round-tripping through a structure library,
    because Protenix's template parser depends on categories that a
    round-trip does not reproduce.
    """
    lines = template_path.read_text(encoding="utf-8").splitlines()

    entity_poly = _find_loop(lines, "_entity_poly")
    strand_col = entity_poly.index("_entity_poly.pdbx_strand_id")
    entity_col = entity_poly.index("_entity_poly.entity_id")
    entity_ids = [row[entity_col] for row in entity_poly.rows if row[strand_col] == peptide_chain]
    if len(entity_ids) != 1:
        raise ValueError(f"expected exactly one entity for chain {peptide_chain}, got {entity_ids}")
    peptide_entity = entity_ids[0]

    atom_site = _find_loop(lines, "_atom_site")
    asym_col = atom_site.index("_atom_site.label_asym_id")
    seq_col = atom_site.index("_atom_site.label_seq_id")
    atom_col = atom_site.index("_atom_site.label_atom_id")
    comp_cols = [
        atom_site.index(name)
        for name in ("_atom_site.label_comp_id", "_atom_site.auth_comp_id")
        if name in atom_site.columns
    ]

    seq_ids = sorted({int(row[seq_col]) for row in atom_site.rows if row[asym_col] == peptide_chain})
    if len(seq_ids) != len(peptide):
        raise ValueError(
            f"template chain {peptide_chain} has {len(seq_ids)} residues but peptide "
            f"{peptide} has {len(peptide)}; only equal-length swaps are supported"
        )
    new_residue_by_seq_id = {seq_id: peptide[i] for i, seq_id in enumerate(seq_ids)}

    kept_rows: List[List[str]] = []
    for row in atom_site.rows:
        if row[asym_col] != peptide_chain:
            kept_rows.append(row)
            continue
        aa = new_residue_by_seq_id[int(row[seq_col])]
        keep = set(BACKBONE_ATOMS) - ({"CB"} if aa == "G" else set())
        if row[atom_col] not in keep:
            continue
        for col in comp_cols:
            row[col] = AA_1_TO_3[aa]
        kept_rows.append(row)
    atom_site.rows = kept_rows

    if "_atom_site.id" in atom_site.columns:
        id_col = atom_site.index("_atom_site.id")
        for serial, row in enumerate(atom_site.rows, start=1):
            row[id_col] = str(serial)

    poly_seq = _find_loop(lines, "_entity_poly_seq")
    poly_entity_col = poly_seq.index("_entity_poly_seq.entity_id")
    poly_mon_col = poly_seq.index("_entity_poly_seq.mon_id")
    poly_num_col = poly_seq.index("_entity_poly_seq.num")
    for row in poly_seq.rows:
        if row[poly_entity_col] != peptide_entity:
            continue
        row[poly_mon_col] = AA_1_TO_3[new_residue_by_seq_id[int(row[poly_num_col])]]

    # Splice the rewritten loops back, later block first so offsets stay valid.
    output = list(lines)
    for loop in sorted((atom_site, poly_seq), key=lambda item: item.header_start, reverse=True):
        output[loop.header_start : loop.end] = _render_loop(loop)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(output) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Case assembly
# ---------------------------------------------------------------------------

def write_fasta(
    records: Dict[str, Dict[str, str]],
    peptide: str,
    out_path: Path,
    peptide_chain: str,
    case_name: str,
) -> None:
    order = sorted(records)
    lines: List[str] = []
    for chain_id in order:
        record = records[chain_id]
        sequence = peptide if chain_id == peptide_chain else record["sequence"]
        description = (
            f"designed target peptide {peptide}"
            if chain_id == peptide_chain
            else record["header"].split("|", 2)[-1]
        )
        lines.append(f">{case_name}|Chain {chain_id}|{description}")
        lines.append(sequence)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaffold-fasta", required=True)
    parser.add_argument("--scaffold-template", required=True)
    parser.add_argument("--peptide", action="append", required=True, help="repeatable 9-mer")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--peptide-chain", default="C")
    parser.add_argument("--alpha-chain", default="D")
    parser.add_argument("--beta-chain", default="E")
    parser.add_argument("--scaffold-name", default=None, help="label used in file names")
    parser.add_argument(
        "--skip-template",
        action="store_true",
        help="only write FASTA/spec (use when gemmi is unavailable)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scaffold_fasta = Path(args.scaffold_fasta)
    scaffold_template = Path(args.scaffold_template)
    out_dir = Path(args.out_dir)
    scaffold_name = args.scaffold_name or scaffold_template.stem.split("_")[0]

    records = read_fasta_records(scaffold_fasta)
    for chain_id in (args.peptide_chain, args.alpha_chain, args.beta_chain):
        if chain_id not in records:
            raise ValueError(f"scaffold FASTA is missing chain {chain_id}")

    cdr_spec = {
        "scaffold": scaffold_name,
        "scaffold_fasta": str(scaffold_fasta),
        "scaffold_template": str(scaffold_template),
        "numbering": "IMGT (anchor-derived: Cys23 / Trp41 / Cys104 / [FW]118)",
        "chains": {
            args.alpha_chain: describe_cdrs(
                records[args.alpha_chain]["sequence"], args.alpha_chain, "TCR_alpha"
            ),
            args.beta_chain: describe_cdrs(
                records[args.beta_chain]["sequence"], args.beta_chain, "TCR_beta"
            ),
        },
        "cases": [],
    }

    for peptide in args.peptide:
        peptide = peptide.strip().upper()
        case_name = f"{scaffold_name}_{peptide}"
        fasta_path = out_dir / f"{case_name}.fasta"
        template_path = out_dir / f"{case_name}.cif"
        write_fasta(records, peptide, fasta_path, args.peptide_chain, case_name)
        if args.skip_template:
            template_path = scaffold_template
        else:
            thread_peptide_into_template(
                scaffold_template, peptide, template_path, args.peptide_chain
            )
        cdr_spec["cases"].append(
            {
                "peptide": peptide,
                "case_name": case_name,
                "fasta": fasta_path.name,
                "template": template_path.name,
            }
        )
        print(f"[ok] {case_name}: {fasta_path.name}, {template_path.name}")

    spec_path = out_dir / "cdr_spec.json"
    spec_path.write_text(json.dumps(cdr_spec, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] CDR spec -> {spec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
