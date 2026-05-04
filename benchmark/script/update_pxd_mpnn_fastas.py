#!/usr/bin/env python3
"""Update PXDesign FASTA files with ProteinMPNN-designed binder sequences.

PXDesign raw diffusion outputs use xpb placeholder residues for the binder chain.
This script keeps the original PDB files unchanged, creates temporary MPNN input
PDBs with xpb mapped to GLY, designs the binder chain using ProteinMPNN, and
rewrites the corresponding FASTA files with the designed binder sequence.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import textwrap
from pathlib import Path


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
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
    "xpb": "X",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/data/huangyanjie/binder_design/BindCraft-v2/benchmark/outputs/pxd_rf3"
        ),
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-seqs", type=int, default=1)
    parser.add_argument("--temperature", default="0.0001")
    parser.add_argument("--rm-aa", default="C")
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def prepare_mpnn_pdb(src: Path, dst: Path) -> None:
    with src.open() as inp, dst.open("w") as out:
        for line in inp:
            if line.startswith(("ATOM  ", "HETATM")):
                line = standardize_pdb_atom_line(line)
            out.write(line)


def source_chain_id(line: str) -> str:
    """Read standard or PXDesign compact chain IDs such as GLYA0/xpbB0."""
    if len(line) > 21 and line[21].strip() and line[21].strip() != "0":
        return line[21].strip()
    if len(line) > 20 and line[20].strip():
        return line[20].strip()
    if len(line) > 21:
        return line[21].strip()
    return ""


def standardize_pdb_atom_line(line: str) -> str:
    chain = source_chain_id(line)
    if line[17:20] == "xpb":
        if line.startswith("HETATM"):
            line = "ATOM  " + line[6:]
        line = line[:17] + "GLY" + line[20:]
    if chain:
        line = line.rstrip("\n").ljust(22) + ("\n" if line.endswith("\n") else "")
        line = line[:20] + " " + chain[:1] + line[22:]
    return line


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    header: str | None = None
    seq_parts: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                entries.append((header, "".join(seq_parts)))
            header = line[1:]
            seq_parts = []
        else:
            seq_parts.append(line)
    if header is not None:
        entries.append((header, "".join(seq_parts)))
    return entries


def format_entry(header: str, seq: str) -> str:
    return f">{header}\n" + "\n".join(textwrap.wrap(seq, 80)) + "\n"


def fasta_has_placeholder(fasta: Path) -> bool:
    return any("XXXX" in seq.upper() for _, seq in parse_fasta(fasta))


def binder_chain_from_fasta(fasta: Path) -> str:
    for header, seq in parse_fasta(fasta):
        if "role=binder" not in header:
            continue
        match = re.search(r"\|chain_([^|]+)\|", header)
        if match and ("XXXX" in seq.upper() or "role=binder_mpnn" not in header):
            return match.group(1)
    raise ValueError(f"No binder chain found in {fasta}")


def rewrite_single_fasta(fasta: Path, designed_seq: str, binder_chain: str) -> None:
    old_entries = parse_fasta(fasta)
    new_entries: list[tuple[str, str]] = []
    replaced = False
    for header, seq in old_entries:
        if f"|chain_{binder_chain}|" in header and "role=binder" in header:
            if "role=binder_xpb" in header:
                header = header.replace("role=binder_xpb", "role=binder_mpnn")
            elif "role=binder_mpnn" not in header:
                header = header.replace("role=binder", "role=binder_mpnn")
            new_entries.append((header, designed_seq))
            replaced = True
        else:
            new_entries.append((header, seq))
    if not replaced:
        stem = fasta.stem
        new_entries.insert(0, (f"{stem}|chain_{binder_chain}|role=binder_mpnn", designed_seq))
    fasta.write_text("\n".join(format_entry(h, s) for h, s in new_entries))


def read_chain_sequence_from_pdb(pdb_path: Path, chain_id: str) -> str:
    seen: set[tuple[str, str]] = set()
    seq: list[str] = []
    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if source_chain_id(line) != chain_id:
                continue
            resname = line[17:20].strip()
            key = (line[22:26].strip(), line[26].strip())
            if key in seen:
                continue
            seen.add(key)
            seq.append(AA3_TO_1.get(resname, "X"))
    return "".join(seq)


def rebuild_summaries(out_dir: Path) -> None:
    all_entries: list[str] = []
    binder_entries: list[str] = []
    for fasta in sorted(out_dir.glob("*.fasta")):
        if fasta.name in {"all_sequences.fasta", "all_binder_sequences.fasta"}:
            continue
        for header, seq in parse_fasta(fasta):
            entry = format_entry(header, seq)
            all_entries.append(entry)
            if "role=binder" in header:
                binder_entries.append(entry)
    (out_dir / "all_sequences.fasta").write_text("\n".join(all_entries))
    (out_dir / "all_binder_sequences.fasta").write_text("\n".join(binder_entries))


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    from ml_collections import ConfigDict
    from pxdbench.tools.protmpnn.main_mpnn import design_binder

    out_dir = args.out_dir
    pxd_pdbs = [
        pdb
        for pdb in sorted(out_dir.glob("pxd_*.pdb"), key=natural_key)
        if pdb.with_suffix(".fasta").exists() and fasta_has_placeholder(pdb.with_suffix(".fasta"))
    ]
    if not pxd_pdbs:
        raise FileNotFoundError(f"No pxd_*.pdb/FASTA pairs with placeholder sequences found in {out_dir}")

    temp_dir = Path("/tmp") / f"pxd_mpnn_inputs_{os.getpid()}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        names: list[str] = []
        binder_chain_by_name: dict[str, str] = {}
        cond_chain_by_name: dict[str, tuple[str, ...]] = {}
        for pdb in pxd_pdbs:
            dst = temp_dir / pdb.name
            prepare_mpnn_pdb(pdb, dst)
            names.append(pdb.stem)
            binder_chain = binder_chain_from_fasta(pdb.with_suffix(".fasta"))
            chain_ids = sorted({source_chain_id(line) for line in pdb.read_text().splitlines() if line.startswith(("ATOM  ", "HETATM"))})
            cond_chains = tuple(chain for chain in chain_ids if chain and chain != binder_chain)
            if not cond_chains:
                raise ValueError(f"No condition chains found for {pdb.name}")
            binder_chain_by_name[pdb.stem] = binder_chain
            cond_chain_by_name[pdb.stem] = cond_chains

        cfg = ConfigDict(
            {
                "fix_interface": False,
                "rm_aa": args.rm_aa,
                "temperature": args.temperature,
                "weights": "original",
            }
        )
        results = []
        grouped_names: dict[tuple[str, tuple[str, ...]], list[str]] = {}
        for name in names:
            grouped_names.setdefault((binder_chain_by_name[name], cond_chain_by_name[name]), []).append(name)
        for (binder_chain, cond_chains), group in sorted(grouped_names.items()):
            results.extend(
                design_binder(
                    str(temp_dir),
                    group,
                    args.num_seqs,
                    binder_chains=[binder_chain],
                    cond_chains=list(cond_chains),
                    mpnn_cfg=cfg,
                    if_print=True,
                )
            )

        seq_by_name: dict[str, str] = {}
        for row in results:
            if int(row.get("seq_idx", 0)) == 0:
                seq_by_name[row["name"]] = row["sequence"]

        missing: list[str] = []
        updated = 0
        for pdb in pxd_pdbs:
            seq = seq_by_name.get(pdb.stem)
            if not seq:
                missing.append(pdb.stem)
                continue
            binder_chain = binder_chain_by_name[pdb.stem]
            expected_len = len(read_chain_sequence_from_pdb(pdb, binder_chain))
            if expected_len and len(seq) != expected_len:
                raise ValueError(
                    f"Length mismatch for {pdb.name}: MPNN={len(seq)} PDB_chain_{binder_chain}={expected_len}"
                )
            rewrite_single_fasta(pdb.with_suffix(".fasta"), seq, binder_chain)
            updated += 1

        warnings = out_dir / "mpnn_update_warnings.txt"
        warnings.write_text("\n".join(missing) + ("\n" if missing else ""))
        print(f"pxd_pdbs={len(pxd_pdbs)}")
        print(f"mpnn_sequences={len(seq_by_name)}")
        print(f"updated_fastas={updated}")
        print(f"missing={len(missing)}")
        print(f"out_dir={out_dir}")
    finally:
        if args.keep_temp:
            print(f"temp_dir={temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
