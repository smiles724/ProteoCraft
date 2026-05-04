#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import gemmi
import yaml


@dataclass(frozen=True)
class ManifestRow:
    method: str
    case: str
    condition: str
    length: int
    samples: int
    config: Path
    out_dir: Path


_PX_PRED_RE = re.compile(r"_sample_(\d+)\.cif$")


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _normalize_chain_id(chain_id: str) -> str:
    """
    Some outputs use chain IDs like A0/B0 while entity tables may list A/B.
    Normalize by stripping trailing digits.
    """
    return re.sub(r"\d+$", "", chain_id.strip())


def _read_manifest(path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            rows.append(
                ManifestRow(
                    method=r["method"],
                    case=_slug(r["case"]),
                    condition=_slug(r["condition"]),
                    length=int(r["length"]),
                    samples=int(r["samples"]),
                    config=Path(r["config"]),
                    out_dir=Path(r["out_dir"]),
                )
            )
    return rows


def _read_text_maybe_gz(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as h:
            return h.read()
    return path.read_text(encoding="utf-8", errors="replace")


def _read_structure(path: Path) -> gemmi.Structure:
    if path.suffix == ".gz":
        doc = gemmi.cif.read_string(_read_text_maybe_gz(path))
        block = doc.sole_block()
        return gemmi.make_structure_from_block(block)
    return gemmi.read_structure(str(path))


def _is_aa_residue(res: gemmi.Residue) -> bool:
    info = gemmi.find_tabulated_residue(res.name)
    return bool(info.is_amino_acid())


def _residue_one_letter(res: gemmi.Residue) -> str:
    info = gemmi.find_tabulated_residue(res.name)
    if not info.is_amino_acid():
        return ""
    olc = info.one_letter_code
    if not olc or olc == "?":
        return "X"
    return olc


def _chain_sequence(chain: gemmi.Chain) -> str:
    letters: list[str] = []
    for res in chain:
        aa = _residue_one_letter(res)
        if aa:
            letters.append(aa)
    return "".join(letters)


def _chain_aa_len(chain: gemmi.Chain) -> int:
    n = 0
    for res in chain:
        if _is_aa_residue(res):
            n += 1
    return n


def _copy_chain(chain: gemmi.Chain) -> gemmi.Chain:
    new_chain = gemmi.Chain(chain.name)
    for res in chain:
        # Keep all residues present in the chain (pxdesign may use placeholder
        # residue names like 'xpb' for the designed binder backbone).
        new_res = gemmi.Residue()
        new_res.name = res.name
        new_res.seqid = gemmi.SeqId(res.seqid.num, res.seqid.icode)
        new_res.subchain = res.subchain
        for atom in res:
            new_atom = gemmi.Atom()
            new_atom.name = atom.name
            new_atom.altloc = atom.altloc
            new_atom.occ = atom.occ
            new_atom.b_iso = atom.b_iso
            new_atom.pos = gemmi.Position(atom.pos.x, atom.pos.y, atom.pos.z)
            new_atom.element = atom.element
            new_atom.charge = atom.charge
            new_res.add_atom(new_atom)
        new_chain.add_residue(new_res)
    return new_chain


def _format_fasta_entry(header: str, seq: str, width: int = 80) -> str:
    seq = seq.strip()
    lines = [f">{header}"]
    for i in range(0, len(seq), width):
        lines.append(seq[i : i + width])
    lines.append("")
    return "\n".join(lines)


def _write_full_fasta(
    *,
    out_fasta: Path,
    stem: str,
    source_path: Path,
    target_chain_norm: str | None,
    chain_order: list[gemmi.Chain],
    seq_by_chain_norm: dict[str, str],
    placeholder_len_by_chain_norm: dict[str, int],
) -> None:
    parts: list[str] = []
    for ch in chain_order:
        chain_id = ch.name
        chain_norm = _normalize_chain_id(chain_id)
        seq = seq_by_chain_norm.get(chain_norm)
        if seq is None:
            # If we cannot recover sequence (e.g. pxdesign binder uses xpb placeholders),
            # emit 'X' repeated to the chain residue count as a safe placeholder.
            L = placeholder_len_by_chain_norm.get(chain_norm, len(list(ch)))
            seq = "X" * int(L)

        role = "target" if (target_chain_norm and chain_norm == target_chain_norm) else "binder"
        header = f"{stem}|chain_{chain_norm}|role={role}|source={source_path}"
        parts.append(_format_fasta_entry(header, seq))

    out_fasta.write_text("\n".join(parts), encoding="utf-8")


def _write_full_pdb(
    *,
    structure: gemmi.Structure,
    chain_order: list[gemmi.Chain],
    out_pdb: Path,
) -> None:
    # Write a structure containing all chains (target + binder) to match 5_tcr style.
    s = gemmi.Structure()
    s.name = structure.name
    s.cell = structure.cell
    s.spacegroup_hm = structure.spacegroup_hm
    m = gemmi.Model("1")
    for ch in chain_order:
        m.add_chain(_copy_chain(ch))
    s.add_model(m)
    s.setup_entities()
    s.write_pdb(str(out_pdb))


def _write_binder_outputs(
    *,
    structure: gemmi.Structure,
    source_path: Path,
    stem: str,
    target_chain_norm: str | None,
    chain_order: list[gemmi.Chain],
    seq_by_chain_norm: dict[str, str],
    placeholder_len_by_chain_norm: dict[str, int],
    out_pdb: Path,
    out_fasta: Path,
    overwrite: bool,
) -> None:
    if not overwrite and (out_pdb.exists() or out_fasta.exists()):
        return

    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)

    _write_full_fasta(
        out_fasta=out_fasta,
        stem=stem,
        source_path=source_path,
        target_chain_norm=target_chain_norm,
        chain_order=chain_order,
        seq_by_chain_norm=seq_by_chain_norm,
        placeholder_len_by_chain_norm=placeholder_len_by_chain_norm,
    )
    _write_full_pdb(structure=structure, chain_order=chain_order, out_pdb=out_pdb)


def _pxd_target_chain_id(config_yaml: Path) -> str:
    data = yaml.safe_load(config_yaml.read_text(encoding="utf-8"))
    chains = (data or {}).get("target", {}).get("chains", {})
    if not isinstance(chains, dict) or not chains:
        raise ValueError(f"{config_yaml} missing target.chains")
    # exactly one chain expected
    chain_id = next(iter(chains.keys()))
    if not isinstance(chain_id, str) or not chain_id.strip():
        raise ValueError(f"{config_yaml} invalid target chain id: {chain_id!r}")
    return chain_id.strip()


def _pick_pxdesign_binder_chain(
    structure: gemmi.Structure, *, binder_len: int, target_chain_id: str
) -> gemmi.Chain:
    if len(structure) == 0:
        raise ValueError("No models in pxdesign CIF")
    model = structure[0]
    chains = list(model)
    if not chains:
        raise ValueError("No chains in pxdesign CIF model")

    scored: list[tuple[int, int, gemmi.Chain]] = []
    for c in chains:
        # pxdesign may emit placeholder residue names; use raw residue count.
        raw_len = len(list(c))
        if raw_len == 0:
            continue
        # primary: closest to expected binder length
        # secondary: prefer not being the declared target chain id (if it exists)
        scored.append((abs(raw_len - binder_len), 0 if c.name != target_chain_id else 1, c))
    if not scored:
        raise ValueError("No chains with residues found in pxdesign CIF")
    scored.sort(key=lambda x: (x[0], x[1]))
    return scored[0][2]


def _pxdesign_chain_sequences(cif_path: Path) -> dict[str, str]:
    """
    Return a mapping of chain_id -> sequence (1-letter) from mmCIF entity tables.

    pxdesign outputs can have placeholder residue names (e.g. xpb) in atom records,
    so extracting sequence from coordinates is unreliable. The entity_poly_seq table
    contains the intended monomer identities.
    """
    doc = gemmi.cif.read_file(str(cif_path))
    block = doc.sole_block()

    # entity_id -> [strand_ids]
    entity_to_strands: dict[str, list[str]] = {}
    try:
        ent_poly = block.find("_entity_poly.", ["entity_id", "pdbx_strand_id"])
    except Exception:
        ent_poly = None
    if ent_poly is not None:
        for row in ent_poly:
            entity_id = str(row[0]).strip()
            strand_id = str(row[1]).strip()
            if not entity_id or not strand_id:
                continue
            strands = [s.strip() for s in strand_id.split(",") if s.strip()]
            if strands:
                entity_to_strands.setdefault(entity_id, []).extend(strands)

    # entity_id -> mon_id list ordered by num
    entity_seq: dict[str, list[tuple[int, str]]] = {}
    try:
        ent_seq = block.find("_entity_poly_seq.", ["entity_id", "mon_id", "num"])
    except Exception:
        ent_seq = None
    if ent_seq is None:
        return {}
    for row in ent_seq:
        entity_id = str(row[0]).strip()
        mon_id = str(row[1]).strip()
        try:
            num = int(str(row[2]).strip())
        except Exception:
            continue
        if not entity_id or not mon_id:
            continue
        entity_seq.setdefault(entity_id, []).append((num, mon_id))

    chain_to_seq: dict[str, str] = {}
    for entity_id, items in entity_seq.items():
        items.sort(key=lambda t: t[0])
        letters: list[str] = []
        for _, mon in items:
            info = gemmi.find_tabulated_residue(mon)
            olc = info.one_letter_code if info.is_amino_acid() else ""
            letters.append(olc if olc and olc != "?" else "X")
        seq = "".join(letters).strip()
        if not seq:
            continue
        for strand in entity_to_strands.get(entity_id, []):
            chain_to_seq[strand] = seq

    return chain_to_seq


def _pick_rf3_binder_chain(structure: gemmi.Structure, binder_len: int) -> gemmi.Chain:
    if len(structure) == 0:
        raise ValueError("No models in rf3 CIF")
    model = structure[0]
    chains = list(model)
    if not chains:
        raise ValueError("No chains in rf3 CIF model")

    scored: list[tuple[int, gemmi.Chain]] = []
    for c in chains:
        aa_len = _chain_aa_len(c)
        if aa_len == 0:
            continue
        scored.append((abs(aa_len - binder_len), c))
    if not scored:
        raise ValueError("No protein chains found in rf3 CIF")
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def _model_chain_list(structure: gemmi.Structure) -> list[gemmi.Chain]:
    if len(structure) == 0:
        return []
    return list(structure[0])


def _iter_pxdesign_predictions(out_dir: Path) -> Iterable[tuple[int, Path]]:
    for p in sorted(out_dir.rglob("*_sample_*.cif")):
        if "predictions" not in p.parts:
            continue
        m = _PX_PRED_RE.search(p.name)
        if not m:
            continue
        yield int(m.group(1)), p


def _expected_rf3_path(out_dir: Path, condition_slug: str, sample_i: int) -> Path:
    # Observed naming: input_<condition>_<i>_model_0.cif.gz
    return out_dir / f"input_{condition_slug}_{sample_i}_model_0.cif.gz"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Flatten 8_bcv1 pxdesign+rf3 outputs into a flat folder (pxd_rf3 style)."
    )
    p.add_argument(
        "--output-root",
        default="/data/huangyanjie/binder_design/BindCraft-v2/benchmark/outputs/8_bcv1/pxd_rf3",
        help="BCv1 output root containing bcv1_design_manifest.csv and pxd_bcv1/rf3_bcv1.",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Flat output dir (default: <output-root>/pxd_rf3).",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = p.parse_args()

    output_root = Path(args.output_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (output_root / "pxd_rf3")
    manifest = output_root / "bcv1_design_manifest.csv"
    rows = _read_manifest(manifest)

    wrote = 0
    skipped = 0

    for row in rows:
        case = row.case
        cond = row.condition
        L = row.length

        if row.method == "pxdesign":
            target_chain_id = _pxd_target_chain_id(row.config)
            found_any = False
            for i, cif_path in _iter_pxdesign_predictions(row.out_dir):
                found_any = True
                stem = f"pxd_{case}_{cond}_len_{L}_sample_{i}"
                out_pdb = out_dir / f"{stem}.pdb"
                out_fasta = out_dir / f"{stem}.fasta"
                if not args.overwrite and (out_pdb.exists() or out_fasta.exists()):
                    skipped += 1
                    continue
                s = _read_structure(cif_path)
                binder_chain = _pick_pxdesign_binder_chain(
                    s, binder_len=L, target_chain_id=target_chain_id
                )
                chain_to_seq_raw = _pxdesign_chain_sequences(cif_path)
                # normalize keys because structures often use A0/B0 while entity tables use A/B
                seq_by_chain_norm: dict[str, str] = {}
                for k, v in chain_to_seq_raw.items():
                    seq_by_chain_norm[_normalize_chain_id(k)] = v
                # For pxdesign, binder sequence is often unknown (xpb placeholders). We will emit X*L placeholder.
                placeholder_len_by_chain_norm = {_normalize_chain_id(binder_chain.name): L}
                chain_order = _model_chain_list(s)
                _write_binder_outputs(
                    structure=s,
                    source_path=cif_path,
                    stem=stem,
                    target_chain_norm=_normalize_chain_id(target_chain_id),
                    chain_order=chain_order,
                    seq_by_chain_norm=seq_by_chain_norm,
                    placeholder_len_by_chain_norm=placeholder_len_by_chain_norm,
                    out_pdb=out_pdb,
                    out_fasta=out_fasta,
                    overwrite=args.overwrite,
                )
                wrote += 1
            if not found_any:
                raise FileNotFoundError(f"No pxdesign predictions found under {row.out_dir}")

        elif row.method == "rf3":
            # rf3 condition string in files is NOT slugged in manifest (it is already e.g. B24 or B33_B..).
            # Our manifest condition was slugged for output naming, but rf3 filenames use original (non-sanitized)
            # condition token from the directory name. Reconstruct it from the directory path.
            #
            # We trust the actual output files in out_dir, and fall back to a glob if the exact expected path is absent.
            condition_token = row.out_dir.parent.name  # .../<condition>/len_<L>
            for i in range(row.samples):
                expected = _expected_rf3_path(row.out_dir, condition_token, i)
                if not expected.exists():
                    matches = sorted(row.out_dir.glob(f"input_*_{i}_model_0.cif.gz"))
                    if len(matches) != 1:
                        raise FileNotFoundError(
                            f"Missing rf3 output for {row.out_dir} sample={i}: "
                            f"expected {expected.name} (found {len(matches)} matches)"
                        )
                    cif_gz = matches[0]
                else:
                    cif_gz = expected

                stem = f"rf3_{case}_{cond}_len_{L}_sample_{i}"
                out_pdb = out_dir / f"{stem}.pdb"
                out_fasta = out_dir / f"{stem}.fasta"
                if not args.overwrite and (out_pdb.exists() or out_fasta.exists()):
                    skipped += 1
                    continue

                s = _read_structure(cif_gz)
                binder_chain = _pick_rf3_binder_chain(s, binder_len=L)
                chain_order = _model_chain_list(s)
                seq_by_chain_norm: dict[str, str] = {}
                placeholder_len_by_chain_norm: dict[str, int] = {}
                for ch in chain_order:
                    seq_by_chain_norm[_normalize_chain_id(ch.name)] = _chain_sequence(ch)
                # No explicit target chain id in rf3 outputs; mark binder by length match, others as target.
                # We'll set target_chain_norm=None and let role be derived later (binder default), but we still
                # want to label non-binder chains as target. To do so, set target_chain_norm to the first
                # non-binder chain if present.
                binder_norm = _normalize_chain_id(binder_chain.name)
                non_binder = [c for c in chain_order if _normalize_chain_id(c.name) != binder_norm]
                target_chain_norm = _normalize_chain_id(non_binder[0].name) if non_binder else None
                _write_binder_outputs(
                    structure=s,
                    source_path=cif_gz,
                    stem=stem,
                    target_chain_norm=target_chain_norm,
                    chain_order=chain_order,
                    seq_by_chain_norm=seq_by_chain_norm,
                    placeholder_len_by_chain_norm=placeholder_len_by_chain_norm,
                    out_pdb=out_pdb,
                    out_fasta=out_fasta,
                    overwrite=args.overwrite,
                )
                wrote += 1
        else:
            raise ValueError(f"Unknown method: {row.method!r}")

    print(f"out_dir={out_dir}")
    print(f"wrote_binders={wrote} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

