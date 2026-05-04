#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestRow:
    paper_case: str
    design_id: str
    designed_complex_pdb_path: str
    target_chain_in_complex: str
    binder_chain_in_complex: str
    target_seq: str
    binder_seq: str


def _wrap_seq(seq: str, width: int = 80) -> str:
    seq = re.sub(r"\s+", "", seq.strip())
    return "\n".join(seq[i : i + width] for i in range(0, len(seq), width))


def _write_fasta(out_path: Path, *, stem: str, source_pdb: Path, row: ManifestRow) -> None:
    t_chain = row.target_chain_in_complex.strip()
    b_chain = row.binder_chain_in_complex.strip()
    t_seq = re.sub(r"\s+", "", row.target_seq.strip())
    b_seq = re.sub(r"\s+", "", row.binder_seq.strip())
    if not t_chain or not b_chain:
        raise ValueError(f"{row.design_id}: empty chain ids: target={t_chain!r} binder={b_chain!r}")
    if not t_seq or not b_seq:
        raise ValueError(f"{row.design_id}: empty sequences: target_len={len(t_seq)} binder_len={len(b_seq)}")

    text = (
        f">{stem}|chain_{t_chain}|role=target|source={source_pdb}\n"
        f"{_wrap_seq(t_seq)}\n\n"
        f">{stem}|chain_{b_chain}|role=binder|source={source_pdb}\n"
        f"{_wrap_seq(b_seq)}\n"
    )
    out_path.write_text(text, encoding="utf-8")


def _read_manifest(path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for r in reader:
            # Some manifests can be header-only (0 rows)
            rows.append(
                ManifestRow(
                    paper_case=r["paper_case"],
                    design_id=r["design_id"],
                    designed_complex_pdb_path=r["designed_complex_pdb_path"],
                    target_chain_in_complex=r["target_chain_in_complex"],
                    binder_chain_in_complex=r["binder_chain_in_complex"],
                    target_seq=r["target_seq"],
                    binder_seq=r["binder_seq"],
                )
            )
    return rows


def main() -> int:
    p = argparse.ArgumentParser(
        description="Flatten benchmark/results/bcv1_yanchao into outputs/8_bcv1/bcv1 (pdb + full-complex fasta)."
    )
    p.add_argument(
        "--bcv1-yanchao-root",
        default="/data/huangyanjie/binder_design/BindCraft-v2/benchmark/results/bcv1_yanchao",
        help="Root containing work/manifests and work/designed_complex_pdbs.",
    )
    p.add_argument(
        "--out-dir",
        default="/data/huangyanjie/binder_design/BindCraft-v2/benchmark/outputs/8_bcv1/bcv1",
        help="Flat output directory.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = p.parse_args()

    root = Path(args.bcv1_yanchao_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    manifests = sorted(glob.glob(str(root / "work" / "manifests" / "*.selected.manifest.csv")))
    if not manifests:
        raise FileNotFoundError(f"No manifests found under {root}/work/manifests")

    out_dir.mkdir(parents=True, exist_ok=True)

    wrote = 0
    skipped = 0

    for man in manifests:
        man_path = Path(man)
        rows = _read_manifest(man_path)
        if not rows:
            continue

        for row in rows:
            paper_case = row.paper_case.strip()
            design_id = row.design_id.strip()
            if not paper_case or not design_id:
                raise ValueError(f"{man_path}: missing paper_case/design_id in a row")

            stem = f"bcv1_{paper_case}_{design_id}"
            src_pdb = (root / row.designed_complex_pdb_path).resolve()
            if not src_pdb.exists():
                raise FileNotFoundError(f"Missing designed_complex_pdb_path: {src_pdb} (from {man_path})")

            out_pdb = out_dir / f"{stem}.pdb"
            out_fasta = out_dir / f"{stem}.fasta"

            if not args.overwrite and (out_pdb.exists() or out_fasta.exists()):
                skipped += 1
                continue

            shutil.copyfile(src_pdb, out_pdb)
            _write_fasta(out_fasta, stem=stem, source_pdb=src_pdb, row=row)
            wrote += 1

    print(f"out_dir={out_dir}")
    print(f"wrote_designs={wrote} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

