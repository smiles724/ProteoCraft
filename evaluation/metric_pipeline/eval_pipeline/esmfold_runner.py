"""ESMFold runner — build input, run inference, strip linker, rechain, dump artifacts.

Pure CPU helpers (build_input, segments, strip, rechain) at the top.
GPU functions (load_esmfold, run_esmfold) at the bottom; importing torch is lazy.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

LINKER = "G" * 25
LINKER_LEN = len(LINKER)
_PDB_CHAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ─── Pure CPU helpers ───────────────────────────────────────────────
def build_esmfold_input(target_seq: str, binder_seq: str, linker: str = LINKER):
    """target_seq may be ':' separated for multi-chain.

    Returns (esm_seq, segments_meta).
    Each segment dict: kind, length, dirty_start, dirty_end (含 linker 索引).
    """
    target_segments = target_seq.split(":") if target_seq else []
    all_segments = list(target_segments) + [binder_seq]
    parts = []
    cursor = 0
    meta = []
    for i, seg in enumerate(all_segments):
        kind = "binder" if i == len(all_segments) - 1 else f"target_chain_{i}"
        meta.append({
            "kind": kind,
            "length": len(seg),
            "dirty_start": cursor,
            "dirty_end": cursor + len(seg),
        })
        parts.append(seg)
        cursor += len(seg)
        if i < len(all_segments) - 1:
            parts.append(linker)
            cursor += len(linker)
    return "".join(parts), meta


def compute_clean_segments(segments_meta: list[dict]) -> list[dict]:
    """Add clean_start/clean_end (post-linker-strip indices) to each segment."""
    cursor = 0
    out = []
    for s in segments_meta:
        ss = dict(s)
        ss["clean_start"] = cursor
        ss["clean_end"] = cursor + s["length"]
        cursor += s["length"]
        out.append(ss)
    return out


def get_clean_index_arrays(clean_segments: list[dict]) -> dict:
    """target_idx_clean (concat of all target chains) + binder_idx_clean."""
    target_parts = [
        np.arange(s["clean_start"], s["clean_end"])
        for s in clean_segments
        if s["kind"].startswith("target_chain_")
    ]
    target_idx = (
        np.concatenate(target_parts) if target_parts else np.array([], dtype=int)
    )
    binder_seg = next(s for s in clean_segments if s["kind"] == "binder")
    binder_idx = np.arange(binder_seg["clean_start"], binder_seg["clean_end"])
    return {"target_idx_clean": target_idx, "binder_idx_clean": binder_idx}


def strip_linker(plddt: np.ndarray, pae: np.ndarray, segments_meta: list[dict]):
    """Remove linker rows/cols; return (plddt_clean, pae_clean, clean_segments)."""
    non_linker = np.concatenate([
        np.arange(s["dirty_start"], s["dirty_end"]) for s in segments_meta
    ])
    plddt_clean = plddt[non_linker]
    pae_clean = pae[np.ix_(non_linker, non_linker)]
    return plddt_clean, pae_clean, compute_clean_segments(segments_meta)


def _assign_pdb_chain_ids(clean_segments: list[dict]) -> list[dict]:
    """v1.1 reviewer P0 #1: assign A, B, C, ... to segments in order."""
    if len(clean_segments) > len(_PDB_CHAIN_LETTERS):
        raise ValueError(
            f"too many segments ({len(clean_segments)}) for single-letter chain IDs"
        )
    return [
        {**s, "pdb_chain_id": _PDB_CHAIN_LETTERS[i]} for i, s in enumerate(clean_segments)
    ]


def rechain_stripped_pdb_by_segments(structure, clean_segments_with_chain: list[dict]):
    """In-place: drop linker residues + rename chains per segments.

    ESMFold with G25-linker input produces a PDB with all residues in one chain. We
    must explicitly partition by clean_segments (using dirty_start/end) and rewrite
    chain IDs so downstream chain-aware code works.
    """
    if len(structure) != 1:
        raise ValueError(f"expected exactly 1 model, got {len(structure)}")
    model = structure[0]
    flat_residues = []
    for chain in list(model):
        for res in list(chain):
            if res.id[0] == " ":
                flat_residues.append((chain, res))
    keep_dirty_idx = set()
    for s in clean_segments_with_chain:
        keep_dirty_idx.update(range(s["dirty_start"], s["dirty_end"]))
    if not keep_dirty_idx:
        raise ValueError("no segments to keep")
    if max(keep_dirty_idx) >= len(flat_residues):
        raise ValueError(
            f"PDB has {len(flat_residues)} residues but max kept index is "
            f"{max(keep_dirty_idx)}; ESMFold output truncated?"
        )
    by_chain_letter: dict[str, list] = {}
    for dirty_idx, (_orig_chain, res) in enumerate(flat_residues):
        if dirty_idx not in keep_dirty_idx:
            continue
        seg = next(
            s for s in clean_segments_with_chain
            if s["dirty_start"] <= dirty_idx < s["dirty_end"]
        )
        by_chain_letter.setdefault(seg["pdb_chain_id"], []).append(res)

    from Bio.PDB.Chain import Chain

    for cid in [c.id for c in list(model)]:
        model.detach_child(cid)
    for chain_letter, residues in by_chain_letter.items():
        new_chain = Chain(chain_letter)
        for res in residues:
            try:
                if res.parent is not None:
                    res.parent.detach_child(res.id)
            except Exception:
                pass
            # Residues retain their original IDs from the dirty PDB. Works because
            # ESMFold output numbers residues sequentially across the whole input,
            # so within any single clean segment IDs are already unique. If a future
            # change ever produces overlapping IDs across kept residues, Bio.Chain.add
            # will raise — at which point renumbering would need to be added here.
            new_chain.add(res)
        model.add(new_chain)
    return structure


# ─── GPU section (lazy-import torch / transformers) ────────────────
@dataclass
class ESMFoldResult:
    pdb_str: str
    plddt: np.ndarray   # shape (L_dirty,) range [0, 100]
    pae: np.ndarray     # shape (L_dirty, L_dirty) Å
    ptm: float
    runtime_s: float


def load_esmfold(device: str = "cuda:0", chunk_size: int = 64):
    """Load HuggingFace facebook/esmfold_v1 on the given device."""
    import torch  # noqa: F401  (kept to fail fast if missing)
    from transformers import AutoTokenizer, EsmForProteinFolding

    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1", low_cpu_mem_usage=True
    )
    model = model.to(device)
    model.esm = model.esm.half()
    model.trunk.set_chunk_size(chunk_size)
    model.eval()
    return tokenizer, model


def run_esmfold(esm_seq: str, tokenizer, model, device: str = "cuda:0") -> ESMFoldResult:
    """Run a single forward pass; return raw arrays + PDB string."""
    import torch
    from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
    from transformers.models.esm.openfold_utils.protein import Protein as OFProtein
    from transformers.models.esm.openfold_utils.protein import to_pdb

    t0 = time.time()
    inputs = tokenizer([esm_seq], return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    # plddt: HF returns shape (B, L, 37) per-atom; reduce to per-residue mean.
    # HF's EsmForProteinFolding returns plddt in [0, 1]; rescale to the standard
    # 0-100 pLDDT scale used by AF2/ESMFold reports, downstream tier thresholds
    # (binder_plddt_loose=80, strict=90), and PDB B-factor convention.
    if outputs.plddt.dim() == 3:
        plddt = outputs.plddt[0].mean(dim=-1).cpu().float().numpy()
    else:
        plddt = outputs.plddt[0].cpu().float().numpy()
    plddt = plddt * 100.0
    pae = outputs.predicted_aligned_error[0].cpu().float().numpy()
    ptm = (
        float(outputs.ptm.cpu())
        if hasattr(outputs, "ptm") and outputs.ptm is not None
        else float("nan")
    )

    final_atom_positions = atom14_to_atom37(outputs.positions[-1], outputs)
    final_atom_mask = outputs.atom37_atom_exists
    # b_factors per atom = pLDDT broadcast across atoms (standard convention)
    b_factors = np.broadcast_to(
        plddt[:, None], (plddt.shape[0], final_atom_mask.shape[-1])
    ).copy()
    pred = OFProtein(
        aatype=outputs.aatype[0].cpu().numpy(),
        atom_positions=final_atom_positions[0].cpu().numpy(),
        atom_mask=final_atom_mask[0].cpu().numpy(),
        residue_index=outputs.residue_index[0].cpu().numpy() + 1,
        b_factors=b_factors,
        chain_index=(
            outputs.chain_index[0].cpu().numpy()
            if hasattr(outputs, "chain_index") and outputs.chain_index is not None
            else None
        ),
    )
    pdb_str = to_pdb(pred)
    return ESMFoldResult(
        pdb_str=pdb_str, plddt=plddt, pae=pae, ptm=ptm, runtime_s=time.time() - t0
    )


def dump_design_artifacts(
    out_dir: Path,
    esm_seq: str,
    segments_meta: list[dict],
    result: ESMFoldResult,
) -> dict:
    """Write esmfold.pdb + plddt.npy + pae.npy + meta.json (linker-stripped + rechained)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plddt_clean, pae_clean, clean_segments = strip_linker(
        result.plddt, result.pae, segments_meta
    )
    clean_segments_with_chain = _assign_pdb_chain_ids(clean_segments)

    from io import StringIO

    from Bio.PDB import PDBIO, PDBParser

    structure = PDBParser(QUIET=True).get_structure("p", StringIO(result.pdb_str))
    structure = rechain_stripped_pdb_by_segments(structure, clean_segments_with_chain)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_dir / "esmfold.pdb"))

    np.save(out_dir / "plddt.npy", plddt_clean.astype(np.float32))
    np.save(out_dir / "pae.npy", pae_clean.astype(np.float32))
    target_len = sum(
        s["length"] for s in clean_segments_with_chain if s["kind"].startswith("target_chain_")
    )
    binder_len = next(s["length"] for s in clean_segments_with_chain if s["kind"] == "binder")
    meta = {
        "target_len": target_len,
        "binder_len": binder_len,
        "ptm": result.ptm,
        "esm_runtime_s": result.runtime_s,
        "esmfold_version": "v1",
        "linker": "G*25",
        "clean_segments": clean_segments_with_chain,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta
