"""AF3 evaluator runner: input JSON, subprocess invocation, output parsing."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


def build_clean_segments_af3(
    binder_seq: str,
    target_seqs: list[str],
) -> list[dict]:
    """AF3 native multichain layout: binder=A, targets=B,C,... with no linker."""
    segs = []
    cursor = 0
    segs.append({
        "kind": "binder",
        "length": len(binder_seq),
        "dirty_start": cursor,
        "dirty_end": cursor + len(binder_seq),
        "clean_start": cursor,
        "clean_end": cursor + len(binder_seq),
        "pdb_chain_id": "A",
    })
    cursor += len(binder_seq)
    for i, tseq in enumerate(target_seqs):
        chain_id = chr(ord("B") + i)
        segs.append({
            "kind": f"target_chain_{i}",
            "length": len(tseq),
            "dirty_start": cursor,
            "dirty_end": cursor + len(tseq),
            "clean_start": cursor,
            "clean_end": cursor + len(tseq),
            "pdb_chain_id": chain_id,
        })
        cursor += len(tseq)
    return segs


def build_af3_input_json(
    job_name: str,
    target_seqs: list[str],
    binder_seq: str,
    target_msa_paths: list[str],
    seed: int = 42,
) -> dict:
    """Construct Alphafold3 dialect v4 JSON for one binder-target design."""
    if len(target_msa_paths) != len(target_seqs):
        raise ValueError(
            f"target_msa_paths ({len(target_msa_paths)}) length must match "
            f"target_seqs ({len(target_seqs)})"
        )

    sequences: list[dict] = [
        {
            "protein": {
                "id": "A",
                "sequence": binder_seq,
                "unpairedMsa": "",
                "pairedMsa": "",
                "templates": [],
            }
        }
    ]
    for i, tseq in enumerate(target_seqs):
        chain_id = chr(ord("B") + i)
        sequences.append({
            "protein": {
                "id": chain_id,
                "sequence": tseq,
                "unpairedMsaPath": target_msa_paths[i],
                "pairedMsa": "",
                "templates": [],
            }
        })

    return {
        "name": job_name,
        "modelSeeds": [seed],
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": 4,
    }


def write_input_json(input_obj: dict, json_path: Path) -> None:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(input_obj, indent=2))


def mmcif_to_pdb(cif_path: Path, pdb_path: Path, decompress_first: bool = False) -> None:
    """Convert AF3 mmCIF output to PDB via BioPython."""
    from Bio.PDB import MMCIFParser, PDBIO

    cif_path = Path(cif_path)
    pdb_path = Path(pdb_path)
    pdb_path.parent.mkdir(parents=True, exist_ok=True)

    if decompress_first:
        import zstandard as zstd

        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with open(tmp_path, "wb") as fout, zstd.open(cif_path, "rb") as src:
                shutil.copyfileobj(src, fout)
            structure = MMCIFParser(QUIET=True).get_structure("af3", tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    else:
        structure = MMCIFParser(QUIET=True).get_structure("af3", str(cif_path))

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_path))


def extract_residue_plddt_from_pdb(
    pdb_path: Path,
    clean_segments: list[dict],
) -> np.ndarray:
    """Return residue-level pLDDT from PDB B-factors in clean_segments order."""
    from Bio.PDB import PDBParser

    structure = PDBParser(QUIET=True).get_structure("af3", str(pdb_path))
    model = structure[0]
    chain_ids = [c.id for c in model]
    plddt: list[float] = []

    for seg in clean_segments:
        chain_id = seg["pdb_chain_id"]
        if chain_id not in chain_ids:
            raise ValueError(f"chain {chain_id!r} missing from {pdb_path} (have {chain_ids})")

        seg_plddt = []
        seg_resseqs = []
        for res in model[chain_id]:
            if res.id[0] != " ":
                continue
            atom_b = [a.bfactor for a in res if a.element != "H"]
            seg_plddt.append(float(np.mean(atom_b)) if atom_b else float("nan"))
            seg_resseqs.append(res.id[1])

        if len(seg_plddt) != seg["length"]:
            raise ValueError(
                f"chain {chain_id} ({seg['kind']}) has {len(seg_plddt)} residues, "
                f"expected {seg['length']} (clean_segments mismatch)"
            )
        expected_resseqs = list(range(1, seg["length"] + 1))
        if seg_resseqs != expected_resseqs:
            raise ValueError(
                f"chain {chain_id} ({seg['kind']}) resseq order "
                f"{seg_resseqs[:5]}{'...' if len(seg_resseqs) > 5 else ''} "
                f"!= expected 1..{seg['length']} monotonic; AF3 layout may have drifted"
            )
        plddt.extend(seg_plddt)

    arr = np.array(plddt, dtype=np.float32)
    if arr.size and float(np.nanmax(arr)) > 100.5:
        raise AssertionError(f"plddt max {float(np.nanmax(arr)):.2f} exceeds 100.5")
    return arr


def extract_pae_from_confidences_json(
    confidences_json_path: Path,
    is_compressed: bool = False,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Load AF3 confidences JSON and return PAE plus token chain/residue ids."""
    from eval_pipeline.metrics import _assert_pae_unit

    confidences_json_path = Path(confidences_json_path)
    if is_compressed:
        import zstandard as zstd

        with zstd.open(confidences_json_path, "rt", encoding="utf-8") as f:
            obj = json.load(f)
    else:
        obj = json.loads(confidences_json_path.read_text())

    pae = np.array(obj["pae"], dtype=np.float32)
    _assert_pae_unit(pae)
    token_chain_ids = list(obj.get("token_chain_ids", []))
    token_res_ids = list(obj.get("token_res_ids", []))
    if len(token_chain_ids) != pae.shape[0] or len(token_res_ids) != pae.shape[0]:
        raise ValueError(
            "AF3 confidences token id arrays length mismatch: "
            f"pae[{pae.shape[0]}], chains[{len(token_chain_ids)}], "
            f"res[{len(token_res_ids)}]; AF3 schema may have changed"
        )
    return pae, token_chain_ids, token_res_ids


NATIVE_SUMMARY_KEYS = (
    "ranking_score",
    "ptm",
    "iptm",
    "has_clash",
    "fraction_disordered",
)


def load_native_summary(summary_json_path: Path) -> dict:
    """Load AF3 summary_confidences plain JSON and return native scalar columns."""
    obj = json.loads(Path(summary_json_path).read_text())
    out = {}
    for key in NATIVE_SUMMARY_KEYS:
        value = obj.get(key, float("nan"))
        if isinstance(value, bool):
            value = 1.0 if value else 0.0
        out[key] = value
    return out


def _sanitize_af3_job_name(job_name: str) -> str:
    sanitized = "".join(
        c for c in str(job_name).replace(" ", "_")
        if c.isalnum() or c == "_"
    )
    return sanitized or "job"


def find_top_ranked_outputs(
    af3_sub_run_output_dir: Path,
    job_name: str,
) -> dict:
    """Locate top-ranked model, summary, and confidence files for one AF3 job."""
    af3_sub_run_output_dir = Path(af3_sub_run_output_dir)
    job_dir = af3_sub_run_output_dir / job_name
    if not job_dir.is_dir():
        sanitized_job_name = _sanitize_af3_job_name(job_name)
        sanitized_job_dir = af3_sub_run_output_dir / sanitized_job_name
        if sanitized_job_name != job_name and sanitized_job_dir.is_dir():
            job_dir = sanitized_job_dir
        else:
            raise FileNotFoundError(
                f"AF3 output dir not found for job {job_name!r} under {af3_sub_run_output_dir}"
            )

    model_candidates = sorted(job_dir.glob("*_model.cif")) + sorted(
        job_dir.glob("*_model.cif.zst")
    )
    if not model_candidates:
        raise FileNotFoundError(f"No top-ranked model cif in {job_dir}")

    model_cif = model_candidates[0]
    is_compressed = model_cif.name.endswith(".cif.zst")
    stem = model_cif.name
    if is_compressed:
        stem = stem[:-len(".zst")]
    stem = stem[:-len("_model.cif")]

    summary = job_dir / f"{stem}_summary_confidences.json"
    confidences = job_dir / (
        f"{stem}_confidences.json.zst" if is_compressed else f"{stem}_confidences.json"
    )
    if not summary.exists():
        raise FileNotFoundError(f"summary plain JSON missing: {summary}")
    if not confidences.exists():
        raise FileNotFoundError(f"confidences JSON missing: {confidences}")

    return {
        "model_cif": model_cif,
        "summary_json": summary,
        "confidences_json": confidences,
        "is_compressed": is_compressed,
    }


def stage_inputs_for_sub_run(
    rows: list[dict],
    staging_dir: Path,
    target_msa_dir: Path,
    seed: int = 42,
) -> list[dict]:
    """Write one AF3 input JSON per manifest row into a per-sub-run staging dir."""
    staging_dir = Path(staging_dir)
    target_msa_dir = Path(target_msa_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    staged: list[dict] = []
    for row in rows:
        target_seqs = str(row["target_seq"]).split(":")
        target_msa_paths: list[str] = []
        for i in range(len(target_seqs)):
            chain_id = chr(ord("B") + i)
            msa_path = target_msa_dir / f"target_chain_{chain_id}.a3m"
            if not msa_path.exists():
                raise FileNotFoundError(
                    f"target MSA missing: {msa_path} for chain {chain_id} "
                    f"(case={row['case']}, design={row['design_id']})"
                )
            target_msa_paths.append(str(msa_path.resolve()))

        job_name = f"{row['case']}__{row['design_id']}"
        json_path = staging_dir / f"{job_name}.json"
        input_obj = build_af3_input_json(
            job_name=job_name,
            target_seqs=target_seqs,
            binder_seq=str(row["binder_seq"]),
            target_msa_paths=target_msa_paths,
            seed=seed,
        )
        write_input_json(input_obj, json_path)
        staged.append({"row": row, "job_name": job_name, "json_path": json_path})
    return staged


def cleanup_for_rerun(
    staging_dir: Path,
    output_dir: Path,
    rows_to_rerun: Optional[Any] = None,
) -> None:
    """Remove stale staging/output files before rerunning AF3 jobs."""
    staging_dir = Path(staging_dir)
    output_dir = Path(output_dir)
    if rows_to_rerun is None:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        return

    if hasattr(rows_to_rerun, "iterrows"):
        iterator = (row for _, row in rows_to_rerun.iterrows())
    elif isinstance(rows_to_rerun, list):
        if not all(isinstance(row, dict) for row in rows_to_rerun):
            raise TypeError("rows_to_rerun list entries must be dict")
        iterator = iter(rows_to_rerun)
    else:
        raise TypeError(
            "rows_to_rerun must be None, DataFrame, or list[dict]; "
            f"got {type(rows_to_rerun).__name__}"
        )

    for row in iterator:
        job_name = f"{row['case']}__{row['design_id']}"
        (staging_dir / f"{job_name}.json").unlink(missing_ok=True)
        job_out = output_dir / job_name
        if job_out.exists():
            shutil.rmtree(job_out)


def run_af3_batched(
    staging_dir: Path,
    output_dir: Path,
    af3_env: dict,
    timeout_s: int = 43200,
    extra_flags: Optional[list[str]] = None,
) -> dict:
    """Launch one AF3 subprocess for all input JSON files in a sub-run staging dir."""
    staging_dir = Path(staging_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        af3_env["AF3_PYTHON"],
        af3_env["AF3_RUN_SCRIPT"],
        f"--input_dir={staging_dir}",
        f"--output_dir={output_dir}",
        f"--model_dir={af3_env['AF3_MODEL_DIR']}",
        f"--db_dir={af3_env['AF3_DB_DIR']}",
        "--run_data_pipeline=False",
        "--force_output_dir",
    ]
    if extra_flags:
        cmd.extend(extra_flags)

    if os.environ.get("AF3_MOCK") == "1":
        return _run_af3_mock_fixture(staging_dir, output_dir)

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            check=False,
            timeout=timeout_s,
            capture_output=True,
            text=True,
        )
        return {
            "exit_code": result.returncode,
            "wall_clock_s": time.time() - t0,
            "stdout_tail": _stream_tail(result.stdout),
            "stderr_tail": _stream_tail(result.stderr),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "exit_code": -1,
            "wall_clock_s": time.time() - t0,
            "stdout_tail": _stream_tail(exc.stdout),
            "stderr_tail": _stream_tail(exc.stderr),
            "timed_out": True,
        }


def _stream_tail(value, max_lines: int = 30) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return "\n".join(str(value).splitlines()[-max_lines:])


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _run_af3_mock_fixture(staging_dir: Path, output_dir: Path) -> dict:
    """Generate minimal AF3-like outputs for local AF3_MOCK end-to-end tests."""
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Structure import Structure
    from Bio.PDB import MMCIFIO

    staging_dir = Path(staging_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(staging_dir.glob("*.json"))
    for json_path in json_paths:
        job_name = _sanitize_af3_job_name(json_path.stem)
        input_obj = json.loads(json_path.read_text())
        job_out = output_dir / job_name
        job_out.mkdir(parents=True, exist_ok=True)

        structure = Structure("mock")
        model = Model(0)
        structure.add(model)
        token_chain_ids: list[str] = []
        token_res_ids: list[int] = []
        for entry in input_obj["sequences"]:
            protein = entry.get("protein")
            if not protein:
                continue
            chain_id = protein["id"]
            sequence = protein["sequence"]
            chain = Chain(chain_id)
            model.add(chain)
            for i, _aa in enumerate(sequence, start=1):
                residue = Residue((" ", i, " "), "ALA", "")
                residue.add(Atom("N", (i * 3.8, 0.0, 0.0), 80.0, 1.0, " ", " N  ", 1, "N"))
                residue.add(Atom("CA", (i * 3.8 + 1.0, 0.0, 0.0), 80.0, 1.0, " ", " CA ", 2, "C"))
                chain.add(residue)
                token_chain_ids.append(chain_id)
                token_res_ids.append(i)

        cif_path = job_out / f"{job_name}_model.cif"
        io = MMCIFIO()
        io.set_structure(structure)
        io.save(str(cif_path))

        n_tokens = len(token_chain_ids)
        confidences = {
            "atom_chain_ids": token_chain_ids,
            "atom_plddts": [80.0] * n_tokens,
            "contact_probs": [[0.0] * n_tokens for _ in range(n_tokens)],
            "pae": [[1.0] * n_tokens for _ in range(n_tokens)],
            "token_chain_ids": token_chain_ids,
            "token_res_ids": token_res_ids,
        }
        (job_out / f"{job_name}_confidences.json").write_text(json.dumps(confidences))
        summary = {
            "ranking_score": 0.5,
            "ptm": 0.7,
            "iptm": 0.5,
            "has_clash": False,
            "fraction_disordered": 0.1,
        }
        (job_out / f"{job_name}_summary_confidences.json").write_text(json.dumps(summary))

    return {
        "exit_code": 0,
        "wall_clock_s": 0.1,
        "stdout_tail": f"AF3_MOCK: generated {len(json_paths)} fixture outputs",
        "stderr_tail": "",
        "timed_out": False,
    }


@dataclass
class AF3Result:
    pdb_path: str
    plddt: np.ndarray
    pae: np.ndarray
    clean_segments: list[dict]
    native: dict
    runtime_s: float
    raw_output_subdir: str


def parse_one_af3_design(
    af3_sub_run_output_dir: Path,
    staged_design: dict,
    target_seqs: list[str],
    binder_seq: str,
    amortized_runtime_s: float,
    out_dir: Path,
) -> AF3Result:
    """Parse one AF3 design output and dump per-design artifacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    job_name = staged_design["job_name"]

    paths = find_top_ranked_outputs(af3_sub_run_output_dir, job_name)
    is_compressed = paths["is_compressed"]

    pdb_out = out_dir / "af3_pred.pdb"
    mmcif_to_pdb(paths["model_cif"], pdb_out, decompress_first=is_compressed)

    clean_segments = build_clean_segments_af3(binder_seq=binder_seq, target_seqs=target_seqs)
    plddt = extract_residue_plddt_from_pdb(pdb_out, clean_segments)
    pae, token_chain_ids, token_res_ids = extract_pae_from_confidences_json(
        paths["confidences_json"],
        is_compressed=is_compressed,
    )
    native = load_native_summary(paths["summary_json"])

    np.save(out_dir / "plddt.npy", plddt)
    np.save(out_dir / "pae.npy", pae)
    meta = {
        "evaluator": "af3",
        "case": staged_design["row"]["case"],
        "design_id": staged_design["row"]["design_id"],
        "clean_segments": clean_segments,
        "native": native,
        "token_chain_ids": token_chain_ids,
        "token_res_ids": token_res_ids,
        "is_compressed": is_compressed,
        "amortized_runtime_s": amortized_runtime_s,
    }
    (out_dir / "meta.json").write_text(json.dumps(_to_jsonable(meta), indent=2))

    staged_json = Path(staged_design["json_path"])
    if staged_json.exists():
        shutil.copy(staged_json, out_dir / "input.json")

    return AF3Result(
        pdb_path=str(pdb_out),
        plddt=plddt,
        pae=pae,
        clean_segments=clean_segments,
        native=native,
        runtime_s=amortized_runtime_s,
        raw_output_subdir=str(paths["model_cif"].parent),
    )
