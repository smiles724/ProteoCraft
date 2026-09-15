"""Pinned AtlasFold-M sequence-only cross-evaluator adapter.

The official AtlasFold checkout and its environment remain external to
ProteoCraft.  This module stages manifest rows for the official multimer CLI,
checks source provenance, and converts the resulting files into the metric
pipeline's chain and array conventions.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from eval_pipeline import af3_runner, esmfold2_runner
from eval_pipeline.metrics import _assert_pae_unit


EVALUATOR_NAME = "atlasfold_m"
MODEL_NAME = "atlasfold-m-260725"
MODEL_REPO_ID = "SeonghwanSeo/atlasfold-m-260725"
MODEL_REVISION = "a9f07668b7ef544d4470795ef03aa492f243a461"
LM_REPO_ID = "SeonghwanSeo/atlaslm-3b-base"
LM_REVISION = "b4e6d8abb92d3432e3a65d37f1cfafc18575f78e"
UPSTREAM_VERSION = "1.0.2"
UPSTREAM_COMMIT = "444f376d85b9954a5f2f5f3f8b3cbcae1201ebb1"

DEFAULT_NUM_RECYCLES = 10
DEFAULT_MLM_PROB = 0.20
DEFAULT_NUM_SAMPLES = 5
DEFAULT_SEEDS = (1,)
DEFAULT_NUM_STEPS = 200
DEFAULT_MAX_TOKENS_PER_BATCH = 1024
DEFAULT_KERNEL = "auto"

STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class ChainLayout:
    """AtlasFold chain order for one manifest row: targets, then binder."""

    sequences: tuple[str, ...]
    chain_ids: tuple[str, ...]
    target_roles: tuple[str, ...]
    clean_segments: tuple[dict[str, Any], ...]

    @property
    def target_chain_ids(self) -> tuple[str, ...]:
        return self.chain_ids[:-1]

    @property
    def binder_chain_id(self) -> str:
        return self.chain_ids[-1]

    @property
    def target_metric_chain_ids(self) -> tuple[str, ...]:
        return tuple(
            chain_id
            for chain_id, role in zip(
                self.target_chain_ids, self.target_roles, strict=True
            )
            if role == "target"
        )

    @property
    def scaffold_chain_ids(self) -> tuple[str, ...]:
        return tuple(
            chain_id
            for chain_id, role in zip(
                self.target_chain_ids, self.target_roles, strict=True
            )
            if role == "scaffold"
        )


@dataclass
class AtlasFoldResult:
    pdb_path: str
    mmcif_path: str
    plddt: np.ndarray
    pae: np.ndarray
    pde: np.ndarray
    clean_segments: list[dict[str, Any]]
    native: dict[str, Any]
    runtime_s: float
    confidence_json_path: str
    raw_output_dir: str


def _default_chain_ids(num_chains: int) -> tuple[str, ...]:
    """Match AtlasFold's deterministic chain naming."""

    chain_ids: list[str] = []
    for index in range(num_chains):
        chain_id = ""
        value = index
        while True:
            chain_id = chr(ord("A") + (value % 26)) + chain_id
            value //= 26
            if value == 0:
                break
        chain_ids.append(chain_id)
    return tuple(chain_ids)


def _validate_sequence(sequence: str, label: str) -> str:
    normalized = "".join(str(sequence).split()).upper()
    if not normalized:
        raise ValueError(f"{label} is empty")
    invalid = sorted(set(normalized) - STANDARD_AA)
    if invalid:
        raise ValueError(
            f"{label} contains nonstandard residues {invalid}; refusing AtlasFold's "
            "implicit X replacement"
        )
    return normalized


def build_chain_layout(
    target_seq: str | Sequence[str],
    binder_seq: str,
    target_chain_roles: Any = None,
) -> ChainLayout:
    """Build AtlasFold-M input order and metric-pipeline clean segments."""

    raw_targets = esmfold2_runner.parse_target_sequences(target_seq)
    targets = tuple(
        _validate_sequence(sequence, f"target chain {index}")
        for index, sequence in enumerate(raw_targets)
    )
    binder = _validate_sequence(binder_seq, "binder chain")
    if target_chain_roles is not None and str(target_chain_roles).strip().lower() in {
        "",
        "<na>",
        "nan",
        "none",
    }:
        target_chain_roles = None
    roles = esmfold2_runner.parse_target_chain_roles(target_chain_roles, len(targets))
    sequences = targets + (binder,)
    chain_ids = _default_chain_ids(len(sequences))

    segments: list[dict[str, Any]] = []
    cursor = 0
    for index, (sequence, chain_id) in enumerate(
        zip(targets, chain_ids[:-1], strict=True)
    ):
        end = cursor + len(sequence)
        segments.append(
            {
                "kind": f"target_chain_{index}",
                "role": roles[index],
                "length": len(sequence),
                "dirty_start": cursor,
                "dirty_end": end,
                "clean_start": cursor,
                "clean_end": end,
                "pdb_chain_id": chain_id,
            }
        )
        cursor = end
    binder_end = cursor + len(binder)
    segments.append(
        {
            "kind": "binder",
            "role": "binder",
            "length": len(binder),
            "dirty_start": cursor,
            "dirty_end": binder_end,
            "clean_start": cursor,
            "clean_end": binder_end,
            "pdb_chain_id": chain_ids[-1],
        }
    )
    return ChainLayout(
        sequences=sequences,
        chain_ids=chain_ids,
        target_roles=roles,
        clean_segments=tuple(segments),
    )


def _safe_job_name(row: dict[str, Any]) -> str:
    identity = "\0".join(
        str(row.get(key, ""))
        for key in (
            "method",
            "case",
            "design_id",
            "target_seq",
            "target_chain_roles",
            "binder_seq",
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    prefix = "__".join(
        str(row.get(key, "unknown")) for key in ("case", "design_id")
    )
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix).strip("._-")
    prefix = prefix[:72] or "design"
    return f"{prefix}__{digest}"


def stage_inputs_for_sub_run(
    *, rows: list[dict[str, Any]], staging_dir: Path
) -> list[dict[str, Any]]:
    """Write one official multimer FASTA plus an auditable chain mapping."""

    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    fasta_lines: list[str] = []

    for row in rows:
        layout = build_chain_layout(
            row["target_seq"],
            row["binder_seq"],
            row.get("target_chain_roles"),
        )
        job_name = _safe_job_name(row)
        if job_name in seen_names:
            raise ValueError(f"duplicate staged AtlasFold job name: {job_name}")
        seen_names.add(job_name)
        fasta_lines.extend([f">{job_name}", ":".join(layout.sequences)])
        staged.append({"row": row, "job_name": job_name, "layout": layout})

    fasta_path = staging_dir / "atlasfold_m_inputs.fasta"
    fasta_path.write_text("\n".join(fasta_lines) + ("\n" if fasta_lines else ""))
    mapping = {
        "schema_version": 1,
        "evaluator": EVALUATOR_NAME,
        "model": MODEL_NAME,
        "upstream_version": UPSTREAM_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "chain_order_contract": "target/scaffold chains in manifest order, binder last",
        "jobs": [
            {
                "job_name": item["job_name"],
                "method": item["row"].get("method"),
                "case": item["row"].get("case"),
                "design_id": item["row"].get("design_id"),
                "chain_ids": list(item["layout"].chain_ids),
                "chain_lengths": [len(seq) for seq in item["layout"].sequences],
                "target_chain_roles": list(item["layout"].target_roles),
                "binder_chain_id": item["layout"].binder_chain_id,
                "sequence_sha256": [
                    hashlib.sha256(seq.encode("ascii")).hexdigest()
                    for seq in item["layout"].sequences
                ],
            }
            for item in staged
        ],
    }
    (staging_dir / "chain_mapping.json").write_text(
        json.dumps(mapping, indent=2, sort_keys=True) + "\n"
    )
    return staged


def verify_source_checkout(source_root: Path) -> dict[str, Any]:
    """Require the reviewed AtlasFold release and reject tracked modifications."""

    source_root = Path(source_root).resolve()
    if not (source_root / "src" / "atlasfold").is_dir():
        raise ValueError(f"not an AtlasFold source checkout: {source_root}")
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(
            f"AtlasFold source commit mismatch: {commit}; expected {UPSTREAM_COMMIT}"
        )
    for args in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        status = subprocess.run(args, cwd=source_root, check=False)
        if status.returncode != 0:
            raise ValueError(f"AtlasFold checkout has tracked modifications: {source_root}")

    pyproject = (source_root / "pyproject.toml").read_text()
    version_match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    version = version_match.group(1) if version_match else None
    if version != UPSTREAM_VERSION:
        raise ValueError(
            f"AtlasFold source version mismatch: {version!r}; expected {UPSTREAM_VERSION}"
        )
    return {
        "source_root": str(source_root),
        "source_commit": commit,
        "source_version": version,
    }


def _subprocess_env(source_root: Path, cache_dir: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    source_path = str(Path(source_root).resolve() / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = source_path if not existing else source_path + os.pathsep + existing
    if cache_dir is not None:
        env["HF_HOME"] = str(Path(cache_dir).resolve())
    return env


def probe_environment(atlasfold_python: str, source_root: Path) -> dict[str, Any]:
    """Verify that the external Python imports the pinned source checkout."""

    probe = """
import importlib.metadata as metadata
import json
import pathlib
import platform
import atlasfold
import torch

packages = {}
for name in (
    "atlasfold", "torch", "numpy", "huggingface-hub", "scipy", "einops",
    "gemmi", "omegaconf", "numba", "cuequivariance-torch",
):
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None
print(json.dumps({
    "version": packages["atlasfold"],
    "module_file": str(pathlib.Path(atlasfold.__file__).resolve()),
    "python": platform.python_version(),
    "packages": packages,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
}))
"""
    result = subprocess.run(
        [str(atlasfold_python), "-c", probe],
        env=_subprocess_env(source_root),
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("AtlasFold environment probe produced no JSON")
    metadata = json.loads(lines[-1])
    expected_src = (Path(source_root).resolve() / "src").resolve()
    module_file = Path(metadata["module_file"]).resolve()
    if metadata.get("version") != UPSTREAM_VERSION:
        raise ValueError(
            f"installed AtlasFold version {metadata.get('version')!r}; "
            f"expected {UPSTREAM_VERSION}"
        )
    if expected_src not in module_file.parents:
        raise ValueError(
            f"AtlasFold imported from {module_file}, not pinned source {expected_src}"
        )
    return metadata


def validate_weights_receipt(
    receipt_path: Path, *, model_path: Path, lm_path: Path
) -> dict[str, Any]:
    """Require local weights tied to immutable Hugging Face snapshot commits."""

    receipt_path = Path(receipt_path).resolve()
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema_version") != 1:
        raise ValueError(
            f"unsupported AtlasFold weights receipt schema: "
            f"{receipt.get('schema_version')!r}"
        )
    downloads = receipt.get("downloads")
    if not isinstance(downloads, dict):
        raise ValueError(f"invalid AtlasFold weights receipt: {receipt_path}")
    expected = {
        "atlasfold_m": (
            MODEL_REPO_ID,
            MODEL_REVISION,
            Path(model_path).resolve(),
        ),
        "atlaslm_3b": (
            LM_REPO_ID,
            LM_REVISION,
            Path(lm_path).resolve(),
        ),
    }
    for key, (repo_id, expected_revision, expected_path) in expected.items():
        entry = downloads.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"weights receipt lacks {key}")
        if entry.get("repo_id") != repo_id:
            raise ValueError(
                f"weights receipt {key} repo {entry.get('repo_id')!r}; expected {repo_id}"
            )
        resolved_revision = str(entry.get("resolved_revision", ""))
        if not re.fullmatch(r"[0-9a-f]{40}", resolved_revision):
            raise ValueError(
                f"weights receipt {key} has non-immutable revision {resolved_revision!r}"
            )
        if resolved_revision != expected_revision:
            raise ValueError(
                f"weights receipt {key} revision {resolved_revision}; "
                f"expected {expected_revision}"
            )
        receipt_weight_path = Path(entry.get("weight_path", "")).resolve()
        if receipt_weight_path != expected_path:
            raise ValueError(
                f"weights receipt {key} path {receipt_weight_path} does not match "
                f"configured {expected_path}"
            )
        if not expected_path.is_file():
            raise FileNotFoundError(f"AtlasFold weight file missing: {expected_path}")
        expected_size = entry.get("weight_size_bytes")
        if expected_size is not None and expected_path.stat().st_size != int(expected_size):
            raise ValueError(f"AtlasFold weight size changed: {expected_path}")
    receipt["receipt_path"] = str(receipt_path)
    return receipt


def build_command(
    *,
    atlasfold_python: str,
    fasta_path: Path,
    output_dir: Path,
    device: str,
    num_recycles: int = DEFAULT_NUM_RECYCLES,
    mlm_prob: float = DEFAULT_MLM_PROB,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    num_steps: int = DEFAULT_NUM_STEPS,
    max_tokens_per_batch: int = DEFAULT_MAX_TOKENS_PER_BATCH,
    kernel: str = DEFAULT_KERNEL,
    cache_dir: Path | None = None,
    model_path: Path | None = None,
    lm_path: Path | None = None,
    overwrite: bool = False,
) -> list[str]:
    if not seeds:
        raise ValueError("AtlasFold-M requires at least one seed")
    command = [
        str(atlasfold_python),
        "-m",
        "atlasfold.cli",
        "multimer",
        "--input-fasta",
        str(fasta_path),
        "--out-dir",
        str(output_dir),
        "--device",
        str(device),
        "--kernel",
        str(kernel),
        "--num-recycles",
        str(int(num_recycles)),
        "--mlm-prob",
        str(float(mlm_prob)),
        "--num-samples",
        str(int(num_samples)),
        "--num-steps",
        str(int(num_steps)),
        "--max-tokens-per-batch",
        str(int(max_tokens_per_batch)),
        "--seed",
        *[str(int(seed)) for seed in seeds],
        "--format",
        "cif",
        "--save-confidence-arrays",
    ]
    if overwrite:
        command.append("--overwrite")
    for flag, value in (
        ("--cache-dir", cache_dir),
        ("--model-path", model_path),
        ("--lm-path", lm_path),
    ):
        if value is not None:
            command.extend([flag, str(value)])
    return command


def cleanup_for_rerun(
    staging_dir: Path,
    output_dir: Path,
    *,
    reset_outputs: bool,
) -> None:
    shutil.rmtree(staging_dir, ignore_errors=True)
    if reset_outputs:
        shutil.rmtree(output_dir, ignore_errors=True)


def _mock_write_outputs(fasta_path: Path, output_dir: Path) -> None:
    """Write official-layout synthetic outputs for CPU integration tests only."""

    from Bio.PDB import MMCIFIO
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Polypeptide import protein_letters_3to1

    protein_letters_1to3 = {
        one_letter: three_letter
        for three_letter, one_letter in protein_letters_3to1.items()
    }

    records: list[tuple[str, str]] = []
    name: str | None = None
    parts: list[str] = []
    for raw_line in Path(fasta_path).read_text().splitlines():
        line = raw_line.strip()
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(parts)))
            name = line[1:].split()[0]
            parts = []
        elif line:
            parts.append(line)
    if name is not None:
        records.append((name, "".join(parts)))

    for job_name, sequence_spec in records:
        sequences = sequence_spec.split(":")
        chain_ids = _default_chain_ids(len(sequences))
        total_length = sum(map(len, sequences))
        structure = Structure(job_name)
        model = Model(0)
        structure.add(model)
        atom_serial = 1
        for chain_index, (chain_id, sequence) in enumerate(
            zip(chain_ids, sequences, strict=True)
        ):
            chain = Chain(chain_id)
            model.add(chain)
            for residue_index, aa in enumerate(sequence, start=1):
                residue = Residue(
                    (" ", residue_index, " "), protein_letters_1to3[aa], ""
                )
                x = (residue_index - 1) * 3.8
                y = chain_index * 4.0
                for atom_name, dx, element in (
                    ("N", 0.0, "N"),
                    ("CA", 1.3, "C"),
                    ("C", 2.5, "C"),
                    ("O", 3.1, "O"),
                ):
                    residue.add(
                        Atom(
                            atom_name,
                            np.array([x + dx, y, 0.0], dtype=float),
                            90.0,
                            1.0,
                            " ",
                            f" {atom_name:<3}",
                            atom_serial,
                            element,
                        )
                    )
                    atom_serial += 1
                chain.add(residue)

        job_dir = Path(output_dir) / job_name
        job_dir.mkdir(parents=True, exist_ok=True)
        sample_stem = f"{job_name}_seed-1_sample-0"
        sample_cif = job_dir / f"{sample_stem}_model.cif"
        io = MMCIFIO()
        io.set_structure(structure)
        io.save(str(sample_cif))
        ranked_cif = job_dir / f"{job_name}_ranked_model.cif"
        shutil.copy2(sample_cif, ranked_cif)

        chain_scores = {
            chain_id: {"avg_plddt": 90.0, "avg_pde": 1.5, "ptm": 0.82}
            for chain_id in chain_ids
        }
        interfaces = {
            f"{left}-{right}": {"iptm": 0.75}
            for index, left in enumerate(chain_ids)
            for right in chain_ids[index + 1 :]
        }
        confidence = {
            "complex": {
                "avg_plddt": 90.0,
                "avg_pde": 1.5,
                "ptm": 0.82,
                "iptm": 0.75,
                "fraction_disordered": 0.02,
                "ranking_score": 0.774,
            },
            "chains": chain_scores,
            "interfaces": interfaces,
        }
        confidence_text = json.dumps(confidence, indent=2) + "\n"
        (job_dir / f"{sample_stem}_confidence.json").write_text(confidence_text)
        (job_dir / f"{job_name}_ranked_confidence.json").write_text(confidence_text)
        plddt = np.full(total_length, 0.9, dtype=np.float32)
        pae = np.full((total_length, total_length), 4.0, dtype=np.float32)
        np.fill_diagonal(pae, 0.0)
        pde = np.full((total_length, total_length), 1.5, dtype=np.float32)
        np.savez(
            job_dir / f"{sample_stem}_confidence.npz",
            plddt=plddt,
            pae=pae,
            pde=pde,
        )
        (job_dir / f"{job_name}_summary.csv").write_text(
            "seed,sample_index,ranking_score,ptm,iptm,"
            "fraction_disordered,avg_pde,avg_plddt\n"
            "1,0,0.774,0.82,0.75,0.02,1.5,90.0\n"
        )
        (job_dir / "done.txt").touch()


def run_atlasfold_batched(
    *,
    staging_dir: Path,
    output_dir: Path,
    atlasfold_python: str,
    source_root: Path,
    device: str,
    timeout_s: int,
    num_recycles: int = DEFAULT_NUM_RECYCLES,
    mlm_prob: float = DEFAULT_MLM_PROB,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    num_steps: int = DEFAULT_NUM_STEPS,
    max_tokens_per_batch: int = DEFAULT_MAX_TOKENS_PER_BATCH,
    kernel: str = DEFAULT_KERNEL,
    cache_dir: Path | None = None,
    model_path: Path | None = None,
    lm_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run one batched official AtlasFold-M CLI process."""

    staging_dir = Path(staging_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = staging_dir / "atlasfold_m_inputs.fasta"
    command = build_command(
        atlasfold_python=atlasfold_python,
        fasta_path=fasta_path,
        output_dir=output_dir,
        device=device,
        num_recycles=num_recycles,
        mlm_prob=mlm_prob,
        num_samples=num_samples,
        seeds=seeds,
        num_steps=num_steps,
        max_tokens_per_batch=max_tokens_per_batch,
        kernel=kernel,
        cache_dir=cache_dir,
        model_path=model_path,
        lm_path=lm_path,
        overwrite=overwrite,
    )
    (output_dir / "adapter_command.json").write_text(
        json.dumps(command, indent=2) + "\n"
    )
    started = time.perf_counter()
    timed_out = False
    if os.environ.get("ATLASFOLD_MOCK") == "1":
        _mock_write_outputs(fasta_path, output_dir)
        completed = subprocess.CompletedProcess(command, 0, "ATLASFOLD_MOCK=1\n", "")
    else:
        try:
            completed = subprocess.run(
                command,
                env=_subprocess_env(source_root, cache_dir),
                check=False,
                capture_output=True,
                text=True,
                timeout=int(timeout_s),
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            completed = subprocess.CompletedProcess(
                command,
                124,
                exc.stdout or "",
                exc.stderr or "",
            )
    wall_clock_s = time.perf_counter() - started
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    (output_dir / "adapter_stdout.log").write_text(stdout)
    (output_dir / "adapter_stderr.log").write_text(stderr)
    result = {
        "exit_code": int(completed.returncode),
        "wall_clock_s": float(wall_clock_s),
        "timed_out": timed_out,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }
    (output_dir / "adapter_run_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def _load_best_sample(job_dir: Path, job_name: str) -> tuple[dict[str, str], dict[str, Any]]:
    summary_path = job_dir / f"{job_name}_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"AtlasFold summary missing: {summary_path}")
    with summary_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"AtlasFold summary has no samples: {summary_path}")
    best = max(rows, key=lambda row: float(row["ranking_score"]))
    seed = int(best["seed"])
    sample_index = int(best["sample_index"])
    sample_stem = f"{job_name}_seed-{seed}_sample-{sample_index}"
    paths = {
        "summary": str(summary_path),
        "ranked_cif": str(job_dir / f"{job_name}_ranked_model.cif"),
        "ranked_confidence": str(job_dir / f"{job_name}_ranked_confidence.json"),
        "sample_confidence": str(job_dir / f"{sample_stem}_confidence.json"),
        "sample_npz": str(job_dir / f"{sample_stem}_confidence.npz"),
    }
    for label, path in paths.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"AtlasFold {label} missing: {path}")
    return paths, best


def _structure_sequences(pdb_path: Path) -> dict[str, str]:
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import protein_letters_3to1

    model = PDBParser(QUIET=True).get_structure("atlasfold", str(pdb_path))[0]
    observed: dict[str, str] = {}
    for chain in model:
        residues = [residue for residue in chain if residue.id[0] == " "]
        observed[chain.id] = "".join(
            protein_letters_3to1.get(residue.resname.upper(), "X")
            for residue in residues
        )
    return observed


def _mean_chain_plddt(
    plddt_percent: np.ndarray, layout: ChainLayout, chain_ids: Sequence[str]
) -> float:
    selected = set(chain_ids)
    indices = [
        np.arange(segment["clean_start"], segment["clean_end"])
        for segment in layout.clean_segments
        if segment["pdb_chain_id"] in selected
    ]
    if not indices:
        return float("nan")
    return float(plddt_percent[np.concatenate(indices)].mean())


def _interface_value(interfaces: dict[str, Any], left: str, right: str) -> float:
    key = f"{left}-{right}"
    reverse = f"{right}-{left}"
    entry = interfaces.get(key, interfaces.get(reverse))
    if not isinstance(entry, dict) or "iptm" not in entry:
        raise ValueError(f"AtlasFold confidence lacks interface {left}-{right}")
    return float(entry["iptm"])


def parse_one_atlasfold_design(
    *,
    atlasfold_output_dir: Path,
    staged_design: dict[str, Any],
    amortized_runtime_s: float,
    out_dir: Path,
    run_config: dict[str, Any],
) -> AtlasFoldResult:
    """Parse, validate, and persist one AtlasFold-M prediction."""

    job_name = staged_design["job_name"]
    layout: ChainLayout = staged_design["layout"]
    job_dir = Path(atlasfold_output_dir) / job_name
    if not (job_dir / "done.txt").is_file():
        raise FileNotFoundError(f"AtlasFold completion marker missing: {job_dir / 'done.txt'}")
    paths, best_summary = _load_best_sample(job_dir, job_name)
    ranked_confidence = json.loads(Path(paths["ranked_confidence"]).read_text())
    sample_confidence = json.loads(Path(paths["sample_confidence"]).read_text())
    if ranked_confidence != sample_confidence:
        raise ValueError("ranked confidence does not match summary-selected sample")

    with np.load(paths["sample_npz"]) as arrays:
        missing = sorted({"plddt", "pae", "pde"} - set(arrays.files))
        if missing:
            raise ValueError(f"AtlasFold confidence NPZ missing arrays: {missing}")
        plddt_raw = np.asarray(arrays["plddt"], dtype=np.float32)
        pae = np.asarray(arrays["pae"], dtype=np.float32)
        pde = np.asarray(arrays["pde"], dtype=np.float32)
    total_length = sum(len(sequence) for sequence in layout.sequences)
    if plddt_raw.shape != (total_length,):
        raise ValueError(
            f"AtlasFold pLDDT shape {plddt_raw.shape}, expected {(total_length,)}"
        )
    if pae.shape != (total_length, total_length):
        raise ValueError(
            f"AtlasFold PAE shape {pae.shape}, expected {(total_length, total_length)}"
        )
    if pde.shape != (total_length, total_length):
        raise ValueError(
            f"AtlasFold PDE shape {pde.shape}, expected {(total_length, total_length)}"
        )
    if plddt_raw.size and (float(plddt_raw.min()) < 0.0 or float(plddt_raw.max()) > 1.001):
        raise AssertionError("AtlasFold NPZ pLDDT must use the documented 0-1 scale")
    _assert_pae_unit(pae)
    plddt_percent = plddt_raw * 100.0

    complex_scores = ranked_confidence.get("complex")
    chain_scores = ranked_confidence.get("chains")
    interfaces = ranked_confidence.get("interfaces")
    if not isinstance(complex_scores, dict):
        raise ValueError("AtlasFold confidence lacks complex metrics")
    if not isinstance(chain_scores, dict) or not isinstance(interfaces, dict):
        raise ValueError("AtlasFold confidence lacks chain/interface metrics")
    if set(chain_scores) != set(layout.chain_ids):
        raise ValueError(
            f"AtlasFold confidence chain IDs {sorted(chain_scores)} do not match "
            f"expected {list(layout.chain_ids)}"
        )
    if not np.isclose(
        float(complex_scores["avg_plddt"]), float(plddt_percent.mean()), atol=1e-3
    ):
        raise ValueError("AtlasFold JSON and NPZ complex pLDDT disagree")
    for key in (
        "ranking_score",
        "ptm",
        "iptm",
        "fraction_disordered",
        "avg_pde",
        "avg_plddt",
    ):
        if key in best_summary and not np.isclose(
            float(best_summary[key]), float(complex_scores[key]), atol=1e-6
        ):
            raise ValueError(f"AtlasFold summary and ranked confidence disagree for {key}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mmcif_path = out_dir / "atlasfold_m.cif"
    pdb_path = out_dir / "atlasfold_m.pdb"
    confidence_path = out_dir / "atlasfold_m_confidence.json"
    confidence_npz_path = out_dir / "atlasfold_m_confidence.npz"
    summary_path = out_dir / "atlasfold_m_summary.csv"
    shutil.copy2(paths["ranked_cif"], mmcif_path)
    shutil.copy2(paths["ranked_confidence"], confidence_path)
    shutil.copy2(paths["sample_npz"], confidence_npz_path)
    shutil.copy2(paths["summary"], summary_path)
    af3_runner.mmcif_to_pdb(mmcif_path, pdb_path)

    observed_sequences = _structure_sequences(pdb_path)
    expected_sequences = dict(zip(layout.chain_ids, layout.sequences, strict=True))
    if observed_sequences != expected_sequences:
        raise ValueError(
            "AtlasFold structure chain/sequence mismatch: "
            f"observed={observed_sequences}, expected={expected_sequences}"
        )

    target_interface_values = [
        _interface_value(interfaces, chain_id, layout.binder_chain_id)
        for chain_id in layout.target_metric_chain_ids
    ]
    scaffold_interface_values = [
        _interface_value(interfaces, chain_id, layout.binder_chain_id)
        for chain_id in layout.scaffold_chain_ids
    ]
    native = {
        "ranking_score": float(complex_scores["ranking_score"]),
        "pair_chain_iptm": float(np.mean(target_interface_values)),
        "scaffold_pair_chain_iptm": (
            float(np.mean(scaffold_interface_values))
            if scaffold_interface_values
            else float("nan")
        ),
        "iptm": float(complex_scores["iptm"]),
        "ptm": float(complex_scores["ptm"]),
        "complex_plddt": float(complex_scores["avg_plddt"]),
        "target_plddt": _mean_chain_plddt(
            plddt_percent, layout, layout.target_metric_chain_ids
        ),
        "binder_plddt": _mean_chain_plddt(
            plddt_percent, layout, [layout.binder_chain_id]
        ),
        "binder_ptm": float(chain_scores[layout.binder_chain_id]["ptm"]),
        "avg_pde": float(complex_scores["avg_pde"]),
        "fraction_disordered": float(complex_scores["fraction_disordered"]),
        "interface_iptm": interfaces,
        "selected_seed": int(best_summary["seed"]),
        "selected_sample_index": int(best_summary["sample_index"]),
    }
    provenance = {
        "schema_version": 1,
        "evaluator": EVALUATOR_NAME,
        "model": MODEL_NAME,
        "upstream_version": UPSTREAM_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "job_name": job_name,
        "raw_output_dir": str(job_dir),
        "chain_ids": list(layout.chain_ids),
        "target_chain_roles": list(layout.target_roles),
        "run_config": run_config,
        "native_metrics": native,
    }
    (out_dir / "atlasfold_m_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    return AtlasFoldResult(
        pdb_path=str(pdb_path),
        mmcif_path=str(mmcif_path),
        plddt=plddt_percent,
        pae=pae,
        pde=pde,
        clean_segments=[dict(segment) for segment in layout.clean_segments],
        native=native,
        runtime_s=float(amortized_runtime_s),
        confidence_json_path=str(confidence_path),
        raw_output_dir=str(job_dir),
    )
