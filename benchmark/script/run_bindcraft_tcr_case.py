#!/usr/bin/env python3
"""Run one BindCraft-v2 TCR binder case directly, without benchmark wrapper."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import traceback
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bindcraft.hallucination.pipeline import run_hybrid_hallucination  # noqa: E402
from bindcraft.utils import configure_protenix_common_paths  # noqa: E402
from protenix.data.utils import pdb_to_cif  # noqa: E402
from scripts.run_hallucination import load_protenix_model  # noqa: E402


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
    "MSE": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

DEFAULT_INIT_BINDER = (
    "AAAQPAAVTAAVGATATLTCTLSAAAAGVVTWTKVATGATLAAAATSASSASLTLAAVTAADAGLYVCTVTDAAGNV"
    "VAAGSALLTVT"
)

DEFAULT_LOSS_WEIGHTS = {
    "plddt": 0.1,
    "pae": 0.4,
    "i_pae": 0.1,
    "con": 1.0,
    "i_con": 3.0,
    "iptm": 0.05,
    "rg": 0.3,
    "helix": -0.3,
    "clash": 0.5,
    "i_clash": 0.5,
    "pde": 0.02,
    "i_pde": 0.01,
}

LOSS_WEIGHT_KEYS = tuple(DEFAULT_LOSS_WEIGHTS)


def setup_logging(log_file: Optional[Path]) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def add_bool_argument(parser: argparse.ArgumentParser, name: str) -> None:
    """Add a boolean flag with --no- prefix for disabling."""
    dest = name.replace("-", "_")
    parser.add_argument(
        f"--{name}",
        dest=dest,
        action="store_true",
        default=argparse.SUPPRESS,
        help=f"Enable {name}",
    )
    parser.add_argument(
        f"--no-{name}",
        dest=dest,
        action="store_false",
        default=argparse.SUPPRESS,
        help=f"Disable {name}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb-id", required=True)
    parser.add_argument("--pdb-path")
    parser.add_argument("--alpha-chain")
    parser.add_argument("--beta-chain")
    parser.add_argument("--hotspot-chain", choices=("alpha", "beta"), default="alpha")
    parser.add_argument("--hotspot-start", type=int)
    parser.add_argument("--hotspot-end", type=int)
    parser.add_argument(
        "--no-hotspot-params",
        action="store_true",
        help="Do not pass target hotspot indices into the loss function.",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--init-binder-seq", default=DEFAULT_INIT_BINDER)
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "outputs" / "bindcraft_v2_tcr_cases"))
    parser.add_argument("--pdb-cache-dir", default=str(PROJECT_ROOT / "inputs" / "pdb_cache"))
    parser.add_argument("--model-checkpoint-dir", default="/data/huangyanjie/Alphaenzyme/LGOS/checkpoints")
    parser.add_argument("--components-cache-dir", default="/data/huangyanjie/Alphaenzyme/LGOS/protenix/release_data/ccd_cache")
    parser.add_argument("--protenix-root-dir", default=str(PROJECT_ROOT / "benchmark" / "protenix_runtime"))
    parser.add_argument("--model-name", default="protenix_base_20250630_v1.0.0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task-type", default=None)
    parser.add_argument("--target-chain-roles", default=None, help="逗号分隔，如 peptide,hla")
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--tau-noise", type=float, default=4.0)
    parser.add_argument("--k-top", type=int, default=8)
    parser.add_argument("--b-candidates", type=int, default=4)
    parser.add_argument("--t-sample", type=float, default=1.0)
    parser.add_argument("--max-mutations-per-seq", type=int, default=3)
    parser.add_argument("--t-init", type=float, default=1.0)
    parser.add_argument("--t-min", type=float, default=0.01)
    parser.add_argument("--log-file")
    parser.add_argument("--no-download", action="store_true")
    # Boolean flags for annealing and tau scheduling
    add_bool_argument(parser, "use-annealing")
    add_bool_argument(parser, "use-tau-scheduling")
    add_bool_argument(parser, "tau-jitter")
    # Loss weights
    for key in LOSS_WEIGHT_KEYS:
        parser.add_argument(f"--w-{key.replace('_', '-')}", dest=f"w_{key}", type=float)
    return parser.parse_args()


def resolve_bool_arg(args: argparse.Namespace, name: str, default: bool) -> bool:
    """Resolve a boolean argument, returning default if not set."""
    dest = name.replace("-", "_")
    value = getattr(args, dest, None)
    return default if value is None else value


def resolve_loss_weights(args: argparse.Namespace) -> Dict[str, float]:
    weights = dict(DEFAULT_LOSS_WEIGHTS)
    for key in LOSS_WEIGHT_KEYS:
        value = getattr(args, f"w_{key}")
        if value is not None:
            weights[key] = value
    return weights


def ensure_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() and Path(os.path.realpath(dst)) == src.resolve():
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def prepare_runtime_root(args: argparse.Namespace) -> Path:
    root = Path(args.protenix_root_dir)
    common = root / "common"
    common.mkdir(parents=True, exist_ok=True)
    ensure_link(Path(args.components_cache_dir) / "components.cif", common / "components.cif")
    ensure_link(
        Path(args.components_cache_dir) / "components.cif.rdkit_mol.pkl",
        common / "components.cif.rdkit_mol.pkl",
    )
    ensure_link(Path(args.model_checkpoint_dir), root / "checkpoint")
    os.environ["PROTENIX_ROOT_DIR"] = str(root)
    os.environ["PROTENIX_CHECKPOINT_DIR"] = args.model_checkpoint_dir
    configure_protenix_common_paths(str(common))
    return root


def candidate_pdb_paths(pdb_id: str, cache_dir: Path) -> Iterable[Path]:
    names = [f"{pdb_id}.pdb", f"{pdb_id.lower()}.pdb", f"{pdb_id.upper()}.pdb"]
    roots = [
        cache_dir,
        PROJECT_ROOT / "inputs",
        PROJECT_ROOT / "examples",
        Path("/data/huangyanjie/binder_design/rf3_binder_design/input_pdbs"),
        Path("/data/huangyanjie/binder_design/AF3Score/input"),
    ]
    extra = os.environ.get("BINDCRAFT_PDB_DIRS")
    if extra:
        roots.extend(Path(part) for part in extra.split(":") if part)
    for root in roots:
        for name in names:
            yield root / name


def download_pdb(pdb_id: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdb_id = pdb_id.upper()
    out = cache_dir / f"{pdb_id}.pdb"
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    logging.info("Downloading %s -> %s", url, out)
    with urllib.request.urlopen(url, timeout=120) as response:
        text = response.read().decode("utf-8")
    if not text.startswith(("HEADER", "TITLE", "COMPND", "ATOM")):
        raise RuntimeError(f"Unexpected response while downloading {pdb_id}")
    out.write_text(text, encoding="utf-8")
    return out


def resolve_pdb_path(args: argparse.Namespace) -> Path:
    if args.pdb_path:
        path = Path(args.pdb_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    cache_dir = Path(args.pdb_cache_dir)
    for path in candidate_pdb_paths(args.pdb_id, cache_dir):
        if path.exists():
            return path

    if args.no_download:
        raise FileNotFoundError(f"No local PDB found for {args.pdb_id}")
    return download_pdb(args.pdb_id, cache_dir)


def parse_compnd_chains(pdb_path: Path) -> Dict[str, str]:
    records: List[str] = []
    for line in pdb_path.read_text(errors="ignore").splitlines():
        if line.startswith("SOURCE"):
            break
        if line.startswith("COMPND"):
            records.append(line[10:].strip())
    text = " ".join(records)
    entries = re.split(r"\bMOL_ID:\s*", text)
    chain_to_molecule: Dict[str, str] = {}
    for entry in entries:
        if not entry.strip():
            continue
        molecule_match = re.search(r"MOLECULE:\s*([^;]+)", entry, flags=re.I)
        chain_match = re.search(r"CHAIN:\s*([^;]+)", entry, flags=re.I)
        if not molecule_match or not chain_match:
            continue
        molecule = molecule_match.group(1).strip()
        chains = [part.strip() for part in chain_match.group(1).split(",")]
        for chain in chains:
            if chain:
                chain_to_molecule[chain] = molecule
    return chain_to_molecule


def extract_chain_residues(pdb_path: Path, chain_id: str) -> List[Dict[str, Any]]:
    residues: List[Dict[str, Any]] = []
    seen = set()
    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            if (line[21].strip() or "_") != chain_id:
                continue
            key = (line[22:26].strip(), line[26].strip())
            if key in seen:
                continue
            seen.add(key)
            residues.append(
                {
                    "residue_number": int(line[22:26].strip()),
                    "insertion_code": line[26].strip(),
                    "aa": AA3_TO_1.get(line[17:20].strip(), "X"),
                }
            )
    if not residues:
        raise ValueError(f"No CA residues found for chain {chain_id} in {pdb_path}")
    return residues


def infer_tcr_chains(pdb_path: Path, alpha_chain: Optional[str], beta_chain: Optional[str]) -> Tuple[str, str]:
    if alpha_chain and beta_chain:
        return alpha_chain, beta_chain
    chain_to_molecule = parse_compnd_chains(pdb_path)
    inferred_alpha = alpha_chain
    inferred_beta = beta_chain
    for chain, molecule in chain_to_molecule.items():
        low = molecule.lower()
        if inferred_alpha is None and "receptor" in low and "alpha" in low:
            inferred_alpha = chain
        if inferred_beta is None and "receptor" in low and "beta" in low:
            inferred_beta = chain
    if inferred_alpha is None or inferred_beta is None:
        raise ValueError(
            "Could not infer TCR alpha/beta chains from COMPND. "
            "Pass --alpha-chain and --beta-chain explicitly."
        )
    return inferred_alpha, inferred_beta


def hotspot_indices(
    residues: List[Dict[str, Any]],
    start: int,
    end: int,
    *,
    label_prefix: str,
    offset: int = 0,
) -> Tuple[List[int], List[str]]:
    if end < start:
        raise ValueError("--hotspot-end must be >= --hotspot-start")
    indices: List[int] = []
    labels: List[str] = []
    for idx, residue in enumerate(residues):
        if start <= int(residue["residue_number"]) <= end:
            indices.append(offset + idx)
            labels.append(f"{label_prefix}{residue['residue_number']}{residue['insertion_code']}")
    expected = end - start + 1
    if len(indices) != expected:
        raise ValueError(
            f"Resolved {len(indices)} hotspot residues for {label_prefix} {start}-{end}, expected {expected}. "
            f"Resolved labels={labels}"
        )
    return indices, labels


def insert_struct_asym_if_missing(cif_text: str) -> str:
    if "_struct_asym.id" in cif_text:
        return cif_text
    pattern = re.compile(
        r"(loop_\n_entity_poly\.entity_id\s*\n_entity_poly\.pdbx_strand_id\s*\n_entity_poly\.type\s*\n"
        r"(?P<rows>(?:[^\n#]+\n)+)#\n)",
        flags=re.M,
    )
    match = pattern.search(cif_text)
    if not match:
        return cif_text
    rows = []
    for raw in match.group("rows").splitlines():
        parts = raw.split()
        if len(parts) < 2:
            continue
        entity_id, strand_field = parts[0], parts[1]
        for chain_id in strand_field.split(","):
            chain_id = chain_id.strip()
            if chain_id:
                rows.append((chain_id, entity_id))
    if not rows:
        return cif_text
    block = "loop_\n_struct_asym.id\n_struct_asym.entity_id\n"
    block += "".join(f"{chain_id} {entity_id}\n" for chain_id, entity_id in rows)
    block += "#\n"
    return cif_text[: match.end()] + block + cif_text[match.end() :]


def prepare_template_cif(pdb_path: Path, output_dir: Path, pdb_id: str) -> Path:
    template_dir = output_dir / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    cif_path = template_dir / f"{pdb_id.upper()}_fixed_template.cif"
    pdb_to_cif(str(pdb_path), str(cif_path))
    text = cif_path.read_text(encoding="utf-8")
    text = text.replace("L-PEPTIDE LINKING", "l-peptide linking")
    text = text.replace("PEPTIDE LINKING", "peptide linking")
    text = insert_struct_asym_if_missing(text)
    cif_path.write_text(text, encoding="utf-8")
    return cif_path


def main() -> int:
    args = parse_args()
    pdb_id = args.pdb_id.upper()
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.no_hotspot_params:
        output_dir = Path(args.output_root) / f"{pdb_id}_no_hotspot"
    else:
        if args.hotspot_start is None or args.hotspot_end is None:
            raise ValueError("--hotspot-start and --hotspot-end are required unless --no-hotspot-params is set")
        output_dir = (
            Path(args.output_root)
            / f"{pdb_id}_{args.hotspot_chain}_{args.hotspot_start}_{args.hotspot_end}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(Path(args.log_file) if args.log_file else output_dir / "run.log")

    result_path = output_dir / "result.json"
    try:
        runtime_root = prepare_runtime_root(args)
        pdb_path = resolve_pdb_path(args)
        alpha_chain, beta_chain = infer_tcr_chains(pdb_path, args.alpha_chain, args.beta_chain)
        alpha_residues = extract_chain_residues(pdb_path, alpha_chain)
        beta_residues = extract_chain_residues(pdb_path, beta_chain)
        target_alpha = "".join(res["aa"] for res in alpha_residues)
        target_beta = "".join(res["aa"] for res in beta_residues)
        hs_indices: List[int] = []
        hs_labels: List[str] = []
        if args.no_hotspot_params:
            pass
        elif args.hotspot_start is None or args.hotspot_end is None:
            raise ValueError("--hotspot-start and --hotspot-end are required unless --no-hotspot-params is set")
        elif args.hotspot_chain == "alpha":
            hs_indices, hs_labels = hotspot_indices(
                alpha_residues,
                args.hotspot_start,
                args.hotspot_end,
                label_prefix="A",
            )
        else:
            hs_indices, hs_labels = hotspot_indices(
                beta_residues,
                args.hotspot_start,
                args.hotspot_end,
                label_prefix="B",
                offset=len(alpha_residues),
            )
        template_path = prepare_template_cif(pdb_path, output_dir, pdb_id)
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        init_binder = args.init_binder_seq.strip().upper()
        loss_weights = resolve_loss_weights(args)
        hotspot_params = None
        if not args.no_hotspot_params:
            hotspot_params = {"target_hotspot_concat_indices": hs_indices}

        logging.info(
            "Case %s | pdb=%s | alpha=%s len=%d | beta=%s len=%d | hotspot=%s indices=%s | hotspot_params=%s | template=%s",
            pdb_id,
            pdb_path,
            alpha_chain,
            len(target_alpha),
            beta_chain,
            len(target_beta),
            hs_labels,
            hs_indices,
            hotspot_params,
            template_path,
        )
        logging.info("Loss weights override=%s", loss_weights)
        logging.info("PROTENIX_ROOT_DIR=%s", runtime_root)

        model, _ = load_protenix_model(
            model_name=args.model_name,
            device=args.device,
            checkpoint_dir=args.model_checkpoint_dir,
        )
        # Resolve boolean arguments with defaults
        use_annealing = resolve_bool_arg(args, "use-annealing", True)
        use_tau_scheduling = resolve_bool_arg(args, "use-tau-scheduling", True)
        tau_jitter = resolve_bool_arg(args, "tau-jitter", True)
        best_seq, final_seq, losses = run_hybrid_hallucination(
            model=model,
            target_seq=[target_alpha, target_beta],
            L_binder=len(init_binder),
            N_steps=args.n_steps,
            learning_rate=args.learning_rate,
            tau_noise=args.tau_noise,
            device=torch.device(args.device),
            loss_weights=loss_weights,
            K_top=args.k_top,
            B_candidates=args.b_candidates,
            T_sample=args.t_sample,
            max_mutations_per_seq=args.max_mutations_per_seq,
            use_annealing=use_annealing,
            T_init=args.t_init,
            T_min=args.t_min,
            use_tau_scheduling=use_tau_scheduling,
            tau_jitter=tau_jitter,
            init_binder_seq=init_binder,
            checkpoint_dir=str(checkpoint_dir),
            hotspot_params=hotspot_params,
            target_template_path=str(template_path),
            task_type=args.task_type,
            target_chain_roles=args.target_chain_roles,
        )
        result = {
            "status": "ok",
            "pdb_id": pdb_id,
            "pdb_path": str(pdb_path),
            "target": {"alpha_chain": alpha_chain, "beta_chain": beta_chain},
            "task_type": args.task_type,
            "target_chain_roles": args.target_chain_roles,
            "hotspot": {
                "chain": args.hotspot_chain,
                "start": args.hotspot_start,
                "end": args.hotspot_end,
                "labels": hs_labels,
                "target_concat_indices": hs_indices,
            "params_enabled": hotspot_params is not None,
            } if not args.no_hotspot_params else None,
            "template_path": str(template_path),
            "output_dir": str(output_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "init_binder_seq": init_binder,
            "best_seq": best_seq,
            "final_seq": final_seq,
            "loss_history": [float(v) for v in losses],
            "best_loss": float(min(losses)) if losses else None,
            "best_step": int(losses.index(min(losses))) if losses else None,
            "final_loss": float(losses[-1]) if losses else None,
            "params": {
                "n_steps": args.n_steps,
                "learning_rate": args.learning_rate,
                "tau_noise": args.tau_noise,
                "k_top": args.k_top,
                "b_candidates": args.b_candidates,
                "t_sample": args.t_sample,
                "max_mutations_per_seq": args.max_mutations_per_seq,
                "use_annealing": use_annealing,
                "t_init": args.t_init,
                "t_min": args.t_min,
                "use_tau_scheduling": use_tau_scheduling,
                "tau_jitter": tau_jitter,
                "model_name": args.model_name,
                "loss_weights": loss_weights,
            },
        }
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        logging.info("Saved result: %s", result_path)
        return 0
    except Exception as exc:
        result = {
            "status": "failed",
            "pdb_id": pdb_id,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "output_dir": str(output_dir),
        }
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        logging.exception("Case %s failed", pdb_id)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
