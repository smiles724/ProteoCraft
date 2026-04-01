#!/usr/bin/env python3
"""
identify_cdr.py - CDR Region Identifier for BindCraft v2

从抗体-抗原复合物 PDB 中识别 CDR（互补决定区）残基，输出可用于 BCv2
binder design 优化的 mutable mask。

背景:
    抗体通过 CDR loop 与抗原结合。在 binder design 中，只需突变 CDR 区域
    的氨基酸（通常占全链 20-30%），而非整条抗体链。本脚本自动识别 CDR
    位置，输出布尔 mask 标记哪些位置可突变。

    CDR 定义参考: http://www.bioinf.org.uk/abs/info.html

安装依赖（一次性）:
    conda install -c bioconda anarci -y   # 包含 HMMER
    pip install biopython                  # 如未安装

基本用法:
    # 输入 PDB ID（自动从 RCSB 下载，缓存到 structures/ 目录）
    python scripts/identify_cdr.py 8YAF

    # 输入本地 PDB 文件
    python scripts/identify_cdr.py structures/8YAF.pdb

    # 指定 binder chain（多拷贝 PDB 时推荐）
    python scripts/identify_cdr.py 8YAF --binder_chain B

    # 切换 CDR 定义（默认 contact，见下方说明）
    python scripts/identify_cdr.py 8YAF --cdr_def chothia

    # 自定义输出路径
    python scripts/identify_cdr.py 8YAF -o my_output/8YAF_cdr.json

CDR 定义说明:
    contact  - 基于抗原接触分析，覆盖所有参与结合的残基（默认，最宽）
    chothia  - 基于结构 loop 位置（结构分析标准）
    kabat    - 基于序列可变性（文献引用最多）
    abm      - Kabat/Chothia 折中
    imgt     - 国际标准（最窄）

输出:
    1. 终端打印：人类可读的 CDR 分析摘要
    2. JSON 文件（默认 outputs/<PDB_ID>_cdr.json），包含：
       - binder_sequence:    抗体链完整序列
       - binder_cdr_mask:    布尔数组，true = CDR 位置（可突变）
       - binder_cdr_residues: CDR 残基总数
       - target_sequence:    抗原序列（所有抗原链拼接）
       - antibody_chains:    每条抗体链的详细 CDR 信息
       - antigen_chains:     每条抗原链的序列和长度

在 Python 中读取 JSON 输出:
    import json
    with open("outputs/8YAF_cdr.json") as f:
        data = json.load(f)
    mask = data["binder_cdr_mask"]          # [False, ..., True, True, ...]
    binder_seq = data["binder_sequence"]    # 抗体链序列
    target_seq = data["target_sequence"]    # 抗原序列

注意事项:
    - 多拷贝 PDB（如 8YAF 有 6 组重复）：用 --binder_chain 指定具体链
    - 所有抗体链均为 ANARCI 自动检测（Heavy/Light），非抗体链归为抗原
    - target_sequence 是所有抗原链的拼接；如需单条抗原链，
      从 JSON 的 antigen_chains 字段中选取
"""

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# CDR definitions: {scheme: {region: (start_num, end_num)}}
# A residue is in a CDR if start <= residue_number <= end
# (insertion codes are automatically included)
# Reference: http://www.bioinf.org.uk/abs/info.html
# ---------------------------------------------------------------------------

CDR_DEFINITIONS = {
    "contact": {
        "L1": (30, 36), "L2": (46, 55), "L3": (89, 96),
        "H1": (30, 35), "H2": (47, 58), "H3": (93, 101),
    },
    "chothia": {
        "L1": (26, 32), "L2": (50, 52), "L3": (91, 96),
        "H1": (26, 32), "H2": (52, 56), "H3": (96, 101),
    },
    "kabat": {
        "L1": (24, 34), "L2": (50, 56), "L3": (89, 97),
        "H1": (31, 35), "H2": (50, 65), "H3": (95, 102),
    },
    "abm": {
        "L1": (24, 34), "L2": (50, 56), "L3": (89, 97),
        "H1": (26, 35), "H2": (50, 58), "H3": (95, 102),
    },
    "imgt": {
        "L1": (27, 32), "L2": (50, 51), "L3": (89, 97),
        "H1": (26, 33), "H2": (51, 56), "H3": (93, 102),
    },
}


# ---------------------------------------------------------------------------
# Core CDR logic
# ---------------------------------------------------------------------------

def is_in_cdr(residue_num: int, chain_type: str, cdr_def: str) -> bool:
    """Check if a residue number falls in any CDR for the given chain type.

    Args:
        residue_num: Kabat/Chothia residue number (integer part)
        chain_type: 'H' for heavy, 'L' or 'K' for light
        cdr_def: CDR definition name
    """
    prefix = "H" if chain_type == "H" else "L"
    for region_name, (start, end) in CDR_DEFINITIONS[cdr_def].items():
        if region_name.startswith(prefix) and start <= residue_num <= end:
            return True
    return False


def build_cdr_mask(
    numbered_residues: List[Tuple[int, str, str]],
    chain_type: str,
    cdr_def: str,
) -> Tuple[List[bool], Dict[str, List[dict]]]:
    """Build a boolean CDR mask from ANARCI-numbered residues.

    Args:
        numbered_residues: [(position_num, insertion_code, amino_acid), ...]
        chain_type: 'H', 'K', or 'L'
        cdr_def: CDR definition name

    Returns:
        mask: boolean list, True = CDR position
        regions: {cdr_name: [{seq_idx, num, aa}, ...]}
    """
    prefix = "H" if chain_type == "H" else "L"
    relevant = {
        name: rng
        for name, rng in CDR_DEFINITIONS[cdr_def].items()
        if name.startswith(prefix)
    }

    regions = {name: [] for name in relevant}
    mask = []

    for seq_idx, (num, ins, aa) in enumerate(numbered_residues):
        matched = False
        for cdr_name, (start, end) in relevant.items():
            if start <= num <= end:
                regions[cdr_name].append({
                    "seq_idx": seq_idx,
                    "num": f"{num}{ins.strip()}",
                    "aa": aa,
                })
                matched = True
                break
        mask.append(matched)

    regions = {k: v for k, v in regions.items() if v}
    return mask, regions


# ---------------------------------------------------------------------------
# PDB handling
# ---------------------------------------------------------------------------

STRUCTURES_DIR = Path(__file__).resolve().parent.parent / "structures"


def download_pdb(pdb_id: str, out_dir: Path = STRUCTURES_DIR) -> Path:
    """Download PDB file from RCSB if not cached locally."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_id = pdb_id.upper()
    out_path = out_dir / f"{pdb_id}.pdb"

    if out_path.exists():
        logging.info(f"Using cached {out_path}")
        return out_path

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    logging.info(f"Downloading {url} ...")
    try:
        urllib.request.urlretrieve(url, out_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download {pdb_id} from RCSB: {e}")

    logging.info(f"Saved to {out_path}")
    return out_path


_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def extract_sequences_from_pdb(pdb_path: Path) -> Dict[str, str]:
    """Extract amino acid sequences per chain from a PDB file."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))

    chains = {}
    for model in structure:
        for chain in model:
            residues = []
            for res in chain.get_residues():
                if res.id[0] != " ":
                    continue
                resname = res.get_resname().strip()
                aa = _THREE_TO_ONE.get(resname, None)
                if aa is not None:
                    residues.append(aa)
            if residues:
                chains[chain.id] = "".join(residues)
        break  # first model only

    return chains


# ---------------------------------------------------------------------------
# ANARCI integration
# ---------------------------------------------------------------------------

def number_antibody_chain(
    sequence: str, scheme: str = "chothia"
) -> Tuple[Optional[List[Tuple[int, str, str]]], Optional[str]]:
    """Number an antibody chain using ANARCI.

    Returns:
        numbered: [(num, insertion_code, aa), ...] or None if not antibody
        chain_type: 'H', 'K', 'L' or None
    """
    from anarci import anarci as run_anarci

    results = run_anarci([("query", sequence)], scheme=scheme, output=False)
    numbering_results, alignment_details, hit_tables = results

    if numbering_results[0] is None:
        return None, None

    # ANARCI returns [(numbering_list, start, end), ...] per domain
    domain_numbering = numbering_results[0][0][0]

    # Chain type is in alignment_details
    chain_type = alignment_details[0][0]["chain_type"]

    numbered = []
    for (num, ins), aa in domain_numbering:
        if aa != "-":
            numbered.append((num, ins, aa))

    return numbered, chain_type


def classify_chains(
    chains: Dict[str, str], scheme: str = "chothia"
) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """Classify chains as antibody (H/L) or antigen using ANARCI.

    Returns:
        antibody_chains: {chain_id: {numbered, chain_type, sequence}}
        antigen_chains: {chain_id: sequence}
    """
    antibody_chains = {}
    antigen_chains = {}

    for chain_id, seq in chains.items():
        numbered, chain_type = number_antibody_chain(seq, scheme=scheme)
        if numbered is not None:
            antibody_chains[chain_id] = {
                "numbered": numbered,
                "chain_type": chain_type,
                "sequence": seq,
            }
        else:
            antigen_chains[chain_id] = seq

    return antibody_chains, antigen_chains


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_chain_result(chain_info: dict, cdr_def: str) -> dict:
    """Format a single antibody chain's CDR analysis."""
    numbered = chain_info["numbered"]
    chain_type = chain_info["chain_type"]
    sequence = chain_info["sequence"]

    mask, regions = build_cdr_mask(numbered, chain_type, cdr_def)

    region_summaries = {}
    for name, residues in regions.items():
        indices = [r["seq_idx"] for r in residues]
        region_summaries[name] = {
            "start_idx": min(indices),
            "end_idx": max(indices),
            "length": len(residues),
            "sequence": "".join(r["aa"] for r in residues),
            "residues": residues,
        }

    type_name = "heavy" if chain_type == "H" else "light"
    return {
        "chain_type": chain_type,
        "chain_type_name": type_name,
        "sequence": sequence,
        "length": len(sequence),
        "cdr_regions": region_summaries,
        "cdr_mask": mask,
        "n_cdr_residues": sum(mask),
        "n_framework_residues": len(mask) - sum(mask),
    }


def print_summary(result: dict):
    """Print human-readable summary to terminal."""
    print("=" * 70)
    print(f"  CDR Analysis: {result['pdb_id']}")
    print(f"  Numbering: {result['scheme']}  |  CDR definition: {result['cdr_definition']}")
    print("=" * 70)

    for cid, info in result["antibody_chains"].items():
        chain_label = "Heavy" if info["chain_type"] == "H" else "Light"
        print(f"\n  Chain {cid} ({chain_label}, {info['length']} aa)")
        pct = info["n_cdr_residues"] / info["length"] * 100 if info["length"] else 0
        print(f"  CDR residues: {info['n_cdr_residues']} / {info['length']}"
              f" ({pct:.1f}%)")
        for region, rinfo in sorted(info["cdr_regions"].items()):
            print(f"    {region}: idx [{rinfo['start_idx']:3d}-{rinfo['end_idx']:3d}]"
                  f"  ({rinfo['length']:2d} aa)  {rinfo['sequence']}")

    if result["antigen_chains"]:
        print(f"\n  Antigen chain(s):")
        for cid, info in result["antigen_chains"].items():
            print(f"    Chain {cid}: {info['length']} aa")

    binder_id = result["binder_chain_id"]
    print(f"\n  >> Binder: chain {binder_id}"
          f" ({result['binder_cdr_residues']} CDR residues to mutate)")
    print(f"  >> Target: {len(result['target_sequence'])} aa"
          f" (concatenated antigen chains)")

    out = result.get("output_file")
    if out:
        print(f"\n  JSON saved to: {out}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Identify CDR regions from antibody-antigen PDB complexes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CDR definitions (--cdr_def):
  contact  - Based on antigen contact analysis (default, broadest)
  chothia  - Based on structural loop positions
  kabat    - Based on sequence variability (most cited)
  abm      - Kabat/Chothia compromise
  imgt     - International standard (narrowest)

Examples:
  python scripts/identify_cdr.py 8YAF
  python scripts/identify_cdr.py 8YAF --cdr_def chothia
  python scripts/identify_cdr.py structures/8YAF.pdb --binder_chain L
""",
    )
    parser.add_argument(
        "input",
        help="PDB file path or 4-character PDB ID (auto-downloads from RCSB)",
    )
    parser.add_argument(
        "--scheme", default="chothia", choices=["chothia", "kabat", "imgt"],
        help="ANARCI numbering scheme (default: chothia)",
    )
    parser.add_argument(
        "--cdr_def", default="contact",
        choices=list(CDR_DEFINITIONS.keys()),
        help="CDR definition (default: contact)",
    )
    parser.add_argument(
        "--binder_chain", default=None,
        help="Chain ID to use as binder (default: first heavy chain)",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output JSON path (default: outputs/<pdb_id>_cdr.json)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    # --- Resolve input ---
    input_str = args.input
    pdb_path = Path(input_str)

    if pdb_path.is_file():
        pdb_id = pdb_path.stem.upper()
    elif len(input_str) == 4 and input_str.isalnum():
        pdb_id = input_str.upper()
        pdb_path = download_pdb(pdb_id)
    else:
        parser.error(f"Input '{input_str}' is not a valid PDB file or 4-char PDB ID")

    # --- Parse PDB ---
    logging.info(f"Parsing {pdb_path} ...")
    chains = extract_sequences_from_pdb(pdb_path)
    if not chains:
        sys.exit(f"Error: No protein chains found in {pdb_path}")

    logging.info(f"Found {len(chains)} chain(s): "
                 + ", ".join(f"{k}({len(v)}aa)" for k, v in chains.items()))

    # --- Classify chains ---
    logging.info("Running ANARCI numbering ...")
    antibody_chains, antigen_chains = classify_chains(chains, scheme=args.scheme)

    if not antibody_chains:
        sys.exit("Error: No antibody chains detected by ANARCI. "
                 "Check that the PDB contains antibody sequences.")

    # --- Select binder chain ---
    if args.binder_chain:
        if args.binder_chain not in antibody_chains:
            avail = ", ".join(antibody_chains.keys())
            sys.exit(f"Error: Chain '{args.binder_chain}' is not an antibody chain. "
                     f"Available: {avail}")
        binder_chain_id = args.binder_chain
    else:
        heavy = [c for c, i in antibody_chains.items() if i["chain_type"] == "H"]
        binder_chain_id = heavy[0] if heavy else list(antibody_chains.keys())[0]

    # --- Build CDR results ---
    ab_results = {}
    for chain_id, chain_info in antibody_chains.items():
        ab_results[chain_id] = format_chain_result(chain_info, args.cdr_def)

    ag_results = {}
    for chain_id, seq in antigen_chains.items():
        ag_results[chain_id] = {"sequence": seq, "length": len(seq)}

    target_sequence = "".join(antigen_chains.values())

    binder_info = ab_results[binder_chain_id]
    result = {
        "pdb_id": pdb_id,
        "pdb_file": str(pdb_path),
        "scheme": args.scheme,
        "cdr_definition": args.cdr_def,
        "antibody_chains": ab_results,
        "antigen_chains": ag_results,
        "binder_chain_id": binder_chain_id,
        "binder_sequence": binder_info["sequence"],
        "binder_cdr_mask": binder_info["cdr_mask"],
        "binder_cdr_residues": binder_info["n_cdr_residues"],
        "target_sequence": target_sequence,
    }

    # --- Save JSON ---
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path(__file__).resolve().parent.parent / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{pdb_id}_cdr.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    result["output_file"] = str(out_path)

    # --- Print summary ---
    print_summary(result)

    return result


if __name__ == "__main__":
    main()
