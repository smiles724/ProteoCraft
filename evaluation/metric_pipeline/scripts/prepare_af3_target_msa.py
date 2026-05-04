"""Prepare per-paper_case AF3 target MSA cache via AF3 data pipeline only."""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _build_target_only_input_json(paper_case: str, target_seqs: list[str]) -> dict:
    sequences = []
    for i, target_seq in enumerate(target_seqs):
        chain_id = chr(ord("B") + i)
        sequences.append({
            "protein": {
                "id": chain_id,
                "sequence": target_seq,
                "templates": [],
            }
        })
    return {
        "name": f"{paper_case}__target_only",
        "modelSeeds": [42],
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": 4,
    }


def _extract_msas_from_data_json(data_json_path: Path, out_dir: Path) -> dict:
    """Parse AF3 target-only *_data.json and write target_chain_<id>.a3m files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(data_json_path).read_text())
    out_paths = {}
    for entry in data["sequences"]:
        protein = entry.get("protein")
        if not protein:
            continue
        chain_id = protein["id"]
        a3m = protein.get("unpairedMsa")
        if a3m is None:
            raise RuntimeError(
                f"AF3 left unpairedMsa=null for chain {chain_id} in {data_json_path}; "
                "data pipeline did not run / failed silently"
            )
        if not a3m.strip():
            raise RuntimeError(
                f"empty MSA string for chain {chain_id} in {data_json_path}"
            )
        path = out_dir / f"target_chain_{chain_id}.a3m"
        path.write_text(a3m)
        out_paths[chain_id] = path
    return out_paths


def run_prep_for_paper_case(
    paper_case: str,
    target_seqs: list[str],
    work_root: Path,
    af3_python: str,
    af3_run_script: str,
    af3_model_dir: str,
    af3_db_dir: str,
    keep_raw: bool = False,
) -> None:
    target_msa_dir = work_root / "per_paper_case_msa" / paper_case
    expected_chains = [chr(ord("B") + i) for i in range(len(target_seqs))]
    present = []
    for chain_id in expected_chains:
        path = target_msa_dir / f"target_chain_{chain_id}.a3m"
        if path.exists() and path.stat().st_size > 0:
            present.append(chain_id)
    if len(present) == len(expected_chains):
        print(f"  [skip] {paper_case} (MSA cache complete)")
        return
    if 0 < len(present) < len(expected_chains):
        raise RuntimeError(
            f"{paper_case}: partial MSA cache present={present}, "
            f"expected={expected_chains}; manually clean {target_msa_dir} before re-running"
        )

    tmp_root = work_root / "_msa_prep_tmp" / paper_case
    tmp_root.mkdir(parents=True, exist_ok=True)
    input_json = tmp_root / "input.json"
    input_json.write_text(json.dumps(_build_target_only_input_json(paper_case, target_seqs), indent=2))

    out_root = tmp_root / "out"
    cmd = [
        af3_python,
        af3_run_script,
        f"--json_path={input_json}",
        f"--output_dir={out_root}",
        f"--model_dir={af3_model_dir}",
        f"--db_dir={af3_db_dir}",
        "--run_data_pipeline=True",
        "--run_inference=False",
        "--force_output_dir",
    ]
    print(f"  [run] {paper_case}: {shlex.join([str(c) for c in cmd])}")
    t0 = time.time()
    result = subprocess.run(cmd, check=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"AF3 data pipeline failed for {paper_case} (exit {result.returncode}); "
            "see AF3 log above for diagnosis"
        )
    print(f"  [done] {paper_case} ({time.time() - t0:.0f}s)")

    job_name = f"{paper_case}__target_only"
    data_json = out_root / job_name / f"{job_name}_data.json"
    if not data_json.exists():
        candidates = list(out_root.glob(f"{job_name}*/{job_name}*_data.json"))
        if not candidates:
            raise FileNotFoundError(
                f"AF3 _data.json not found in {out_root}"
            )
        data_json = candidates[0]

    _extract_msas_from_data_json(data_json, target_msa_dir)
    if not keep_raw:
        shutil.rmtree(tmp_root)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", required=True,
                    help="one or more selected manifest CSVs (shell glob OK)")
    ap.add_argument("--work-root", required=True, type=Path)
    ap.add_argument("--af3-python", required=True)
    ap.add_argument("--af3-run-script", required=True)
    ap.add_argument("--af3-model-dir", required=True)
    ap.add_argument("--af3-db-dir", required=True)
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--paper-case", default=None)
    args = ap.parse_args()

    manifest_dfs: list[tuple[Path, pd.DataFrame, str]] = []
    available_paper_cases = set()
    for manifest in args.manifest:
        df = pd.read_csv(manifest)
        paper_case_col = "paper_case" if "paper_case" in df.columns else "case"
        available_paper_cases.update(df[paper_case_col].dropna().astype(str).unique())
        manifest_dfs.append((Path(manifest), df, paper_case_col))

    if args.paper_case and args.paper_case not in available_paper_cases:
        raise SystemExit(
            f"--paper-case {args.paper_case!r} not found in manifest; "
            f"available: {sorted(available_paper_cases)}"
        )

    seen: dict[str, list[str]] = {}
    for manifest, df, paper_case_col in manifest_dfs:
        for _, row in df.iterrows():
            paper_case = str(row[paper_case_col])
            if args.paper_case and paper_case != args.paper_case:
                continue
            target_seq = row.get("target_seq")
            if pd.isna(target_seq) or not str(target_seq).strip():
                raise SystemExit(
                    f"manifest row for {paper_case} "
                    f"(design_id={row.get('design_id', '?')}) has empty/NaN target_seq; "
                    "investigate manifest upstream"
                )
            chains = str(target_seq).split(":")
            if paper_case not in seen:
                seen[paper_case] = chains
            # Chain order matters in AF3 input layout, so list equality is intentional.
            elif seen[paper_case] != chains:
                raise SystemExit(
                    f"inconsistent target_seq for {paper_case}: previously saw "
                    f"{seen[paper_case]}, now {chains}; investigate manifest upstream"
                )

    print(f"prep MSA for {len(seen)} paper_cases")
    for paper_case in sorted(seen):
        run_prep_for_paper_case(
            paper_case=paper_case,
            target_seqs=seen[paper_case],
            work_root=args.work_root,
            af3_python=args.af3_python,
            af3_run_script=args.af3_run_script,
            af3_model_dir=args.af3_model_dir,
            af3_db_dir=args.af3_db_dir,
            keep_raw=args.keep_raw,
        )
    print("done.")


if __name__ == "__main__":
    main()
