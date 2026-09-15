#!/usr/bin/env python3
"""Validate the pinned AtlasFold-M source/runtime without loading weights."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import subprocess
import sys
from pathlib import Path


EXPECTED_VERSION = "1.0.2"
EXPECTED_COMMIT = "444f376d85b9954a5f2f5f3f8b3cbcae1201ebb1"
EXPECTED_MODEL_REVISIONS = {
    "atlasfold_m": "a9f07668b7ef544d4470795ef03aa492f243a461",
    "atlaslm_3b": "b4e6d8abb92d3432e3a65d37f1cfafc18575f78e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--model-path")
    parser.add_argument("--lm-path")
    parser.add_argument("--weights-receipt")
    parser.add_argument("--require-models", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    errors: list[str] = []
    warnings: list[str] = []

    if sys.version_info < (3, 10):
        errors.append(f"Python >=3.10 required; found {sys.version.split()[0]}")
    else:
        print(f"[OK] Python {sys.version.split()[0]}")

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception as exc:
        errors.append(f"cannot inspect AtlasFold source: {exc}")
        commit = None
    if commit == EXPECTED_COMMIT:
        print(f"[OK] AtlasFold source commit {commit}")
    elif commit is not None:
        errors.append(f"AtlasFold commit {commit}; expected {EXPECTED_COMMIT}")

    try:
        import atlasfold
        from atlasfold.model import AtlasFold_Multimer, SamplingConfig  # noqa: F401
        from atlasfold.runner_multimer import MultimerFoldingRunner  # noqa: F401
    except Exception as exc:
        errors.append(f"AtlasFold-M import failed: {type(exc).__name__}: {exc}")
    else:
        module_path = Path(atlasfold.__file__).resolve()
        if source_root / "src" not in module_path.parents:
            errors.append(f"atlasfold imports from {module_path}, not {source_root / 'src'}")
        else:
            print(f"[OK] atlasfold imports from {module_path}")
        try:
            version = importlib.metadata.version("atlasfold")
        except importlib.metadata.PackageNotFoundError:
            version = None
        if version != EXPECTED_VERSION:
            errors.append(f"atlasfold version {version or 'missing'}; expected {EXPECTED_VERSION}")
        else:
            print(f"[OK] atlasfold {version}")

    try:
        import torch
    except Exception as exc:
        errors.append(f"torch import failed: {exc}")
    else:
        print(f"[OK] torch {torch.__version__}, CUDA build {torch.version.cuda}")
        if torch.cuda.is_available():
            print(f"[OK] CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif args.require_cuda:
            errors.append("CUDA required but torch.cuda.is_available() is false")
        else:
            warnings.append("CUDA unavailable; real inference not tested")

    model_paths = {
        "AtlasFold-M": args.model_path,
        "AtlasLM-3B": args.lm_path,
    }
    for label, raw_path in model_paths.items():
        if raw_path and Path(raw_path).is_file():
            print(f"[OK] {label} weight file: {Path(raw_path).resolve()}")
        elif args.require_models:
            errors.append(f"{label} weight path missing or not a file: {raw_path}")
        else:
            warnings.append(f"{label} local weight path not checked")

    if args.weights_receipt:
        try:
            weights_receipt = json.loads(Path(args.weights_receipt).read_text())
            downloads = weights_receipt["downloads"]
            expected = {
                "atlasfold_m": (
                    "SeonghwanSeo/atlasfold-m-260725",
                    args.model_path,
                ),
                "atlaslm_3b": ("SeonghwanSeo/atlaslm-3b-base", args.lm_path),
            }
            for key, (repo_id, configured_path) in expected.items():
                entry = downloads[key]
                revision = str(entry["resolved_revision"])
                if entry["repo_id"] != repo_id or not re.fullmatch(
                    r"[0-9a-f]{40}", revision
                ):
                    raise ValueError(f"invalid immutable provenance for {key}")
                if revision != EXPECTED_MODEL_REVISIONS[key]:
                    raise ValueError(
                        f"unexpected weight revision for {key}: {revision}"
                    )
                if Path(entry["weight_path"]).resolve() != Path(
                    configured_path
                ).resolve():
                    raise ValueError(f"configured path disagrees for {key}")
            print(f"[OK] immutable weight receipt: {Path(args.weights_receipt).resolve()}")
        except Exception as exc:
            errors.append(f"invalid weight receipt: {exc}")
    elif args.require_models:
        errors.append("--weights-receipt is required with --require-models")
    else:
        warnings.append("immutable weight receipt not checked")

    receipt = {
        "python": sys.version.split()[0],
        "source_root": str(source_root),
        "source_commit": commit,
        "expected_commit": EXPECTED_COMMIT,
        "expected_version": EXPECTED_VERSION,
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(receipt, indent=2))
    for warning in warnings:
        print(f"[WARN] {warning}")
    for error in errors:
        print(f"[ERROR] {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
