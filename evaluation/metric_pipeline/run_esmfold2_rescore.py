#!/usr/bin/env python3
"""Prepare, run or inspect independent stock ESMFold2 candidate rescoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_pipeline import esmfold2_rescore as rescore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Freeze candidates and protocol without loading models")
    prepare.add_argument("--manifest", type=Path, required=True, help="Candidate JSONL, or completed-run JSONL with --from-search")
    prepare.add_argument("--config", type=Path, default=rescore.DEFAULT_CONFIG)
    prepare.add_argument("--from-search", action="store_true", help="Select Top K unique sequences from each completed run")
    for name in ("run", "report", "_worker"):
        command = commands.add_parser(name)
        command.add_argument("--out-dir", type=Path, required=True)
        if name != "report":
            command.add_argument("--device", default="cuda:0")
            command.add_argument("--retry-failed", action="store_true")
    prepare.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        rescore.freeze(args.manifest, args.config, args.out_dir, from_search=args.from_search)
        with rescore.locked(args.out_dir):
            print(json.dumps(rescore.report(args.out_dir), indent=2))
        return 0
    if args.command == "report":
        with rescore.locked(args.out_dir):
            print(json.dumps(rescore.report(args.out_dir), indent=2))
        return 0
    if args.command == "run":
        command = [sys.executable, str(Path(__file__).resolve()), "_worker", "--out-dir",
                   str(args.out_dir.resolve()), "--device", args.device]
        if args.retry_failed:
            command.append("--retry-failed")
        return subprocess.run(command, check=False).returncode
    return rescore.run_worker(args.out_dir, args.device, retry_failed=args.retry_failed)


if __name__ == "__main__":
    raise SystemExit(main())
