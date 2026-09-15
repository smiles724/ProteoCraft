#!/usr/bin/env python3
"""Download AtlasFold-M weights and record immutable Hub revisions."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = {
    "atlasfold_m": (
        "SeonghwanSeo/atlasfold-m-260725",
        "weights/atlasfold-m-260725.pth",
        "a9f07668b7ef544d4470795ef03aa492f243a461",
    ),
    "atlaslm_3b": (
        "SeonghwanSeo/atlaslm-3b-base",
        "weights/atlaslm_3b_base.pth",
        "b4e6d8abb92d3432e3a65d37f1cfafc18575f78e",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--atlasfold-revision", default=MODELS["atlasfold_m"][2])
    parser.add_argument("--atlaslm-revision", default=MODELS["atlaslm_3b"][2])
    parser.add_argument("--receipt", default=None)
    parser.add_argument("--max-attempts", type=int, default=12)
    return parser.parse_args()


def _download(
    repo_id: str,
    filename: str,
    revision: str,
    cache_dir: Path,
    max_attempts: int,
) -> dict:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    for attempt in range(1, max_attempts + 1):
        try:
            snapshot = Path(
                snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    cache_dir=cache_dir,
                    allow_patterns=[filename],
                )
            ).resolve()
            weight_path = snapshot / filename
            if not weight_path.is_file():
                raise FileNotFoundError(
                    f"downloaded snapshot lacks {filename}: {snapshot}"
                )
            break
        except Exception as exc:
            if attempt == max_attempts:
                raise
            delay_s = min(60, 5 * attempt)
            print(
                f"Download attempt {attempt}/{max_attempts} failed for {repo_id}: "
                f"{type(exc).__name__}: {exc}; retrying in {delay_s}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay_s)
    return {
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": snapshot.name,
        "snapshot_path": str(snapshot),
        "weight_path": str(weight_path),
        "weight_size_bytes": weight_path.stat().st_size,
    }


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    revisions = {
        "atlasfold_m": args.atlasfold_revision,
        "atlaslm_3b": args.atlaslm_revision,
    }
    downloads = {
        name: _download(
            repo_id,
            filename,
            revisions[name],
            cache_dir,
            args.max_attempts,
        )
        for name, (repo_id, filename, _) in MODELS.items()
    }
    receipt_path = Path(args.receipt) if args.receipt else cache_dir / "atlasfold_m_weights_receipt.json"
    receipt = {
        "schema_version": 1,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "downloads": downloads,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"Receipt: {receipt_path.resolve()}")
    print(f"export ATLASFOLD_MODEL_PATH={downloads['atlasfold_m']['weight_path']}")
    print(f"export ATLASFOLD_LM_PATH={downloads['atlaslm_3b']['weight_path']}")
    print(f"export ATLASFOLD_CACHE_DIR={cache_dir}")
    print(f"export ATLASFOLD_WEIGHTS_RECEIPT={receipt_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
