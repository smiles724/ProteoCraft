#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${MANIFEST:-$ROOT/outputs/8_bcv1/pxd_rf3/bcv1_design_manifest.csv}"
METHOD="${METHOD:-all}"
CASE_FILTER="${CASE:-all}"
CONDITION_FILTER="${CONDITION:-all}"
LENGTH_FILTER="${LENGTH:-all}"
LIMIT="${LIMIT:-0}"
DRY_RUN="${DRY_RUN:-0}"
SHARD_INDEX="${SHARD_INDEX:--1}"
SHARD_COUNT="${SHARD_COUNT:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
FAILURES_FILE="${FAILURES_FILE:-}"

usage() {
  cat <<EOF
Usage:
  bash benchmark/script/run_bcv1_design_batch.sh [options]

Environment filters:
  MANIFEST=<csv>       Default: benchmark/outputs/8_bcv1/pxd_rf3/bcv1_design_manifest.csv
  METHOD=pxdesign|rf3|all
  CASE=BetV1|DerF21|DerF7|HER2|IFNAR2|PD-L1|PD1|SpCas9|all
  CONDITION=<name>|all
  LENGTH=80|100|120|all
  LIMIT=<n>            Stop after n matching tasks, 0 means no limit
  DRY_RUN=1            Print commands without running them
  SHARD_INDEX=<i>      Run only shard i, 0-based. Default: disabled
  SHARD_COUNT=<n>      Total number of shards. Default: 1
  CONTINUE_ON_ERROR=1  Continue after a failed task and record it
  FAILURES_FILE=<path> Append failed task rows here when continuing
  PX_GPU_ID=<gpu>      Used by PXDesign commands
  RF3_GPU_ID=<gpu>     Used by RF3 commands

Examples:
  DRY_RUN=1 METHOD=pxdesign CASE=BetV1 LENGTH=80 bash benchmark/script/run_bcv1_design_batch.sh
  METHOD=rf3 CASE=DerF21 CONDITION=B10 LENGTH=80 LIMIT=1 bash benchmark/script/run_bcv1_design_batch.sh
  SHARD_INDEX=0 SHARD_COUNT=8 PX_GPU_ID=0 RF3_GPU_ID=0 bash benchmark/script/run_bcv1_design_batch.sh
EOF
}

if [[ "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  echo "Run: python benchmark/script/prepare_bcv1_pxdesign_rf3.py" >&2
  exit 1
fi

python - "$MANIFEST" "$METHOD" "$CASE_FILTER" "$CONDITION_FILTER" "$LENGTH_FILTER" "$LIMIT" "$DRY_RUN" "$SHARD_INDEX" "$SHARD_COUNT" "$CONTINUE_ON_ERROR" "$FAILURES_FILE" <<'PY'
import csv
import subprocess
import sys
from pathlib import Path

(
    manifest,
    method,
    case_filter,
    condition_filter,
    length_filter,
    limit,
    dry_run,
    shard_index,
    shard_count,
    continue_on_error,
    failures_file,
) = sys.argv[1:]
limit = int(limit)
dry_run = dry_run == "1"
shard_index = int(shard_index)
shard_count = int(shard_count)
continue_on_error = continue_on_error == "1"

if shard_count < 1:
    print(f"Invalid SHARD_COUNT={shard_count}", file=sys.stderr)
    sys.exit(2)
if shard_index >= shard_count:
    print(
        f"Invalid SHARD_INDEX={shard_index} for SHARD_COUNT={shard_count}",
        file=sys.stderr,
    )
    sys.exit(2)
sharding_enabled = shard_index >= 0


def complete(row):
    out_dir = Path(row["out_dir"])
    expected = int(row["samples"])
    if row["method"] == "pxdesign":
        return len(list(out_dir.glob("*/*/SUCCESS_FILE"))) >= 1 and len(
            list(out_dir.glob("*/*/predictions/*_sample_*.cif"))
        ) >= expected
    if row["method"] == "rf3":
        return (
            len(list(out_dir.glob("*.cif.gz"))) >= expected
            and len(list(out_dir.glob("*.json"))) >= expected
        )
    return False


def match(row):
    return (
        (method == "all" or row["method"] == method)
        and (case_filter == "all" or row["case"] == case_filter)
        and (condition_filter == "all" or row["condition"] == condition_filter)
        and (length_filter == "all" or row["length"] == length_filter)
    )


with open(manifest, newline="", encoding="utf-8") as handle:
    matched_rows = [row for row in csv.DictReader(handle) if match(row)]

if not matched_rows:
    print("No matching tasks.", file=sys.stderr)
    sys.exit(1)

if sharding_enabled:
    rows = [
        row
        for index, row in enumerate(matched_rows)
        if index % shard_count == shard_index
    ]
else:
    rows = matched_rows

if not rows:
    print(
        f"No tasks selected for shard {shard_index}/{shard_count}; "
        f"matched={len(matched_rows)}",
        file=sys.stderr,
    )
    sys.exit(0)

ran = 0
failed = 0
for row in rows:
    tag = f"{row['method']} {row['case']} {row['condition']} len_{row['length']}"
    if complete(row):
        print(f"[skip complete] {tag}")
        continue
    cmd = row["command"]
    print(f"[run] {tag}")
    print(cmd)
    if not dry_run:
        try:
            subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")
        except subprocess.CalledProcessError as err:
            failed += 1
            print(f"[failed rc={err.returncode}] {tag}", file=sys.stderr)
            if failures_file:
                Path(failures_file).parent.mkdir(parents=True, exist_ok=True)
                with open(failures_file, "a", encoding="utf-8") as handle:
                    handle.write(
                        "\t".join(
                            [
                                row["method"],
                                row["case"],
                                row["condition"],
                                row["length"],
                                str(err.returncode),
                                row["config"],
                                row["out_dir"],
                            ]
                        )
                        + "\n"
                    )
            if not continue_on_error:
                raise
    ran += 1
    if limit and ran >= limit:
        break

print(
    f"matched={len(matched_rows)} selected={len(rows)} ran={ran} "
    f"failed={failed} dry_run={dry_run} shard_index={shard_index} "
    f"shard_count={shard_count}"
)
if failed:
    sys.exit(1)
PY
