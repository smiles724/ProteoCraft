#!/usr/bin/env bash
# AF3 evaluator smoke test: run one selected-manifest design through AF3 eval.
# Usage: bash scripts/af3_smoke.sh <selected_manifest.csv> <work_root>
set -euo pipefail

MANIFEST="${1:?manifest path required}"
WORK="${2:?work root required}"

_first_value_for_col() {
  local col_name="$1"
  awk -F, -v name="$col_name" '
    NR==1 {
      for (i=1; i<=NF; i++) if ($i == name) col=i
      next
    }
    NR==2 {
      if (col) print $col
      exit
    }
  ' "$MANIFEST"
}

SUB_RUN="$(_first_value_for_col case)"
PAPER_CASE="$(_first_value_for_col paper_case)"
if [ -z "$PAPER_CASE" ]; then
  PAPER_CASE="$SUB_RUN"
fi
TARGET_SEQ="$(_first_value_for_col target_seq)"
N_CHAINS=$(awk -v seq="$TARGET_SEQ" 'BEGIN { print split(seq, a, ":") }')

if [ "${AF3_MOCK:-0}" = "1" ]; then
  echo "=== AF3_MOCK=1: skipping real MSA prep, using fixture mode ==="
  mkdir -p "$WORK/per_paper_case_msa/$PAPER_CASE"
  for i in $(seq 0 $((N_CHAINS - 1))); do
    chain_octal=$(printf '\\%03o' $((66 + i)))
    chain_letter=$(printf "$chain_octal")
    printf ">mock\nAAAAA\n" > "$WORK/per_paper_case_msa/$PAPER_CASE/target_chain_${chain_letter}.a3m"
  done
  AF3_PYTHON=${AF3_PYTHON:-/usr/bin/python}
  AF3_RUN_SCRIPT=${AF3_RUN_SCRIPT:-/tmp/fake_run.py}
  AF3_MODEL_DIR=${AF3_MODEL_DIR:-/tmp}
  AF3_DB_DIR=${AF3_DB_DIR:-/tmp}
else
  : "${AF3_PYTHON:?set AF3_PYTHON env var first, or use AF3_MOCK=1}"
  : "${AF3_RUN_SCRIPT:?set AF3_RUN_SCRIPT env var first, or use AF3_MOCK=1}"
  : "${AF3_MODEL_DIR:?set AF3_MODEL_DIR env var first, or use AF3_MOCK=1}"
  : "${AF3_DB_DIR:?set AF3_DB_DIR env var first, or use AF3_MOCK=1}"

  mkdir -p "$WORK"
  echo "=== Step 1: prep target MSA cache ==="
  python scripts/prepare_af3_target_msa.py \
    --manifest "$MANIFEST" \
    --work-root "$WORK" \
    --af3-python "$AF3_PYTHON" \
    --af3-run-script "$AF3_RUN_SCRIPT" \
    --af3-model-dir "$AF3_MODEL_DIR" \
    --af3-db-dir "$AF3_DB_DIR"
fi

echo "=== Step 2: AF3 evaluator (smoke, --limit 1) ==="
python -m eval_pipeline.cli \
  --manifest "$MANIFEST" \
  --out-dir "$WORK" \
  --evaluator af3 \
  --paper-case "$PAPER_CASE" \
  --sub-run "$SUB_RUN" \
  --af3-python "$AF3_PYTHON" \
  --af3-run-script "$AF3_RUN_SCRIPT" \
  --af3-model-dir "$AF3_MODEL_DIR" \
  --af3-db-dir "$AF3_DB_DIR" \
  --limit 1

echo "=== Step 3: verify output csv ==="
EVAL_CSV="$WORK/eval/${PAPER_CASE}__${SUB_RUN}__af3.csv"
test -f "$EVAL_CSV" || { echo "FAIL: $EVAL_CSV missing"; exit 1; }
N_LINES=$(wc -l < "$EVAL_CSV")
[ "$N_LINES" -ge 2 ] || { echo "FAIL: $EVAL_CSV has fewer than 2 lines"; exit 1; }
echo "AF3 smoke test PASSED ($EVAL_CSV: $N_LINES lines, AF3_MOCK=${AF3_MOCK:-0})"
