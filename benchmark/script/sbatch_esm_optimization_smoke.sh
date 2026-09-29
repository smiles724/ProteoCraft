#!/usr/bin/env bash
# Supply account/partition/GPU/output with sbatch; this script does not choose a cluster.
set -euo pipefail
REPO_ROOT="${PROTEOCRAFT_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
: "${ESMOPT_PYTHON:?Set ESMOPT_PYTHON to the isolated Python executable}"
: "${PROTEOCRAFT_UPLIFT_ROOT:?Set PROTEOCRAFT_UPLIFT_ROOT}"
: "${HF_HOME:?Set HF_HOME to the model cache}"
: "${ESMCFOLD_CCD_PATH:?Set ESMCFOLD_CCD_PATH}"
ESMOPT_OUTPUT="${ESMOPT_OUTPUT:-$REPO_ROOT/outputs/esm_optimization/smoke}"
mkdir -p "$ESMOPT_OUTPUT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TRITON_CACHE_DIR="$ESMOPT_OUTPUT/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$ESMOPT_OUTPUT/inductor_cache"
cd "$REPO_ROOT"
if [[ "${ESMOPT_RUN_TESTS:-1}" == 1 ]]; then
  "$ESMOPT_PYTHON" -m pytest bindcraft/tests/test_esmfold2_optimizations.py -q \
    --junitxml="$ESMOPT_OUTPUT/gpu_tests.xml"
fi
for phase in ${ESMOPT_PHASES:-prior proposal update verifier}; do
  case "$phase" in
    prior) arms="${ESMOPT_PRIOR_ARMS:-baseline esmc_exact}" ;;
    proposal) arms="${ESMOPT_PROPOSAL_ARMS:-baseline proposal_exact proposal_fast esmc_exact}" ;;
    update) arms="${ESMOPT_UPDATE_ARMS:-baseline proposal_exact proposal_fast esmc_exact}" ;;
    verifier) arms="${ESMOPT_VERIFIER_ARMS:-baseline verifier_exact verifier_fast big}" ;;
    *) echo "Unknown phase: $phase" >&2; exit 2 ;;
  esac
  arms="${arms//,/ }"
  arms="${arms//:/ }"
  for arm in $arms; do
    output="$ESMOPT_OUTPUT/$phase/$arm"
    reference=()
    if [[ "$arm" != baseline ]]; then reference=(--reference "$ESMOPT_OUTPUT/$phase/baseline"); fi
    timeout --signal=TERM --kill-after=30s "${ESMOPT_COMMAND_TIMEOUT:-30m}" \
      "$ESMOPT_PYTHON" benchmark/script/run_esm_optimization.py micro \
      --case "${ESMOPT_CASE:-denovo_PD1}" --phase "$phase" --arm "$arm" --output "$output" \
      --loops "${ESMOPT_LOOPS:-1}" --sampling-steps "${ESMOPT_SAMPLING_STEPS:-2}" \
      --warmup "${ESMOPT_WARMUP:-1}" --repeats "${ESMOPT_REPEATS:-2}" \
      --soft-steps "${ESMOPT_SOFT_STEPS:-1}" \
      --deterministic \
      --proposal-model "$HF_HOME/hub/models--biohub--ESMFold2-Experimental-Fast/snapshots/04dec820ede9283c9893e318e5ca5a9ac2ab93bc" \
      --verifier-model "$HF_HOME/hub/models--biohub--ESMFold2/snapshots/1ebf0e3481a5184eb6171d40615c79e384b48796" \
      "${reference[@]}" > "$ESMOPT_OUTPUT/${phase}_${arm}.log" 2>&1
  done
done
