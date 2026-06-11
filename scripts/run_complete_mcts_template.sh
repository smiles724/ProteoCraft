#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TASK="${TASK:-denovo}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
MODEL_NAME="${MODEL_NAME:-protenix_base_default_v1.0.0}"
PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-}"
PROTENIX_CHECKPOINT_DIR="${PROTENIX_CHECKPOINT_DIR:-${PROTENIX_ROOT_DIR:+${PROTENIX_ROOT_DIR}/checkpoints}}"
COMPONENTS_CACHE_DIR="${COMPONENTS_CACHE_DIR:-${PROTENIX_ROOT_DIR:+${PROTENIX_ROOT_DIR}/ccd_cache}}"
export PROTENIX_ROOT_DIR
export PROTENIX_CHECKPOINT_DIR
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"

N_STEPS="${N_STEPS:-}"
LEARNING_RATE="${LEARNING_RATE:-0.05}"
TAU_NOISE="${TAU_NOISE:-4.0}"
SEED="${SEED:-42}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
PRIOR_STRENGTH="${PRIOR_STRENGTH:-0.1}"
MCTS_C="${MCTS_C:-1.0}"
MCTS_MAX_NODES="${MCTS_MAX_NODES:-5000}"
MCTS_TAU_FIXED="${MCTS_TAU_FIXED:-4.0}"
MCTS_NUM_BEAMS="${MCTS_NUM_BEAMS:-8}"
MCTS_BOOTSTRAP_STEPS="${MCTS_BOOTSTRAP_STEPS:-50}"
MCTS_BOOTSTRAP_PATIENCE="${MCTS_BOOTSTRAP_PATIENCE:-10}"
MCTS_BOOTSTRAP_MIN_DELTA="${MCTS_BOOTSTRAP_MIN_DELTA:-0.02}"
MCTS_BOOTSTRAP_MAX_STEPS="${MCTS_BOOTSTRAP_MAX_STEPS:-80}"
MCTS_EXPAND_ACTIONS_PER_STEP="${MCTS_EXPAND_ACTIONS_PER_STEP:-2}"
MCTS_HANDOFF_TOP_K="${MCTS_HANDOFF_TOP_K:-8}"
MCTS_PRIOR_MODE="${MCTS_PRIOR_MODE:-full}"
MCTS_PROPOSAL_METHOD="${MCTS_PROPOSAL_METHOD:-one_step_fixed}"

usage() {
  cat <<'EOF'
Run current complete MCTS templates.

Usage:
  TASK=denovo TARGET_SEQ=... TARGET_TEMPLATE_PATH=... TARGET_HOTSPOTS=24 OUTPUT_DIR=outputs/case bash scripts/run_complete_mcts_template.sh
  TASK=tcr PDB_ID=1QRN PDB_PATH=/path/1QRN.pdb ALPHA_CHAIN=D BETA_CHAIN=E HOTSPOT_CHAIN=alpha HOTSPOT_START=72 HOTSPOT_END=76 OUTPUT_DIR=outputs/tcr_case bash scripts/run_complete_mcts_template.sh
  TASK=cdr CDR_CONFIG=benchmark/yaml/cdr_tcr/cdr_tcr_domain_redesign.yaml CDR_CASE=1QRN_alpha OUTPUT_DIR=outputs/cdr_case bash scripts/run_complete_mcts_template.sh

Common environment:
  PYTHON_BIN=python
  DEVICE=cuda:0
  PROTENIX_ROOT_DIR=/path/to/resources
  PROTENIX_CHECKPOINT_DIR=/path/to/resources/checkpoints
  COMPONENTS_CACHE_DIR=/path/to/resources/ccd_cache
  LAYERNORM_TYPE=torch

Current complete MCTS defaults:
  two_stage + full prior + one_step_fixed + tuned loss.

Notes:
  - TASK=denovo uses scripts/run_hallucination.py.
  - TASK=tcr uses benchmark/script/run_bindcraft_tcr_case.py.
  - TASK=cdr uses benchmark/script/run_cdr_tcr_domain_redesign.py and requires a CDR YAML with valid input/model paths.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    echo "Run with --help for examples." >&2
    exit 2
  fi
}

common_mcts_denovo_args=()
common_mcts_denovo_args+=(--mode mcts)
common_mcts_denovo_args+=(--learning_rate "${LEARNING_RATE}")
common_mcts_denovo_args+=(--tau_noise "${TAU_NOISE}")
common_mcts_denovo_args+=(--device "${DEVICE}")
common_mcts_denovo_args+=(--model_name "${MODEL_NAME}")
common_mcts_denovo_args+=(--checkpoint_dir "${PROTENIX_CHECKPOINT_DIR}")
common_mcts_denovo_args+=(--dtype bf16)
common_mcts_denovo_args+=(--t_sample 1.0)
common_mcts_denovo_args+=(--max_mutations_per_seq 3)
common_mcts_denovo_args+=(--mcts_c "${MCTS_C}")
common_mcts_denovo_args+=(--mcts_max_nodes "${MCTS_MAX_NODES}")
common_mcts_denovo_args+=(--mcts_tau_fixed "${MCTS_TAU_FIXED}")
common_mcts_denovo_args+=(--mcts_num_beams "${MCTS_NUM_BEAMS}")
common_mcts_denovo_args+=(--mcts_strategy two_stage)
common_mcts_denovo_args+=(--mcts_bootstrap_steps "${MCTS_BOOTSTRAP_STEPS}")
common_mcts_denovo_args+=(--mcts_bootstrap_patience "${MCTS_BOOTSTRAP_PATIENCE}")
common_mcts_denovo_args+=(--mcts_bootstrap_min_delta "${MCTS_BOOTSTRAP_MIN_DELTA}")
common_mcts_denovo_args+=(--mcts_bootstrap_max_steps "${MCTS_BOOTSTRAP_MAX_STEPS}")
common_mcts_denovo_args+=(--mcts_handoff_top_k "${MCTS_HANDOFF_TOP_K}")
common_mcts_denovo_args+=(--mcts_expand_actions_per_step "${MCTS_EXPAND_ACTIONS_PER_STEP}")
common_mcts_denovo_args+=(--prior_strength "${PRIOR_STRENGTH}")
common_mcts_denovo_args+=(--chunk_size "${CHUNK_SIZE}")
common_mcts_denovo_args+=(--seed "${SEED}")
common_mcts_denovo_args+=(--mcts_prior_mode "${MCTS_PRIOR_MODE}")
common_mcts_denovo_args+=(--mcts_proposal_method "${MCTS_PROPOSAL_METHOD}")

case "${TASK}" in
  denovo)
    require_var TARGET_SEQ
    require_var OUTPUT_DIR
    require_var PROTENIX_CHECKPOINT_DIR
    : "${N_STEPS:=150}"
    : "${BINDER_LENGTH:=120}"
    : "${K_TOP:=10}"
    : "${B_CANDIDATES:=8}"
    mkdir -p "${OUTPUT_DIR}"
    cmd=("${PYTHON_BIN}" -u scripts/run_hallucination.py)
    cmd+=(--target "${TARGET_SEQ}")
    cmd+=(--output_dir "${OUTPUT_DIR}")
    cmd+=(--save_checkpoint_dir "${OUTPUT_DIR}/checkpoints")
    cmd+=(--log_file "${OUTPUT_DIR}/optim.log")
    cmd+=(--binder_length "${BINDER_LENGTH}")
    cmd+=(--n_steps "${N_STEPS}")
    cmd+=(--k_top "${K_TOP}")
    cmd+=(--b_candidates "${B_CANDIDATES}")
    cmd+=(--loss_preset tuned)
    cmd+=("${common_mcts_denovo_args[@]}")
    [[ -n "${TARGET_TEMPLATE_PATH:-}" ]] && cmd+=(--target_template_path "${TARGET_TEMPLATE_PATH}")
    [[ -n "${TARGET_HOTSPOTS:-}" ]] && cmd+=(--target_hotspots "${TARGET_HOTSPOTS}")
    [[ -n "${INIT_BINDER_SEQ:-}" ]] && cmd+=(--init_binder_seq "${INIT_BINDER_SEQ}")
    ;;

  tcr)
    require_var PDB_ID
    require_var OUTPUT_DIR
    require_var PROTENIX_CHECKPOINT_DIR
    : "${N_STEPS:=120}"
    : "${K_TOP:=8}"
    : "${B_CANDIDATES:=4}"
    mkdir -p "${OUTPUT_DIR}"
    cmd=("${PYTHON_BIN}" -u benchmark/script/run_bindcraft_tcr_case.py)
    cmd+=(--pdb-id "${PDB_ID}")
    cmd+=(--output-dir "${OUTPUT_DIR}")
    cmd+=(--log-file "${OUTPUT_DIR}/optim.log")
    cmd+=(--mode mcts)
    cmd+=(--model-checkpoint-dir "${PROTENIX_CHECKPOINT_DIR}")
    [[ -n "${COMPONENTS_CACHE_DIR:-}" ]] && cmd+=(--components-cache-dir "${COMPONENTS_CACHE_DIR}")
    cmd+=(--model-name "${MODEL_NAME}")
    cmd+=(--device "${DEVICE}")
    cmd+=(--n-steps "${N_STEPS}")
    cmd+=(--learning-rate "${LEARNING_RATE}")
    cmd+=(--tau-noise "${TAU_NOISE}")
    cmd+=(--k-top "${K_TOP}")
    cmd+=(--b-candidates "${B_CANDIDATES}")
    cmd+=(--t-sample 1.0)
    cmd+=(--max-mutations-per-seq 3)
    cmd+=(--prior-strength "${PRIOR_STRENGTH}")
    cmd+=(--mcts-c "${MCTS_C}")
    cmd+=(--mcts-max-nodes "${MCTS_MAX_NODES}")
    cmd+=(--mcts-tau-fixed "${MCTS_TAU_FIXED}")
    cmd+=(--mcts-num-beams "${MCTS_NUM_BEAMS}")
    cmd+=(--mcts-strategy two_stage)
    cmd+=(--mcts-bootstrap-steps "${MCTS_BOOTSTRAP_STEPS}")
    cmd+=(--mcts-bootstrap-patience "${MCTS_BOOTSTRAP_PATIENCE}")
    cmd+=(--mcts-bootstrap-min-delta "${MCTS_BOOTSTRAP_MIN_DELTA}")
    cmd+=(--mcts-bootstrap-max-steps "${MCTS_BOOTSTRAP_MAX_STEPS}")
    cmd+=(--mcts-expand-actions-per-step "${MCTS_EXPAND_ACTIONS_PER_STEP}")
    cmd+=(--chunk-size "${CHUNK_SIZE}")
    cmd+=(--seed "${SEED}")
    cmd+=(--mcts-prior-mode "${MCTS_PRIOR_MODE}")
    cmd+=(--mcts-proposal-method "${MCTS_PROPOSAL_METHOD}")
    [[ -n "${PDB_PATH:-}" ]] && cmd+=(--pdb-path "${PDB_PATH}")
    [[ -n "${ALPHA_CHAIN:-}" ]] && cmd+=(--alpha-chain "${ALPHA_CHAIN}")
    [[ -n "${BETA_CHAIN:-}" ]] && cmd+=(--beta-chain "${BETA_CHAIN}")
    [[ -n "${HOTSPOT_CHAIN:-}" ]] && cmd+=(--hotspot-chain "${HOTSPOT_CHAIN}")
    [[ -n "${HOTSPOT_START:-}" ]] && cmd+=(--hotspot-start "${HOTSPOT_START}")
    [[ -n "${HOTSPOT_END:-}" ]] && cmd+=(--hotspot-end "${HOTSPOT_END}")
    [[ -n "${INIT_BINDER_SEQ:-}" ]] && cmd+=(--init-binder-seq "${INIT_BINDER_SEQ}")
    ;;

  cdr)
    require_var OUTPUT_DIR
    CDR_CONFIG="${CDR_CONFIG:-benchmark/yaml/cdr_tcr/cdr_tcr_domain_redesign.yaml}"
    : "${N_STEPS:=150}"
    : "${K_TOP:=8}"
    : "${B_CANDIDATES:=4}"
    mkdir -p "${OUTPUT_DIR}"
    cmd=("${PYTHON_BIN}" -u benchmark/script/run_cdr_tcr_domain_redesign.py)
    cmd+=(--config "${CDR_CONFIG}")
    cmd+=(--output-dir "${OUTPUT_DIR}")
    cmd+=(--log-file "${OUTPUT_DIR}/launcher.log")
    cmd+=(--mode mcts)
    cmd+=(--device "${DEVICE}")
    cmd+=(--model-name "${MODEL_NAME}")
    [[ -n "${PROTENIX_CHECKPOINT_DIR:-}" ]] && cmd+=(--model-checkpoint-dir "${PROTENIX_CHECKPOINT_DIR}")
    [[ -n "${COMPONENTS_CACHE_DIR:-}" ]] && cmd+=(--components-cache-dir "${COMPONENTS_CACHE_DIR}")
    [[ -n "${PROTENIX_ROOT_DIR:-}" ]] && cmd+=(--protenix-root-dir "${PROTENIX_ROOT_DIR}")
    cmd+=(--n-steps "${N_STEPS}")
    cmd+=(--k-top "${K_TOP}")
    cmd+=(--b-candidates "${B_CANDIDATES}")
    cmd+=(--mcts-strategy two_stage)
    cmd+=(--mcts-bootstrap-steps "${MCTS_BOOTSTRAP_STEPS}")
    cmd+=(--mcts-bootstrap-patience "${MCTS_BOOTSTRAP_PATIENCE}")
    cmd+=(--mcts-bootstrap-min-delta "${MCTS_BOOTSTRAP_MIN_DELTA}")
    cmd+=(--mcts-bootstrap-max-steps "${MCTS_BOOTSTRAP_MAX_STEPS}")
    cmd+=(--mcts-handoff-top-k "${MCTS_HANDOFF_TOP_K}")
    cmd+=(--mcts-expand-actions-per-step "${MCTS_EXPAND_ACTIONS_PER_STEP}")
    cmd+=(--mcts-prior-mode "${MCTS_PRIOR_MODE}")
    cmd+=(--mcts-proposal-method "${MCTS_PROPOSAL_METHOD}")
    cmd+=(--chunk-size "${CHUNK_SIZE}")
    cmd+=(--checkpoint-save-policy best_only)
    [[ -n "${CDR_CASE:-}" ]] && cmd+=(--case "${CDR_CASE}")
    ;;

  *)
    echo "Unsupported TASK=${TASK}. Use denovo, tcr, or cdr." >&2
    exit 2
    ;;
esac

echo "Running TASK=${TASK} with complete MCTS defaults"
printf ' %q' "${cmd[@]}"
echo
exec "${cmd[@]}"
