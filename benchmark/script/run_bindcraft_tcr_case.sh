#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/hpc2hdd/home/hlin199/.conda/envs/protenix/bin/python}"

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
    --config FILE          YAML config file (required)
    --pdb-id ID            PDB ID (overrides yaml)
    --alpha-chain CHAIN    Alpha chain ID (overrides yaml)
    --beta-chain CHAIN     Beta chain ID (overrides yaml)
    --hotspot-chain CHAIN  Hotspot chain: alpha or beta (overrides yaml)
    --hotspot-start N      Hotspot start residue (overrides yaml)
    --hotspot-end N        Hotspot end residue (overrides yaml)
    --mode MODE            hybrid or mcts (overrides yaml; default mcts)
    --prepare-only         Validate config only, don't run
    -h, --help            Show this help message

Examples:
    $(basename "$0") --config ../yaml/tcr/1qrn_alpha_53_56_template.yaml
    $(basename "$0") --config ../yaml/tcr/9c3e_alpha_91_94_custom_loss_hotspot.yaml --prepare-only
EOF
}

# Parse arguments
CONFIG_FILE=""
PDB_ID=""
ALPHA_CHAIN=""
BETA_CHAIN=""
HOTSPOT_CHAIN=""
HOTSPOT_START=""
HOTSPOT_END=""
MODE=""
PREPARE_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --config=*)
            CONFIG_FILE="${1#--config=}"
            shift
            ;;
        --pdb-id)
            PDB_ID="$2"
            shift 2
            ;;
        --alpha-chain)
            ALPHA_CHAIN="$2"
            shift 2
            ;;
        --beta-chain)
            BETA_CHAIN="$2"
            shift 2
            ;;
        --hotspot-chain)
            HOTSPOT_CHAIN="$2"
            shift 2
            ;;
        --hotspot-start)
            HOTSPOT_START="$2"
            shift 2
            ;;
        --hotspot-end)
            HOTSPOT_END="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --mode=*)
            MODE="${1#--mode=}"
            shift
            ;;
        --prepare-only)
            PREPARE_ONLY=1
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            show_help >&2
            exit 1
            ;;
    esac
done

# Validate config file
if [[ -z "${CONFIG_FILE}" ]]; then
    echo "Error: --config is required" >&2
    exit 1
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Error: Config file not found: ${CONFIG_FILE}" >&2
    exit 1
fi

CONFIG_FILE="$(cd "$(dirname "${CONFIG_FILE}")" && pwd)/$(basename "${CONFIG_FILE}")"

# Parse YAML using python
parse_yaml() {
    local yaml_file="$1"
    local key="$2"
    "${PYTHON_BIN}" -c "
import yaml
import sys
try:
    with open('${yaml_file}', 'r') as f:
        config = yaml.safe_load(f)
    value = config.get('${key}', '')
    print(value if value is not None else '')
except Exception as e:
    print('', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null || echo ""
}

yaml_key_is_null() {
    local yaml_file="$1"
    local key="$2"
    "${PYTHON_BIN}" -c "
import yaml
import sys
try:
    with open('${yaml_file}', 'r') as f:
        config = yaml.safe_load(f) or {}
    print('1' if '${key}' in config and config.get('${key}') is None else '0')
except Exception:
    print('0')
" 2>/dev/null || echo "0"
}

# Extract values from YAML (command-line overrides take precedence)
YAML_PDB_ID=$(parse_yaml "${CONFIG_FILE}" "pdb_id")
YAML_ALPHA_CHAIN=$(parse_yaml "${CONFIG_FILE}" "alpha_chain")
YAML_BETA_CHAIN=$(parse_yaml "${CONFIG_FILE}" "beta_chain")
YAML_HOTSPOT_CHAIN=$(parse_yaml "${CONFIG_FILE}" "hotspot_chain")
YAML_HOTSPOT_START=$(parse_yaml "${CONFIG_FILE}" "hotspot_start")
YAML_HOTSPOT_END=$(parse_yaml "${CONFIG_FILE}" "hotspot_end")
YAML_PDB_PATH=$(parse_yaml "${CONFIG_FILE}" "pdb_path")
YAML_OUTPUT_DIR=$(parse_yaml "${CONFIG_FILE}" "output_dir")
YAML_NO_DOWNLOAD=$(parse_yaml "${CONFIG_FILE}" "no_download")
YAML_TARGET_HOTSPOT=$(parse_yaml "${CONFIG_FILE}" "target_hotspot_residues")
YAML_HOTSPOT_PARAMS=$(parse_yaml "${CONFIG_FILE}" "hotspot_params")
YAML_HOTSPOT_PARAMS_NULL=$(yaml_key_is_null "${CONFIG_FILE}" "hotspot_params")
YAML_SOURCE_DIR=$(parse_yaml "${CONFIG_FILE}" "source_dir")
YAML_MODE=$(parse_yaml "${CONFIG_FILE}" "mode")
YAML_N_STEPS=$(parse_yaml "${CONFIG_FILE}" "n_steps")
YAML_LEARNING_RATE=$(parse_yaml "${CONFIG_FILE}" "learning_rate")
YAML_TAU_NOISE=$(parse_yaml "${CONFIG_FILE}" "tau_noise")
YAML_K_TOP=$(parse_yaml "${CONFIG_FILE}" "k_top")
YAML_B_CANDIDATES=$(parse_yaml "${CONFIG_FILE}" "b_candidates")
YAML_T_SAMPLE=$(parse_yaml "${CONFIG_FILE}" "t_sample")
YAML_MAX_MUTATIONS_PER_SEQ=$(parse_yaml "${CONFIG_FILE}" "max_mutations_per_seq")
YAML_T_INIT=$(parse_yaml "${CONFIG_FILE}" "t_init")
YAML_T_MIN=$(parse_yaml "${CONFIG_FILE}" "t_min")
YAML_CHUNK_SIZE=$(parse_yaml "${CONFIG_FILE}" "chunk_size")
YAML_USE_PRIOR=$(parse_yaml "${CONFIG_FILE}" "use_prior")
YAML_PRIOR_STRENGTH=$(parse_yaml "${CONFIG_FILE}" "prior_strength")
YAML_TASK_TYPE=$(parse_yaml "${CONFIG_FILE}" "task_type")
YAML_TARGET_CHAIN_ROLES=$(parse_yaml "${CONFIG_FILE}" "target_chain_roles")
YAML_MCTS_C=$(parse_yaml "${CONFIG_FILE}" "mcts_c")
YAML_MCTS_MAX_NODES=$(parse_yaml "${CONFIG_FILE}" "mcts_max_nodes")
YAML_MCTS_TAU_FIXED=$(parse_yaml "${CONFIG_FILE}" "mcts_tau_fixed")
YAML_MCTS_NUM_BEAMS=$(parse_yaml "${CONFIG_FILE}" "mcts_num_beams")
YAML_MCTS_STRATEGY=$(parse_yaml "${CONFIG_FILE}" "mcts_strategy")
YAML_MCTS_BOOTSTRAP_STEPS=$(parse_yaml "${CONFIG_FILE}" "mcts_bootstrap_steps")
YAML_MCTS_BOOTSTRAP_PATIENCE=$(parse_yaml "${CONFIG_FILE}" "mcts_bootstrap_patience")
YAML_MCTS_BOOTSTRAP_MIN_DELTA=$(parse_yaml "${CONFIG_FILE}" "mcts_bootstrap_min_delta")
YAML_MCTS_BOOTSTRAP_MAX_STEPS=$(parse_yaml "${CONFIG_FILE}" "mcts_bootstrap_max_steps")
YAML_MCTS_HANDOFF_TOP_K=$(parse_yaml "${CONFIG_FILE}" "mcts_handoff_top_k")
YAML_MCTS_EXPAND_ACTIONS_PER_STEP=$(parse_yaml "${CONFIG_FILE}" "mcts_expand_actions_per_step")
YAML_MCTS_PRIOR_MODE=$(parse_yaml "${CONFIG_FILE}" "mcts_prior_mode")
YAML_MCTS_PROPOSAL_METHOD=$(parse_yaml "${CONFIG_FILE}" "mcts_proposal_method")
YAML_MCTS_ADAPTIVE_TAU_START=$(parse_yaml "${CONFIG_FILE}" "mcts_adaptive_tau_start")
YAML_MCTS_ADAPTIVE_TAU_END=$(parse_yaml "${CONFIG_FILE}" "mcts_adaptive_tau_end")
YAML_MODEL_CHECKPOINT_DIR=$(parse_yaml "${CONFIG_FILE}" "model_checkpoint_dir")
YAML_COMPONENTS_CACHE_DIR=$(parse_yaml "${CONFIG_FILE}" "components_cache_dir")
YAML_MODEL_NAME=$(parse_yaml "${CONFIG_FILE}" "model_name")
YAML_DEVICE=$(parse_yaml "${CONFIG_FILE}" "device")
YAML_W_PLDDT=$(parse_yaml "${CONFIG_FILE}" "w_plddt")
YAML_W_PAE=$(parse_yaml "${CONFIG_FILE}" "w_pae")
YAML_W_I_PAE=$(parse_yaml "${CONFIG_FILE}" "w_i_pae")
YAML_W_CON=$(parse_yaml "${CONFIG_FILE}" "w_con")
YAML_W_I_CON=$(parse_yaml "${CONFIG_FILE}" "w_i_con")
YAML_W_IPTM=$(parse_yaml "${CONFIG_FILE}" "w_iptm")
YAML_W_RG=$(parse_yaml "${CONFIG_FILE}" "w_rg")
YAML_W_HELIX=$(parse_yaml "${CONFIG_FILE}" "w_helix")
YAML_W_CLASH=$(parse_yaml "${CONFIG_FILE}" "w_clash")
YAML_W_I_CLASH=$(parse_yaml "${CONFIG_FILE}" "w_i_clash")
YAML_W_PDE=$(parse_yaml "${CONFIG_FILE}" "w_pde")
YAML_W_I_PDE=$(parse_yaml "${CONFIG_FILE}" "w_i_pde")
NO_HOTSPOT_PARAMS=0
if [[ "${YAML_HOTSPOT_PARAMS_NULL}" == "1" ]]; then
    NO_HOTSPOT_PARAMS=1
fi

# Determine final values (CLI > YAML)
FINAL_PDB_ID="${PDB_ID:-${YAML_PDB_ID}}"
FINAL_ALPHA_CHAIN="${ALPHA_CHAIN:-${YAML_ALPHA_CHAIN}}"
FINAL_BETA_CHAIN="${BETA_CHAIN:-${YAML_BETA_CHAIN}}"
FINAL_HOTSPOT_CHAIN="${HOTSPOT_CHAIN:-${YAML_HOTSPOT_CHAIN}}"
FINAL_HOTSPOT_START="${HOTSPOT_START:-${YAML_HOTSPOT_START}}"
FINAL_HOTSPOT_END="${HOTSPOT_END:-${YAML_HOTSPOT_END}}"
FINAL_MODE="${MODE:-${YAML_MODE:-mcts}}"

# Validate required parameters
if [[ -z "${FINAL_PDB_ID}" ]]; then
    echo "Error: pdb_id is required (set in yaml or --pdb-id)" >&2
    exit 1
fi

if [[ -z "${FINAL_ALPHA_CHAIN}" ]]; then
    echo "Error: alpha_chain is required (set in yaml or --alpha-chain)" >&2
    exit 1
fi

if [[ -z "${FINAL_BETA_CHAIN}" ]]; then
    echo "Error: beta_chain is required (set in yaml or --beta-chain)" >&2
    exit 1
fi

if [[ "${NO_HOTSPOT_PARAMS}" -eq 0 && -z "${FINAL_HOTSPOT_CHAIN}" ]]; then
    echo "Error: hotspot_chain is required (set in yaml or --hotspot-chain)" >&2
    exit 1
fi

if [[ "${NO_HOTSPOT_PARAMS}" -eq 0 && -z "${FINAL_HOTSPOT_START}" ]]; then
    echo "Error: hotspot_start is required (set in yaml or --hotspot-start)" >&2
    exit 1
fi

if [[ "${NO_HOTSPOT_PARAMS}" -eq 0 && -z "${FINAL_HOTSPOT_END}" ]]; then
    echo "Error: hotspot_end is required (set in yaml or --hotspot-end)" >&2
    exit 1
fi

# Resolve PDB path
if [[ -n "${YAML_PDB_PATH}" ]]; then
    FINAL_PDB_PATH="${YAML_PDB_PATH}"
elif [[ -n "${YAML_SOURCE_DIR}" ]]; then
    FINAL_PDB_PATH="${YAML_SOURCE_DIR}/${FINAL_PDB_ID}.pdb"
else
    FINAL_PDB_PATH=""
fi

# Set output dir
if [[ -n "${YAML_OUTPUT_DIR}" ]]; then
    FINAL_OUTPUT_DIR="${YAML_OUTPUT_DIR}"
elif [[ "${NO_HOTSPOT_PARAMS}" -eq 1 ]]; then
    FINAL_OUTPUT_DIR="${PROJECT_ROOT}/outputs/bindcraft_v2_tcr_cases/${FINAL_PDB_ID}_no_hotspot"
else
    FINAL_OUTPUT_DIR="${PROJECT_ROOT}/outputs/bindcraft_v2_tcr_cases/${FINAL_PDB_ID}_${FINAL_HOTSPOT_CHAIN}_${FINAL_HOTSPOT_START}_${FINAL_HOTSPOT_END}"
fi

echo "[$(date '+%F %T')] TCR Case Runner"
echo "[$(date '+%F %T')] Config: ${CONFIG_FILE}"
echo "[$(date '+%F %T')] PDB ID: ${FINAL_PDB_ID}"
echo "[$(date '+%F %T')] Alpha chain: ${FINAL_ALPHA_CHAIN}, Beta chain: ${FINAL_BETA_CHAIN}"
echo "[$(date '+%F %T')] Mode: ${FINAL_MODE}"
if [[ "${NO_HOTSPOT_PARAMS}" -eq 1 ]]; then
    echo "[$(date '+%F %T')] Hotspot params: disabled"
else
    echo "[$(date '+%F %T')] Hotspot: ${FINAL_HOTSPOT_CHAIN} ${FINAL_HOTSPOT_START}-${FINAL_HOTSPOT_END}"
fi

if [[ -n "${FINAL_PDB_PATH}" ]]; then
    echo "[$(date '+%F %T')] PDB path: ${FINAL_PDB_PATH}"
fi
echo "[$(date '+%F %T')] Output dir: ${FINAL_OUTPUT_DIR}"

if [[ "${PREPARE_ONLY}" -eq 1 ]]; then
    echo "[$(date '+%F %T')] prepare-only mode, validation passed"
    exit 0
fi

# Build command
CMD=("${PYTHON_BIN}")
CMD+=("${SCRIPT_DIR}/run_bindcraft_tcr_case.py")
CMD+=(--pdb-id "${FINAL_PDB_ID}")
CMD+=(--alpha-chain "${FINAL_ALPHA_CHAIN}")
CMD+=(--beta-chain "${FINAL_BETA_CHAIN}")
CMD+=(--mode "${FINAL_MODE}")
CMD+=(--output-dir "${FINAL_OUTPUT_DIR}")

if [[ "${NO_HOTSPOT_PARAMS}" -eq 1 ]]; then
    CMD+=(--no-hotspot-params)
else
    CMD+=(--hotspot-chain "${FINAL_HOTSPOT_CHAIN}")
    CMD+=(--hotspot-start "${FINAL_HOTSPOT_START}")
    CMD+=(--hotspot-end "${FINAL_HOTSPOT_END}")
fi

if [[ -n "${FINAL_PDB_PATH}" ]]; then
    CMD+=(--pdb-path "${FINAL_PDB_PATH}")
fi

add_yaml_arg() {
    local value="$1"
    local flag="$2"
    if [[ -n "${value}" ]]; then
        CMD+=("${flag}" "${value}")
    fi
}

add_yaml_arg "${YAML_N_STEPS}" "--n-steps"
add_yaml_arg "${YAML_LEARNING_RATE}" "--learning-rate"
add_yaml_arg "${YAML_TAU_NOISE}" "--tau-noise"
add_yaml_arg "${YAML_K_TOP}" "--k-top"
add_yaml_arg "${YAML_B_CANDIDATES}" "--b-candidates"
add_yaml_arg "${YAML_T_SAMPLE}" "--t-sample"
add_yaml_arg "${YAML_MAX_MUTATIONS_PER_SEQ}" "--max-mutations-per-seq"
add_yaml_arg "${YAML_T_INIT}" "--t-init"
add_yaml_arg "${YAML_T_MIN}" "--t-min"
add_yaml_arg "${YAML_CHUNK_SIZE}" "--chunk-size"
add_yaml_arg "${YAML_PRIOR_STRENGTH}" "--prior-strength"
add_yaml_arg "${YAML_TASK_TYPE}" "--task-type"
add_yaml_arg "${YAML_TARGET_CHAIN_ROLES}" "--target-chain-roles"
add_yaml_arg "${YAML_MCTS_C}" "--mcts-c"
add_yaml_arg "${YAML_MCTS_MAX_NODES}" "--mcts-max-nodes"
add_yaml_arg "${YAML_MCTS_TAU_FIXED}" "--mcts-tau-fixed"
add_yaml_arg "${YAML_MCTS_NUM_BEAMS}" "--mcts-num-beams"
add_yaml_arg "${YAML_MCTS_STRATEGY}" "--mcts-strategy"
add_yaml_arg "${YAML_MCTS_BOOTSTRAP_STEPS}" "--mcts-bootstrap-steps"
add_yaml_arg "${YAML_MCTS_BOOTSTRAP_PATIENCE}" "--mcts-bootstrap-patience"
add_yaml_arg "${YAML_MCTS_BOOTSTRAP_MIN_DELTA}" "--mcts-bootstrap-min-delta"
add_yaml_arg "${YAML_MCTS_BOOTSTRAP_MAX_STEPS}" "--mcts-bootstrap-max-steps"
add_yaml_arg "${YAML_MCTS_HANDOFF_TOP_K}" "--mcts-handoff-top-k"
add_yaml_arg "${YAML_MCTS_EXPAND_ACTIONS_PER_STEP}" "--mcts-expand-actions-per-step"
add_yaml_arg "${YAML_MCTS_PRIOR_MODE}" "--mcts-prior-mode"
add_yaml_arg "${YAML_MCTS_PROPOSAL_METHOD}" "--mcts-proposal-method"
add_yaml_arg "${YAML_MCTS_ADAPTIVE_TAU_START}" "--mcts-adaptive-tau-start"
add_yaml_arg "${YAML_MCTS_ADAPTIVE_TAU_END}" "--mcts-adaptive-tau-end"
add_yaml_arg "${YAML_MODEL_CHECKPOINT_DIR}" "--model-checkpoint-dir"
add_yaml_arg "${YAML_COMPONENTS_CACHE_DIR}" "--components-cache-dir"
add_yaml_arg "${YAML_MODEL_NAME}" "--model-name"
add_yaml_arg "${YAML_DEVICE}" "--device"
add_yaml_arg "${YAML_W_PLDDT}" "--w-plddt"
add_yaml_arg "${YAML_W_PAE}" "--w-pae"
add_yaml_arg "${YAML_W_I_PAE}" "--w-i-pae"
add_yaml_arg "${YAML_W_CON}" "--w-con"
add_yaml_arg "${YAML_W_I_CON}" "--w-i-con"
add_yaml_arg "${YAML_W_IPTM}" "--w-iptm"
add_yaml_arg "${YAML_W_RG}" "--w-rg"
add_yaml_arg "${YAML_W_HELIX}" "--w-helix"
add_yaml_arg "${YAML_W_CLASH}" "--w-clash"
add_yaml_arg "${YAML_W_I_CLASH}" "--w-i-clash"
add_yaml_arg "${YAML_W_PDE}" "--w-pde"
add_yaml_arg "${YAML_W_I_PDE}" "--w-i-pde"

if [[ -n "${YAML_NO_DOWNLOAD}" && "${YAML_NO_DOWNLOAD}" == "true" ]]; then
    CMD+=(--no-download)
fi

if [[ -n "${YAML_USE_PRIOR}" ]]; then
    if [[ "${YAML_USE_PRIOR}" == "false" || "${YAML_USE_PRIOR}" == "False" || "${YAML_USE_PRIOR}" == "0" ]]; then
        CMD+=(--no-use-prior)
    else
        CMD+=(--use-prior)
    fi
fi

# Run
echo "[$(date '+%F %T')] Starting TCR case..."
cd "${PROJECT_ROOT}"
PYTHONPATH="${PROJECT_ROOT}" "${CMD[@]}"
