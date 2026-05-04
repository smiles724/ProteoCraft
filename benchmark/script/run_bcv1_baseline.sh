#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCHMARK_DIR="${ROOT_DIR}/benchmark"
YAML_DIR="${BENCHMARK_DIR}/yaml/bcv1/baseline"
LOG_BASE="${BENCHMARK_DIR}/log/bcv1/baseline"
RESULTS_DIR="${BENCHMARK_DIR}/results/bcv1/baseline"
PYTHON_BIN="${PYTHON_BIN:-/home/huangyanjie/miniconda3/bin/python}"

mkdir -p "$LOG_BASE" "$RESULTS_DIR"

configs=(
  "betv1_baseline:0"
  "derf21_baseline:1"
  "derf7_baseline:2"
  "her2_baseline:3"
  "ifnar2_baseline:4"
  "pd1_baseline:5"
  "pd_l1_baseline:6"
  "spcas9_baseline:7"
)

for item in "${configs[@]}"; do
  config_name="${item%%:*}"
  gpu="${item##*:}"
  config_file="${YAML_DIR}/${config_name}.yaml"
  log_dir="${LOG_BASE}/${config_name}"
  mkdir -p "$log_dir"

  echo "Launching ${config_name} on GPU ${gpu}..."
  CUDA_VISIBLE_DEVICES="$gpu" \
    LOG_DIR="$log_dir" \
    nohup "${PYTHON_BIN}" "${BENCHMARK_DIR}/run_benchmark.py" \
    --config "$config_file" \
    > "${log_dir}/run.log" 2>&1 &
  echo "  pid=$!"
done

echo "All jobs launched. Use 'jobs' to see running jobs."
