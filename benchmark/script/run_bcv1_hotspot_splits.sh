#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCHMARK_DIR="${ROOT_DIR}/benchmark"
YAML_DIR="${BENCHMARK_DIR}/yaml/bcv1"
LOG_BASE="${BENCHMARK_DIR}/log/bcv1"
RESULTS_DIR="${BENCHMARK_DIR}/results/bcv1"
PYTHON_BIN="${PYTHON_BIN:-/home/huangyanjie/miniconda3/bin/python}"

mkdir -p "$RESULTS_DIR"

configs=(
  "derf21_custom_loss_hotspot_B10:0"
  "derf21_custom_loss_hotspot_B28:1"
  "derf21_custom_loss_hotspot_B63:2"
  "derf21_custom_loss_hotspot_B81:3"
  "derf21_custom_loss_hotspot_B92:4"
  "derf7_custom_loss_hotspot_B132:5"
  "derf7_custom_loss_hotspot_B178:6"
  "derf7_custom_loss_hotspot_B201:7"
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

echo "All jobs launched."
