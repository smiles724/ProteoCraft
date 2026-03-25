#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH -J retrodiction_v2
#SBATCH -n 8
#SBATCH --gres=gpu:a40:1
#SBATCH -o /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft/logs/retrodiction_%j.txt
#SBATCH -e /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft/logs/retrodiction_%j.err
#SBATCH -t 7-00:00:00

set -e

WORKSPACE="/hpc2hdd/home/bduan436/Work_space/Stanford"
PROJECT="$WORKSPACE/Protenix_BindCraft"
CONDA_ENV="protenix"

echo "========== Retrodiction Pipeline v2 =========="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

# 激活 conda 环境
source ~/miniconda3/bin/activate "$CONDA_ENV"
cd "$PROJECT"

echo ">>> Step 1: 重新筛选案例（target_length <= 180）"
python evaluation/select_cases.py --top_n 10
echo ""

echo ">>> Step 2: 批量设计剩余案例"
python evaluation/run_design.py --device cuda:0
echo ""

echo ">>> Step 3: 零阶 AF3 评估（4组: designed/random/real_vh/real_full）"
python evaluation/run_eval.py --device cuda:0
echo ""

echo ">>> Step 4: 结果分析"
python evaluation/analyze_results.py
echo ""

echo "========== Pipeline 完成 =========="
echo "End: $(date)"
