#!/bin/bash
#SBATCH --job-name=bindcraft_full
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/full_test_%j.out
#SBATCH --error=logs/full_test_%j.err

# BindCraft v2 完整测试脚本
# 运行 200 步优化，验证 x_coarse 修复后的完整流程

echo "=========================================="
echo "BindCraft v2 完整测试（200 步）"
echo "=========================================="

# 激活环境
source ~/.bashrc
conda activate protenix

# 创建日志目录
mkdir -p logs outputs

# 运行完整测试（200 步）
cd /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft

python scripts/run_hallucination.py \
    --target GIVEQCCTSICSLYQLENYCN \
    --binder_length 40 \
    --n_steps 200 \
    --learning_rate 0.05 \
    --tau_noise 4.0 \
    --device cuda \
    --output_dir outputs/full_test \
    --log_file logs/full_test.log

echo "=========================================="
echo "完整测试完成！"
echo "=========================================="
