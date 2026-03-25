#!/bin/bash
#SBATCH --job-name=bindcraft_v2_test
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

# BindCraft v2 快速测试脚本
# 在 debug 节点上运行 10 步测试

echo "=========================================="
echo "BindCraft v2 快速测试"
echo "=========================================="

# 激活环境
source ~/.bashrc
conda activate protenix

# 创建日志目录
mkdir -p logs outputs

# 运行测试（10 步）
cd /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft

python scripts/run_hallucination.py \
    --target GIVEQCCTSICSLYQLENYCN \
    --binder_length 40 \
    --n_steps 10 \
    --learning_rate 0.05 \
    --tau_noise 4.0 \
    --device cuda \
    --output_dir outputs/test \
    --log_file logs/test.log

echo "=========================================="
echo "测试完成！"
echo "=========================================="
