#!/bin/bash
#SBATCH --job-name=bindcraft_retro
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/retrodiction_9d41_%j.out
#SBATCH --error=logs/retrodiction_9d41_%j.err

# Retrodiction 测试：PXM-22to25-Ab-Ag entry 9d41（target 55 aa）
# 检查当前 BindCraft 版本在真实 target 上的效果

echo "=========================================="
echo "BindCraft Retrodiction 测试（9d41, 200 步）"
echo "=========================================="

source ~/.bashrc
conda activate protenix

mkdir -p logs outputs
cd /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft

TARGET_9d41="LKDFDREVEIRIKQIESDRQNLLKEVDNLYNIEILRLPKALREMNWLDYFALGGN"

python scripts/run_hallucination.py \
    --target "$TARGET_9d41" \
    --binder_length 40 \
    --n_steps 200 \
    --learning_rate 0.05 \
    --tau_noise 4.0 \
    --device cuda \
    --output_dir outputs/retrodiction_9d41 \
    --log_file logs/retrodiction_9d41.log

echo "=========================================="
echo "Retrodiction 测试完成！"
echo "=========================================="
