#!/bin/bash
#SBATCH --job-name=bindcraft_tuned
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/tuned_%j.out
#SBATCH --error=logs/tuned_%j.err

# 调参实验：提高 iptm 权重，降低 attraction 权重，减小 tau_noise
# 目标：让优化真正朝 iPTM 方向走，而非仅靠近 target

echo "=========================================="
echo "BindCraft 调参实验（iptm 权重 2.0）"
echo "=========================================="

source ~/.bashrc
conda activate protenix

mkdir -p logs outputs
cd /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft

TARGET_9d41="LKDFDREVEIRIKQIESDRQNLLKEVDNLYNIEILRLPKALREMNWLDYFALGGN"

python scripts/run_hallucination.py \
    --target "$TARGET_9d41" \
    --binder_length 40 \
    --n_steps 400 \
    --learning_rate 0.02 \
    --tau_noise 2.0 \
    --device cuda \
    --output_dir outputs/tuned \
    --log_file logs/tuned.log

echo "=========================================="
echo "调参实验完成！"
echo "=========================================="
