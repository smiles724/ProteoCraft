#!/bin/bash
#SBATCH --job-name=bindcraft_stage3
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/stage3_test_%j.out
#SBATCH --error=logs/stage3_test_%j.err

# 阶段 3 验证测试：无 Gradient Checkpoint 的 Pairformer 前向
# 验证 pairformer_forward_no_ckpt 能正常运行并梯度反传

echo "=========================================="
echo "BindCraft 阶段 3 验证（无 checkpoint Pairformer）"
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
    --output_dir outputs/stage3_test \
    --log_file logs/stage3_test.log

echo "=========================================="
echo "阶段 3 验证完成！"
echo "=========================================="
