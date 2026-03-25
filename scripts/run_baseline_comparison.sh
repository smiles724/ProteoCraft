#!/bin/bash
#SBATCH --job-name=bindcraft_baseline
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

# BindCraft v2 Baseline 对比实验
# 在同一 target 上对比：BindCraft v2 vs Poly-Ala vs Random
#
# 使用上次 hallucination 优化的最佳结果：
#   best_loss = 3.7950 @step 33
#   best_seq  = PPHDKKYGKDQQDPSGHTMGSSSSSPSHAQHEDDRPSQHD

echo "=========================================="
echo "BindCraft v2 Baseline 对比实验"
echo "=========================================="

source ~/.bashrc
conda activate protenix

mkdir -p logs outputs/baseline
cd /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft

TARGET="LKDFDREVEIRIKQIESDRQNLLKEVDNLYNIEILRLPKALREMNWLDYFALGGN"
BINDCRAFT_BEST_SEQ="PPHDKKYGKDQQDPSGHTMGSSSSSPSHAQHEDDRPSQHD"
BINDCRAFT_BEST_LOSS="3.7950"

echo "Target: $TARGET"
echo "BindCraft best seq: $BINDCRAFT_BEST_SEQ"
echo "BindCraft best loss: $BINDCRAFT_BEST_LOSS"
echo ""

python scripts/run_baseline.py \
    --target "$TARGET" \
    --binder_length 40 \
    --tau_noise 4.0 \
    --n_samples 10 \
    --device cuda \
    --output_dir outputs/baseline \
    --bindcraft_best_loss "$BINDCRAFT_BEST_LOSS" \
    --bindcraft_best_seq "$BINDCRAFT_BEST_SEQ"

echo ""
echo "=========================================="
echo "Baseline 对比完成！"
echo "=========================================="
