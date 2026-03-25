#!/bin/bash
#SBATCH --job-name=bindcraft_eval
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/full_eval_%j.out
#SBATCH --error=logs/full_eval_%j.err

# BindCraft v2 完整结构评估
# 使用 Protenix 完整多步 ODE 推理，计算独立于优化目标的真实指标
# 指标：pLDDT, iPTM, pTM, ranking_score, chain_pair_iPTM

echo "=========================================="
echo "BindCraft v2 完整结构评估（独立指标）"
echo "=========================================="

source ~/.bashrc
conda activate protenix

mkdir -p logs outputs/full_eval
cd /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft

TARGET="LKDFDREVEIRIKQIESDRQNLLKEVDNLYNIEILRLPKALREMNWLDYFALGGN"
BINDCRAFT_BEST_SEQ="PPHDKKYGKDQQDPSGHTMGSSSSSPSHAQHEDDRPSQHD"
BINDCRAFT_BEST_LOSS="3.7950"

echo "Target: $TARGET"
echo "BindCraft best seq: $BINDCRAFT_BEST_SEQ"
echo ""

python scripts/run_full_eval.py \
    --target "$TARGET" \
    --binder_length 40 \
    --n_repeats 3 \
    --device cuda \
    --output_dir outputs/full_eval \
    --bindcraft_best_seq "$BINDCRAFT_BEST_SEQ" \
    --bindcraft_best_loss "$BINDCRAFT_BEST_LOSS"

echo ""
echo "=========================================="
echo "完整评估完成！"
echo "=========================================="
