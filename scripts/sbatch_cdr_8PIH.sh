#!/bin/bash
#SBATCH -p debug
#SBATCH -J cdr_8PIH
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -o logs/cdr_8PIH_%j.txt
#SBATCH -e logs/cdr_8PIH_%j.err
#SBATCH -t 02:00:00

set -e
source ~/miniconda3/bin/activate protenix
cd /root/BindcraftV2/Bindcraftv2

# CUDA_VISIBLE_DEVICES=X 隔离后只有1卡可见，设备索引重置为0
python scripts/run_hallucination_cdr.py \
    --pdb_id 8PIH \
    --mode hybrid \
    --n_steps 50 \
    --learning_rate 0.05 \
    --b_candidates 2 \
    --output_dir outputs/cdr_8PIH \
    --device cuda:0 \
    --log_file logs/cdr_8PIH_$(date +%Y%m%d_%H%M%S).log

echo ">>> 完成: $(date)"
