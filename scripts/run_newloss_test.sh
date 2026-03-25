#!/bin/bash
#SBATCH --job-name=bc_newloss
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/newloss_%j.out
#SBATCH --error=logs/newloss_%j.err

echo "=========================================="
echo "BindCraft v2 - 新 Loss 体系测试"
echo "Loss: pLDDT + pAE + i_pAE + con + i_con + iPTM + Rg + helix"
echo "=========================================="

source ~/.bashrc
conda activate protenix

mkdir -p logs outputs
cd /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft

TARGET="LKDFDREVEIRIKQIESDRQNLLKEVDNLYNIEILRLPKALREMNWLDYFALGGN"

python scripts/run_hallucination.py \
    --target "$TARGET" \
    --binder_length 40 \
    --n_steps 200 \
    --learning_rate 0.05 \
    --tau_noise 4.0 \
    --device cuda \
    --output_dir "outputs/newloss_test" \
    --log_file "logs/newloss_test.log"

echo "=========================================="
echo "测试完成"
echo "=========================================="
