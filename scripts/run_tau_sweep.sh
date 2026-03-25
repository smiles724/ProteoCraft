#!/bin/bash
#SBATCH --job-name=bc_tau_sweep
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/tau_sweep_%j.out
#SBATCH --error=logs/tau_sweep_%j.err

# tau_noise 扫描实验
# Protenix EDM 框架：tau 是噪声标准差，范围 [4e-4, 160]
# 当前默认 tau=4.0（中间偏低），本实验对比 tau=4 / 20 / 80

echo "=========================================="
echo "tau_noise 扫描实验（4 / 20 / 80）"
echo "=========================================="

source ~/.bashrc
conda activate protenix

mkdir -p logs outputs
cd /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft

TARGET="LKDFDREVEIRIKQIESDRQNLLKEVDNLYNIEILRLPKALREMNWLDYFALGGN"

for TAU in 4.0 20.0 80.0; do
    echo ""
    echo "=========================================="
    echo ">>> tau_noise = $TAU"
    echo "=========================================="

    python scripts/run_hallucination.py \
        --target "$TARGET" \
        --binder_length 40 \
        --n_steps 200 \
        --learning_rate 0.05 \
        --tau_noise "$TAU" \
        --device cuda \
        --output_dir "outputs/tau_${TAU}" \
        --log_file "logs/tau_${TAU}.log"

    echo ">>> tau=$TAU 完成，best seq:"
    grep "Best Sequence" "outputs/tau_${TAU}/hallucination_result.txt"
done

echo ""
echo "=========================================="
echo "tau 扫描完成！汇总："
for TAU in 4.0 20.0 80.0; do
    echo -n "tau=$TAU | "
    grep "Best Sequence" "outputs/tau_${TAU}/hallucination_result.txt" 2>/dev/null || echo "未完成"
done
echo "=========================================="
