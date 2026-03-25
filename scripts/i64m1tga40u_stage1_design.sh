#!/bin/bash
# Stage 1: 批量设计 10 个 case（充足时间）
#SBATCH -p i64m1tga40u
#SBATCH -J retro_design
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft/logs/stage1_design_%%j.txt
#SBATCH -e /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft/logs/stage1_design_%%j.err
#SBATCH -t 7-00:00:00

set -e
source ~/miniconda3/bin/activate protenix
cd /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft

echo ">>> Stage 1: 设计 10 个 case（50 步）"
python evaluation/run_design.py --device cuda:0 --n_steps 50

echo ">>> Stage 1 完成: $(date)"
echo ">>> 设计结果:"
ls -lh evaluation/outputs/designs/
