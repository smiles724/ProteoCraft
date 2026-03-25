#!/bin/bash
#SBATCH -p debug
#SBATCH -J retro_design
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -o /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft/logs/stage1_design_%%j.txt
#SBATCH -e /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft/logs/stage1_design_%%j.err
#SBATCH -t 00:35:00

set -e
source ~/miniconda3/bin/activate protenix
cd /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft

echo ">>> Stage 1: 批量设计 10 个 case（35 步）"
python evaluation/run_design.py --device cuda:0 --n_steps 35

echo ">>> Stage 1 完成: $(date)"
echo ">>> 等待 Stage 2 eval 作业..."
ls -lh evaluation/outputs/designs/
