#!/bin/bash
#SBATCH -p debug
#SBATCH -J retro_8r9z
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -o /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft/logs/retro_8r9z_%%j.txt
#SBATCH -e /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft/logs/retro_8r9z_%%j.err
#SBATCH -t 00:30:00

set -e
source ~/miniconda3/bin/activate protenix
cd /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft

echo ">>> 设计: 8r9z"
python evaluation/run_design.py --device cuda:0 --entry_ids 8r9z
echo ""
echo ">>> 评估: 8r9z"
python evaluation/run_eval.py --device cuda:0 --entry_ids 8r9z
echo ">>> 完成: $(date)"
