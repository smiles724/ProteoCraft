#!/bin/bash
#SBATCH -p debug
#SBATCH -J retro_<ENTRY>
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -o /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft/logs/retro_<ENTRY>_%%j.txt
#SBATCH -e /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft/logs/retro_<ENTRY>_%%j.err
#SBATCH -t 00:30:00

set -e
source ~/miniconda3/bin/activate protenix
cd /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft

echo ">>> 设计: <ENTRY>"
python evaluation/run_design.py --device cuda:0 --entry_ids <ENTRY>
echo ""
echo ">>> 评估: <ENTRY>"
python evaluation/run_eval.py --device cuda:0 --entry_ids <ENTRY>
echo ">>> 完成: $(date)"
