#!/bin/bash
#SBATCH -p debug
#SBATCH -J eval_8pih
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -o /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft/logs/eval_8pih_%%j.txt
#SBATCH -e /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft/logs/eval_8pih_%%j.err
#SBATCH -t 00:30:00

set -e
source ~/miniconda3/bin/activate protenix
cd /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft

echo ">>> 评估: 8pih"
python evaluation/run_eval.py --device cuda:0 --entry_ids "8pih"
echo ">>> 完成: $(date)"
