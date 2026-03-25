#!/bin/bash
#SBATCH -p debug
#SBATCH -J eval_8trt
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -o /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft/logs/eval_8trt_%%j.txt
#SBATCH -e /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft/logs/eval_8trt_%%j.err
#SBATCH -t 00:30:00

set -e
source ~/miniconda3/bin/activate protenix
cd /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft

echo ">>> 评估: 8trt"
python evaluation/run_eval.py --device cuda:0 --entry_ids "8trt"
echo ">>> 完成: $(date)"
