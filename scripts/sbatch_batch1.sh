#!/bin/bash
#SBATCH -p i64m1tga40u
#SBATCH -J retro_batch1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft/logs/batch1_%j.txt
#SBATCH -e /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft/logs/batch1_%j.err
#SBATCH -t 7-00:00:00

set -e
source ~/miniconda3/bin/activate protenix
cd /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft

echo ">>> 设计阶段: $ENTRY_IDS"
python evaluation/run_design.py --device cuda:0 --entry_ids "8trt,9ho4"
echo ""

echo ">>> 评估阶段: $ENTRY_IDS"
python evaluation/run_eval.py --device cuda:0 --entry_ids "8trt,9ho4"
echo ""

echo ">>> 完成: 2026-03-20T19:02:55 CST"
