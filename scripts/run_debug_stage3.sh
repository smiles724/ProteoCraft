#!/bin/bash
#SBATCH --job-name=bc_debug3
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=00:15:00
#SBATCH --output=logs/debug_stage3_%j.out
#SBATCH --error=logs/debug_stage3_%j.err

source ~/.bashrc
conda activate protenix

mkdir -p logs
cd /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft

python scripts/debug_stage3.py 2>&1 | tee logs/debug_stage3.log
