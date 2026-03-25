#!/bin/bash
# 补交 8r9z 和 8f60（等前 8 个跑完、QOS 配额释放后运行）
PROJECT="/hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft"
CONDA_ENV="protenix"

for ENTRY in "8r9z" "8f60"; do
    SCRIPT="$PROJECT/scripts/sbatch_debug_${ENTRY}.sh"
    cat > "$SCRIPT" << EOF
#!/bin/bash
#SBATCH -p debug
#SBATCH -J retro_\${ENTRY}
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -o $PROJECT/logs/debug_\${ENTRY}_%%j.txt
#SBATCH -e $PROJECT/logs/debug_\${ENTRY}_%%j.err
#SBATCH -t 00:30:00

set -e
source ~/miniconda3/bin/activate $CONDA_ENV
cd $PROJECT

echo ">>> 设计: \$ENTRY"
python evaluation/run_design.py --device cuda:0 --n_steps 25 --entry_ids "\$ENTRY"
echo ">>> 设计完成，开始评估..."
python evaluation/run_eval.py --device cuda:0 --entry_ids "\$ENTRY"
echo ">>> 全部完成: \$(date)"
EOF
    chmod +x "$SCRIPT"
    echo ">>> 提交 debug: $ENTRY"
    sbatch "$SCRIPT"
    sleep 3
done

squeue -u bduan436 -p debug
