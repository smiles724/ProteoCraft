#!/bin/bash
# 10 个 debug 作业（每 case 25 步设计 + 4 组 eval，约 20 分钟/作业，安全留 10 分钟余量）
# QOS 限制每用户最多 8 个并发，先提交 8 个，跑完配额释放后再补 2 个

PROJECT="/hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft"
CONDA_ENV="protenix"
mkdir -p "$PROJECT/logs"

ENTRIES=(
    "8trt" "9ho4" "8yat" "8yaf"
    "8trs" "8pih" "8zd5" "9g2a"
    # 8r9z 和 8f60 等前 8 个跑完后补（运行: bash .../submit_last2.sh）
)

for ENTRY in "${ENTRIES[@]}"; do
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

echo ">>> 设计: \$ENTRY (25 步)"
python evaluation/run_design.py --device cuda:0 --n_steps 25 --entry_ids "\$ENTRY"
echo ">>> 评估: \$ENTRY"
python evaluation/run_eval.py --device cuda:0 --entry_ids "\$ENTRY"
echo ">>> 完成: \$(date)"
EOF
    chmod +x "$SCRIPT"
    echo ">>> 提交: $ENTRY"
    sbatch "$SCRIPT"
    sleep 3
done

echo ""
echo "=== 前 8 个已提交 ==="
squeue -u bduan436 -p debug
echo ""
echo ">>> 等这 8 个跑完后补交 8r9z 和 8f60:"
echo "   bash $PROJECT/scripts/submit_last2.sh"
