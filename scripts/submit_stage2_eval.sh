#!/bin/bash
# Stage 2: 分 5 批提交评估作业（每批 2 个 case，每作业约 15-20 分钟）
# 每批只做 eval（设计结果已由 Stage 1 产出）
# 等 Stage 1 (9590964) 完成后立即提交

PROJECT="/hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft"
CONDA_ENV="protenix"
TEMPLATE="$PROJECT/scripts/debug_retro_TEMPLATE.sh"
mkdir -p "$PROJECT/logs"

# 5 批次: (批次名, entry_ids)
BATCHES=(
    "eval1:8trt,9ho4"
    "eval2:8yat,8yaf"
    "eval3:8trs,8pih"
    "eval4:8zd5,9g2a"
    "eval5:8r9z,8f60"
)

for item in "${BATCHES[@]}"; do
    name="${item%%:*}"
    ids="${item##*:}"
    SCRIPT="$PROJECT/scripts/sbatch_debug_${name}.sh"

    cat > "$SCRIPT" << EOF
#!/bin/bash
#SBATCH -p debug
#SBATCH -J $name
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -o $PROJECT/logs/${name}_%%j.txt
#SBATCH -e $PROJECT/logs/${name}_%%j.err
#SBATCH -t 00:30:00

set -e
source ~/miniconda3/bin/activate $CONDA_ENV
cd $PROJECT
echo ">>> 评估: $ids"
python evaluation/run_eval.py --device cuda:0 --entry_ids "$ids"
echo ">>> 完成: \$(date)"
EOF
    chmod +x "$SCRIPT"
done

# 逐个提交，每批等 1 秒避免 QOS 限制
for item in "${BATCHES[@]}"; do
    name="${item%%:*}"
    SCRIPT="$PROJECT/scripts/sbatch_debug_${name}.sh"
    echo ">>> 提交 debug eval: $name"
    sbatch "$SCRIPT"
    sleep 2
done

echo ""
echo "=== 全部提交完成 ==="
squeue -u bduan436 -p debug
