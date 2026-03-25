#!/bin/bash
# 分批提交 retropipeline v2 作业
# 10 个 case 分 5 批，每批 2 个（8f60 单独一批）
# 5 个作业同时排队，哪个先拿到 GPU 先跑

PROJECT="/hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft"
CONDA_ENV="protenix"

# 批次定义：(批次名, entry_ids)
declare -a BATCHES=(
    "batch1:8trt,9ho4"
    "batch2:8yat,8yaf"
    "batch3:8trs,8pih"
    "batch4:8zd5,9g2a"
    "batch5:8r9z,8f60"
)

for item in "${BATCHES[@]}"; do
    name="${item%%:*}"
    ids="${item##*:}"
    job_name="retro_${name}"
    log_dir="$PROJECT/logs"
    mkdir -p "$log_dir"

    script="$PROJECT/scripts/sbatch_${name}.sh"
    cat > "$script" << EOF
#!/bin/bash
#SBATCH -p i64m1tga40u
#SBATCH -J $job_name
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o $log_dir/${name}_%j.txt
#SBATCH -e $log_dir/${name}_%j.err
#SBATCH -t 7-00:00:00

set -e
source ~/miniconda3/bin/activate $CONDA_ENV
cd $PROJECT

echo ">>> 设计阶段: \$ENTRY_IDS"
python evaluation/run_design.py --device cuda:0 --entry_ids "$ids"
echo ""

echo ">>> 评估阶段: \$ENTRY_IDS"
python evaluation/run_eval.py --device cuda:0 --entry_ids "$ids"
echo ""

echo ">>> 完成: $(date)"
EOF
    chmod +x "$script"

    echo ">>> 提交 $name (entry_ids=$ids)"
    sbatch "$script"
done

echo ""
echo "=== 提交完成，当前队列状态 ==="
squeue -u bduan436
