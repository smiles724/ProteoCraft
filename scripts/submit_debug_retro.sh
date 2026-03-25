#!/bin/bash
# 提交 10 个 debug 作业（每个 1 个 case，30 分钟）
# debug 队列免费，每个作业独立排队

PROJECT="/hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft"
CONDA_ENV="protenix"
TEMPLATE="$PROJECT/scripts/debug_retro_TEMPLATE.sh"

# 10 个 entry_id
ENTRIES=(
    "8trt" "9ho4" "8yat" "8yaf"
    "8trs" "8pih" "8zd5" "9g2a"
    "8r9z" "8f60"
)

mkdir -p "$PROJECT/logs"

for ENTRY in "${ENTRIES[@]}"; do
    SCRIPT="$PROJECT/scripts/sbatch_debug_${ENTRY}.sh"

    # 替换模板中的 <ENTRY> 占位符
    sed "s/<ENTRY>/$ENTRY/g" "$TEMPLATE" > "$SCRIPT"
    chmod +x "$SCRIPT"

    echo ">>> 提交 debug: $ENTRY"
    sbatch "$SCRIPT"
done

echo ""
echo "=== 提交完成 ==="
squeue -u bduan436
