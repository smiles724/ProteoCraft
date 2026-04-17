#!/bin/bash
# 监控所有正在运行的设计任务进度
# 用法: bash monitor_designs.sh

LOG_DIR="/root/BindcraftV2/Stanford/Protenix_BindCraft/logs"
DESIGNS_DIR="/root/BindcraftV2/Stanford/Protenix_BindCraft/evaluation/outputs/designs"

echo "============================================================"
echo "BindCraft v2 — 批量设计监控 $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# 1. 已完成的设计
echo "【已完成】"
completed=$(ls $DESIGNS_DIR/*.json 2>/dev/null | xargs -I{} basename {} .json)
if [ -z "$completed" ]; then
    echo "  暂无"
else
    for eid in $completed; do
        loss=$(python3 -c "
import json
with open('$DESIGNS_DIR/${eid}.json') as f:
    d = json.load(f)
hist = d.get('design_loss_history', [])
best = min(hist) if hist else None
print(f'best={best:.4f}' if best else 'N/A')
" 2>/dev/null)
        echo "  ✅ $eid  ($loss)"
    done
fi

echo ""

# 2. 正在运行的设计（通过GPU显存判断）
echo "【正在运行】"
running=0
total=0
for log in $(ls -t $LOG_DIR/retrodiction_design_20260327_*.log 2>/dev/null | grep -v "171946\|161836" | head -20); do
    last_step=$(grep "Step  " "$log" 2>/dev/null | tail -1)
    if [ -n "$last_step" ]; then
        entry_id=$(grep "设计 " "$log" 2>/dev/null | head -1 | grep -oP '\d[a-z]{3}')
        if [ -z "$entry_id" ]; then
            entry_id=$(grep "正在其他实例中\|已存在" "$log" 2>/dev/null | tail -1 | grep -oP '\d[a-z]{3}')
        fi
        step=$(echo "$last_step" | grep -oP 'Step\s+\d+/50')
        phase=$(echo "$last_step" | grep -oP 'phase=\w+')
        gbest=$(echo "$last_step" | grep -oP 'global_best=\S+')
        accepted=$(echo "$last_step" | grep -oP 'accepted=\w+')
        echo "  🟡 ${entry_id:-?} | $step | $phase | $gbest | $accepted"
        running=$((running+1))
    fi
    total=$((total+1))
done

if [ $running -eq 0 ]; then
    echo "  暂无"
fi

echo ""

# 3. 锁文件检测
echo "【锁文件（其他实例正在设计中）】"
locks=$(ls $DESIGNS_DIR/*.lock 2>/dev/null | xargs -I{} basename {} .lock | sort)
if [ -z "$locks" ]; then
    echo "  无"
else
    for lk in $locks; do
        echo "  🔒 $lk (被其他实例持有)"
    done
fi

echo ""

# 4. 汇总
target_total=11
done_count=$(ls $DESIGNS_DIR/*.json 2>/dev/null | wc -l)
remaining=$((target_total - done_count))
echo "进度: $done_count/$target_total 完成 | $remaining 待完成"
echo ""
echo "============================================================"
