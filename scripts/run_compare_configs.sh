#!/bin/bash
#SBATCH --job-name=bc_compare
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/compare_%j.out
#SBATCH --error=logs/compare_%j.err

echo "=========================================="
echo "BindCraft 两组配置对比评估"
echo "=========================================="

source ~/.bashrc
conda activate protenix

mkdir -p logs outputs/compare
cd /hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft

TARGET="LKDFDREVEIRIKQIESDRQNLLKEVDNLYNIEILRLPKALREMNWLDYFALGGN"

# 旧配置 best（iptm=0.1, lr=0.05, tau=4.0, 200步）
SEQ_OLD="PPHDKKYGKDQQDPSGHTMGSSSSSPSHAQHEDDRPSQHD"
# 新配置 best（iptm=2.0, lr=0.02, tau=2.0, 400步）
SEQ_NEW="SPGEHSKNENSTGPEPTCTTTPTPDNPSDDPGDDNSDDHS"

python - <<EOF
import os, sys, json, logging, torch, numpy as np
sys.path.insert(0, '/hpc2hdd/home/bduan436/Work_space/Stanford/Protenix_BindCraft')

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

from scripts.run_hallucination import load_protenix_model
from scripts.run_full_eval import evaluate_seq_full

model, _ = load_protenix_model("protenix_base_default_v1.0.0", "cuda")
device = torch.device("cuda")

TARGET = "$TARGET"
L = 40
np.random.seed(0)
aa_chars = list("ACDEFGHIKLMNPQRSTVWY")

seqs = {
    "Poly-Ala":         "A" * L,
    "Random":           "".join(np.random.choice(aa_chars, size=L)),
    "Old(iptm=0.1,tau=4.0)": "$SEQ_OLD",
    "New(iptm=2.0,tau=2.0)": "$SEQ_NEW",
}

all_stats = {}
for label, seq in seqs.items():
    logger.info("\n" + "=" * 60)
    logger.info(f"{label}: {seq}")
    logger.info("=" * 60)
    all_stats[label] = evaluate_seq_full(
        model=model, binder_seq=seq, target_seq=TARGET,
        device=device, n_repeats=3, label=label,
    )

# 报告
logger.info("\n" + "=" * 80)
logger.info("配置对比报告（Protenix 完整推理，独立指标）")
logger.info("=" * 80)
logger.info(f"{'Method':<28} {'pLDDT↑':>10} {'iPTM↑':>10} {'pTM↑':>10} {'ranking↑':>10}")
logger.info("-" * 68)
for label, stats in all_stats.items():
    plddt = stats.get("plddt", {})
    iptm  = stats.get("iptm",  {})
    ptm   = stats.get("ptm",   {})
    rank  = stats.get("ranking_score", {})
    logger.info(
        f"{label:<28} "
        f"{plddt.get('mean',0):>6.2f}±{plddt.get('std',0):.2f}  "
        f"{iptm.get('mean',0):>6.3f}±{iptm.get('std',0):.3f}  "
        f"{ptm.get('mean',0):>6.3f}±{ptm.get('std',0):.3f}  "
        f"{rank.get('mean',0):>6.3f}±{rank.get('std',0):.3f}"
    )

# 直接比较两个 BindCraft 配置
old = all_stats["Old(iptm=0.1,tau=4.0)"]
new = all_stats["New(iptm=2.0,tau=2.0)"]
logger.info("")
logger.info("新配置 vs 旧配置：")
for metric, label in [("plddt","pLDDT"), ("iptm","iPTM"), ("ranking_score","ranking")]:
    o = old.get(metric,{}).get("mean",0)
    n = new.get(metric,{}).get("mean",0)
    better = "✅ 新更好" if n > o else ("❌ 旧更好" if o > n else "= 相同")
    logger.info(f"  {label}: 旧={o:.3f}  新={n:.3f}  {better}")

out = {"target": TARGET, "sequences": seqs, "stats": all_stats}
with open("outputs/compare/compare_results.json","w") as f:
    json.dump(out, f, indent=2)
logger.info("\n结果已保存到 outputs/compare/compare_results.json")
EOF

echo "=========================================="
echo "对比评估完成！"
echo "=========================================="
