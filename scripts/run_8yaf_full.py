#!/usr/bin/env python3
"""
8YAF Retrodiction 完整运行脚本（CDR-Aware + Eval）

1. CDR 设计 → 保存到 evaluation/outputs/designs/8yaf.json
2. Retrodiction 评估 → 保存到 evaluation/outputs/eval_results.csv

用法:
    CUDA_VISIBLE_DEVICES=1 nohup python scripts/run_8yaf_full.py > logs/run_8yaf_full.log 2>&1 &
"""
import os, sys, json, time, random, csv, logging, datetime
from datetime import datetime as dt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"run_8yaf_full_{dt.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger("8yaf_full")

# ── Constants ─────────────────────────────────────────────────────────────────
DEVICE = "cuda:0"
MODEL_NAME = "protenix_base_default_v1.0.0"
DESIGNS_DIR = os.path.join(PROJECT_ROOT, "evaluation", "outputs", "designs")
os.makedirs(DESIGNS_DIR, exist_ok=True)

TARGET_SEQ = "ATKAVCVLKGDGPVQGIINFEQKESNGPVKVWGSIKGLTEGLHGFHVHEFGDNTAGCTSAGPHFNPLSRKHGGPKDEERHVGDLGNVTADKDGVADVSIEDSVISLSGDHCIIGRTLVVHEKADDLGKGGNEESTKTGNAGSRLACGVIGIA"
BINDER_SEQ = "QLQESGGGLVQPGGSLRLSCRASETLFSLYAMGWYRQAPGKQPELIATISGGGEGTGNYADPVKGRFTISRNNADNMVFLQMNNLKPEDTAVYYCNVYGTNLAPWGQGTQVTVS"
RANDOM_SEQ = "".join(random.choices(list("ACDEFGHIKLMNPQRSTVWY"), k=len(BINDER_SEQ)))

# CDR mask from TEST_8YAF_cdr.json (19 CDR positions out of 114)
CDR_MASK = [False] * len(BINDER_SEQ)
with open(os.path.join(PROJECT_ROOT, "outputs", "TEST_8YAF_cdr.json")) as f:
    cdr_data = json.load(f)
CDR_MASK = cdr_data["binder_cdr_mask"]

# Design params
N_STEPS = 200
LEARNING_RATE = 0.05
K_TOP = 8
B_CANDIDATES = 8
T_SAMPLE = 1.0
MAX_MUTATIONS = 3

# ── Phase 1: CDR Design ───────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("Phase 1: CDR-Aware Design for 8YAF")
logger.info(f"  target_len={len(TARGET_SEQ)} aa | binder_len={len(BINDER_SEQ)} aa")
logger.info(f"  CDR positions: {sum(CDR_MASK)}/{len(CDR_MASK)}")
logger.info(f"  N_steps={N_STEPS} | B_candidates={B_CANDIDATES}")
logger.info("=" * 60)

import torch
from scripts.run_hallucination_cdr import load_protenix_model
from bindcraft.hallucination.pipeline_cdr import run_hybrid_hallucination_cdr

t0 = time.time()
model, configs = load_protenix_model(MODEL_NAME, DEVICE)
logger.info("模型加载完成")

best_seq, final_seq, losses_history = run_hybrid_hallucination_cdr(
    model=model,
    binder_sequence=BINDER_SEQ,
    cdr_mask=CDR_MASK,
    target_seq=TARGET_SEQ,
    N_steps=N_STEPS,
    learning_rate=LEARNING_RATE,
    K_top=K_TOP,
    B_candidates=B_CANDIDATES,
    T_sample=T_SAMPLE,
    max_mutations_per_seq=MAX_MUTATIONS,
    use_annealing=True,
    T_init=1.0,
    T_min=0.01,
    use_tau_scheduling=True,
    tau_jitter=True,
    use_prior=True,
    prior_strength=0.1,
    checkpoint_dir=os.path.join(PROJECT_ROOT, "outputs", "cdr_8YAF_checkpoints"),
)
design_elapsed = time.time() - t0

best_loss = float(min(losses_history))
best_step = int(losses_history.index(best_loss))

logger.info(f"设计完成: {design_elapsed:.1f}s | best_loss={best_loss:.4f}@{best_step}")

# ── Save design JSON (eval format) ────────────────────────────────────────────
out_json = os.path.join(DESIGNS_DIR, "8yaf.json")
result = {
    "entry_id": "8yaf",
    "target_seq": TARGET_SEQ,
    "target_length": len(TARGET_SEQ),
    "designed_seq": best_seq,
    "designed_final_seq": final_seq,
    "random_seq": RANDOM_SEQ,
    "real_binder_seq": BINDER_SEQ,
    "real_binder_heavy": BINDER_SEQ,
    "design_loss_history": [float(l) for l in losses_history],
    "best_loss": best_loss,
    "best_step": best_step,
    "design_time_sec": round(design_elapsed, 1),
    "design_params": {
        "L_binder": len(BINDER_SEQ),
        "N_steps": N_STEPS,
        "num_beams": 1,
        "B_candidates": B_CANDIDATES,
        "use_annealing": True,
        "use_tau_scheduling": True,
    },
}
with open(out_json, "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
logger.info(f"设计结果已保存: {out_json}")

# ── Phase 2: Retrodiction Eval ─────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("Phase 2: Retrodiction Evaluation for 8YAF")
logger.info("=" * 60)

from bindcraft.hallucination.zeroth_order import zeroth_order_eval

RESULT_COLUMNS = [
    "entry_id", "seq_type", "binder_length", "target_length",
    "total_loss", "plddt", "pae", "i_pae", "con", "i_con", "iptm", "rg", "helix",
    "mean_plddt_conf", "iptm_conf", "ptm_conf",
    "dockq_ref", "lddt_ref", "eval_time_sec",
]

sequences = {
    "designed": best_seq,
    "random": RANDOM_SEQ,
    "real_vh": BINDER_SEQ,
    "real_full": BINDER_SEQ,
}

EVAL_MAX_TOTAL_SEQ_LEN = 650
eval_csv = os.path.join(PROJECT_ROOT, "evaluation", "outputs", "eval_results.csv")
file_exists = os.path.exists(eval_csv)
with open(eval_csv, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
    if not file_exists:
        writer.writeheader()

for seq_type, binder_seq in sequences.items():
    if len(TARGET_SEQ) + len(binder_seq) > EVAL_MAX_TOTAL_SEQ_LEN:
        logger.warning(f"  跳过 {seq_type}: {len(binder_seq)}aa + {len(TARGET_SEQ)}aa > {EVAL_MAX_TOTAL_SEQ_LEN}")
        continue

    logger.info(f"  {seq_type}: {binder_seq[:20]}... ({len(binder_seq)}aa) [AF3 推断]")
    t1 = time.time()
    result_eval = zeroth_order_eval(
        binder_seq=binder_seq,
        target_seq=TARGET_SEQ,
        model=model,
        device=torch.device(DEVICE),
    )
    elapsed = time.time() - t1

    sc = result_eval.summary_confidence
    row = {
        "entry_id": "8yaf",
        "seq_type": seq_type,
        "binder_length": len(binder_seq),
        "target_length": len(TARGET_SEQ),
        "total_loss": result_eval.total_loss,
        "eval_time_sec": round(elapsed, 1),
    }
    for key in ["plddt", "pae", "i_pae", "con", "i_con", "iptm", "rg", "helix"]:
        row[key] = result_eval.loss_dict.get(key, 0.0)
    row["mean_plddt_conf"] = sc.get("plddt", sc.get("mean_plddt", 0.0))
    row["iptm_conf"] = sc.get("iptm", 0.0)
    row["ptm_conf"] = sc.get("ptm", 0.0)
    row["dockq_ref"] = ""
    row["lddt_ref"] = ""

    with open(eval_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writerow(row)

    logger.info(
        f"    total_loss={result_eval.total_loss:.4f} | "
        f"plddt_conf={row['mean_plddt_conf']:.1f} | "
        f"iptm_conf={row['iptm_conf']:.3f} | "
        f"ptm_conf={row['ptm_conf']:.3f} | {elapsed:.1f}s"
    )

logger.info("=" * 60)
logger.info(f"8YAF 完整流程完成！设计={design_elapsed:.1f}s | 日志={log_file}")
logger.info(f"设计: {out_json}")
logger.info(f"评估: {eval_csv}")
logger.info("=" * 60)
