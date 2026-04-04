"""
Retrodiction 评估 Pipeline - 配置文件

集中管理路径、默认参数，供 select / design / eval / analyze 各模块共享。
"""

import os

# ============================================================
# 路径
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 工作区根目录（数据 / 模型均存放于此）
WORKSPACE_ROOT = PROJECT_ROOT

RETRODICTION_CSV = os.path.join(
    WORKSPACE_ROOT, "datasets", "PXM-22to25-Ab-Ag", "retrodiction_list.csv"
)
AB_RAW_CSV = os.path.join(
    WORKSPACE_ROOT, "datasets", "PXM-22to25-Ab-Ag", "ab-raw.csv"
)
# Protenix 模型权重目录
PROTENIX_CHECKPOINT_DIR = os.path.join(WORKSPACE_ROOT, "checkpoint")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "evaluation", "outputs")
DESIGNS_DIR = os.path.join(OUTPUT_DIR, "designs")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
MODEL_NAME = "protenix_base_default_v1.0.0"

# ============================================================
# 案例筛选参数
# ============================================================

SELECT_TARGET_LENGTH_MIN = 80
SELECT_TARGET_LENGTH_MAX = 570  # target + 80aa binder ≤ 650aa（VRAM 安全边界）
SELECT_TOP_N = 10

# ============================================================
# 设计参数
# ============================================================

DESIGN_L_BINDER = 80
DESIGN_N_STEPS = 50
DESIGN_NUM_BEAMS = 1  # 1=单路径（显存友好）；2=Beam Search（显存压力大，长序列易OOM）
DESIGN_LEARNING_RATE = 0.05
DESIGN_K_TOP = 8
DESIGN_B_CANDIDATES = 2  # 长序列安全值（B=4 在长序列上易 OOM，已成功 entries 保留 B=2 结果）
DESIGN_T_SAMPLE = 1.0
DESIGN_MAX_MUTATIONS = 3
DESIGN_USE_ANNEALING = True
DESIGN_USE_TAU_SCHEDULING = True

# ═══════════════════════════════════════════════════════════════════════════
# 评估参数
# ═══════════════════════════════════════════════════════════════════════════

# ── zeroth_order_eval 模式（快速批量评估）──────────────────────────────

# 三组（designed / random / real_vh / real_full）全部走 zeroth_order_eval() AF3 推断，
# 用同一把尺子（iPTM / pLDDT / total_loss）打分。
# designed/random 组：80 aa de novo binder（BindCraft 最优设计区间 60-180 aa）
# real_vh 组：仅重链（~115 aa），与 designed/random 长度最接近，做公平长度对比
# real_full 组：完整 H+L 链（~430 aa），与 ab-raw.csv DockQ 参考条件一致
# target + binder 总长超过此阈值的案例，对应组跳过推断（VRAM 安全边界）。
EVAL_MAX_TOTAL_SEQ_LEN = 650

# ab-raw.csv 中用于读取 DockQ 参考值的模型名前缀
AB_RAW_MODEL_PREFIX = "Protenix-v1"

# ── full ODE 推理模式（高精度深度评估）────────────────────────────────

# 每个序列的完整 ODE 推理次数（扩散随机种子不同，反映结构预测随机性）
EVAL_N_REPEATS_FULL = 3

# ODE diffusion 采样步数（4 = 快速推理的默认值；可增大到 8-10 以提高精度）
EVAL_N_CYCLES = 4
