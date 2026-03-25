"""
Retrodiction 评估 Pipeline - 配置文件

集中管理路径、默认参数，供 select / design / eval / analyze 各模块共享。
"""

import os

# ============================================================
# 路径
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)

RETRODICTION_CSV = os.path.join(
    WORKSPACE_ROOT, "datasets", "PXM-22to25-Ab-Ag", "retrodiction_list.csv"
)
AB_RAW_CSV = os.path.join(
    WORKSPACE_ROOT, "datasets", "PXM-22to25-Ab-Ag", "ab-raw.csv"
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "evaluation", "outputs")
DESIGNS_DIR = os.path.join(OUTPUT_DIR, "designs")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

CHECKPOINT_DIR = "/hpc2hdd/home/bduan436/checkpoint"
MODEL_NAME = "protenix_base_default_v1.0.0"

# ============================================================
# 案例筛选参数
# ============================================================

SELECT_TARGET_LENGTH_MIN = 80
SELECT_TARGET_LENGTH_MAX = 180
SELECT_TOP_N = 10

# ============================================================
# 设计参数
# ============================================================

DESIGN_L_BINDER = 40
DESIGN_N_STEPS = 50
DESIGN_LEARNING_RATE = 0.05
DESIGN_K_TOP = 8
DESIGN_B_CANDIDATES = 4
DESIGN_T_SAMPLE = 1.0
DESIGN_MAX_MUTATIONS = 3
DESIGN_USE_ANNEALING = True
DESIGN_USE_TAU_SCHEDULING = True

# ============================================================
# 评估参数
# ============================================================

# 三组（designed / random / real_vh / real_full）全部走 zeroth_order_eval() AF3 推断，
# 用同一把尺子（iPTM / pLDDT / total_loss）打分。
# real_vh 组使用仅重链序列（~115aa），与 designed/random 长度最接近，可做长度接近的公平对比
# real_full 组使用完整 H+L 链（~430aa），与 ab-raw.csv DockQ 参考条件一致
# target + binder 总长超过此阈值的案例，对应组跳过推断（VRAM 安全边界）。
EVAL_MAX_TOTAL_SEQ_LEN = 650

# ab-raw.csv 中用于读取 DockQ 参考值的模型名前缀
AB_RAW_MODEL_PREFIX = "Protenix-v1"
