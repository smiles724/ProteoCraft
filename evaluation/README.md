# Retrodiction 评估 Pipeline

## 背景与动机

BindCraft v2 通过 AlphaFold 3 (Protenix) 的网络逆向优化，从零设计能与给定靶标蛋白 (target) 结合的 binder 序列。但优化过程中使用的 pLDDT / iPTM / contact 等指标都来自模型自身的预测——设计出来的序列是否具有**真实世界的结合意义**？

为回答这个问题，我们构建 **Retrodiction（回顾性设计）评估 Pipeline**：以真实的抗体-抗原复合物为基准，将抗原作为 target 输入 BindCraft 设计 binder，再将设计结果与真实抗体（ground truth）和随机序列（负对照）进行系统对比。

## 数据来源

```
SAbDab (20,700 条抗体记录 + 10,516 个 PDB 结构)
    │
    ├── sabdab_summary_all.tsv     序列/物种/分辨率/亲和力等元数据
    └── all_structures/imgt/*.pdb  IMGT 编号的 PDB 结构文件
           │
           ▼
datasets/scripts/prepare_pxm_retrodiction.py
           │  交叉匹配 SAbDab 与 Protenix 基准中的 entry_id
           ▼
datasets/PXM-22to25-Ab-Ag/retrodiction_list.csv  (450 条)
    │
    │  字段: entry_id, target_sequence, binder_sequence_heavy,
    │        binder_sequence_light, binder_sequence, target_length, binder_length
    │
    └── 每条 = 一个已知的 (抗原, 抗体) 配对，可作为设计基准
```

辅助数据：`ab-raw.csv`（55 万行）含 Protenix-v1 等模型对这些复合物的预测结构 vs 实验 PDB 结构的 DockQ / LDDT 评分，作为参考信息。

## 评估方法论

### 核心原则：同一把尺子

四组序列全部通过 `zeroth_order_eval()` 做完整 AF3 推断，用同一套指标（iPTM / pLDDT / total_loss）打分，实现完全公平的对比。

| 组别 | 序列来源 | 序列长度 | 评估方式 |
|------|----------|----------|----------|
| **designed** | BindCraft v2 设计的 80aa binder | 80aa | AF3 完整推断 |
| **random** | 等长随机氨基酸 | 80aa | AF3 完整推断 |
| **real_vh** | 仅重链（~115aa） | ~115aa | AF3 完整推断 |
| **real_full** | 完整 H+L 链（~430aa） | ~430aa | AF3 完整推断 |

### real_vh vs real_full 的设计意图

- **real_vh**：仅重链（~115aa），与 designed/random 长度最接近，做公平长度对比
- **real_full**：完整 H+L 链（~430aa），与 ab-raw.csv DockQ 参考条件一致

### DockQ 参考

`ab-raw.csv` 中 Protenix-v1 在完整真实序列上预测结构与实验 PDB 结构对比得出的 DockQ，附在 `real_full` 行的 `dockq_ref` 列作为额外参考。这不是评估指标，而是该案例的"结构预测质量上界"。

### 预期结果

- **real > designed > random**: 真实抗体重链的 AF3 置信度应最高，设计序列居中，随机序列最低
- designed vs random 的差距量化了优化的有效性
- designed vs real_vh 的差距量化了设计质量离真实抗体还有多远

## Pipeline 使用方法

所有脚本从 `Protenix_BindCraft/` 根目录运行，需激活 `protenix` conda 环境。

### Step 1: 筛选测试案例

```bash
python evaluation/select_cases.py --top_n 10
```

从 450 条中筛选 target 长度在 [80, 570] 之间、真实抗体 DockQ 最高的 10 条。

### Step 2: 批量设计 binder

```bash
CUDA_VISIBLE_DEVICES=3 python evaluation/run_design.py \
    --cases evaluation/outputs/selected_cases.csv \
    --n_steps 50 --b_candidates 2 --l_binder 80
```

对每个 target 运行 `run_hybrid_hallucination()`，同时生成随机基线。支持 `--num_beams 1`（单路径）或 `--num_beams 2`（Beam Search，显存压力大）。

### Step 3: 三组统一评估

```bash
CUDA_VISIBLE_DEVICES=3 python evaluation/run_eval.py
```

四组序列（designed / random / real_vh / real_full）全部通过 AF3 推断打分。

### Step 4: 生成对比报告

```bash
python evaluation/analyze_results.py
```

生成对比表、汇总统计、配对比较（designed vs random, designed vs real_vh）、DockQ 参考。

## 评估指标说明

| 指标 | 来源 | 含义 | 方向 | 适用 |
|------|------|------|------|------|
| total_loss | BindCraft loss | 8 项加权和 | 越低越好 | 全部 |
| mean_pLDDT | summary_confidence | 平均 per-residue 置信度 (0-100) | 越高越好 | 全部 |
| iPTM | summary_confidence | 界面 pTM 置信度 (0-1) | 越高越好 | 全部 |
| pTM | summary_confidence | 全局 pTM 置信度 (0-1) | 越高越好 | 全部 |
| DockQ_ref | ab-raw.csv | 预测结构 vs 实验 PDB 对接质量 (0-1) | 越高越好 | 仅 real_full 参考 |

## 已知局限

1. **长度不对称**: designed/random (80aa) vs real_vh (~115aa) vs real_full (~430aa)，长度不同对 AF3 置信度可能有天然偏好
2. **计算成本**: 零阶推断 ~8-10s/条（80aa），~15-25s/条（115aa）；设计阶段 ~40s/step
3. **模型偏差**: AF3/Protenix 的置信度指标不等同于实验结合亲和力
4. **序列形式**: mini-binder (80aa) 与天然抗体重链的功能空间不同
