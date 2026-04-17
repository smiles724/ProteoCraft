# BindCraft v2 — 评估模块

## 目录

- [模块概览](#模块概览)
- [四组序列的精确定义](#四组序列的精确定义)
- [输出文件清单](#输出文件清单)
- [快速开始](#快速开始)
- [分步 Pipeline](#分步-pipeline)
- [脚本详解](#脚本详解)
  - [run_design.py — 批量设计 Binder](#run_designpy--批量设计-binder)
  - [run_eval.py — 评估](#run_evalpy--评估)
  - [analyze_results.py — 生成报告](#analyze_resultspy--生成报告)
- [四组序列的填充逻辑](#四组序列的填充逻辑)
- [评估指标说明](#评估指标说明)
- [常见使用场景](#常见使用场景)
- [CDR 设计任务说明](#cdr-设计任务说明)

---

## 模块概览

评估模块用于系统评估 BindCraft v2 的 binder 设计能力，采用 **Retrodiction（回顾性设计）** 策略：以真实抗体-抗原复合物为基准，将抗原作为 target 输入，设计 binder，再将结果与真实抗体和随机序列进行多维度对比。

**核心原则：同一把尺子** — 所有序列组（designed / real_vh / real_full）均通过同一套 Protenix 推理流程评估，保证公平比较。

### 四个模块的协作关系

```
datasets/PXM-22to25-Ab-Ag/retrodiction_list.csv
        │
        ▼
    select_cases.py          ← 案例筛选，输出 selected_cases.csv
        │
        ▼
    run_design.py            ← 批量设计 binder，输出 designs/*.json
        │
        ├── designed_seq     ← BindCraft 设计的 binder
        ├── real_binder_seq  ← 完整抗体（H+L 拼接），用于 real_full
        └── real_binder_heavy ← 仅重链，用于 real_vh
        │
        ▼
    run_eval.py              ← zeroth / full ODE 两种精度评估
        │
        ├── eval_results.csv         ← zeroth 快速评估结果
        └── full_eval_results.csv   ← full ODE 高精度结果
        │
        ▼
    analyze_results.py        ← 生成对比报告
        │
        ├── evaluation_report.md         ← zeroth 报告
        └── full_evaluation_report.md    ← full ODE 报告
```

---

## 四组序列的精确定义

| 组别 | 含义 | 来源 | 设计 JSON 中的字段 |
|------|------|------|-------------------|
| **designed** | BindCraft v2 设计的 binder | `run_hybrid_hallucination()` 输出 | `designed_seq` |
| **real_vh** | 仅重链（Variable Region） | 原始抗体重链序列 | `real_binder_heavy` |
| **real_full** | 完整抗体（H+L 拼接） | 所有 antibody_chains 拼接 | `real_binder_seq` |
| random | 等长随机序列（负对照） | 程序生成 | `random_seq`（仅存于 design JSON，未参与结构评估） |

> **重要**：`real_binder_seq` 和 `real_binder_heavy` 的填充逻辑见[四组序列的填充逻辑](#四组序列的填充逻辑)章节。

### 为什么 real_full 比 real_vh 长很多？

以 8PIH_cdr 为例：
- `real_vh` = chain B（重链）= **127 aa**
- `real_full` = chain B + chain C（VH+VL）= **245 aa**

`real_full` 对应 Protenix 在 ab-raw.csv 中的训练条件（完整抗体-抗原复合物），是模型最熟悉的场景，可作为 DockQ 参考的上界。但因为长度更长，其 pLDDT 通常比 designed/real_vh 更低（模型对长序列的 per-residue 置信度更保守），比较时需注意长度公平性。

---

## 输出文件清单

```
evaluation/outputs/
├── selected_cases.csv          # 筛选后的案例列表（run_design.py 的输入）
├── designs/                    # 每个案例的设计结果
│   ├── 8PIH_cdr.json
│   └── 8yaf.json
├── eval_results.csv            # zeroth 快速评估原始数据
├── full_eval_results.csv       # full ODE 高精度评估原始数据
├── evaluation_report.md         # zeroth 评估报告（自动生成）
├── full_evaluation_report.md    # full ODE 评估报告（自动生成）
└── checkpoints/                # 设计过程保存的中间 checkpoint
    └── <entry_id>/
        └── best.json
```

---

## 快速开始

完整 Pipeline（设计 + 评估 + 报告）：

```bash
# Step 1: 筛选案例（从 retrodiction_list.csv 选 10 个）
CUDA_VISIBLE_DEVICES=0 python evaluation/run_design.py --select_only

# Step 2: 批量设计
CUDA_VISIBLE_DEVICES=0 python evaluation/run_design.py \
    --n_steps 50 --b_candidates 2 --l_binder 80

# Step 3: zeroth 快速评估（批量）
CUDA_VISIBLE_DEVICES=0 python evaluation/run_eval.py --eval_mode zeroth

# Step 4: full ODE 高精度评估（批量）
CUDA_VISIBLE_DEVICES=0 python evaluation/run_eval.py --eval_mode full

# Step 5: 生成报告
python evaluation/analyze_results.py --mode both
```

或一步到位（zeroth + full 依次运行）：

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/run_eval.py --eval_mode all
python evaluation/analyze_results.py --mode both
```

---

## 分步 Pipeline

### Step 1: 案例筛选（select_cases）

从 `datasets/PXM-22to25-Ab-Ag/retrodiction_list.csv`（450 条）筛选出适合 BindCraft 的案例，输出 `selected_cases.csv`。

筛选条件（见 `configs.py`）：
- `target_length` ∈ [80, 570]（target + 80aa binder ≤ 650aa VRAM 安全边界）
- 取前 N 条（默认 `--top_n 10`）

此步骤通过 `run_design.py --select_only` 触发，也可手动创建/修改 `selected_cases.csv`。

### Step 2: 批量设计（run_design.py）

对 `selected_cases.csv` 中每个案例调用 `run_hybrid_hallucination()` 设计 binder，输出 `designs/<entry_id>.json`。

### Step 3: 评估（run_eval.py）

两种精度，详见下文的[脚本详解](#脚本详解)。

### Step 4: 生成报告（analyze_results.py）

分析 CSV 结果，输出 markdown 对比报告。

---

## 脚本详解

### run_design.py — 批量设计 Binder

**输入**：`selected_cases.csv`（每行一个案例）

**输出**：`evaluation/outputs/designs/<entry_id>.json`

#### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n_steps` | 50 | 优化迭代步数 |
| `--b_candidates` | 2 | 每步保留的候选序列数（B 值；长序列建议 ≤2） |
| `--l_binder` | 80 | binder 设计长度（aa）；不指定则用 `DESIGN_L_BINDER` |
| `--num_beams` | 1 | Beam Search 宽度；1=单路径；2=多路径（显存压力大） |
| `--entry_ids` | None | 逗号分隔；只处理指定的 case（如 `--entry_ids 8PIH_cdr,8yaf`） |
| `--cases` | `selected_cases.csv` | 案例列表路径 |
| `--checkpoint_dir` | `outputs/checkpoints/` | 中间 checkpoint 保存目录；设为空字符串可禁用 |

#### 使用示例

```bash
# 设计全部案例（默认 80aa）
CUDA_VISIBLE_DEVICES=0 python evaluation/run_design.py

# 指定长度设计（对照实验）
CUDA_VISIBLE_DEVICES=0 python evaluation/run_design.py --l_binder 114

# 只设计特定案例
CUDA_VISIBLE_DEVICES=0 python evaluation/run_design.py --entry_ids 8PIH_cdr

# 筛选案例（不执行设计）
CUDA_VISIBLE_DEVICES=0 python evaluation/run_design.py --select_only
```

#### Design JSON 字段说明

```json
{
  "entry_id": "8PIH_cdr",
  "target_seq": "TLWCGFKHTDACCRTHDMCPDVMSAGES...",   // 抗原序列
  "target_length": 116,
  "designed_seq": "QVQLVESGGGLVQAGGSLRL...",           // BindCraft 设计的最优 binder
  "designed_final_seq": "...",                         // 最终迭代后的序列
  "random_seq": "...",                                 // 等长随机序列（负对照）
  "real_binder_seq": "Bchain+Cchain...",               // 完整抗体（H+L），用于 real_full 对照
  "real_binder_heavy": "Bchain...",                     // 仅重链，用于 real_vh 对照
  "design_loss_history": [5.68, 1.92, ...],            // 每步 total_loss 历史
  "best_loss": 1.1041,
  "best_step": 173,
  "design_time_sec": 869.5,
  "design_params": { ... }
}
```

> `real_binder_seq` 和 `real_binder_heavy` 的来源见[四组序列的填充逻辑](#四组序列的填充逻辑)章节。

---

### run_eval.py — 评估

统一入口，支持两种精度模式。

#### 三种评估模式

| 模式 | 说明 | 推理方式 | 输出文件 |
|------|------|----------|----------|
| `zeroth` | 快速批量评估 | zeroth_order_eval（1次 ODE） | `eval_results.csv` |
| `full` | 高精度深度评估 | 多 seed 完整 ODE（默认 3 seeds × 4 cycles） | `full_eval_results.csv` |
| `all` | zeroth + full 依次运行 | 两者结合 | 两者结合 |

#### zeroth 模式

调用 `zeroth_order_eval()`，对每个序列做一次完整 ODE 推理，输出 12 项 loss + 3 项置信度指标。

```bash
# 评估所有 designs/*.json
CUDA_VISIBLE_DEVICES=0 python evaluation/run_eval.py --eval_mode zeroth

# 只评估特定案例
CUDA_VISIBLE_DEVICES=0 python evaluation/run_eval.py --eval_mode zeroth --entry_ids 8PIH_cdr
```

#### full ODE 模式

对每个序列执行多次完整 ODE 推理（不同 seed），统计均值±标准差。

```bash
# 从 designs/*.json 批量评估
CUDA_VISIBLE_DEVICES=0 python evaluation/run_eval.py --eval_mode full

# 只评估特定案例
CUDA_VISIBLE_DEVICES=0 python evaluation/run_eval.py --eval_mode full --entry_ids 8PIH_cdr,8yaf

# 指定精度参数
CUDA_VISIBLE_DEVICES=0 python evaluation/run_eval.py --eval_mode full \
    --n_repeats 5 --n_cycles 8

# 单序列评估（直接传入序列）
CUDA_VISIBLE_DEVICES=0 python evaluation/run_eval.py --eval_mode full \
    --binder_seq QVQLVESGGGLVQAGGSLRLSCAASGRTFSRYAMGWFRQAPGKEREFVSAISGSGGFTDYADSVKGRFTISRDNAKSTVYL... \
    --target_seq TLWCGFKHTDACCRTHDMCPDVMSAGESKHGLTNTASHTRLSCDCDDKFYDCLKNSADTISSYFVGKMYFNLIDTKCYKLE... \
    --label 8PIH_DesignedBest \
    --n_repeats 3
```

#### all 模式

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/run_eval.py --eval_mode all --entry_ids 8PIH_cdr
```

---

### analyze_results.py — 生成报告

读取 CSV 结果，生成 markdown 对比报告。

```bash
python evaluation/analyze_results.py --mode both   # zeroth + full ODE 都分析
python evaluation/analyze_results.py --mode zeroth  # 只分析 zeroth
python evaluation/analyze_results.py --mode full    # 只分析 full ODE
```

报告内容：
- 逐案例对比表（所有序列组的各项指标）
- 汇总统计（按 seq_type 分组）
- 配对比较（designed vs real_vh：胜率、平均差异）
- DockQ 参考值（zeroth 模式，real_full 组）

---

## 四组序列的填充逻辑

设计 JSON 中 `real_binder_seq` 和 `real_binder_heavy` 的填充规则如下：

### 规则 1：优先使用 `antibody_chains`（CDR 任务等结构化 case）

如果 case 中存在 `antibody_chains` 字段（JSON 对象，key 为 chain_id，value 含 `sequence` 和 `chain_type`）：

```python
# real_binder_seq: 所有 chain 拼接 = 完整抗体
real_binder_seq = "".join(v["sequence"] for v in ab_chains.values())

# real_binder_heavy: 仅 chain_type=H 的链
for v in ab_chains.values():
    if v.get("chain_type") == "H":
        real_binder_heavy = v["sequence"]
        break
```

**示例（8PIH_cdr）**：

| 字段 | 来源 | 长度 |
|------|------|------|
| `real_binder_seq` | chain B（127 aa） + chain C（118 aa） | **245 aa** |
| `real_binder_heavy` | chain B（type=H） | **127 aa** |

### 规则 2：降级使用 `binder_sequence` / `binder_sequence_heavy`（旧格式或单链任务）

如果 case 中没有 `antibody_chains`：

```python
real_binder_seq    = case.get("binder_sequence", "")
real_binder_heavy  = case.get("binder_sequence_heavy", "")
```

**示例（8yaf）**：

| 字段 | 来源 | 长度 |
|------|------|------|
| `real_binder_seq` | `binder_sequence` | 114 aa |
| `real_binder_heavy` | `binder_sequence_heavy` | 114 aa |

（8yaf 的 binder_sequence 和 binder_sequence_heavy 相同，因为它是单链 binder 设计，两个字段值一样。）

### 手动 patch 已有 design JSON

如果已有的 design JSON 缺少正确的 `real_binder_seq`/`real_binder_heavy`（比如用旧版本 run_design.py 生成），可以直接修改 JSON：

```python
import json
d = json.load(open("evaluation/outputs/designs/8PIH_cdr.json"))

# 方式 A：从 PDB 文件提取序列
# （需自行从本地 structures/*.pdb 或 PDB 数据库获取链序列）
d["real_binder_seq"] = chain_b + chain_c   # 完整抗体
d["real_binder_heavy"] = chain_b           # 仅重链

# 方式 B：从 antibody_chains 字段拼接（如果 design JSON 中保留了该字段）
chains = d.get("antibody_chains", {})
d["real_binder_seq"] = "".join(v["sequence"] for v in chains.values())

json.dump(d, open("evaluation/outputs/designs/8PIH_cdr.json", "w"), indent=2)
```

patch 后重新运行评估即可：

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/run_eval.py --eval_mode all --entry_ids 8PIH_cdr
```

---

## 评估指标说明

### zeroth 模式指标

来自 `zeroth_order_eval()` 的 `loss_dict` 和 `summary_confidence`：

| 指标 | 来源 | 含义 | 方向 |
|------|------|------|------|
| `total_loss` | loss_dict 加权和 | BindCraft 优化目标，越低越好 | ↓ |
| `plddt` | loss_dict | per-residue 置信度 loss，越低越好 | ↓ |
| `pae` | loss_dict | 位置误差 loss，越低越好 | ↓ |
| `iptm` | loss_dict | 接口 TM loss，越低越好 | ↓ |
| `mean_plddt_conf` | summary_confidence | 平均 per-residue 置信度 (0-100) | ↑ |
| `iptm_conf` | summary_confidence | 接口 TM 分数 (0-1) | ↑ |
| `ptm_conf` | summary_confidence | 全局 TM 分数 (0-1) | ↑ |
| `dockq_ref` | ab-raw.csv | real_full 组的 Protenix-v1 预测 DockQ 参考值 | — |

### full ODE 模式指标

来自 Protenix 完整多步 ODE 推理的 `summary_confidence`，独立于 BindCraft 优化目标：

| 指标 | 含义 | 方向 | 阈值 |
|------|------|------|------|
| **pLDDT** | 结构置信度，per-residue 预测准确性 | 越高越好 | ≥70 为可信结构 |
| **iPTM** | 接口 TM 分数，binder-target 界面相似性 | 越高越好 | ≥0.6 为可信界面 |
| **pTM** | 全局 TM 分数，整体折叠相似性 | 越高越好 | ≥0.5 为可信折叠 |
| **ranking** | Protenix 综合排名 = 0.8×iPTM + 0.2×pTM | 越高越好 | — |
| **chain_iPTM** | Binder→Target 链对接口置信度（最直接结合指标） | 越高越好 | ≥0.6 |
| **gpde** | 全局预测 DockQ，结构质量估计 | 越低越好 | <3 excellent, <6 good |

> 注：`zeroth` 模式使用模型内部预测置信度（summary_confidence），`full` 模式使用 ODE 推理的原生置信度。两者指标含义相近但数值不可直接比较（推理路径不同）。

---

## 常见使用场景

### 场景 A：跑一批新案例

```bash
# 1. 筛选
python evaluation/run_design.py --select_only --top_n 10

# 2. 设计
CUDA_VISIBLE_DEVICES=0 python evaluation/run_design.py \
    --n_steps 50 --b_candidates 2 --l_binder 80

# 3. 评估
CUDA_VISIBLE_DEVICES=0 python evaluation/run_eval.py --eval_mode all

# 4. 报告
python evaluation/analyze_results.py --mode both
```

### 场景 B：单独评估某条序列

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/run_eval.py --eval_mode full \
    --binder_seq <你的序列> \
    --target_seq <抗原序列> \
    --label MyBinderTest \
    --n_repeats 3
```

### 场景 C：重新评估已有设计（修改了对照序列后）

已有 design JSON 的 `real_binder_seq`/`real_binder_heavy` 更新后，直接重新运行评估即可：

```bash
# patch JSON 后
CUDA_VISIBLE_DEVICES=0 python evaluation/run_eval.py --eval_mode all --entry_ids <entry_id>
python evaluation/analyze_results.py --mode both
```

### 场景 D：不同 binder 长度对照实验

```bash
for L in 60 80 100 120; do
    CUDA_VISIBLE_DEVICES=0 python evaluation/run_design.py \
        --l_binder $L --entry_ids 8PIH_cdr
    CUDA_VISIBLE_DEVICES=0 python evaluation/run_eval.py \
        --eval_mode full --entry_ids 8PIH_cdr
done
python evaluation/analyze_results.py --mode full
```

---

## CDR 设计任务说明

CDR（Crystalography-of-Realistic-Interface）任务是指：给定一个 PDB 结构，以其抗体上的 CDR 区域为 target，设计新的 CDR 序列。设计 JSON 的 `design_params` 字段记录了任务元信息：

```json
"design_params": {
  "task_type": "cdr",
  "pdb_id": "8PIH",
  "binder_chain": "B",      // 被 CDR 化的链
  "antigen_chain": "A"      // 抗原链
}
```

CDR 任务的 `designed_seq` 是对 chain B 的 CDR 区域进行优化后的序列，而 `real_binder_seq`（real_full）和 `real_binder_heavy` 均来自原始 PDB 结构（chain B + chain C）。

> 如果已有 CDR design JSON 的 `real_binder_seq` 为空或只含单链，可以通过 PDB 文件手动 patch：
> ```bash
> python -c "
> import json
> d = json.load(open('evaluation/outputs/designs/8PIH_cdr.json'))
> pdb = 'structures/8PIH.pdb'
> # 从 PDB 提取 chain B 和 chain C 序列...
> d['real_binder_seq'] = chain_b + chain_c
> d['real_binder_heavy'] = chain_b
> json.dump(d, open('evaluation/outputs/designs/8PIH_cdr.json','w'), indent=2)
> "
> ```
