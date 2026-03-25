# BindCraft v2: 蛋白质 Binder 设计推理时优化框架

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-green.svg)]()

**BindCraft v2** 是基于 [Protenix](https://github.com/bytedance/Protenix) 构建的蛋白质 binder（结合蛋白）设计框架，引入了**混合一阶 + 零阶优化**流水线，通过将 Protenix 视为冻结的可微分评判器（Differentiable Critic），在不微调基座模型的情况下实现基于梯度的序列优化。

---

## 目录

- [背景与动机](#背景与动机)
- [核心思想：冻结评判器 + 推理时优化](#核心思想冻结评判器--推理时优化)
- [架构：三模块流水线](#架构三模块流水线)
  - [模块 A — STE 桥接（输入离散化桥）](#模块-a--ste-桥接输入离散化桥)
  - [模块 B — 单步扩散外推（输出截断）](#模块-b--单步扩散外推输出截断)
  - [模块 C — 物理约束损失（冻结评判器）](#模块-c--物理约束损失冻结评判器)
- [混合搜索：一阶提议 + 零阶验证](#混合搜索一阶提议--零阶验证)
- [时间步调度：粗到细退火](#时间步调度粗到细退火)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [算法文档](#算法文档)
- [引用](#引用)
- [许可证](#许可证)

---

## 背景与动机

设计能够特异性结合特定靶点（如抗原表位、酶口袋或信号接口）的蛋白质，是治疗性蛋白质工程的核心挑战。传统方法依赖库筛选或基于物理的对接——成本高且速度慢。

**网络求逆设计**的核心洞察是：**将前向结构预测网络视为可微分黑箱**。冻结模型权重，将氨基酸序列设为连续可优化变量，通过网络反向传播物理设计目标（结合距离、空间位阻、结构置信度），直接更新输入序列。

该范式在 AlphaFold 2（AF2）中天然可行——AF2 使用固定残基槽位（每个残基一个 ID，原子数量恒定）。然而，升级至 **AlphaFold 3 / Protenix（AF3）** 后，两座根本性的"物理死锁"出现了：

| | AF2 | AF3 / Protenix |
|---|---|---|
| **输入表示** | 固定残基槽位（每个残基 ID 恒定，原子数固定） | 动态原子列表 — 甘氨酸仅 4 个原子，色氨酸有 14 个。**总原子数是随氨基酸种类动态变化的标量**。 |
| **输出生成** | 几何拼接（角度 → 坐标） | 扩散生成（高斯噪声 → 全原子坐标）。 |

这些变化**切断了从 3D 坐标回到 1D 序列的梯度路径**：
1. **输入端死锁**：无法对动态变长原子表做朴素反向传播。
2. **输出端死锁**：在完整计算图中运行 50+ 步扩散 ODE 导致 OOM 和梯度粉碎。

BindCraft v2 从架构层面同时解决这两个死锁。

---

## 核心思想：冻结评判器 + 推理时优化

BindCraft v2 借鉴了 **Score Distillation / SDS** 方法（DreamFusion, ICLR 2023）的"生成 → 评分 → 修订"循环，针对 AF3 物理模型进行适配：

```
┌──────────────────────────────────────────────────────────────────────┐
│  当前 Binder 序列 (logits Θ)                                          │
│       ↓                                                               │
│  ┌─────────────┐   硬采样    ┌──────────────────────────┐            │
│  │  STE 桥接   │ ─────────→  │  Protenix Pairformer     │            │
│  │  (模块 A)   │  软特征     │  (48 层，冻结权重)        │            │
│  └─────────────┘ ←─────────  └──────────────────────────┘             │
│                                              ↓                         │
│                                    ┌──────────────────────┐            │
│                                    │  单步扩散外推         │            │
│                                    │  (模块 B, τ = 0.1)   │            │
│                                    └──────────────────────┘            │
│                                              ↓                         │
│                               ┌───────────────────────────┐           │
│                               │  物理约束损失              │            │
│                               │  距离 + 碰撞 + iPTM        │            │
│                               │  (模块 C — 冻结评判器)     │            │
│                               └───────────────────────────┘            │
│                                              ↓                         │
│                                  ┌────────────────────┐               │
│                                  │  backward(∂L/∂Θ)   │               │
│                                  │  Adam 更新 Θ        │               │
│                                  └────────────────────┘                │
└──────────────────────────────────────────────────────────────────────┘
```

冻结的 Protenix 模型充当**单步可微分物理评判器**。一次前向 + 反向传播即可产生切线梯度，在数学期望上等效于驱动一维序列演化的最速下降方向。

---

## 架构：三模块流水线

### 模块 A — STE 桥接（输入离散化桥）

*(映射 CV 领域：PEZ 算法，ICLR 2023 — "软特征前向，硬 token 验证")*

**问题**：AF3 的 `InputEmbedder` 执行离散查表，产生变长原子列表。任何连续梯度在此处被彻底截断。

**解决方案 — 直通估计器（STE）+ 软加权单体特征**：

1. 定义连续 logits 变量 $\mathbf{\Theta} \in \mathbb{R}^{L_{binder} \times 20}$（requires_grad=True），其中 20 为氨基酸种类数。

2. 用温度参数 $\tau_{temp}$ 转换为软概率矩阵：
   $$\mathbf{P} = \text{Softmax}\left(\mathbf{\Theta} / \tau_{temp}\right) \in \mathbb{R}^{L_{binder} \times 20}$$

3. **硬采样**（前向验证）：对 $\mathbf{P}$ 取 `argmax` 得到离散序列 $\mathbf{a}_{hard}$，通过官方 `InputEmbedder` 获取合法的系统原子总数 $N_{atom}$ 和硬编码单体特征 $\mathbf{s}^{init}_{hard}$。此步无梯度流过。

4. **软特征构建**（梯度高速公路）：离线预计算特征字典 $\mathbf{W}_{AAE} \in \mathbb{R}^{20 \times c_s}$（20 种氨基酸的 AtomAttentionEncoder 编码特征），构建平滑特征：
   $$\mathbf{s}^{init}_{soft} = \mathbf{P} \cdot \mathbf{W}_{AAE}$$

5. **STE 融合** — 前向使用硬特征（合法形状/尺寸），反向将梯度路由至软特征分支：
   $$\tilde{\mathbf{s}}^{init} = \mathbf{s}^{init}_{hard}.\text{detach()} + \mathbf{s}^{init}_{soft} - \mathbf{s}^{init}_{soft}.\text{detach()}$$

融合后的 binder 特征与 target 特征拼接（总长度恢复为 $N_{token}$），送入 Pairformer。

---

### 模块 B — 单步扩散外推（输出截断）

*(映射 CV 领域：Universal Guidance，ICCV 2023 — "截断 ODE，从噪声外推清晰结构")*

**问题**：在完整计算图中运行 50+ 步扩散 ODE 会导致 OOM 和梯度粉碎。

**解决方案 — 单步 $\hat{\mathbf{x}}_0$ 外推**：

在选定的微小噪声时间步 $\tau$（如 $\tau = 0.1$），取对应带噪 3D 骨架 $\mathbf{x}_\tau$（可通过无梯度前向 ODE 求解低成本获得）。执行**唯一一次**前向扩散，预测去噪速度场：

$$\mathbf{v}_{pred} = \text{Diffusion}\left(\mathbf{x}_\tau, \tau, \mathbf{s}, \mathbf{z}\right) \in \mathbb{R}^{N_{atom} \times 3}$$

根据流匹配（Flow Matching）常微分方程的几何切线原理，代数外推回干净的全原子坐标：

$$\hat{\mathbf{x}}_0 = \mathbf{x}_\tau - \tau \cdot \mathbf{v}_{pred} \in \mathbb{R}^{N_{atom} \times 3}$$

此 $\hat{\mathbf{x}}_0$ **全链路可导**，且完全规避了多步 ODE 循环。梯度通过 $\tau$（标量缩放）和 $\mathbf{v}_{pred}$ 流回 Pairformer。

**为什么有效**：逆向优化需要的是"方向导数"而非"完美的生成样本"。单步切线梯度——未经成千上万层网络污染——能为优化器提供信噪比最高的更新信号。

---

### 模块 C — 物理约束损失（冻结评判器）

*(映射 CV 领域：SDS / DreamFusion — "冻结模型作为可微分评判器")*

在单步外推的全原子结构 $\hat{\mathbf{x}}_0$ 上，直接施加结构学设计目标：

$$\mathcal{L}_{total} = \lambda_1 \underbrace{\text{Distance}\left(\hat{\mathbf{x}}_0^{(binder)}, \mathbf{x}_{target}\right)}_{\text{Binder 与靶点热点的距离}} + \lambda_2 \underbrace{\text{Clash}\left(\hat{\mathbf{x}}_0\right)}_{\text{空间位阻惩罚}} - \lambda_3 \underbrace{\text{iPTM}\left(\hat{\mathbf{x}}_0, \mathbf{s}, \mathbf{z}\right)}_{\text{复合物置信度（最大化）}}$$

这使冻结的 Protenix 模型成为**单步可微分 3D 物理评判器**。

---

## 混合搜索：一阶提议 + 零阶验证

在离散序列空间（规模约 $20^{L_{binder}}$）中进行纯一阶优化，存在两大风险：
1. **梯度截断误差**：单步外推的近似误差使连续空间更新偏离真实物理可行集合。
2. **STE 间隙**：连续概率空间的下降趋势，在 `argmax` 投影回离散序列后不一定保持一致。

BindCraft v2 通过**一阶 + 零阶混合框架**解决：

```
┌───────────────────────────────────────────────────────────────┐
│  一阶提议 (1st-Order Proposal)                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  反向传播 → 得到 ∇_Θ L                                     │  │
│  │  用泰勒近似对 (位置, 氨基酸) 突变排序                         │  │
│  │  生成 B 条候选序列（如 B = 8–16）                           │  │
│  └─────────────────────────────────────────────────────────┘  │
│                          ↓                                     │
│  零阶验证 (0th-Order Verification, 无梯度)                      │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  对每条候选序列执行完整 AF3 前向（并行，无梯度）             │  │
│  │  在生成结构上计算精确物理损失                                │  │
│  │  按真实损失排序 → 保留 top B                                │  │
│  └─────────────────────────────────────────────────────────┘  │
│                          ↓                                     │
│  状态更新（贪心或 Metropolis-Hastings 模拟退火）                │
└───────────────────────────────────────────────────────────────┘
```

- **一阶**：快速梯度提议有潜力的突变方向。
- **零阶**：以精确 AF3 评估为 Oracle 做严格筛选。
- **集束搜索**：维护 B 条并行分支，实现隐式回退。

---

## 时间步调度：粗到细退火

不同噪声水平携带不同的物理频域信息：

| 阶段 | 训练进度 | τ 范围 | 优化重点 |
|------|---------|--------|---------|
| 1 — 全局拓扑探索 | 0–30% | [0.6, 0.9] | 宏观形状、Binder 取向 |
| 2 — 界面塑造 | 30–70% | [0.2, 0.6] | 接触面、iPTM |
| 3 — 原子级细化 | 70–100% | [0.01, 0.2] | 碰撞、氢键、侧链堆积 |

**时间步抖动**：在每阶段内采样多个 τ 值并融合梯度，以减少偏差。

**模拟退火**：Metropolis-Hastings 接受准则使系统在早期以受控概率接受略差候选，增强跳出局部最优的能力。

---

## 项目结构

```
BindCraft/
├── bindcraft/                  # 核心 Binder 设计算法
│   ├── hallucination/          # 优化流水线
│   │   ├── pipeline.py         # 三阶段设计流程
│   │   ├── ste_bridge.py      # STE 输入离散化桥接
│   │   ├── diffusion_one_step.py  # 单步 x̂₀ 外推
│   │   ├── zeroth_order.py     # 零阶优化
│   │   ├── mutation_proposer.py # 梯度引导突变提议器
│   │   └── scheduler.py        # 时间步与参数调度器
│   └── losses/                 # 物理约束损失
│       ├── geometry.py         # 距离、碰撞、氢键
│       ├── confidence.py       # iPTM、pLDDT
│       └── total_loss.py       # 加权总损失
│
├── protenix/                   # Protenix 基座模型
│   ├── model/                  # 模型架构
│   │   ├── protenix.py        # 完整模型
│   │   ├── modules/           # Transformer / Pairformer
│   │   └── layer_norm/        # 自定义 CUDA LayerNorm
│   ├── data/                  # 数据处理
│   │   ├── inference/         # JSON 输入解析
│   │   └── msa/              # MSA 处理
│   └── metrics/               # 评估指标
│
├── runner/                     # 训练与推理入口
│   ├── train.py               # 训练 / 微调
│   └── inference.py           # 单结构推理
│
├── scripts/                    # 设计与评估脚本
│   ├── run_hallucination.py  # 主体 Binder 设计脚本
│   └── run_full_eval.py       # 完整评估流程
│
├── configs/                    # 配置文件
│   ├── configs_model_type.py  # 模型选择
│   └── configs_inference.py   # 推理配置
│
├── examples/                   # 示例 JSON 输入文件
├── docs/bindcraft/            # 算法文档
│   ├── BindcraftV2算法结构.md  # 完整算法架构
│   └── Bindcraft v2零阶更新.md  # 零阶优化详解
│
├── inference_demo.sh          # 推理演示
├── finetune_demo.sh           # 微调演示
└── train_demo.sh              # 训练演示
```

---

## 快速开始

### 1. 安装

```bash
pip install protenix
```

### 2. 运行蛋白质 Binder 设计

Binder 设计入口为 `scripts/run_hallucination.py`，支持两种模式：

#### 模式 A — 纯一阶幻觉优化（STE + 单步扩散）

```bash
python scripts/run_hallucination.py \
    --target GIVEQCCTSICSLYQLENYCN \
    --binder_length 40 \
    --n_steps 200 \
    --learning_rate 0.05 \
    --tau_noise 4.0 \
    --lambda_dist 2.0 \
    --lambda_clash 1.0 \
    --lambda_iptm -1.0 \
    --lambda_plddt -0.5
```

#### 模式 B — 混合 1st + 0th 优化（推荐）

在一阶梯度提议的基础上，叠加零阶全链路验证 + 集束搜索：

```bash
python scripts/run_hallucination.py \
    --target GIVEQCCTSICSLYQLENYCN \
    --binder_length 40 \
    --mode hybrid \
    --n_steps 100 \
    --learning_rate 0.05 \
    --tau_noise 4.0 \
    --k_top 10 \
    --b_candidates 8 \
    --max_mutations_per_seq 3 \
    --use_annealing true \
    --t_init 1.0 \
    --t_min 0.01 \
    --use_tau_scheduling true \
    --tau_jitter true \
    --lambda_dist 2.0 \
    --lambda_clash 1.0 \
    --lambda_iptm -1.0 \
    --lambda_plddt -0.5
```

#### 可选：针对特定靶点微调模型

```bash
bash finetune_demo.sh
```

### 3. 关键参数说明

完整参数列表可运行 `python scripts/run_hallucination.py --help`：

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--target` | 靶点抗原序列（单字母氨基酸代码） | **必填** |
| `--mode` | 优化模式：`first_order` 或 `hybrid` | `first_order` |
| `--binder_length` | Binder 序列长度 | `40` |
| `--n_steps` | 优化迭代步数 | `200` |
| `--learning_rate` | Adam 学习率（更新 logits） | `0.05` |
| `--tau_noise` | 单步外推的扩散噪声时间步 | `4.0` |
| `--k_top` | 每分支采样的 Top-K (位置, 氨基酸) 候选（仅 hybrid） | `10` |
| `--b_candidates` | 零阶验证候选序列数量（仅 hybrid） | `8` |
| `--t_sample` | 突变选择采样温度（仅 hybrid） | `1.0` |
| `--max_mutations_per_seq` | 每条候选序列的最大突变数（仅 hybrid） | `3` |
| `--use_annealing` | 是否启用模拟退火（仅 hybrid） | `true` |
| `--t_init` / `--t_min` | 退火初始/最低温度 | `1.0` / `0.01` |
| `--use_tau_scheduling` | 是否启用粗到细时间步退火调度 | `true` |
| `--tau_jitter` | 是否启用时间步抖动 | `true` |
| `--lambda_dist` | Binder 与靶点热点距离损失权重 | `2.0` |
| `--lambda_clash` | 空间位阻惩罚权重 | `1.0` |
| `--lambda_iptm` | iPTM 复合物置信度权重（负值 = 最大化） | `-1.0` |
| `--lambda_plddt` | pLDDT 单体置信度权重（负值 = 最大化） | `-0.5` |
| `--model_name` | Protenix 模型变体 | `protenix_base_default_v1.0.0` |
| `--device` | 计算设备 | `cuda` |
| `--output_dir` | 输出目录 | `./outputs` |

---

## 算法文档

详细算法文档（中文）：

- **[BindcraftV2算法结构.md](docs/bindcraft/BindcraftV2%E7%AE%97%E6%B3%95%E7%BB%93%E6%9E%84.md)** — 完整算法架构、雅可比链推导、AF2 vs AF3 对照表。
- **[Bindcraft v2零阶更新.md](docs/bindcraft/Bindcraft%20v2%E9%9B%B6%E9%98%B6%E6%9B%B4%E6%96%B0.md)** — 零阶优化、集束搜索、模拟退火、阶段退火调度器详解。

---

## 引用

如果您在研究中使用了 BindCraft v2，请同时引用 BindCraft v2 和底层 Protenix 模型：

```bibtex
@article{bindcraftv2,
  title  = {BindCraft v2: Hybrid 1st-Order + 0th-Order Optimization for Protein Binder Design},
  author = {Your Name and Co-Authors},
  year   = {2026}
}

@article{zhang2026protenix,
  author = {Zhang, Yuxuan and Gong, Chengyue and Zhang, Hanyu and Ma, Wenzhi and Liu, Zhenyu and others},
  title  = {Protenix-v1: Toward High-Accuracy Open-Source Biomolecular Structure Prediction},
  journal = {bioRxiv},
  year   = {2026},
  doi    = {10.64898/2026.02.05.703733}
}
```

---

## 许可证

BindCraft v2 采用 [Apache 2.0 许可证](./LICENSE) 发布。代码和模型参数均可免费用于学术研究和商业用途。
