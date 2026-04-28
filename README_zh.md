# BindCraft v2: 推理时优化驱动的蛋白质 Binder 设计

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

**BindCraft v2** 是基于 [Protenix](https://github.com/bytedance/Protenix) 构建的蛋白质 binder 设计框架。核心思想：冻结结构预测模型，将 3D 物理目标（界面接触、空间位阻、复合物置信度）直接反向传播过网络来优化输入氨基酸序列，无需对基座模型做任何微调。

---

## 核心思路

BindCraft v2 解决了 AlphaFold 3 / Protenix 与朴素梯度序列优化之间的两大根本障碍：

1. **输入端死锁** — AF3 使用变长原子列表（甘氨酸 4 原子，色氨酸 14 原子），梯度在原子查表处被截断。BindCraft v2 用 **STE（直通估计器）** 跨越此鸿沟：前向走硬离散特征（合法原子数），反向走软连续特征（梯度高速公路）。

2. **输出端死锁** — AF3 的扩散模块运行 50+ 步 ODE，保留完整计算图会导致 OOM 和梯度粉碎。BindCraft v2 将扩散截断为**单步外推**：在噪声时间步 τ 预测一个速度场，代数外推回干净结构 $\hat{x}_0 = x_\tau - \tau \cdot v_{pred}$，完全规避 ODE 循环且保持可导。

在一阶梯度之上，BindCraft v2 叠加了**零阶验证层**：候选序列通过完整 AF3 前向推断（精确物理、无梯度）重新评估，以贪心或 Metropolis-Hastings 准则决定接受，有效缓解 STE 离散化误差。

> 完整算法推导（雅可比链、AF2 vs AF3 对照、STE 机制、单步外推数学）见 **[docs/bindcraft/BindcraftV2算法结构.md](docs/bindcraft/BindcraftV2%E7%AE%97%E6%B3%95%E7%BB%93%E6%9E%84.md)**

> 零阶搜索、集束搜索、模拟退火、时间步调度详解见 **[docs/bindcraft/Bindcraft v2零阶更新.md](docs/bindcraft/Bindcraft%20v2%E9%9B%B6%E9%98%B6%E6%9B%B4%E6%96%B0.md)**

---

## 快速开始

### 安装

```bash
pip install protenix
```

### 设计 Binder

入口为 `scripts/run_hallucination.py`，支持两种模式：

**模式 A — 纯一阶（STE + 单步扩散）**

```bash
python scripts/run_hallucination.py \
    --target GIVEQCCTSICSLYQLENYCN \
    --binder_length 40 \
    --n_steps 200 \
    --learning_rate 0.05 \
    --tau_noise 4.0 \
    --w_i_con 3.0
```

**模式 B — 混合 1st + 0th（推荐）**

在一阶梯度提议基础上叠加零阶验证 + 集束搜索：

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
    --use_tau_scheduling true \
    --tau_jitter true \
    --w_i_pae 0.1 \
    --w_i_con 3.0
```

**可选：针对特定靶点微调模型**

```bash
bash finetune_demo.sh
```

### 关键参数

完整参数列表：`python scripts/run_hallucination.py --help`

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--target` | 靶点抗原序列（单字母代码） | **必填** |
| `--mode` | `first_order` 或 `hybrid` | `first_order` |
| `--binder_length` | Binder 长度 | `40` |
| `--n_steps` | 优化迭代步数 | `200` |
| `--learning_rate` | Adam 学习率 | `0.05` |
| `--tau_noise` | 扩散噪声时间步 | `4.0` |
| `--w_i_pae` | 链间 PAE 权重；启用 hotspot 后仅统计 binder-hotspot 对 | `0.1` |
| `--w_i_con` | 链间接触权重；启用 hotspot 后仅统计 binder-hotspot 接触 | `3.0` |
| `--w_clash` | binder 内部位阻惩罚权重 | `0.5` |
| `--w_iptm` | iPTM 权重 | `0.05` |
| `--w_plddt` | pLDDT 权重 | `0.1` |
| `--k_top` | 每分支 Top-K 突变候选（仅 hybrid） | `10` |
| `--b_candidates` | 零阶验证候选数（仅 hybrid） | `8` |
| `--use_tau_scheduling` | 粗到细 τ 退火调度 | `true` |

---

## 项目结构

```
bindcraft/
├── hallucination/             # 优化流水线
│   ├── pipeline.py           # 主循环（纯一阶 / 混合模式）
│   ├── ste_bridge.py         # STE 输入离散化桥接
│   ├── diffusion_one_step.py # 单步 x̂₀ 外推
│   ├── zeroth_order.py        # 零阶验证
│   ├── mutation_proposer.py  # 梯度引导突变提议器
│   └── scheduler.py          # 时间步与参数调度器
└── losses/                   # 物理约束损失
    ├── geometry.py           # Rg、螺旋、碰撞
    ├── confidence.py         # iPTM、pLDDT
    └── total_loss.py         # 加权总损失

protenix/                     # Protenix 基座模型

scripts/
└── run_hallucination.py     # 主体入口脚本

configs/                      # 配置文件
```

---

## 引用

```bibtex
@article{bindcraftv2,
  title  = {BindCraft v2: Hybrid 1st-Order + 0th-Order Optimization for Protein Binder Design},
  author = {Your Name and Co-Authors},
  year   = {2026}
}

@article{zhang2026protenix,
  author = {Zhang, Yuxuan and Gong, Chengyue and others},
  title  = {Protenix-v1: Toward High-Accuracy Open-Source Biomolecular Structure Prediction},
  journal = {bioRxiv},
  year   = {2026},
  doi    = {10.64898/2026.02.05.703733}
}
```

## 许可证

Apache 2.0 — 学术研究与商业用途均免费。
