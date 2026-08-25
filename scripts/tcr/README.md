# TCR-Specific Loss — 代码索引

针对 pMHC 的 scaffold-constrained de novo CDR3 设计，在 ESMFold2 backbone 上
加两条 TCR-specific 信号并做消融。分支 `tcr-esmfold2-20260820`，基于
Bingchen 的 `7bf6251`。

## 实现（`bindcraft/`）

| 路径 | 作用 |
|---|---|
| `hallucination/tcr_prior.py` | TCR 序列先验 scorer（TCRLM / ESM-C / LigandMPNN 三个后端）。**沿用旧实现，本轮未改** |
| `hallucination/tcr_contact.py` | TCR 接触损失 + A6 scaffold 的 CDR1/2/3 手工标注 + mask 构建 |
| `hallucination/tcr_natural_prior.py` | 天然结构先验：从 216 个真实人类 class I TCR–pMHC 结构解析接触统计 |
| `data/tcr_contact_stats/` | 上述统计的原始 CSV |
| `hallucination/esmfold2_runtime.py` | **改动点**：先验/接触两条信号接入 proposal 梯度与 verifier |
| `scripts/run_hallucination.py` | **改动点**：CLI 参数 |

关键接入位置（`esmfold2_runtime.py`）：

- proposal 梯度：`autograd.grad(structure_loss, logits)` 之后，经
  `combine_normalized_gradients` 混合。**混合的是单位方向**，所以 λ 不等价于
  标量 loss 权重。
- verifier：`verify()` 的 `extra_loss_dict`。必须在 `verify()` **内部**算，
  因为结果按序列缓存（`verifier_cache[binder_seq]`），在缓存外累加会重复计数。
- 第二次 `autograd.grad` 需要 `retain_graph=True`，否则图已释放。

## 工具（`scripts/tcr/`）

| 脚本 | 作用 |
|---|---|
| `run_experiment.py` | 正式 matched run 的调度器（多 GPU、断点续跑） |
| `run_sweep.py` | 参数扫描调度器，由上者派生 |
| `run_smoke.py` / `check_smoke.py` | 短 smoke 与验收检查 |
| `evaluate.py` | **匹配化后评估**：所有 arm 用同一 verifier、同一固定 eval seed 重新打分，接触指标对没优化过它的 arm 也照算 |
| `report_arms.py` | 分项报表（结构 / 自然度 / 接触 / 搜索分开报，不合成单一 total_loss） |
| `report_2x2.py` | 2×2 表 + 叠加性检验 |
| `plot_search.py` | 逐 step 的各 loss 项轨迹图 + proposal 梯度诊断图 |
| `audit_old_protenix/` | 旧 Protenix 轮次的复盘工具 |

## 实验记录（`experiments/`）

`round_configs/*.json` 记录每一轮的 arm 定义 —— **扫描脚本是逐轮原地改的，
文件本身只反映最后一轮**，所以配置单独存在这里。

| 轮次 | 内容 | 结果文件 |
|---|---|---|
| `formal_tcr6_20260820` | 首轮正式 2×2（旧 nll_hinge 目标），24 jobs | `formal_eval_full.json`、`RESULTS.txt`、实验报告 md |
| `sweep_rpep_20260824` | 新 r_pep_target 目标的首次 λ 探测，2 case | `sweep_eval.json` |
| `sweep_rpep_r2_20260824` | λ 向上探，6 case | `sweep_r2_eval.json` |
| `sweep_rpep_r3_20260824` | 组合 arm，补齐新目标下的 2×2 | `sweep_r3_eval.json`、`RESULTS_2x2.txt` |
| `sweep_rpep_r4_20260825` | 接触项 verifier 权重阶梯 | `sweep_r4_eval.json` |

## 两条信号的实测性质

| | cos vs 结构梯度 | 辅助项占选择压力 |
|---|---|---|
| 序列先验 | −0.007（正交） | 22.1% |
| 接触（旧 nll_hinge） | **−0.348（对抗）** | — |
| 接触（新 r_pep_target） | **+0.16（同向）** | 37.7% (w=4.0) |

「选择压力」按各 loss 项在候选间的**标准差**算，不是均值 ——
`tcr_prior_hard ≈ 0.1 × NLL` 均值大但方差小，是近似常数偏移，不改变排序。

## 两个必须记住的坑

1. **`r_pep` 有天然目标值，不是越高越好**：α 0.2953 / β 0.3468。按
   `|r_pep − 天然值|` 评分，基线 arm A 本来就最接近；「最大化 r_pep」会把设计
   推得更不自然。
2. **framework mask 不能用 `~mutable_mask`**：那会把 CDR1/CDR2 算进惩罚，而
   天然数据显示 α CDR1 是所有区域里接触 pMHC 概率最高的（0.589）。

## 复现

```bash
P=/root/miniforge3/envs/proteocraft_esmfold2_py312/bin/python
export HF_HOME=/root/Bingchen/resources/hf_cache HF_HUB_OFFLINE=1

$P scripts/tcr/run_experiment.py --arms A,B,C,D --gpus 4,6,7 --out <outdir>
$P scripts/tcr/evaluate.py --root <outdir> --arms A,B,C,D --out eval.json
$P scripts/tcr/report_arms.py --eval eval.json
$P scripts/tcr/plot_search.py --root <outdir> --out plots/
```

模型与数据全部复用 Bingchen 的路径，未复制任何 checkpoint。节点无外网，
`esmc_300m` 不在本地 cache，外部 ESM-C 通道只能用 `esmc:biohub/ESMC-6B`。
