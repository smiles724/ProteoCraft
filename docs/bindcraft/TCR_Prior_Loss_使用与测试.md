# TCR Sequence Prior Loss — 算法说明 + 使用 / 测试指南

维护文件：`bindcraft/hallucination/tcr_prior.py`（backend 实现）、`bindcraft/hallucination/pipeline.py` / `pipeline_cdr.py`（接入点）、`bindcraft/hallucination/TCR_PRIOR_BACKENDS.md`（原始英文技术文档，本篇是内部中文版 + 使用/测试记录）。

## 一、要解决的问题

TCR CDR 重新设计原本只由结构类 loss（plddt / pae / i_pae / clash …）驱动，不知道什么是"合理"的 TCR 序列。TCR sequence prior 在优化/选择过程中额外引入一个 naturalness 信号，把搜索拉回自然 TCR 会出现的序列空间。三个 naturalness 来源从不同角度回答这个问题：

| 来源 | 回答什么问题 | 方法 |
|---|---|---|
| TCRLM | 这段 CDR3 本身自然吗 | masked-marginal pseudo-log-likelihood，专门在 &gt;1 亿条 CDR3 上训练 |
| ESM-C | 整条 TCR 自然吗 | 通用蛋白语言模型（EvolutionaryScale），单遍前向或逐位置遮盖两种打分模式 |
| LigandMPNN | 换了这段 CDR 之后跟 pMHC + TCR 骨架结构搭吗 | 结构条件化，给定 backbone（+ 可选的非蛋白配体原子）预测每个位置的 log-prob |

## 二、两条接入路径（算法）

两条路径都在 `pipeline.py`（整段 binder 的 hybrid / mcts 循环）和 `pipeline_cdr.py`（CDR-限定的 hybrid / mcts 循环）里独立实现，可以同时开启。

### 1. Gradient-based proposal（梯度回传路径，思路同 germinal）

在一阶梯度更新的 proposal loss 里加一项 soft prior loss：

$$
L_{prop}^{total} = L_{prop} + \lambda \cdot L_{prior}^{soft}
$$

$$
L_{prior}^{soft} = \text{mean}_i \; \text{CE}\big(\text{softmax}(\theta_i),\; p_{prior}(\cdot \mid \text{context})_i\big)
$$

- $\theta$ 是当前正在优化的序列 logits；$p_{prior}$ 是 prior 模型给出的目标分布，对 prior 模型 **detach**，梯度只流回 $\theta$，prior 模型本身永远不参与训练。
- 代码位置：`pipeline.py`（`979,1979,2329,2410`）和 `pipeline_cdr.py`（`847,1750,2121,2204`）各 4 处、共 8 处调用点，在 `.backward()` 之前把 `L_prior^{soft}` 加到 `first_order_loss` / `proposal_loss` 上。
- `mutation_proposer.py` 的 `compute_mutation_benefit()` 不需要感知 tcr_prior 的存在——它消费的梯度矩阵 $G = \nabla_\theta L$ 已经包含了 prior 的贡献，两者职责分离。

### 2. Tree-value / verifier selection（候选打分路径）

MCTS 展开候选节点时，在 verifier 的 total_loss 里加一项 hard prior loss（naturalness 的负数）：

$$
L_{verify}^{total} = L_{verify} + \lambda \cdot L_{prior}^{hard}
$$

$$
L_{prior}^{hard} = -\,\text{mean}_i \log p_{prior}(a_i \mid \text{context})
$$

- `naturalness = -L_{prior}^{hard}`（即 $\text{mean}_i \log p_{prior}$）；`apply_tcr_verifier_score()` 把这一项写回候选的 `total_loss`。
- 这个 `total_loss` 就是 `mcts_controller.MCTSTree.expand()` 计算 `tree_reward` 时用的值。**`mcts_controller.py` 本身完全不 import `tcr_prior`**——它只认 `total_loss`，naturalness 是在算 `total_loss` 之前被加进去的，两者通过这一层解耦。
- 调用点：`_apply_tcr_verifier` 闭包，覆盖 init / baseline / phase0 / 候选评分，`pipeline.py` 5 处 + `pipeline_cdr.py` 6 处，共 11 处。

<aside>
💡

两条路径不是互斥选项，`--use_tcr_prior_gradient` 和 `--use_tcr_prior_verifier` 是两个独立开关，可以只开一个，也可以同时开。

</aside>

## 三、三个 Prior 来源的实现细节

### TCRLM（CDR3 naturalness）
- 仓库：`https://github.com/hliulab/tcrLM`，BERT 风格 masked LM（`FLASHTransformer`）。**不是** pip/HF 包，需要 clone 仓库 + 单独下载权重（158MB，仓库和 Zenodo 包里都只有 LFS 指针文件，需要用 LFS batch API 单独拉取真实对象）。
- 打分方式：masked-marginal pseudo-log-likelihood——把每个位置替换成 `MASK`，读模型对真实残基的 log-prob。
- 权重加载做了 $\ge$95% tensor 匹配校验，防止 key 不完全对齐导致的静默 partial load（会产出乱码级别的困惑度）。
- 推荐 `--tcr_prior_region cdr`，只在 CDR3 位置上打分。

### ESM-C（whole-TCR naturalness）
- 包：`esm`（EvolutionaryScale），权重首次使用时自动从 HuggingFace 下载。
- 两种打分模式（对应 `--tcr_prior_masked` 开关）：
  - **单遍前向（默认）**：完整序列一次性编码，直接读 $p(a_i \mid \text{完整序列})$。ESM-C 和 ESM-2 一样是双向（bidirectional）masked language model，不是自回归模型——由于是双向 encoder，位置 $i$ 自己的输入 token 会通过 self-attention/残差直接影响位置 $i$ 自己的输出，相当于自我条件化（self-conditioning），对低复杂度/重复序列容易给出虚高的置信度。优点是只需一次前向，适合作为可微分的 soft prior 目标。
  - **逐位置遮盖（`--tcr_prior_masked true`）**：对区间内每个位置单独替换成 mask token 重新编码，读 $p(a_i \mid \text{位置 } i \text{ 被遮盖的序列})$，即标准的 masked-marginal PLL，没有自我条件化偏差，但对长度 $L$ 的区间需要 $L$ 次独立前向，更慢，建议只在 verifier-only 场景使用。
- 与 ProteoCraft 自带的 `fair-esm`（ESM-2）在同一个 Python 进程里互斥（都占用顶层 `esm` 模块名），必须跑在独立 conda env 里。

### LigandMPNN（结构条件化 CDR redesign prior）
- 仓库：`https://github.com/dauparas/LigandMPNN`。需要 `metadata['pdb_path']`（每个候选的 backbone PDB），没有结构时直接退化为 no-op。
- **`model_type` 自动探测**（`tcr_prior.py` 的 `_load()` 方法内）：加载 checkpoint 时检查是否含 `atom_context_num` 这个 key——含有则视为真实 `ligand_mpnn` 权重，从 checkpoint 里读 `atom_context_num` / `k_neighbors`，并按需开启 `ligand_mpnn_use_side_chain_context`；不含则退回 `protein_mpnn` 模式（`atom_context_num=1`）。两种模式下 `parse_PDB(parse_all_atoms=...)` 和 `featurize(use_atom_context=..., number_of_ligand_atoms=..., model_type=...)` 的参数会相应调整，逻辑对齐官方 `run.py`。
- `ligand_mpnn` 模式下"看配体原子"的能力，只对结构文件里被解析为 `not protein and not water`（即非标准蛋白残基、非水）的原子生效——如果 pMHC 结构里的肽段是按标准蛋白残基写的（通常如此），这部分能力目前不会比纯 backbone 条件化多看到什么；真正的增益预期要在 TCR-pMHC 数据上专门 fine-tune 之后才会体现，见「实验结论」里的对照数字。

## 四、Config 参数一览

| flag | 含义 |
|---|---|
| `--use_tcr_prior_gradient` | 开启梯度回传路径（soft prior） |
| `--tcr_prior_gradient_weight` | 路径一的 $\lambda$ |
| `--use_tcr_prior_verifier` | 开启候选打分路径（hard prior / naturalness） |
| `--tcr_prior_verifier_weight` | 路径二的 $\lambda$ |
| `--tcr_prior_region` | `all` / `cdr` / `none`，prior 作用在哪些位置 |
| `--tcr_prior_model_name` | backend 选择：`tcrlm[:/repo]` / `esmc[:model_id]` / `ligandmpnn[:/repo]`；留空/`none`/`dummy`/任意 HF id 走原始占位逻辑 |
| `--tcr_prior_checkpoint` | `tcrlm` / `ligandmpnn` 的权重路径（或环境变量 `TCRLM_CKPT` / `LIGANDMPNN_CKPT`） |
| `--tcr_prior_masked` | ESM-C 是否用逐位置遮盖（更慢更严谨），默认 `false` |
| `--require_real_tcr_prior` | 严格模式：backend 加载失败/无真实权重/无真实输出时直接报错，而不是静默退化为 no-op |

<aside>
💡

默认行为：任何 backend 加载失败（缺仓库/缺权重/缺依赖）都会静默退化为 no-op（prior 贡献 0），主循环不会被打断。只有开启 `--require_real_tcr_prior true` 才会改成报错，适合调试/验证阶段用。

</aside>

## 五、如何使用

### 环境
三个 backend 依赖互相冲突（尤其 ESM-C 的新版 `esm` 包 vs ProteoCraft 自带的 `fair-esm`），推荐跑在同一个独立 conda env 里（不要装进 `base`）：

```bash
conda create -y -p /path/to/tcr_prior_backends/env python=3.11
ENV=/path/to/tcr_prior_backends/env
$ENV/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
$ENV/bin/pip install esm einops prody "numpy<2.3" scipy pytest
```

TCRLM / LigandMPNN 需要各自 clone 仓库并准备权重（`TCRLM_HOME` + `TCRLM_CKPT`，`LIGANDMPNN_HOME` + `LIGANDMPNN_CKPT`）；ESM-C 权重首次调用自动下载（Polaris 上需要走 ALCF 代理，`HF_HOME` 建议放到项目盘而非 `$HOME`）。

### 调用示例

```bash
# TCRLM，CDR3 区域，只用候选打分路径
export TCRLM_HOME=/abs/path/to/tcrLM TCRLM_CKPT=/abs/path/to/pretrain.pt
python scripts/run_hallucination.py --mode mcts ... \
  --use_tcr_prior_verifier true --tcr_prior_verifier_weight 0.1 \
  --tcr_prior_region cdr --tcr_prior_model_name tcrlm

# ESM-C，全序列，梯度 + 候选打分都开
python scripts/run_hallucination.py --mode hybrid ... \
  --use_tcr_prior_verifier true --tcr_prior_verifier_weight 0.1 \
  --use_tcr_prior_gradient true --tcr_prior_gradient_weight 0.1 \
  --tcr_prior_region all --tcr_prior_model_name esmc:esmc_600m

# LigandMPNN，CDR 区域，需要额外把候选 pdb_path 传下去（结构条件化）
export LIGANDMPNN_HOME=/abs/path/to/LigandMPNN
python scripts/run_hallucination.py --mode mcts ... \
  --use_tcr_prior_verifier true --tcr_prior_verifier_weight 0.1 \
  --tcr_prior_region cdr --tcr_prior_model_name ligandmpnn \
  --tcr_prior_checkpoint $LIGANDMPNN_HOME/model_params/ligandmpnn_v_32_010_25.pt
```

<aside>
💡

目前这些参数只在通用 CLI `scripts/run_hallucination.py` 里暴露。实际在跑真实 case 的两个脚本（`scripts/run_bindcraft_tcr_case.py` 配 `configs/tcr5_hotspot_v1like_loss.yaml` 跑 5 个真实 case；`scripts/run_hallucination_cdr.py` 用于 CDR 专项设计）目前调用优化函数时没有转发 `tcr_prior_config`，需要单独加一步「接线」才能在真实 case 里生效——不是本篇的范围，单独跟进。

</aside>

## 六、如何测试

### 1. 单元测试（backend / 接口层）

```bash
# 默认：只测接口逻辑（no-op 退化、region mask、soft/hard 数学关系），不需要真实权重
python -m pytest bindcraft/tests/test_tcr_prior.py -q
# → 17 passed, 5 skipped（5 个 skip 是下面的真实 backend 测试）

# 打开真实 backend 测试（需要计算节点 + 真实权重/网络）
RUN_REAL_TCR_PRIOR=1 \
  TCRLM_HOME=... TCRLM_CKPT=... \
  LIGANDMPNN_HOME=... LIGANDMPNN_CKPT=... TCR_PRIOR_PDB=... \
  python -m pytest bindcraft/tests/test_tcr_prior.py -q
# → 21 passed, 1 skipped（唯一 skip 的前提是"esm 包未安装"，装了 esm 的 env 里 skip 是正常的）
```

`bindcraft/__init__.py` 和 `bindcraft/hallucination/__init__.py` 用了 PEP 562 懒加载，所以 import `tcr_prior` 不会拖进整个 protenix/diffusion 依赖，单测可以在任意有 `torch`/`numpy`/`pytest` 的 env 里跑。

### 2. 真实 backend 验证（不只是接口 smoke，是真的跑一遍推理）

```bash
python scripts/validate_tcr_prior_backends.py \
  --esmc-model esmc_300m \
  --tcrlm-repo /path/to/tcrLM --tcrlm-ckpt /path/to/pretrain.pt \
  --ligandmpnn-repo /path/to/LigandMPNN --ligandmpnn-ckpt /path/to/model_params/ligandmpnn_v_32_010_25.pt \
  --pdb /path/to/example.pdb
```

逐个加载 backend、检查真实权重、跑一遍 `per_position_logprobs`、验证 soft loss 产生非零梯度、跑一遍 strict mode，最后打印状态表。

### 3. Pipeline-level 端到端 smoke test

```bash
python scripts/smoke_tcr_prior_pipeline.py --backends tcrlm,esmc,esmc_masked,ligandmpnn
```

不跑一整套结构预测，而是用真实的 `MCTSTree.expand()` + 真实 backend scorer，喂 3 条合成序列（`cdr3_like` / `repetitive` / `scrambled`，定义见脚本内 `SEQUENCES`），检查：
- `soft_prior_loss` 是否产生非零梯度（梯度路径）
- 候选的 `total_loss` 加入 naturalness 后，`MCTSTree.expand()` 算出的 `tree_reward` 是否真的变化（打分路径）

`--backends` 可选 `tcrlm,esmc,esmc_masked,ligandmpnn` 任意子集。

## 七、实验结论小结

- 三个 backend 均已用真实权重验证：加载成功、输出真实分布、soft gradient 非零（`validate_tcr_prior_backends.py` 结果）。
- TCRLM 判别力实测：正常 CDR3（`CASSLAPGATNEKLFF`）perplexity 6.60，退化重复序列（`MHMHMHMHMHMHMH`）perplexity 570.10，约 90 倍差距；smoke test 里 TCRLM 也是唯一给出"自然 &lt; 退化 &lt; 打乱"这种干净单调排序的来源，其余两个来源都是"极端重复序列"分数异常低（这是重复序列本身信息量低导致的,不是 bug，逐位置遮盖模式下依然如此）。
- LigandMPNN：`protein_mpnn` 模式（普通权重）vs 真实 `ligand_mpnn` 模式（真实 LigandMPNN 权重）对同一候选结构打分，hard score 差 0.005（-0.276 → -0.281），smoke test 里三条序列 NLL 差距在 0.02–0.06 之间——差距很小，初步判断是因为测试用的结构本身没有真正的配体/异质原子（细节见「三、LigandMPNN」）；这不是同一权重开关配体条件化的严格对照，结论有待在真实带异质原子的 pMHC 结构上复核。
- Unit test：接口层 17 passed/5 skipped（login node 秒级跑完），真实 backend 层 21 passed/1 skipped（计算节点，~1 分钟）。
- Pipeline smoke test：4 个 backend × 3 条合成序列，12/12 项检查（梯度非零 + tree value 确实变化）全部通过。
