# TCR Sequence Prior Loss — 4-Case 实验报告

日期：2026-07-15 ～ 2026-07-17；跑在 ProteoCraft 团队的 H200 服务器（`proteocraft` 主机），backbone 为 `protenix_base_20250630_v1.0.0`，数据来自 `CDR_redesign_4cases_minimal_package_20260714.zip`（9NMW_alpha/beta、1QRN_alpha/beta 共 4 个 case）。

<aside>💡 这份文档记录的是这一轮 4-case 验证的完整过程、发现的问题、修复方式，以及最终的实验结果和局限性。跟 `TCR_Prior_Loss_使用与测试.md`（算法说明 + 使用指南）是互补关系，这份更侧重"这次具体跑了什么、结果怎么样、能不能信"。</aside>

---

## 一、Loss 是什么，为什么会是负数

`total_loss`（有时也写作 `L_real`）不是"越接近 0 越好"的传统损失，而是一个可以任意为负的**综合目标函数**，写法沿用 AfDesign / ColabDesign / BindCraft 的惯例。计算逻辑集中在 `bindcraft/losses/total_loss.py::compute_all_losses`，一阶优化和零阶验证共用同一份代码，保证两处一致。

$$
L_{\text{total}} = w_{\text{plddt}} \cdot (1-\overline{\text{pLDDT}})
+ w_{\text{pae}} \cdot \overline{\text{PAE}}_{\text{intra}}
+ w_{\text{i\_pae}} \cdot \overline{\text{PAE}}_{\text{inter}}
+ w_{\text{con}} \cdot L_{\text{con}}
+ w_{\text{i\_con}} \cdot L_{\text{i\_con}}
+ w_{\text{iptm}} \cdot (1-\text{ipTM})
$$
$$
+ w_{\text{rg}} \cdot \text{elu}(R_g - R_g^{\text{理论紧凑值}})
+ w_{\text{helix}} \cdot \big(-\log p_{\text{helix接触}}\big)
+ w_{\text{clash}} \cdot L_{\text{clash}}
+ w_{\text{i\_clash}} \cdot L_{\text{i\_clash}}
+ w_{\text{pde}} \cdot L_{\text{pde}}
+ w_{\text{i\_pde}} \cdot L_{\text{i\_pde}}
$$

这次用的权重（yaml 里的 `w_*`）：

| 权重 | 值 | 权重 | 值 |
|---|---|---|---|
| `w_plddt` | 0.1 | `w_helix` | **-0.3** |
| `w_pae` | 0.4 | `w_clash` | 0.5 |
| `w_i_pae` | 0.1 | `w_i_clash` | 0.5 |
| `w_con` | 1.0 | `w_pde` | 0.02 |
| `w_i_con` | 3.0 | `w_i_pde` | 0.01 |
| `w_iptm` | 0.05 | `w_rg` | 0.3 |

**关键在 `w_helix = -0.3`（唯一的负权重）。** `l_helix = -\log(\text{binder 内 } i,i{+}3 \text{ 残基对的接触概率})`（`bindcraft/losses/geometry.py::compute_helix_loss`）——这个值衡量的是"binder 有多不像螺旋"：越不像螺旋（CDR 保持天然的柔性 loop 构象），`l_helix` 越大。乘上负权重后，**CDR 越不螺旋化，这一项对 `total_loss` 的贡献就越负**。这是有意设计成"奖励项"，因为 CDR 本来就该是柔性 loop，不该被优化成人工螺旋。

再叠加权重最大的 `w_i_con=3.0`（界面接触，接触越好这项越小），如果一个设计**界面接触质量好 + CDR 保持 loop 形态**，这两项加起来能把 `total_loss` 拉到明显为负。**所以 `total_loss` 越负，代表结构质量越好，不是计算出错。**

---

## 二、这一轮验证发现并修复的 3 个真实 bug

这是第一次把 tcr_prior 三个 backend 接进真实 GPU 上的完整 CDR pipeline 跑，之前的验证（PBS job、smoke test）都没有触发这些问题：

### 1. TCRLM 设备不匹配
`TCRLMBackend.per_position_logprobs()` 里 `length_t`、`reorder` 两个 tensor 一直建在 CPU 上，模型输出在 GPU 上时会直接报 `RuntimeError: Expected all tensors to be on the same device`。已修复（`tcr_prior.py`）。

### 2. LigandMPNN 拿不到候选结构的 PDB 路径
`pipeline.py` 里已经写好的 `_attach_tcr_prior_structure_metadata()`（把零阶验证算出来的坐标 dump 成临时 PDB、挂到 `metadata['pdb_path']`）从来没有被 `pipeline_cdr.py` 的两个 `_apply_tcr_verifier` 闭包调用过，导致 LigandMPNN 在 CDR pipeline 里永远拿不到结构、只能 no-op。已修复：在 `pipeline_cdr.py` 里补上这个调用。

### 3. TCRLM 的"均匀分布兜底"bug（这次最重要的发现）
`TCRLMBackend` 是 **CDR3-only 模型，`SEQ_MAX_LEN=20`**。CDR pipeline 传给它的是整条 binder 序列（150-250 个残基）+ CDR 在整条序列里的绝对位置（比如 88-103）。原来的代码会把序列从头截断到前 20 个残基——CDR 位置永远在这个截断窗口之外，于是每次都触发"越界 → 均匀分布兜底"，返回一个跟具体 CDR3 序列**完全无关的常数分数**（`NLL = log(20) ≈ 2.9957`，对应"20 个氨基酸等概率"）。

验证方式：检查已经跑完的 checkpoint 里 `loss_dict.tcr_prior_hard`，发现 4 个 case **完全相同**：`0.29957323...` = `log(20) × 0.1`，这不是巧合，说明模型根本没有真正看到任何具体序列。

修复：让 `per_position_logprobs()` 自动提取"真正包含 CDR 区域"的 20 残基窗口（以 CDR 为中心），而不是无脑从头截断；并加了 `strict_window` 开关（跟 `require_real_tcr_prior` 联动）——修复后如果还有位置落在窗口外，直接报错而不是静默兜底。

修复验证（真实 CDR3 vs 打乱后的序列，同一个位置窗口）：

| 序列 | NLL | Perplexity |
|---|---|---|
| 真实 CDR3（`CAVNNARLMFGDGTQL`） | 3.34 | 28.3 |
| 打乱（`MHMHMHMHMHMHMHMH`） | 5.61 | 274.0 |
| 均匀分布参考值（bug 时的输出） | 2.996 | 20.0 |

修复前，这两个序列会打出**完全相同**的分数；修复后明显区分开，且方向正确（自然序列更"可预测"）。

**这次报告里的所有 TCRLM 数据，都是修复后重新跑出来的。** 修复前的 10.6 小时那次跑（本次会话早些时候完成的）已经确认是"TCRLM 分支等效于 prior 关闭"，不代表 TCRLM prior 真实生效的结果，不采用。

---

## 三、实验方法

### 环境
- Backbone：`protenix_base_20250630_v1.0.0`（`/root/Bingchen/resources/checkpoints/`，checkpoint 和 CCD cache 复用团队已有配置，没有重新下载）。
- 三个 backend 统一在同一个 conda 环境（`proteocraft_protenix_py311`）里跑，避免了 `fair-esm`（Protenix 自带）和 ESM-C 的 `esm` 包名冲突：额外装了 `transformers==4.57.6`（并从另一个环境补齐了缺失的 `models/esmc` 子模块）、`prody`（LigandMPNN 需要解析 PDB）。
- ESM-C 用的是服务器本地已缓存的 `biohub/ESMC-6B`（走 `transformers.models.esmc.modeling_esmc.ESMCForMaskedLM`，`HF_HUB_OFFLINE=1` 强制离线，不碰网络）。

### Pipeline 配置
- `mode: mcts`，`n_steps: 150`，`mcts_max_nodes: 5000`，`mcts_strategy: two_stage`（StageA legacy_bootstrap + StageB document_puct_refine），其余超参跟包里原始 yaml 一致。
- `tcr_prior_region: cdr`（只对 CDR 区域打分），`tcr_prior_gradient_weight = tcr_prior_verifier_weight = 0.1`。
- TCRLM、ESM-C：gradient + verifier 两条路径都打开。
- LigandMPNN：**只开 verifier**，`use_tcr_prior_gradient: false`——因为 gradient 路径发生在候选结构还没生成的阶段，LigandMPNN 是结构条件化模型，此时天然没有 PDB 可用，会直接报错（`require_real_tcr_prior=true` 下）。这是设计上的限制，不是 bug，已经写进 `TCR_PRIOR_BACKENDS.md`。

### Baseline 来源
`/root/Bingchen/evaluations_archive/evaluations/20260621_cdr_v2_faithful_verify_all/`，2026-06-21 跑的，早于 tcr_prior 整个工作，完全没有 sequence prior。取每个 case 的 rank01（已经过 Rosetta/verifier 综合筛选的最优候选）。

<aside>⚠️ Baseline 对比存在下面几个已知的不对等，见第六节"局限性"，先说清楚：backbone 版本大概率不同（baseline 疑似用 `protenix-v2`，我们用的是 `protenix_base_20250630_v1.0.0`）；baseline 的 rank01 是从 59 个 run 里人工筛出来的最优候选，我们是单次 run 的直接输出。</aside>

---

## 四、结果一：结构质量对比

`loss_dict.total`（纯结构 loss，pLDDT/PAE/接触/Rg/helix/clash/PDE 的加权和，**不含** TCR prior 项，跟 baseline 同一套计算逻辑，可以直接比较）：

| Case | Baseline（无 prior） | TCRLM（已修复） | ESM-C | LigandMPNN |
|---|---:|---:|---:|---:|
| 9NMW_alpha | -0.9844 | -1.5218 | -0.9082 | -0.8119 |
| 9NMW_beta | -1.6712 | -1.1507 | -1.1113 | -0.9900 |
| 1QRN_alpha | -1.4824 | -1.6369 | -1.9913 | -1.1468 |
| 1QRN_beta | +0.4774 | +0.1792 | -0.8823 | +0.6034 |
| **平均** | **-0.9152** | **-1.0326** | **-1.2233** | **-0.5863** |

- 12 个 case（4 case × 3 backend）全部 `status: ok`，**framework 区域全程未变**（`best_framework_unchanged` / `final_framework_unchanged` 均为 `true`），只在 CDR 区域内变异，符合设计预期。
- 平均来看 ESM-C、TCRLM（修复后）都比 baseline 结构质量更好，LigandMPNN 比 baseline 差；但正如"局限性"一节说的，这是**单次 run** 的对比，不能就此下"prior 让结构变好/变差"的结论。
- 单次跑耗时：TCRLM 约 3.09/3.03/2.93/3.07 小时（4 个 case），ESM-C 约 2.86/2.77/2.81/2.80 小时，LigandMPNN 约 2.90/2.93/2.82/2.81 小时——三个 backend 量级一致，符合预期（GPU 6/7/4 上并行跑的）。

`total_loss`（顶层字段，= `loss_dict.total + tcr_prior_hard`，是我们 pipeline 内部拿来做 MCTS 选择用的那个数，**跟 baseline 不是同一套东西，不能直接比**，附在这里仅供内部参考）：

| Case | TCRLM total_loss | ESM-C total_loss | LigandMPNN total_loss |
|---|---:|---:|---:|
| 9NMW_alpha | -1.1835 | -0.7672 | -0.6399 |
| 9NMW_beta | -0.8376 | -0.9223 | -0.8636 |
| 1QRN_alpha | -1.3031 | -1.8055 | -0.9711 |
| 1QRN_beta | +0.4760 | -0.6912 | +0.7256 |

---

## 五、结果二：序列自然性交叉验证

**这是专门用来回答"加了 prior 之后，CDR3 是不是真的更自然了"这个问题的实验**——跟第四节的结构质量是完全独立的两个维度。

方法：把 baseline 和我们三个 backend 各自最优结果的 CDR3，**都拿去过同一套打分器**（修复后的 TCRLM masked-marginal NLL、ESM-C NLL），跟各自的搜索过程完全脱钩，纯粹事后统一评分。NLL 越低 = 越"自然"（打分器认为这个序列越容易预测/越像真实 TCR）。

| Case | 来源 | TCRLM NLL ↓ | ESM-C NLL ↓ |
|---|---|---:|---:|
| 9NMW_alpha | baseline | 3.2611 | 1.7801 |
| | TCRLM 引导 | 3.3828 | 2.3527 |
| | ESM-C 引导 | **3.2468** | **1.4101** |
| | LigandMPNN 引导 | 3.4698 | 2.2448 |
| 9NMW_beta | baseline | **2.9942** | 3.0927 |
| | TCRLM 引导 | 3.1308 | 2.5079 |
| | ESM-C 引导 | 2.7558 | **1.8894** |
| | LigandMPNN 引导 | 3.0709 | 1.9386 |
| 1QRN_alpha | baseline | 3.4768 | 1.9017 |
| | TCRLM 引导 | **3.3379** | 2.2475 |
| | ESM-C 引导 | 3.6122 | 1.8576 |
| | LigandMPNN 引导 | 3.5825 | **1.7875** |
| 1QRN_beta | baseline | 3.2823 | 2.6916 |
| | TCRLM 引导 | **2.9680** | 3.4022 |
| | ESM-C 引导 | 3.3413 | **1.9115** |
| | LigandMPNN 引导 | 3.2396 | 2.2182 |

（粗体 = 该 case、该打分器下最自然的一个来源）

**观察（如实说，不夸大）**：
- **ESM-C 引导的序列在 ESM-C 自己的打分下，4 个 case 里 3 个都是最自然的**（9NMW_alpha、9NMW_beta、1QRN_beta），1QRN_alpha 也很接近最优——这是相对一致、符合预期的信号：ESM-C prior 确实在把搜索推向 ESM-C 认为更自然的序列。
- **TCRLM 引导的序列在 TCRLM 自己的打分下，4 个里只有 2 个是最优**（1QRN_alpha、1QRN_beta），另外 2 个（9NMW_alpha、9NMW_beta）反而是 baseline 或 ESM-C 引导的序列更自然——这个信号不如 ESM-C 干净，可能是 `tcr_prior_gradient_weight/verifier_weight=0.1` 相对结构 loss 的权重偏小，也可能是单次搜索的随机性掩盖了真实效果。
- **LigandMPNN 引导的序列没有在任何一个 case、任何一个打分器下明显胜出**，这符合预期——LigandMPNN 的"自然性"来自结构/骨架条件化，不是序列语言模型式的 naturalness，不应该期待它在 TCRLM/ESM-C 打分上有系统性优势。
- Baseline（完全没有 prior）在个别 case 下（9NMW_beta 的 TCRLM 打分）反而是最自然的——说明"没加 prior 的设计一定更不自然"这个假设也不成立，naturalness 和这次的搜索/权重设置之间的关系比较微妙，不是单调的。

**结论**：sequence prior loss 这个机制本身是真实生效的（三个 backend 都打出了随候选变化的真实分数，不是常数），但"加了 prior 就一定让 CDR3 更自然"这个假设，在当前的权重设置（0.1）和单次搜索下，**只在 ESM-C 上得到了比较一致的支持，TCRLM 和 LigandMPNN 的证据比较弱或者不适用**。

---

## 六、局限性（诚实列出，不回避）

1. **单次 run，非重复实验**：结构质量对比（第四节）用的是每个 backend 单次 150 步搜索的最优结果。MCTS 搜索路径本身有随机性（候选提案、退火调度、采样），不同 backend 之间的差异，可能是 prior 的真实效果，也可能只是搜索轨迹不同导致的噪声。要严谨的话需要多个随机种子重复跑、报告均值和方差。
2. **Backbone 版本不完全一致**：baseline 大概率用的是 `protenix-v2`，我们这次为了跟之前已验证过的 smoke test 保持一致，用的是 `protenix_base_20250630_v1.0.0`。
3. **Baseline 经过多轮筛选，我们没有**：baseline 的 rank01 是从 59 个 run 里人工挑出来的最优候选（可能还经过 Rosetta 二次筛选），我们的结果是单次 run 的直接输出，没有做等价的多候选筛选。
4. **Verifier 路径的因果效应还没有单独隔离验证**：目前确认了"prior 分数是真实的、会被加进 total_loss、这个 total_loss 确实驱动了日志里看到的 MCTS accept/reject 决策"，但**没有做"同一个随机种子、prior 开 vs 关"的对照实验**，去直接证明"MCTS 最终选中的候选，是因为 prior 的存在才不一样"。这是最值得补的一组实验。
5. **Gradient 路径和 verifier 路径的贡献没有分开测**：TCRLM、ESM-C 这次都是两条路径同时打开，没有做"只开 gradient" vs "只开 verifier" vs "两个都开"的消融，不知道两条路径各自的贡献有多大。
6. **LigandMPNN 的结构条件化能力可能没有充分发挥**：之前的额外测试发现，测试用的 pMHC+TCR 结构里没有真正的配体/异质原子（`other_atoms` 数据源），LigandMPNN 相对 ProteinMPNN 的"看配体原子"优势在这批数据上体现不出来。

---

## 七、后续建议

1. 如果需要更严谨的结论，优先做 **verifier 路径的开关对照实验**（同一个 case、同一个随机种子，prior 权重设为 0 vs 0.1，其余不变）——这是当前证据链里最薄弱的一环，也最容易补。
2. 如果时间允许，TCRLM/ESM-C 各自的 gradient-only / verifier-only 消融，能回答"两条接入路径谁在起作用"。
3. 想跟 baseline 做更严格的对比，需要统一 backbone 版本（`protenix-v2`）、统一候选筛选方式（比如同样跑多个候选再取最优）。
4. LigandMPNN 如果要发挥真正的结构优势，需要在真正含有配体/异质原子的 pMHC-TCR 结构上测试，或者按最初协作者建议的那样，在 TCR-pMHC 数据上做一次 fine-tune。

---

*相关文档：`TCR_Prior_Loss_使用与测试.md`（算法说明与使用指南）、`bindcraft/hallucination/TCR_PRIOR_BACKENDS.md`（三个 backend 的实现细节与已知限制）。*
