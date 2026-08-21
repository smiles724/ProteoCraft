# TCR-Specific Loss — 6-Case 消融实验报告

> **核心问题**：在 A6 scaffold 固定、仅重设计 CDR3 的条件下，显式引入 TCR-specific 的约束，能否把设计出的 CDR3 推向更接近天然 TCR repertoire 的序列空间，同时不牺牲结构与 interface 质量？
>
> **一句话结论**：序列先验通道成立（naturalness↑ 且结构↑，无 trade-off）；接触约束通道不成立（达成了自身指标，但破坏了它本该保护的 `r_pep`）。失效机制已被直接测量定位，改法明确。

- 分支：`tcr-esmfold2-20260820`　backbone：`7bf6251`　结果 commit：`e969afe`
- 规模：3 peptides × α/β = 6 cases × 4 arms = 24 runs，**24/24 完成，0 失败**
- 日期：2026-08-21

---

## 一、两条 TCR-specific 通道

本轮测试两个**独立、可分别 ablate** 的信号，注入位置完全对称。

### 通道一：TCR sequence prior（直接作用于序列空间）

用 TCRLM（在天然 TCR repertoire 上训练）对 CDR3 打分，把「像不像真实 TCR」变成可优化量。

```text
soft prior → proposal gradient
hard prior → verifier / MCTS tree value
```

### 通道二：TCR contact loss（通过相互作用模式间接约束）

奖励 CDR3 真正接触 peptide，惩罚 framework 介导的 pMHC 结合，防止设计退化成泛化 MHC binder。

```text
L_contact = contact_nll(CDR3, peptide)
            + λ_FW · relu(hinge − contact_nll(framework, pMHC))
```

第二项用 hinge 而非字面的 `−Contact(FW,pMHC)`：`contact_nll` 上无界，字面写法会让优化器靠「把框架推到无穷远」拿到无限奖励。hinge 把惩罚限制在 `[0, hinge]`，保留意图但数值安全。

**复用而非重写**：`_contact_loss`（`esmfold2_runtime.py:303`）是 backbone 里已有的可微接触原语（原本给 hotspot loss 用），本轮直接复用，没有新写接触原语。

---

## 二、这一轮发现并修复的 3 个真实问题

### 1. 序列先验从来没有真正接进 ESMFold2（本轮最重要的发现）

上一轮的 `TCRSequencePriorScorer` 只接进了 `pipeline_cdr.py` / `pipeline.py`，也就是 **Protenix** 通路（旧配置里写着 `model_name: protenix_base_20250630_v1.0.0`）。而 `esmfold2_pipeline.py` 里 `tcr_prior` 出现次数为 **0**。

`run_hallucination.py` 会把 `tcr_prior_config=` 传给 `run_esmfold2_mcts_hallucination()`，但该函数既没有这个参数、也没有 `**kwargs`（AST 已验证）——**这是一个 latent TypeError，意味着「ESMFold2 + prior」这条路从来没有真正跑起来过**。

因此本轮不是「验证已有实现」，而是一次**移植**：scorer 本体原样复用，新增 ESMFold2 侧的调用点。

### 2. framework mask 不能用 `~mutable_mask`

`mutable_mask` 只标了 CDR3，所以 `binder & ~mutable` 会把 **CDR1/CDR2 也算进 framework 惩罚**——而 CDR1/CDR2 与 MHC 的接触是天然的，惩罚它们在生物学上是错的。

修法：手工标注 A6 的六个 CDR，framework 定义为**两条 TCR 链的 variable domain 减去全部 CDR**（即 FR1∪FR2∪FR3∪FR4）。α/β scaffold 在三个 peptide 之间字节级相同，所以只需标注两次即可覆盖全部 6 个 case。

| 链 | 长度 | CDR1 | CDR2 | CDR3 | V domain |
|---|---|---|---|---|---|
| A6 α (TRAV12-2) | 200 | `DRGSQS` [25,31) | `IYSNGD` [48,54) | `AVTTDSWGKLQ` [88,99) | [0,110) |
| A6 β (TRBV6-5) | 243 | `MNHEY` [24,29) | `SVGAGI` [46,52) | `ASRPGLAGGRPEQY` [89,103) | [0,113) |

四个 IMGT 锚点（Cys23 / Trp41 / Cys104 / Phe118）全部核对通过，CDR3 span 与 case spec 完全一致。

### 3. verifier cache 会导致 loss 重复累加

verifier 结果按序列缓存（`verifier_cache[binder_seq]`）。若在 cache 命中之后再加 TCR 项，重复访问同一序列时会反复累加。修法：所有额外 loss 一律在 `verify()` **内部**计算，走 `extra_loss_dict`。

### 附：上一轮 TCRLM window bug 的现状

旧实现曾把整条 150–250 aa 的 TCR 截前 20 aa，导致 CDR3 不在窗口内 → uniform fallback → 固定 `NLL = log(20)`、weighted hard loss ≈ `0.29957`。

**该修复仍在，且本轮已重新验证**：TCRLM 与 ESM-C 两个 backend 均确认加载真实权重、soft gradient 非零、无 fallback，hard score 分别为 `−0.18645` / `−0.04451`，均非 `0.29957` 签名。

---

## 三、实验方法

### 环境

```text
server        erran (node-20-2, 8× H200)
python        /root/miniforge3/envs/proteocraft_esmfold2_py312/bin/python
HF_HOME       /root/Bingchen/resources/hf_cache   # 复用，未复制任何 checkpoint
proposal      biohub/ESMFold2-Experimental-Fast
verifier      biohub/ESMFold2 (full)
TCRLM         /root/mingxuan/tcr_prior_backends/tcrLM
```

节点**无外网**，全部模型从本地 cache 解析。`esmc_300m` 不在 cache 中，因此 external ESM-C 通道只能用 `esmc:biohub/ESMC-6B`。

### Pipeline 配置

```text
profile              esmfold2_d
task                 cdr_redesign (CDR3-only)
seed                 42
mcts_strategy        two_stage
verifier             4 loops × 50 sampling steps
λ_contact            0.1   (normalized-gradient 混合系数，非标量 loss 权重)
λ_FW                 1.0
hinge                12.5
```

> **注意**：ESMFold2 用的是 normalized gradient mixing，`λ = 0.1` 缩放的是**单位方向**，与旧 Protenix 里的标量 loss 权重 `0.1` 不数值等价。

### 输入

- 6 case 来自 `20260819_liubo_cdr6_native_20x200` 的 case spec
- 链构成：`chain0 = 9-mer peptide (target)` / `chain1 = HLA-A*0201 180aa (scaffold)` / `chain2 = 对侧 TCR 链 (scaffold)`
- **输入中没有 β2m**，故 pMHC = peptide + HLA
- role 无法区分 HLA 与对侧 TCR 链（两者都是 `scaffold`），所以 mask **按 chain index 构建**
- `template_usage: none_sequence_only` — ESMFold2 确为 sequence-only，A6 只提供 framework 序列，**不固定原始 docking pose**

### 2×2 设计

| Arm | sequence prior | contact loss | 回答 |
|---|---|---|---|
| A | OFF | OFF | 基线（注意 ESMFold2 自带 ESM-C guidance，并非「无先验」） |
| B | ON | OFF | B−A：序列先验单独效果 |
| C | OFF | ON | C−A：接触约束单独效果 |
| D | ON | ON | D−C / D−B：条件增益 |

### 评估纪律（关键）

全部 arm 由**同一 verifier、同一固定 eval seed（4242）**重新打分，**contact 指标对没有优化过它的 arm 也照样计算**。因此没有任何一列依赖 arm-specific 的 `total_loss`——这正是上一轮 Protenix 分析被污染的地方。

`esm_verify_loss` 是纯结构项，不含任何 TCR 项，为 arm 间可比量；`total` 含 TCR 项，**不可跨 arm 比较**。

---

## 四、结果一：序列先验成立

### 分解表（6 case 均值差 / 朝好方向的 case 数）

| 对比 | 结构 loss | TCRLM NLL | ESM-C NLL | CDR3–pep | FW–pMHC 间隔 | r_pep |
|---|---|---|---|---|---|---|
| **B − A** 只加先验 | **−0.042** (5/6) | **−0.691** (5/6) | +0.042 (3/6) | +0.253 (2/6) | **+0.197** (4/6) | +0.054 (3/6) |
| **C − A** 只加接触 | +0.088 (1/6) | −0.055 (4/6) | +0.062 (3/6) | **−0.917** (5/6) | −0.577 (3/6) | **−0.113** (2/6) |
| **D − C** 接触上加先验 | −0.036 (4/6) | **−0.321** (5/6) | −0.153 (3/6) | −0.152 (4/6) | **+0.175** (5/6) | **+0.200** (5/6) |
| **D − B** 先验上加接触 | +0.094 (2/6) | +0.315 (3/6) | −0.132 (3/6) | **−1.321** (5/6) | −0.598 (2/6) | +0.033 (3/6) |

**B−A 是一次没有 trade-off 的改善**：TCRLM NLL −0.691（5/6），同时结构 loss 也更好（5/6）。这是对核心问题的正面回答。

### 与天然 A6 CDR3 的对照

天然 A6 自身 CDR3 的 TCRLM NLL 作为参照（**全程不参与优化**）：α = 3.2263，β = 2.0675。

| case | arm A | arm B | 天然 A6 | B 优于 A | B 越过天然线 |
|---|---|---|---|---|---|
| ALGRRIPPV_α | 3.802 | **2.122** | 3.226 | ✓ | ✓ |
| KLNPRIPRI_α | 3.641 | **2.906** | 3.226 | ✓ | ✓ |
| YLNPRVARI_α | 3.469 | **2.521** | 3.226 | ✓ | ✓ |
| ALGRRIPPV_β | 3.299 | 2.856 | 2.067 | ✓ | ✗ |
| KLNPRIPRI_β | 3.335 | 2.498 | 2.067 | ✓ | ✗ |
| YLNPRVARI_β | 2.742 | 3.240 | 2.067 | ✗ | ✗ |

**α 链 3/3 越过天然线**（设计出的 CDR3 比 A6 本身更「repertoire-典型」），**β 链 0/3**。这个 α/β 不对称值得单独追。

---

## 五、结果二：接触约束不成立，及其失效机制

接触约束确实提高了 CDR3–peptide 接触（5/6，−0.917），但：结构变差（1/6），framework 反而更靠近 pMHC（−0.577），**`r_pep` 下降（2/6）**。即它正在制造它本该阻止的泛化 MHC binder。

三层机制，**均为直接测量所得**：

### 机制一：梯度与结构梯度反向

跨 6 case、1056 个优化步：

| | arm C | arm D |
|---|---|---|
| `cos(contact, structure)` 均值 | **−0.348** | −0.326 |
| 为负的步数占比 | **76.7%** (810/1056) | 74.2% |
| \|contact\| / \|structure\| | 3.23× | 3.30× |
| `cos(contact, tcr_prior)` | +0.000 | −0.008 |

接触梯度并非「在结构允许的方向上加偏好」，而是系统性地对抗结构优化。

### 机制二：目标函数只约束分子

distogram 是全局预测，降低 `contact_nll(CDR3,pep)` 最省力的路径是让模型预测一个**整体更紧的复合物**——因为分母 `Contact(TCR,pMHC)` 完全不在目标里。

签名很清楚，两侧**同向下降**：

| | arm A | arm C | Δ |
|---|---|---|---|
| CDR3–peptide nll（想要更近） | 8.93 | 8.02 | −0.92 ✓ |
| framework–pMHC nll（想要更远） | 13.31 | 12.73 | −0.58 ✗ |

后果体现在 `r_pep`：两个 α case 从 0.451 / 0.161 掉到 **0.025 / 0.018**，CDR3 对 peptide 的接触占比几乎消失。

### 机制三：hinge 挡板放得太低

hinge 设在 12.5，而实测 framework nll 分布是 **11.5–14.7（中位 13.2）**。框架有近 0.7 nats 的自由下滑空间才碰到惩罚，实际下滑量恰好 0.58——滑到门口就停住。24 个 cell 中**只触发 4 次**，等于没有约束力。

---

## 六、改法（机制 ↔ 改动一一对应）

| # | 机制 | 改法 | 性质 |
|---|---|---|---|
| 01 | 梯度反向（77% 步骤为负） | 梯度投影（PCGrad 式）：混合前投影掉与结构梯度反向的分量。退一步可把 `λ_contact` 0.1 → 0.03 | 补丁 |
| 02 | 目标函数只约束分子 | **直接优化 `r_pep` 比值**。它已算好且可微，当前只是包在 `no_grad` 里。比值天然免疫「整体压紧」 | **根本** |
| 03 | hinge 位置过低 | hinge 12.5 → **13.5**（实测中位数之上），一行 config | 补丁 |

**只做一件事的话，做 02。**

### 一个额外的好消息

`cos(contact, tcr_prior)` ≈ 0 —— 两条通道在梯度空间里**几乎完全正交**。这正是当初设计成可分别 ablate 时想要的性质，也解释了 arm D 的效果大致等于 B 与 C 的叠加。**所以问题出在目标函数的形式上，不在两条通道的耦合上**，改 02 就够，不需要重构整条通路。

---

## 七、局限性（诚实列出，不回避）

### 控制项全部干净

| 检查 | 结果 |
|---|---|
| job 完成 | 24/24，0 失败 |
| CDR3 互不相同 | 24/24 |
| framework 被改动 | **0** |
| CDR3 外突变数 | 0（全部 cell） |
| clash | 0 |

### 能说的

- 符号一致的两条：TCRLM prior 提升 CDR3 的 repertoire typicality（5/6）；接触约束提高 CDR3–peptide 接触（5/6）。
- 失效机制是**直接测量到的**（1056 步梯度诊断），不是推测。

### 不能说的

1. **只有 1 个 seed。** 结构那几个 delta（±0.04~0.09）小到一个 seed 分不开 run-to-run variance。要下结论必须补 seed 43 / 44。
2. **ESM-C NLL 每个 arm 都是 3/6**，纯抛硬币——没有任何 arm 改善了通用蛋白 naturalness。
3. **self-scoring 问题。** TCRLM prior 让 TCRLM NLL 变好，但 ESM-C NLL 没动，与上一轮 Protenix audit 抓到的是同一模式。因此对核心问题的**正确表述是「推向了 TCR repertoire 的典型区域」，而非笼统的「更 natural」**。
4. **α/β 不对称尚无解释**（α 3/3 越过天然线、β 0/3）。
5. 搜索预算较短（`esmfold2_d`），未验证更长预算下结论是否保持。
6. 尚未做 post-hoc 与已知 cognate TCR 的比对（cognate TCR 全程仅用于评估，未进入初始化/检索/搜索/优化）。

---

## 八、后续建议

1. **改法 02 + 03 一起跑**（`r_pep` 作为优化目标 + hinge 13.5），仍然 6 cases、seed 42，与本轮完全 matched，直接看 `r_pep` 与结构能否同时守住。
2. **并行补 arm B 的 seed 43 / 44**，把唯一的正面结论钉死。
3. 上述两件事可并行，成本约为本轮的一半。
4. 追一下 α/β 不对称的来源（V 基因？CDR3 长度？β 链参照本身就更典型？）。
5. external ESM-C 通道（arm E/F）本轮未纳入正式实验，smoke 已通过，可在下一轮补上。

---

## 附：复现

```bash
# 正式实验
python run_formal.py --arms A,B,C,D --gpus 4,6,7 --out formal_tcr6_20260820

# 匹配化后评估（全部 arm 同一 verifier / 同一 eval seed）
CUDA_VISIBLE_DEVICES=4 HF_HOME=/root/Bingchen/resources/hf_cache \
  python evaluate_formal.py --out formal_eval_full.json

# 出报告表
python report_formal.py --eval formal_eval_full.json
```

产物已 commit 在 `experiments/formal_tcr6_20260820/`（仅 metadata，72K，不含大文件）。
