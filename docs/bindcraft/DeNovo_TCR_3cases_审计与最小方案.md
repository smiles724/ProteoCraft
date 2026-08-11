# De novo TCR 设计（HLA-A\*02:01 + 3 个 9-mer）— Pipeline 审计与最小接入方案

日期：2026-08-11。目标：给定 HLA-A\*02:01 和 `ALGRRIPPV` / `YLNPRVARI` / `KLNPRIPRI`
三个 9-mer peptide，设计能结合对应 pMHC 且保持天然 α/β TCR topology / framework /
CDR organization 的新 TCR。实验已知的 cognate TCR **只用于最后 post-hoc sanity
check，不进入任何设计初始化**。

---

## 一、现有 CDR pipeline 需要什么输入

入口：`benchmark/script/run_cdr_tcr_domain_redesign.py`
（→ `bindcraft/hallucination/pipeline_cdr.py::run_mcts_hallucination_cdr`）。
每个 case 需要 4 样东西：

| 输入 | 含义 | 硬性要求 |
|---|---|---|
| `fasta_path` | 多链复合物 FASTA，header 必须含 `\|Chain X\|` | 只允许 20 种标准氨基酸 |
| `template_path` | **整个复合物**的 CIF/PDB template | 链数 ≥ partner 链数；每条 query 链要能对上一条 template 链 |
| `binder_chain` / `target_chains` / `scaffold_chains` | 链角色划分 | binder 只能是**一条**链 |
| `cdr_motif` | 一段**连续且唯一**的子串，用来定位 CDR | `make_cdr_mask()` 只支持单个 motif |

两个容易被忽略的事实：

- **MSA 不需要**。`bindcraft/utils.py:522` 把 `msa` 放进 `dummy_feats`，走的是
  dummy MSA，所以离线集群上不需要跑 MSA search。
- **template 决定 framework**。`_cdr_binder_template_mask()`
  （`pipeline_cdr.py:1157`）返回 `[not cdr]`：framework 残基的坐标由 template
  固定喂进去，CDR 位置在 template feature 里被打成 gap。也就是说
  **template 同时提供了 pMHC 结构、TCR framework 构象、以及 TCR–pMHC 的对接位姿**。

所以从"只有 HLA + peptide 序列"到能跑，缺的不是 pMHC 结构本身，而是
**一个已经摆好对接位姿的 TCR–pMHC 复合物 template**。单跑 Protenix 折一个 pMHC
解决不了这一步（还缺 TCR 怎么搭上去）。

---

## 二、选定的 scaffold 与理由

用 **1QRN 的 A6 TCR / HLA-A2 复合物**作为 generic scaffold：

- HLA-A\*02:01（chain A，残基 1-180）+ β2m（chain B）+ **9-mer** peptide（chain C），
  peptide 长度与三个 target 完全一致 → 只换序列不换长度，template 映射走
  `len(query)==len(template)` 的恒等映射路径，不需要 realign。
- TCR α/β 是 TRAV12-2 / TRBV6-5（A6，Tax-specific），与这三个 peptide 的 cognate
  TCR 无关，符合"non-cognate scaffold"的要求。
- 这份 template 已经在 4-case 实验里被验证过能跑通，规避了新 template 的解析风险。

备选 scaffold：`9NMW`（TCR156 / HLA-A2 / 9-mer），同一套流程可直接换。

### CDR 编号（IMGT，anchor 推导，不依赖 ANARCI）

集群上没有 ANARCI，`scripts/prepare_denovo_tcr_cases.py` 改用 IMGT 保守
anchor（Cys23 / Trp41 / Cys104 / `[FW]GxG` 的 118）推导边界。
**验证**：推出的 CDR3 与 4-case yaml 里手工标注的 `cdr_motif` 逐字符一致，
CDR1/CDR2 与文献里 A6 的 CDR 一致。

| 链 | CDR1 (IMGT 27-38) | CDR2 (56-65) | CDR3 (104-118) |
|---|---|---|---|
| D (α) | `DRGSQS` 26-31 | `IYSNGD` 49-54 | `CAVTTDSWGKLQF` 88-100 |
| E (β) | `MNHEY` 25-29 | `SVGAGI` 47-52 | `CASRPGLAGGRPEQYF` 89-104 |

（1-based，闭区间；完整机器可读版本在 `benchmark/input/denovo_tcr_case/cdr_spec.json`。）

---

## 三、已经做好的 data package

`scripts/prepare_denovo_tcr_cases.py` 从 scaffold 生成，产物在
`benchmark/input/denovo_tcr_case/`：

```
A6scaffold_<PEPTIDE>.fasta   # A=HLA-A*0201(1-180) B=β2m C=目标肽 D=TCRα E=TCRβ
A6scaffold_<PEPTIDE>.cif     # scaffold 几何 + chain C 被 thread 成目标肽
cdr_spec.json                # α/β 的 CDR1/2/3 motif + 0/1-based 区间 + IMGT anchor
```

**peptide threading**：template 的 chain C 保留主链 + CB 坐标，残基类型改写成目标
肽，侧链原子删掉，同时同步 `_entity_poly_seq`。这样 template 提供的是"A2 凹槽里
9-mer 的构象"，而不会把 scaffold 原本的 Tax 肽序列泄漏进 template feature。
（第一版用 gemmi 做 round-trip，会丢掉 Protenix template parser 依赖的 category，
报 `No protein chains found`；现在改成 mmCIF 文本级改写，原文件结构原样保留。）

配置：`benchmark/yaml/cdr_tcr/denovo_tcr_a0201_3cases.yaml`，6 个 case
（3 peptide × α/β），loss 权重与 4-case 实验完全一致，便于横向比较。

---

## 四、已验证到哪一步（CPU，无 GPU）

登录节点没有 GPU，所以跑的是全链路的**非 GPU 部分**，都通过：

1. `run_cdr_tcr_domain_redesign.py --prepare-only` → 6 个 case 全部
   `Validated`，CDR 区间与 `cdr_spec.json` 对得上。
2. 直接调 `bindcraft.utils._extract_target_template_features`（真实 Protenix
   template parser）：
   - 4 条链的 template feature 都成功抽出；
   - peptide 的 `template_sequence` = `ALGRRIPPV`（threading 生效，无泄漏）；
   - binder 的 CDR3 在 template 里是 `-------------`、atom mask 全 0；
   - framework atom mask = 681，非零 → framework 被 template 固定。

**还没跑的**：Protenix 前向 / MCTS 搜索 / TCR prior 打分，都需要 GPU。
注意本 shell 里必须 `export OPENBLAS_NUM_THREADS=1`，否则 scipy 导入阶段
OpenBLAS 起 64 线程会失败。

下一步（一个 GPU 节点上）：

```bash
export OPENBLAS_NUM_THREADS=1
python benchmark/script/run_cdr_tcr_domain_redesign.py \
  --config benchmark/yaml/cdr_tcr/denovo_tcr_a0201_3cases.yaml \
  --case ALGRRIPPV_alpha --n-steps 3
```

---

## 五、TCR sequence prior 的接入状态

`tcr_prior_config` 已经从 yaml 一路透传到 `pipeline_cdr`，gradient 与
verifier 两条路径都在，**这次不需要改代码**，yaml 里已经打开：

```yaml
use_tcr_prior_gradient: true    tcr_prior_gradient_weight: 0.1
use_tcr_prior_verifier: true    tcr_prior_verifier_weight: 0.1
tcr_prior_region: cdr           tcr_prior_model_name: esmc:esmc_300m
require_real_tcr_prior: true
```

本地资源：TCRLM 权重在
`/lus/eagle/projects/IMPROVE_Aim1/caom/tcr_prior_backends/tcrLM/pretrained_model/pretrained_model_tcr.bin`（158 MB，真权重）；
ESM-C 只缓存了 **300m**（`.../tcr_prior_backends/hf_cache`），不是 4-case 实验用的
6B——换机器时这是一个已知的不对等。

⚠️ **TCRLM 有 20 残基窗口限制**：`SEQ_MAX_LEN=20`，窗口以 CDR 区域为中心提取。
CDR3-only 没问题（13/16 残基），但一旦把 CDR1+CDR2+CDR3 合成一个 mask
（跨度 ~80 残基），`strict_window` 会直接报错。**多 CDR 场景下 TCRLM 需要按 loop
分段打分，ESM-C 不受影响。**

---

## 六、两个 contact CSV 能不能用

- `human_classI_00_position_contact_proportions.csv`：310 行 = TCR α/β 各 150 个
  IMGT slot + peptide P1-P10，列是 6 类接触的比例（TCR-Peptide / TCR-MHC /
  Bridge / TCRa-TCRb / Pep-MHC anchor / present-no-contact）。
- `human_classI_00_contact_type_LABELED_wide.csv`：216 行 = 216 个人类 class-I
  TCR–pMHC 复合物（行名是 PDB ID，含 1AO7 / 1QRN 等），316 列 = 每个 IMGT slot
  的接触类别标签。

**结论：可以，但只作为 evaluation / 事后 prior，暂不进 loss。** 理由和用法：

按区域聚合"碰到肽的比例"（只统计 n_present ≥ 20 的 slot）：

| 区域 | α | β |
|---|---:|---:|
| CDR3 | 0.315 | 0.340 |
| CDR1 | 0.271 | 0.204 |
| CDR2 | 0.067 | 0.135 |
| FR1 / FR3 / FR4 | 0.000 / 0.005 / 0.000 | 0.000 / 0.006 / 0.000 |

三个直接可用的判断：

1. **framework 固定是对的** —— FR1/FR3/FR4 在 216 个天然复合物里基本零接触。
2. **CDR3 优先、CDR1 其次** —— 支持"先只 redesign CDR3"的最小方案。
3. **CDR2 主要面向 MHC 而不是肽**（α CDR2 slot 59 的 TCR-MHC 比例 0.655，
   FR3 slot 66/67 也有 0.55/0.61）—— 这些 slot 是 docking angle 的指纹，
   适合做"设计出来的 TCR 是不是保持了天然对接几何"的评价指标，
   而不适合当序列 prior。

推荐用法：设计完成后把预测结构按同一套 IMGT slot 算接触类别，与这 216 个天然
复合物的分布比（每个 slot 上做 KL 或 cosine），得到一个独立于 TCRLM/ESM-C 的
"natural-ness"维度。**这条不依赖任何 cognate TCR 信息，可以在正式评估里用。**

---

## 七、还缺什么 / 建议的下一步

按优先级：

1. **多 CDR mask（当前最大的功能缺口）**。`make_cdr_mask()` 只接受一个连续
   motif，所以现在一次只能 redesign 一个 loop。需要加 `cdr_motifs: [...]`
   （约 15 行，纯加法，不影响老 case）。`pipeline_cdr` 侧的 `cdr_mask` 本来就是
   任意 `List[bool]`，不需要动。
2. **α/β 交替设计**。现在 binder 只能是一条链，另一条 TCR 链当 scaffold（CDR 被
   冻在 A6 上）。最简单的做法是 α → β → α 交替跑，每轮把上一轮的结果写回 FASTA。
3. **prior on/off 对照**。yaml 里除了 prior 两个开关全部一致，直接复制一份把
   `use_tcr_prior_*` 设成 false，同一个 seed 跑，这正好补上 4-case 报告第六节
   列的"最薄弱的一环"，也直接回答主线问题。
4. **post-hoc sanity check**（最后一步）：拿实验 cognate TCR 与设计结果比
   CDR3 相似度、以及第六节的 IMGT contact-pattern 距离。cognate 序列在此之前
   不得进入任何配置文件。

---

## 八、与上游 `smiles724/ProteoCraft` 的同步状态

- 我们的分支 `mingxuan` 与 `origin/mingxuan` **一致**，本地没有未推送的提交，
  TCR prior 的工作没有任何遗漏。
- 但 `origin/main` 从 `aea61f8` 前进到了 `43ccf46`（**约 50 个新提交**），全部是
  ESMFold2 方向的：`esmfold2_cdr.py`、`esmfold2_one_step.py`、
  `run_esmfold2_cdr_redesign.py`、`configs/proteocraft_profiles/`、
  hotspot coverage 评估等，`mingxuan` 还没有合过来。
- **冲突面很小**：两边同时改的文件只有 `scripts/run_hallucination.py`
  （我们 +36 行，上游 +1262 行）。`pipeline_cdr.py` 和 `tcr_prior.py` 上游没动。
- 上游新增了独立的 ESMFold2 CDR redesign 入口，与我们现在用的 Protenix 入口是
  并行的两条路，本方案不受影响。是否合并请你定。
