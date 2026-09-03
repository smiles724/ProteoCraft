# ProteoCraft Affibody 生成与已有实验数据验证计划

## 1. 目的和范围

本计划用于指导以下工作：

1. 使用当前 ProteoCraft 为 peptide–HLA-A*02:01（pMHC）靶点生成 Affibody。
2. 利用已有 LibA、LibB 测序数据和 cleavage-retention 数据验证生成结果。
3. 在处理 `CoEvo_target_panel_v3.xlsx` 中的 69 个靶点前，先证明 ProteoCraft 的结构评分和候选选择与已有实验结果一致。
4. 区分直接实验验证、局部 pair 转移、测序支持和纯计算外推，避免过度解释结果。

本计划对应的代码框架现已实现；详细操作见
[`AFFIBODY_WORKFLOW_GUIDE_zh.md`](AFFIBODY_WORKFLOW_GUIDE_zh.md)。当前仍未运行
序列生成、结构预测、GPU 模型评分或湿实验，所有 Go/No-Go 判断需在后续按本计划执行后完成。

## 2. 核心逻辑

可以把整个问题理解为：

```text
pMHC 是“锁”
Affibody 是“钥匙”
Affibody scaffold 是钥匙的固定外形
LibA/LibB 的 4 或 5 个可变位点是钥匙齿
```

已有实验没有测试任意序列的 Affibody，而是只测试固定 Affibody scaffold 上少数可变位点的组合。因此，若要让新生成的 Affibody 可以被已有实验数据验证，ProteoCraft 的设计空间必须与实验库一致：固定 framework，只改变实验中随机化过的位置，并限制为实验库使用过的氨基酸字母表。

整体工作流：

```text
整理并冻结实验数据
        ↓
建立 LibA/LibB scaffold redesign 配置
        ↓
统一构建 HLA-A02 | β2m | peptide 非 binder 输入并分配 chain role
        ↓
先对已有 228 个 cleavage 配对做盲评分
        ↓
进行 closed-set 选择验证
        ↓
进行 open-set Affibody 生成
        ↓
冻结候选后查询已有实验数据
        ↓
检查 pMHC 特异性和 HLA-only binding
        ↓
分批处理 69 个 target
        ↓
输出带证据等级的 Affibody candidate list
```

## 3. 已有数据盘点

### 3.1 Target panel

输入文件：`datafolder/CoEvo_target_panel_v3.xlsx`。

当前包含 69 个 9-mer 靶点：

- Tier 1：61 个，主要为 native P4/P5 register。
- Tier 2：3 个，P5/P6 shifted register。
- Tier 3：5 个，包含 atypical anchor 或展示方式不确定的情况。
- High-confidence：47 个。
- Oncogenic：34 个。
- 含 transfer-dependent residue：27 个。

表格中标记的 10 个 cleavage-supported target 主要对应 LibB peptide pair。若同时考虑 LibA，AFP 158-166、hTERT 540-548 和 HPV16 E7 11-19 也有局部 pair-level cleavage 支持。

### 3.2 测序数据

输入目录：

```text
datafolder/LibA Raw data/
datafolder/LibA top2pct/
datafolder/LibB Raw data/
datafolder/LibB top2pct/
```

轮次含义：

- R000：背景或初始库。
- R001：负选择。
- R002–R014：正选择。
- R009–R010：富集明显且尚未严重发生 family collapse，与 cleavage retention 的相关性最好。
- R011–R014：选择更严格，但后期选择伪影和 family collapse 更明显。

已有报告显示：

- LibA R9–R10 pooled count 与 retention 的 Spearman 相关约为 0.705。
- LibB R9–R10 pooled count 与 retention 的 Spearman 相关约为 0.738。
- LibA R9–R10 binder 分类 MCC 约为 0.491。
- LibB R9–R10 binder 分类 MCC 约为 0.640。

因此建议以 LibB 为主要生成与验证体系，LibA 作为补充和独立复核体系。

### 3.3 Cleavage-retention 数据

输入文件：`datafolder/Affibody-MHC data summary.pptx`。

- LibA：108 个 peptide-pair × Affibody 配对。
- LibB：120 个 peptide-pair × Affibody 配对。
- 总计：228 个有 retention 数值的配对。

采用现有报告中的定义：

```text
retention >= 75%：binder
retention < 75%：non-binder
```

这些数据是本计划中的金标准。测序富集用于更广泛但更弱的支持，不能替代 cleavage retention。

### 3.4 已知结构与阳性对照

输入文件：

```text
datafolder/nyeso_xx133_complex.pdb
datafolder/NYESO-binder N133 with pMHC (1).pse
```

本地 PDB 包含 HLA-A02、β2-microglobulin、peptide `SLLMWITQV` 和已知 Affibody xx133。xx133 对应的 LibB 可变位点 tuple 为 `NNYYF`。在 LibB cleavage matrix 中，`MW + NNYYF` 的 retention 为 83.77%，因此它既是实验 binder，也是结构阳性对照。

## 4. 实验证据等级

每个 target 和每个候选必须标注为以下等级之一。

### Level A：完整序列直接验证

新 target 的完整 peptide 与实验构建体相同，且候选 Affibody tuple 有实验记录。

最明确的案例是 NY-ESO-1 9V `SLLMWITQV`。Co-evolution peptide 构建体为 `SLLXXITQV`，所以 MW 对应的完整构建体就是 `SLLMWITQV`。

### Level B：P4/P5 局部化学转移

完整 peptide 不同，但其 P4/P5 pair 在 cleavage matrix 中测过。

例如 MAGE-A3 target 是 `KVAELVHFL`，P4/P5 为 EL；实验实际测的是 `SLLELITQV`。`SLLELITQV + EFYSV` 的 retention 为 98.45%，这支持 EL–EFYSV 的局部配对化学，但不能直接证明 EFYSV 会结合完整的 `KVAELVHFL/HLA-A02`。

### Level C：测序支持

该 P4/P5 pair 或 Affibody tuple 在 R000–R014 中有覆盖，但没有相应 cleavage retention。它可以支持 R9–R10 enrichment 和局部特异性分析，但不能直接等同于亲和力。

### Level D：纯计算外推

包括：

- P4/P5 含原 peptide library 未系统覆盖的残基。
- Tier 2 的 P5/P6 shifted register。
- Tier 3 atypical anchor。
- 没有可靠序列来源或展示方式不确定的 target。

这类候选只能报告为 computational prediction。

## 5. 第一阶段：整理并冻结验证数据

### 步骤 1：将 cleavage matrix 转成结构化表格

生成：

```text
data_prepared/cleavage_libA.csv
data_prepared/cleavage_libB.csv
```

每行字段：

```text
library
peptide_pair
experimental_peptide
affibody_tuple
retention
binder_label
assay_time_min
source_file
```

其中：

```text
experimental_peptide = SLL{peptide_pair}ITQV
binder_label = 1 if retention >= 75 else 0
```

为什么这样做：PPTX 适合展示但不适合程序分析。正式 CSV 可以防止错位、重复和人工复制错误，并让评分结果与标签通过明确字段连接。

### 步骤 2：合并 R000–R014 原始测序数据

生成 `data_prepared/sequencing_all_rounds.parquet`，每行包含：

```text
library, round, peptide_pair, affibody_tuple, count, frequency, pvalue
```

为每个精确 pair 计算：

```text
R000_frequency
R001_frequency
R009_frequency
R010_frequency
R9_R10_pooled_count
R9_R10_pooled_frequency
R9_R10_vs_R000_log2_enrichment
R9_top2pct
R10_top2pct
```

推荐 enrichment：

```text
log2((frequency_R009 + frequency_R010 + pseudocount) /
     (frequency_R000 + pseudocount))
```

为什么优先使用 frequency：不同轮次测序深度不同，raw count 不能直接跨轮比较。为什么不能把缺失记录当阴性：没有 reads 可能来自库覆盖或抽样，而不一定表示不会结合。

### 步骤 3：建立高置信负集

建议定义为：

```text
R001 中观察到
且未在任何正选择轮次达到支持阈值
且没有 cleavage binder 标签
```

为什么这样做：有些真实 binder 也可能在 R001 出现。不排除后来在正选择中富集的 pair，会把真实 binder 错标为阴性。

### 步骤 4：建立 target manifest

生成 `data_prepared/target_manifest.csv`，至少包含：

```text
target_id
epitope
full_peptide
register
up_facing_pair
tier
oncogenic
source
antigen
pdb
confidence
transfer_dependent_residue
experimental_support_level
applicable_library
```

支持等级应从 LibA/LibB cleavage pair 集合重新推导，不能只依赖 Excel 手工标记。

### 步骤 5：冻结数据版本

为原始文件和结构化输出记录路径、大小、SHA256、生成日期和解析脚本版本，生成 `data_prepared/data_snapshot_manifest.tsv`。

为什么这样做：后续若 PPTX、Excel 或 TSV 被替换，可以确认某次验证使用的是哪一版数据。

## 6. 第二阶段：建立 Affibody scaffold redesign 模式

### 步骤 6：重建并核对完整 scaffold

从 `pep-affibody sequences_updated.pptx` 恢复完整 LibA/LibB mature Affibody 序列，并用 NY-ESO PDB 的 Affibody chain H/I 核对结构可见区域。

需要明确：

1. Construct、mature protein 和 PDB residue 编号之间的关系。
2. LibA 位置 15、19、29、33 的准确序列索引。
3. LibB 位置 8、12、15、16、19 的准确序列索引。
4. 将实验 tuple 写回后能否恢复预期完整序列。

为什么必须先核对：PDB 可能缺少末端残基，PPTX 文本又由多个带格式片段组成。若位置偏移 1–2 aa，生成 tuple 会与实验数据错误对应。

### 步骤 7：定义 LibA/LibB profile

LibA：

```text
可变位置：15、19、29、33
允许字母表：A D E F H I K L N P Q S T V Y
其余位置：固定
```

LibB：

```text
可变位置：8、12、15、16、19
允许字母表：A D E F I K L M N S T V Y
其余位置：固定
```

生成：

```text
configs/affibody/libA_profile.yaml
configs/affibody/libB_profile.yaml
```

为什么固定 framework：实验只验证同一 scaffold 上的少数界面突变。若模型改变整个 scaffold，现有数据不再提供精确对照。

为什么限制字母表：若输出实验库未覆盖的 tuple，即使计算评分高，也无法做 exact lookup。

### 步骤 8：增加 allowed-amino-acid mask

当前 ESMFold2 流程已支持 `init_binder_seq` 和 `mutable_mask`，还需增加 `allowed_amino_acids_by_position`。

该限制必须作用于：

1. 初始 logits。
2. Soft optimization。
3. Mutation proposal。
4. Stage A/Stage B 离散候选。
5. Checkpoint 恢复后的候选。
6. 最终输出检查。

为什么不能只在最后删除非法候选：若优化过程中允许非法残基参与，模型会把搜索资源集中在不可验证空间，最后删除后只剩少量低质量候选。

### 步骤 9：增加输出完整性检查

每条生成序列检查：

```text
长度是否等于 scaffold 长度
非可变位置是否与 scaffold 完全一致
可变位置是否属于允许字母表
是否只含标准氨基酸
能否提取出唯一且正确长度的 tuple
```

任何检查失败的候选不进入后续评分或实验数据查询。

## 7. 第三阶段：统一构建 pMHC 复合物输入

### 步骤 10：定义非 binder 链和 chain role

每个 target 使用：

```text
HLA-A*02:01 | β2-microglobulin | 9-mer peptide
```

不能只输入 9-mer peptide，因为游离 peptide 与 HLA 沟槽中的构象不同；必须把完整 pMHC 三条链都送入 ESMFold2。不过，通用命令的 `--target` 只是“非 binder 序列容器”，不等于三条链都应具有 `role=target`。本 workflow 固定角色为：

```text
HLA-A*02:01       → scaffold
β2-microglobulin  → scaffold
9-mer peptide     → target
Affibody           → binder
```

只有 peptide 是本次需要区分的可变抗原，因此 peptide–Affibody 界面才是主目标；HLA 和 β2m 是所有候选共用的结构背景。将二者标为 scaffold 后，它们仍参与完整四链结构预测和 scaffold/全局结构约束，只是不再和 peptide 一起被平均进 target–binder 主界面分数。

这里的 scaffold 是 loss role，不是坐标模板。当前 ESMFold2 workflow 仍是 sequence-only，不会固定 HLA 或 β2m 的三维坐标。

建议从 NY-ESO PDB 中提取并冻结一套共同使用的 HLA-A02 和 β2m 序列，所有 target 只替换 peptide，以减少非目标变量。

### 步骤 11：统一 chain 顺序与编号

固定顺序：

```text
HLA-A02 → β2m → peptide → Affibody
```

对应命令参数固定为：

```text
--target HLA_SEQUENCE|B2M_SEQUENCE|PEPTIDE_SEQUENCE
--esmfold2_target_chain_roles scaffold,scaffold,target
```

脚本根据实际链长计算 peptide P1–P9 在拼接 target 中的全局 1-based index：

```text
global_P4 = len(HLA) + len(β2m) + 4
global_P5 = len(HLA) + len(β2m) + 5
```

为什么不能手填：ProteoCraft hotspot 使用全部非 binder 链拼接后的 index，scaffold 链也计入编号偏移；若偏移，模型会被引导到 HLA 或 β2m 的错误残基。代码在映射后还会检查 hotspot 必须落在 `role=target` 的 peptide 链上。

### 步骤 12：定义 hotspot

- Tier 1：peptide P4/P5。
- Tier 2：peptide P5/P6，并标记 register extrapolation。
- Tier 3：根据结构或明确假设单独定义，不自动继承 Tier 1。

为什么指定 hotspot：不指定时，模型可能主要优化 Affibody–HLA 接触，得到 HLA binder，而不是 peptide-specific TCR mimic。

### 步骤 13：target 输入检查

每个 target 检查：

```text
peptide 是否恰好为 9 aa
P2/P9 anchor 是否与表格一致
register 是否明确
hotspot 是否落在 peptide chain
PDB 引用是否与 peptide 序列一致
链序列是否只含标准氨基酸
```

## 8. 第四阶段：验证结构表示与评分器

### 步骤 14：NY-ESO 已知复合物 preflight

输入：

```text
pMHC：HLA-A02 | β2m | SLLMWITQV
Affibody：xx133，tuple NNYYF
参考结构：nyeso_xx133_complex.pdb
```

检查：

1. peptide 是否位于 HLA 沟槽。
2. Affibody 是否位于 peptide-facing 表面。
3. Affibody 是否接触 P4/P5 的 MW。
4. 是否存在明显 clash。
5. 是否主要结合 HLA 而几乎不接触 peptide。
6. 多个 paired seed 下界面是否基本稳定。

为什么先做：当前 ESMFold2 workflow 是 sequence-only，参考 PDB 只用于提取 HLA/β2m 序列，不参与结构生成。若已知复合物都不能形成合理几何关系，后续生成结果没有可信基础。此时应停止批量生成，先检查四链表示、chain role、hotspot、verifier 配置和 ESMFold2 环境；在该 workflow 通过已知对照前，不进入开放生成。

### 步骤 15：构建 228 个 exact experimental complexes

对每个 cleavage pair 构建：

```text
experimental_peptide = SLL{P4}{P5}ITQV
```

例如：

```text
MW → SLLMWITQV
EL → SLLELITQV
DP → SLLDPITQV
```

将 matrix 中每个 Affibody tuple 写回完整 LibA/LibB scaffold，得到 228 个精确对应实验构建体的完整序列。

为什么必须使用 `SLLXXITQV`：cleavage 实验测的是这个固定背景。直接使用 MAGE-A3、AFP 等完整新 peptide，会把局部 pair transfer 错当成 exact validation。

### 步骤 16：盲法统一评分

在不加载 retention 和 R9–R10 标签的情况下，使用与正式 `esmfold2_d` 设计相同的 `aligned_common` full-verifier loss 和 paired seed，对 228 个 complexes 评分，保存为：

```text
validation/known_pair_scores_blind.csv
```

主排序量固定为越低越好的：

```text
total_loss =
    1.0 × (1 - pair_chain_iptm)
  + 0.2 × (1 - binder_pLDDT)
  + 0.1 × (1 - pTM)
  + 0.1 × binder_Rg_excess
```

设计和盲评都固定使用 `scaffold,scaffold,target`。因此公式中的 `pair_chain_iptm` 专指 peptide–Affibody 界面；HLA/β2m–Affibody 的 scaffold pair-chain iPTM 只作为 HLA 偏好诊断记录，不参与第一项。

盲评可以使用比搜索阶段更高的 `20 loops × 200 sampling steps` 降低随机噪声，但 loss 的组成和权重必须与设计 workflow 一致。若要比较 search-time 数值本身，则另做 `4 × 50` 诊断；不能把两种计算精度的绝对 loss 值直接混在同一张表中。

记录完整序列、seed、`total_loss`、各 loss 分量、interface 分数、hotspot contacts、peptide/HLA contacts 和 clash。`pair_chain_iptm` 作为辅助诊断保留，但不替代综合 `total_loss`。评分文件写完并记录 checksum 后，才能连接实验标签。

为什么盲法：若一边看 retention 一边调权重，只能证明模型记住答案，不能证明它能预测新 target。

### 步骤 17：计算验证指标

连接实验标签后计算：

1. Spearman correlation。
2. ROC-AUC。
3. PR-AUC。
4. MCC。
5. Top-1、Top-3 binder hit rate。
6. NDCG。
7. 每个 peptide pair 内的排名质量。

置信区间按 peptide pair grouped bootstrap，而不是逐 matrix cell bootstrap，因为同一 peptide row 内的 Affibody 共享相同 pMHC 环境，并非完全独立。

## 9. 第五阶段：验证 ProteoCraft 的生成能力

### 步骤 18：Closed-set 选择验证

对每个 cleavage peptide pair，只允许 ProteoCraft 在该 matrix 已测的 10 或 12 个 Affibody tuple 中选择。

例如 MW：

```text
输入：SLLMWITQV/HLA-A02
候选：LibB matrix 中 MW 行的已测 Affibody
输出：模型认为最好的 Top-1、Top-3
```

因为每个候选都有 retention，可以直接计算：

```text
Top-1 是否为 binder
Top-3 中 binder 数量
Top-3 平均 retention
相对随机选择的 enrichment
```

为什么先做 closed-set：它类似有标准答案的选择题，可以纯粹测试 ProteoCraft 是否会从已知候选中选出真正 binder，不受“新生成序列没有实验标签”的影响。

### 步骤 19：冻结生成参数

根据 preflight 和 closed-set 的开发结果固定：

```text
backend
profile
loss weights
hotspot 设置
搜索步数
候选数
seed 方案
最终 rescoring 设置
```

参数冻结后，不再根据 locked validation 结果反复修改。若确需修改，必须生成新的实验版本号并重新报告全部指标。

为什么这样做：一边查看验证答案一边调参会导致数据泄漏，得到的高分不能代表对新 peptide 的泛化能力。

### 步骤 20：Open-set Affibody 生成

Closed-set 通过后，在完整 LibA 或 LibB 允许空间中生成新 tuple。

建议每个 target：

```text
8 个独立初始 seed
每个 seed 保留 10–20 个候选
去重后保留约 30–50 个候选
```

所有候选先保存为：

```text
validation/frozen_candidates_model_only.csv
```

该文件必须在连接任何 cleavage 或测序标签前冻结并记录 checksum。

为什么先冻结：防止边看实验结果边挑选候选，从而高估 ProteoCraft 的真实预测能力。

### 步骤 21：查询 open-set 候选的已有实验数据

对每个冻结候选依次查询：

1. 是否有 exact cleavage retention。
2. 是否在 R009/R010 中出现。
3. 相对 R000 是否富集。
4. 是否进入 R9/R10 top 2%。
5. 同一 Affibody tuple 对其他 peptide pair 是否也广泛富集。

证据解释：

```text
有 exact cleavage：强实验支持
只有 R9–R10 enrichment：中等支持
只有早期轮次 reads：弱支持
没有 reads：unknown，不自动判为 non-binder
```

## 10. 第六阶段：验证 peptide 特异性

### 步骤 22：建立计算 decoy pMHC panel

每个 Affibody 除了对目标 pMHC 打分，还要对相同 HLA-A02、不同 peptide 的 decoy panel 重新打分。

Decoy 至少包括：

1. 与 target 只差一个 peptide 残基的近邻。
2. P4/P5 化学性质相似的 peptide。
3. P4/P5 完全不同的 peptide。
4. Target panel 中其他肿瘤 peptide。
5. Target panel 中病毒 peptide。
6. NY-ESO、MART-1 等已知对照。

计算：

```text
specificity_margin = target_score - best_off_target_score
```

若模型 score 的方向是越低越好，应先统一转换方向，确保报告中 margin 越大表示越特异。

为什么需要 decoy：所有 target 使用相同 HLA-A02。一个主要结合 HLA 的候选可能对所有 target 都有很高模型分数，却不具备 peptide 特异性。

### 步骤 23：检查 peptide contact 与 HLA contact

每个复合物记录：

```text
number_of_peptide_contacts
number_of_P4_P5_contacts
number_of_HLA_contacts
peptide_contact_fraction
minimum_hotspot_distance
interface_clash_count
```

为什么这样做：高总界面分数可能来自大量 Affibody–HLA 接触。真正的 TCR-mimic 候选必须对 peptide，特别是预定义 hotspot，有明确接触贡献。

### 步骤 24：利用 225-pair 测序空间评估局部特异性

对每个生成 Affibody tuple，查看其在全部 peptide pair 上的 R9–R10 enrichment 分布。

计算：

```text
target_pair_percentile
best_off_target_enrichment
target_vs_off_target_enrichment_margin
number_of_top2pct_peptide_pairs
promiscuity_score
```

理想候选应表现为目标 pair 排名高、off-target pair 数量少。对大量 pair 都富集的 tuple 应标记为 broad binder；即使其目标 retention 很高，也不能优先作为高度特异的 TCR mimic。

## 11. 第七阶段：预先规定 Go/No-Go 门槛

门槛必须在查看 locked validation 结果前预先确定。

### 11.1 结构表示门槛

- NY-ESO peptide 位于合理的 HLA 沟槽位置。
- xx133 位于 peptide-facing 表面。
- xx133 与 MW hotspot 有直接或邻近接触。
- 无严重 interface clash。
- 多个 paired seed 的界面定位基本一致。

### 11.2 评分门槛

- ProteoCraft score 与 retention 的 Spearman 方向正确。
- Peptide-group bootstrap 的 95% CI 下限大于 0，或预先定义的开发/确认拆分表现一致。
- PR-AUC 高于 binder prevalence baseline。
- Top-3 binder hit rate至少为随机选择的 2 倍。
- 已知 xx133 被识别为 binder，但不强制排名第一。

### 11.3 序列完整性门槛

- Framework preservation 为 100%。
- 所有可变位置均在允许字母表内。
- 所有候选都能映射为合法 LibA/LibB tuple。

### 11.4 特异性门槛

- 候选必须形成 peptide contact。
- 候选不能只依靠 HLA contact 获得高分。
- Target score 优于 decoy panel。
- 在测序空间中不能表现为明显的全局 promiscuous binder，除非项目目的明确允许 broad binding。

若关键门槛没有通过，则不进入 69-target 批量生成。应先修正 target 表示、模板使用、hotspot、评分函数或 Affibody 搜索限制。

## 12. 第八阶段：分批处理 target panel

### 步骤 25：Wave 0——实验系统回顾验证

不使用新 target，先完成：

- NY-ESO known-complex preflight。
- 228 个 exact experimental complexes 盲评分。
- Closed-set 选择验证。
- LibA/LibB 分别报告。

这是进入任何新 target 前的必需里程碑。

### 步骤 26：Wave 1——最高可解释性 target

建议顺序：

1. NY-ESO-1 9V `SLLMWITQV`
   - 完整 peptide 直接验证。
   - 有 xx133 binder、cleavage、测序和晶体结构。

2. MAGE-A3 `KVAELVHFL`
   - P4/P5=`EL`。
   - LibB 有局部 cleavage 支持。
   - 属于 pair-level transfer，不是完整 target 直接验证。

3. AFP `FMNKFIYEI`
   - P4/P5=`KF`。
   - LibA 有局部 cleavage 支持。

4. hTERT `ILAKFLHWL`
   - P4/P5=`KF`。
   - LibA 有局部 cleavage 支持。

5. EBV BMLF1 `GLCTLVAML`
   - P4/P5=`TL`。
   - 有 pair-level cleavage 支持和 PDB。
   - 可作为非肿瘤转移对照。

为什么先做这五个：它们覆盖 exact control、LibB transfer、LibA transfer、肿瘤 target 和病毒对照，能在较小规模下暴露主要失败模式。

### 步骤 27：Wave 2——Tier 1/high-confidence 插值靶点

进入条件：Wave 0 和 Wave 1 达到 Go/No-Go 要求。

优先级：

1. Tier 1/native P4/P5 register。
2. High-confidence sequence。
3. Oncogenic target。
4. 有 PDB 或可靠结构来源。
5. P4/P5 pair 在 cleavage 或原测序字母表中有支持。

### 步骤 28：Wave 3——化学外推靶点

包括 P4/P5 含 transfer-dependent residue 的靶点。

要求：

- 与 Wave 2 分开统计成功率。
- 不能用训练/实验字母表内的阈值直接宣称同等可信度。
- 优先进行多 seed 和多 backend 一致性检查。

### 步骤 29：Wave 4——register/anchor 外推

包括：

- Tier 2 的 P5/P6 shifted register。
- Tier 3 atypical anchor。

这些案例可能要求 Affibody 整体改变识别姿态，而不只是改变 interface side chain，因此信息风险最高，最后处理。

## 13. 候选排序方法

在验证充分前，不建议把所有指标直接加权成一个总分。先应用 hard gates，再分层排序。

### 13.1 Hard gates

直接淘汰：

- Framework 被修改。
- 含非法字母表残基。
- 严重 clash。
- Peptide hotspot 无接触。
- 多 seed 下结构完全不稳定。
- 对大多数 decoy pMHC 同样高分。

### 13.2 第一层：结构可信度

考虑 complex/interface confidence、hotspot contact、peptide contact fraction、interface clash 和多 seed 一致性。

### 13.3 第二层：实验支持

证据强度顺序：

```text
exact cleavage
    > pair-level cleavage transfer
    > R9/R10 enrichment
    > 无实验覆盖
```

### 13.4 第三层：特异性

考虑 decoy specificity margin、225-pair 测序特异性 percentile 和 promiscuity score。

最终候选应报告 Pareto 情况，例如“结构更强但实验支持较弱”或“实验支持更强但 off-target 风险较高”，不使用不透明总分掩盖差异。

## 14. 最终输出文件

建议目录：

```text
data_prepared/
├── cleavage_libA.csv
├── cleavage_libB.csv
├── sequencing_all_rounds.parquet
├── target_manifest.csv
└── data_snapshot_manifest.tsv

configs/affibody/
├── libA_profile.yaml
└── libB_profile.yaml

validation/
├── known_pair_scores_blind.csv
├── closed_set_results.csv
├── validation_metrics.json
├── frozen_candidates_model_only.csv
└── validation_report.md

target_results/
├── affibody_candidate_list.csv
├── candidate_structures/
└── target_panel_report.md
```

最终 `affibody_candidate_list.csv` 至少包含：

```text
target_id
full_peptide
register
up_facing_pair
support_level
library
full_affibody_sequence
affibody_tuple
model_rank
structure_scores
hotspot_contacts
peptide_contact_fraction
HLA_contacts
clash_metrics
cleavage_retention
R9_R10_enrichment
target_pair_percentile
decoy_specificity_margin
promiscuity_score
confidence
selection_reason
```

## 15. 推荐里程碑

### Milestone 1：数据可用

完成 cleavage、测序和 target manifest 的结构化与人工复核。

### Milestone 2：Affibody 模式可用

完成 scaffold、位置 mask、字母表 mask 和输出完整性检查。

### Milestone 3：已知结构可重现

NY-ESO–xx133 preflight 通过，确认模型没有把 Affibody 放在错误 HLA 表面。

### Milestone 4：实验排序有效

完成 228 个 exact experimental complexes 的盲评分和统计报告。

### Milestone 5：生成能力有效

Closed-set Top-K 明显优于随机，open-set 候选完成冻结和实验数据查询。

### Milestone 6：Wave 1 完成

完成 NY-ESO、MAGE-A3、AFP、hTERT 和 EBV BMLF1，检查 exact、LibA transfer、LibB transfer 与病毒对照的一致性。

### Milestone 7：Target panel 完成

按 Wave 2–4 分批完成，并分别报告插值和外推结果。

## 16. 主要风险与应对

### 风险 1：ESMFold2 无法稳定构建 pMHC

表现：peptide 离开 HLA 沟槽，或 Affibody 接触错误表面。

应对：停止批量生成，复核 HLA/β2m/peptide 链序列、peptide hotspot 和 paired-seed verifier 稳定性；若 sequence-only ESMFold2 仍不能通过已知结构 preflight，则将该 workflow 判为当前数据上的 no-go，而不是继续扩大生成规模。

### 风险 2：模型偏好 HLA-only binder

表现：对多个不同 peptide/HLA-A02 都评分很高。

应对：强化 peptide hotspot、增加 peptide-contact gate，并使用 contrastive decoy rescoring。

### 风险 3：生成序列超出实验库

表现：Affibody tuple 含实验库未覆盖残基或 framework 被修改。

应对：在 logits、mutation proposal 和输出三层应用 alphabet/mutable mask。

### 风险 4：把局部 pair 数据误当完整 peptide 验证

表现：用 `SLLELITQV + EFYSV` 的结果直接宣称 EFYSV 已被证明结合 MAGE-A3。

应对：强制使用 Level A–D 证据标签；只有完整 experimental construct 相同才能标记 direct validation。

### 风险 5：测序缺失被误判为阴性

表现：没有 reads 的候选被直接删除。

应对：标记为 unknown；只有明确的 cleavage non-binder 或严格定义的高置信负集才作为阴性。

### 风险 6：使用实验答案反复调参

表现：验证指标很好，但对新 peptide 没有泛化能力。

应对：盲评分、候选冻结、参数版本化，并分别报告 model-only 和 experiment-assisted 结果。

## 17. NY-ESO 完整示例

```text
1. 构建 HLA-A02 | β2m | SLLMWITQV
2. 使用固定 LibB Affibody scaffold
3. 只允许位置 8、12、15、16、19 改变
4. 只允许 LibB DHS 13 种氨基酸
5. 先给实验中已测的 10 个 Affibody 打分
6. 检查模型能否选出 retention >= 75% 的候选
7. 检查 NNYYF 是否形成正确 peptide-facing interface
8. Closed-set 通过后，开放生成新 tuple
9. 冻结生成候选
10. 查询候选的 cleavage 或 R9–R10 enrichment
11. 对多个 decoy pMHC 重新评分
12. 淘汰 HLA-only binder 和 broad binder
13. 输出最终候选及证据等级
```

## 18. 最终判断原则

本项目最重要的第一个结果不是 69 个 target 的大量候选，而是证明：

> ProteoCraft 所认为的“好 Affibody”，与已有 cleavage-retention 实验中定义的“好 Affibody”具有一致关系。

只有 228 个已知实验配对的盲评分、closed-set 选择和 NY-ESO 结构 preflight 均达到预设门槛后，才应把相同流程应用到 MAGE-A3、AFP、hTERT 及其余 target panel。

对于最终候选，必须明确回答三个问题：

1. 结构模型认为它是否可能结合目标 pMHC？
2. 现有实验数据提供的是直接支持、局部转移支持、测序支持，还是没有支持？
3. 它是否对目标 peptide 具有特异性，而不是 HLA-only 或广谱 binder？

只有同时回答这三个问题，Affibody candidate list 才具有可解释性，并能为后续湿实验提供合理优先级。
