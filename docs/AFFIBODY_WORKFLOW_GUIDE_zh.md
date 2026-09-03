# ProteoCraft Affibody 工作流使用教程

这份文档说明如何使用本仓库新增的 Affibody 工作流，把 `datafolder` 中的实验数据整理成验证集，限制 ProteoCraft 只设计合法的 LibA/LibB Affibody，并在设计后查询已有 cleavage 和测序证据。

完整的科学设计理由见 [AFFIBODY_VALIDATION_PLAN.md](AFFIBODY_VALIDATION_PLAN.md)。本文件更偏向实际操作。

## 1. 先理解这套代码解决了什么问题

已有实验不是对任意 60 aa 蛋白进行测试，而是对固定 Affibody 骨架上的少数位点进行组合测试：

| 文库 | Affibody 长度 | 可变位置（1-based） | 允许氨基酸 | 默认对照 tuple |
|---|---:|---|---|---|
| LibA | 60 aa | 15、19、29、33 | `ADEFHIKLNPQSTVY` | `DSHK` |
| LibB | 60 aa | 8、12、15、16、19 | `ADEFIKLMNSTVY` | `NNYYF` |

因此代码做了两个硬限制：

1. 其余 framework 位点完全固定，ProteoCraft 不能修改。
2. 可变位点只能使用相应实验文库允许的氨基酸。

这样生成的序列才能重新写成 LibA 的 4-aa tuple 或 LibB 的 5-aa tuple，并与现有测序表、cleavage matrix 对接。

复合物的非 binder 输入统一表示为三条链：

```text
HLA-A*02:01 | β2-microglobulin | 9-mer peptide
```

这里 `--target` 是通用入口对“所有非 binder 链”的参数名，不表示其中每条链在 loss 中都具有 `role=target`。Affibody workflow 固定使用：

| 输入链 | chain role | 含义 |
|---|---|---|
| HLA-A*02:01 | `scaffold` | 保留在完整复合物中，提供固定 pMHC 结构背景 |
| β2-microglobulin | `scaffold` | 保留在完整复合物中，稳定 HLA 结构背景 |
| 9-mer peptide | `target` | peptide–Affibody 界面优化和主排序的对象 |
| Affibody | binder | 只允许实验文库规定的位置发生变化 |

因此实际参数是 `--esmfold2_target_chain_roles scaffold,scaffold,target`。ESMFold2 仍预测完整的 HLA–β2m–peptide–Affibody 四链结构；`scaffold` 只改变 loss mask，不会从模型输入中删除 HLA/β2m，也不表示固定它们的三维坐标。当前 workflow 是 sequence-only。

peptide 的 P4/P5（Tier 2 时为 P5/P6）会被映射为 ProteoCraft hotspot。这样可以把 peptide-facing 表面作为明确目标，降低模型只偏好共同 HLA 表面的风险。

### 1.1 这里只使用 ESMFold2 workflow

Affibody 设计不会调用其他结构后端，也不会建立一套新的搜索算法。它固定使用仓库正式的 `esmfold2_d` workflow：

```text
完整 Affibody 初始序列
        ↓
mutable mask + 实验文库 alphabet
        ↓
ESMFold2 Phase 0：official_exact roots
        ↓
Stage A：soft-logit / ESMC / truncated one-step proposal
        ↓
ESMFold2 full verifier
        ↓
Stage B：MCTS 严格单点突变与离散 rerank
        ↓
ESMFold2 full verifier + 最终候选
        ↓
framework/alphabet 合法性复查
```

Affibody 只改变原 workflow 的任务输入：binder 是完整 60-aa Affibody，mask 是 LibA/LibB 的实验位点，alphabet 是相应文库的合法残基集合；非 binder 输入仍包含 HLA、β2m、peptide 三条链，但角色分别为 `scaffold,scaffold,target`。Stage A、Stage B、结构损失、MCTS、verifier 和 checkpoint 仍是共享实现。

任务类型写成 `affibody_redesign`。它使用局部 scaffold-redesign 的 ESMC prior 权重 `0.05`，而不是 de novo binder 的 `0.15`；除此之外不分叉搜索逻辑。

当前 workflow 是 sequence-only。参考 PDB 只用于提取 HLA 和 β2m 序列，不把模板坐标送入 ESMFold2。每个 run 的 `affibody_case.json` 会明确记录这一点。

### 1.2 文库氨基酸限制如何生效

限制不是生成结束后的筛选，而是搜索空间中的硬约束：

1. `run_affibody_design.py` 把文库 alphabet 传给 `--esmfold2_allowed_binder_aas`。
2. 启动器先检查初始序列的可变位点是否都合法。
3. ESMFold2 把可变位点上所有不允许残基的 logits 设置为 `-inf`，所以 softmax 概率为 0。
4. Phase 0、梯度更新、Stage A proposal、Stage B MCTS proposal、argmax 和 checkpoint 恢复都会重新应用同一 mask。
5. 运行结束和候选收集时，再检查 framework 与 alphabet；不合法结果会直接报错。

固定 framework 位点不套用实验 alphabet，而是由 mutable mask 锚定到初始完整序列。例如 LibB 的固定位置可以保留 `G`、`Q` 或 `R`，即使这些残基不在 LibB 可变位点的 alphabet 中。

## 2. 本次新增了哪些代码

核心模块：

```text
bindcraft/affibody/profiles.py     LibA/LibB 骨架、可变位置和字母表
bindcraft/affibody/targets.py      pMHC 三链构建、PDB 链读取、hotspot 映射
bindcraft/affibody/office.py       不依赖 pandas 的 XLSX/PPTX 表格读取
bindcraft/affibody/data.py         实验数据整理、测序查询和支持等级
bindcraft/affibody/candidates.py   候选合法性检查、冻结和证据连接
```

命令行工具：

```text
scripts/prepare_affibody_validation.py
scripts/score_affibody_complexes.py
scripts/analyze_affibody_blind_scores.py
scripts/run_affibody_design.py
scripts/collect_affibody_candidates.py
scripts/annotate_affibody_candidates.py
scripts/build_affibody_decoy_panel.py
scripts/analyze_affibody_specificity.py
```

ESMFold2 现有入口 `scripts/run_hallucination.py` 也增加了：

```text
--esmfold2_allowed_binder_aas
```

这个参数从 logits 初始化、梯度更新、候选生成直到 checkpoint 恢复都持续生效，不只是生成结束后再做字符串过滤。

Affibody 启动器还会显式锁定：

```text
--model_backend esmfold2
--profile esmfold2_d
--mode mcts
--mcts_strategy two_stage
--esmfold2_task_type affibody_redesign
--esmfold2_logit_init_mode official_exact
```

这样即使通用入口仍兼容仓库中的其他任务，Affibody 生成命令也不会因默认值变化而切换到别的后端或搜索模式。

## 3. 哪些步骤不会运行模型

下列操作只读取表格或生成清单，可以在 CPU/login node 上运行：

- `prepare_affibody_validation.py`
- 不带 `--execute` 的 `score_affibody_complexes.py`
- 不带 `--execute` 的 `run_affibody_design.py`
- `collect_affibody_candidates.py`
- `annotate_affibody_candidates.py`
- `build_affibody_decoy_panel.py`
- `analyze_affibody_blind_scores.py`
- `analyze_affibody_specificity.py`

只有以下两种命令会真正加载 ESMFold2/GPU：

```text
score_affibody_complexes.py --execute
run_affibody_design.py --execute
```

本次代码实现和检查过程中没有运行这两个 `--execute` 路径。

## 4. 进入项目并检查环境

先进入项目目录：

```bash
cd /hai/users/r/y/rycwang/ProteoCraft
```

确认主要输入存在：

```bash
ls -lh \
  datafolder/CoEvo_target_panel_v3.xlsx \
  'datafolder/Affibody-MHC data summary.pptx' \
  datafolder/nyeso_xx133_complex.pdb
```

数据整理只要求 Python 3.10 或更新版本：

```bash
python --version
```

真正进行 ESMFold2 盲评分或设计前，还必须在相应计算环境中确认：

```bash
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

如果出现 `No module named torch`，说明当前 Python 只适合数据准备，不能进行 GPU 评分或设计。应先切换到已安装 ProteoCraft/ESMFold2 依赖的环境；不要直接加 `--execute`。

## 5. 第一阶段：整理并冻结已有实验数据

运行：

```bash
python scripts/prepare_affibody_validation.py \
  --data-dir datafolder \
  --output-dir data_prepared/affibody
```

当前数据应得到：

```text
69 targets
228 cleavage labels
```

228 条标签来自：

```text
LibA: 9 peptide pairs × 12 Affibody tuples = 108
LibB: 12 peptide pairs × 10 Affibody tuples = 120
```

生成文件：

```text
data_prepared/affibody/
├── targets.csv
├── cleavage_labels.csv
├── known_complexes_blind.csv
├── affibody_profiles.json
└── preparation_manifest.json
```

各文件作用：

- `targets.csv`：69 个设计目标和证据等级。
- `cleavage_labels.csv`：含 retention、binder 标签和实验来源。
- `known_complexes_blind.csv`：同样的 228 个复合物，但故意不含 retention/binder 标签，专用于盲评分。
- `affibody_profiles.json`：本次使用的 scaffold、可变位点和允许字母表。
- `preparation_manifest.json`：输入和输出的 SHA-256 checksum，保证以后知道结果来自哪一版文件。

为什么把 blind complexes 和 labels 分开：模型评分完成前不能看到实验答案。否则很容易一边看 retention 一边调参数，得到数据泄漏后的虚高结果。

### 5.1 检查 target 支持等级

可用下面的命令查看前几行：

```bash
python - <<'PY'
import csv
with open('data_prepared/affibody/targets.csv', newline='') as f:
    for row in list(csv.DictReader(f))[:10]:
        print(
            row['target_id'], row['peptide'], row['register'],
            row['up_facing_pair'], row['support_level'],
            row['applicable_libraries']
        )
PY
```

`support_level` 的含义：

| 等级 | 含义 | 能否称为直接实验验证 |
|---|---|---|
| `exact_full_peptide` | 完整 9-mer 和 cleavage 实验肽完全相同 | 可以 |
| `cleavage_pair_transfer` | P4/P5 二肽已有标签，但完整 peptide 不同 | 不可以，只是局部迁移证据 |
| `sequencing_pair_supported` | 二肽属于 15×15 测序库，但没有 cleavage 标签 | 不可以 |
| `register_extrapolation` | 使用 P5/P6 等未训练 register | 不可以，属于高风险外推 |
| `computational_only` | 二肽含未校准残基 | 不可以 |

最重要的例子：

- `T001 / SLLMWITQV` 是完整 peptide 直接验证。
- `T002 / KVAELVHFL` 的 `EL` 虽有 LibB cleavage 标签，但实验背景是 `SLLELITQV`，因此只能叫 pair transfer。

## 6. 第二阶段：先做 228 个已知复合物的盲评

这是正式开放生成前的推荐步骤。它回答的问题是：在完全已知实验答案的集合中，模型分数是否真的能把高 retention 候选排在前面？

### 6.1 先只生成盲评计划

```bash
python scripts/score_affibody_complexes.py \
  --complexes data_prepared/affibody/known_complexes_blind.csv \
  --reference-pdb datafolder/nyeso_xx133_complex.pdb \
  --output validation/known_pair_scores_blind.jsonl \
  --verifier-loss-mode aligned_common \
  --seeds 101,102,103
```

盲评分脚本现在默认使用 `aligned_common`，命令中仍显式写出它，以便计划文件和批处理脚本一眼可审计；正式 `esmfold2_d` Affibody 设计也锁定同一个 full-verifier loss。

盲评与设计也使用同一套链角色：HLA 和 β2m 是 scaffold，只有 peptide 是 target。这样 `pair_chain_iptm` 表示 peptide–Affibody pair-chain iPTM，而不是 HLA、β2m、peptide 三个界面分数的平均值。`scaffold_binder_pair_chain_iptm` 可作为 Affibody 是否偏向 HLA/β2m 的诊断量，但不进入下面主 loss 的第一项。

盲评和设计共用的 loss 定义是：

```text
total_loss =
    1.0 × (1 - pair_chain_iptm)
  + 0.2 × (1 - binder_pLDDT)
  + 0.1 × (1 - pTM)
  + 0.1 × binder_Rg_excess
```

`total_loss` 越低越好。`pair_chain_iptm` 仍会写入结果，但只作为辅助界面指标，不作为本轮盲评的主排序分数。

因为没有 `--execute`，这个命令只会：

1. 检查所有 Affibody 是否保持 framework。
2. 检查 tuple 是否与完整序列一致。
3. 拒绝包含 `retention_percent` 或 `cleavage_binder` 的输入。
4. 写出 `known_pair_scores_blind.jsonl.plan.json`。

它不会加载 ESMFold2。

### 6.2 做一个最小 preflight 计划

第一次可以先只选 1–2 个复合物：

```bash
python scripts/score_affibody_complexes.py \
  --complexes data_prepared/affibody/known_complexes_blind.csv \
  --reference-pdb datafolder/nyeso_xx133_complex.pdb \
  --output validation/preflight_blind.jsonl \
  --verifier-loss-mode aligned_common \
  --limit 2 \
  --seeds 101
```

确认计划、环境和输出路径都正确后，才在有 GPU 的 ESMFold2 环境中显式执行：

```bash
python scripts/score_affibody_complexes.py \
  --complexes data_prepared/affibody/known_complexes_blind.csv \
  --reference-pdb datafolder/nyeso_xx133_complex.pdb \
  --output validation/preflight_blind.jsonl \
  --verifier-loss-mode aligned_common \
  --limit 2 \
  --seeds 101 \
  --device cuda \
  --execute
```

注意：上面最后一个命令会真正运行计算，不要在 login node 上直接执行。

### 6.3 正式盲评

在 preflight 通过、参数冻结后，去掉 `--limit`：

```bash
python scripts/score_affibody_complexes.py \
  --complexes data_prepared/affibody/known_complexes_blind.csv \
  --reference-pdb datafolder/nyeso_xx133_complex.pdb \
  --output validation/known_pair_scores_blind.jsonl \
  --verifier-loss-mode aligned_common \
  --seeds 101,102,103 \
  --device cuda \
  --execute
```

中断后可用同一配置继续：

```bash
python scripts/score_affibody_complexes.py \
  --complexes data_prepared/affibody/known_complexes_blind.csv \
  --reference-pdb datafolder/nyeso_xx133_complex.pdb \
  --output validation/known_pair_scores_blind.jsonl \
  --verifier-loss-mode aligned_common \
  --seeds 101,102,103 \
  --device cuda \
  --resume \
  --execute
```

### 6.4 评分冻结后再连接实验标签

```bash
python scripts/analyze_affibody_blind_scores.py \
  --scores validation/known_pair_scores_blind.jsonl \
  --labels data_prepared/affibody/cleavage_labels.csv \
  --output-dir validation/blind_analysis \
  --score-key total_loss \
  --lower-is-better
```

输出：

```text
validation/blind_analysis/
├── blind_analysis_report.html
├── blind_validation_metrics.json
├── group_rankings.csv
├── ranking_agreement.csv
├── score_retention_scatter.csv
├── library_correlations.csv
└── score_bins.csv
```

分析只包含以下三种直观方案：

1. `group_rankings.csv`：在每个 `library × peptide_pair` 组内并排显示模型名次、实验 retention 名次、模型 Top-1 和实验最佳候选。`ranking_agreement.csv` 逐组报告 Overlap@3、随机校正 Overlap@3 和 Kendall tau-b，并在 HTML 中分别汇总 Overall、LibA、LibB。三个指标的含义与取值范围放在默认折叠的说明区域。
4. `score_retention_scatter.csv`：生成原始 `total_loss` 与 retention 的 LibA/LibB 分面散点图，低值放在左边并标注“越左/越低越好”。点颜色表示 peptide pair，binder 为实心，non-binder 为空心，悬停显示 Affibody tuple。描述性趋势线只帮助观察方向，不解释为 retention 预测模型。原来的组内 percentile 散点图章节 4B 已删除；组内 percentile 只用于方案 6 分箱。`library_correlations.csv` 另外分别报告 LibA、LibB 的 Pearson `r(retention, -total_loss)`；正值表示较低原始 loss 倾向伴随较高 retention。
6. `score_bins.csv`：先在每个 `library × peptide_pair` 组内计算名次中点 percentile，再分为 Top 20%、20–40%、40–60%、60–80% 和 Bottom 20%。每档报告样本数、binder rate、平均/中位 retention、Q1、Q3 和 IQR，并分别给出 Overall、LibA、LibB 结果。

`blind_analysis_report.html` 汇总方案 1、4、6；`blind_validation_metrics.json` 保存汇总数字、输入路径与 SHA256。方案 1 中 cleavage binder 使用绿色，non-binder 使用灰色 retention 条和红色标签。

使用 `--lower-is-better` 时，脚本只在内部排序时把 `total_loss` 取负。所有人类可读表格和散点图继续显示未经变换的原始 `total_loss`，避免负号方向造成混淆。

## 7. 第三阶段：只生成命令，不立即设计

### 7.1 先用最强阳性对照 T001/LibB

```bash
python scripts/run_affibody_design.py \
  --target-manifest data_prepared/affibody/targets.csv \
  --target-id T001 \
  --library libb \
  --reference-pdb datafolder/nyeso_xx133_complex.pdb \
  --output-root outputs/affibody_design_wave0
```

默认不带 `--execute`，因此只生成：

```text
outputs/affibody_design_wave0/
├── commands.sh
├── design_runs.jsonl
├── design_manifest.json
└── T001_LibB_seed43/
    └── affibody_case.json
```

先检查命令：

```bash
sed -n '1,5p' outputs/affibody_design_wave0/commands.sh
```

命令中应该能看到：

```text
--profile esmfold2_d
--model_backend esmfold2
--mode mcts
--mcts_strategy two_stage
--esmfold2_task_type affibody_redesign
--esmfold2_logit_init_mode official_exact
--binder_length 60
--esmfold2_mutable_mask 000...010...
--esmfold2_allowed_binder_aas ADEFIKLMNSTVY
--esmfold2_target_chain_roles scaffold,scaffold,target
--esmfold2_verifier_loss_mode aligned_common
--target_hotspots 379,380
```

`379,380` 是当前参考 PDB 中 HLA 276 aa、β2m 99 aa 后，peptide P4/P5 在完整非 binder 拼接序列中的 1-based 位置。hotspot 编号始终包含前面的 scaffold 链，所以把 HLA/β2m 改为 scaffold 后这里仍是 `379,380`；代码会根据实际提取的链长度重新计算，而不是把数字写死，并验证两个位置确实落在 `role=target` 的 peptide 链上。

这套角色对 loss 的影响是：

- peptide–Affibody 的 iPAE、接触项和 `pair_chain_iptm` 是目标界面项。
- HLA/β2m 不进入 `pair_chain_iptm` 的 target 平均，但仍参与完整四链折叠、全局 pTM，以及 proposal 阶段的 scaffold/几何约束。
- Affibody 固定 framework 也属于 binder scaffold；只有 mutable mask 选中的文库位点能被修改。
- 盲评和开放设计均使用相同的 `aligned_common` verifier loss 与相同链角色，避免评价口径改变。

`official_exact` 与原正式 ESMFold2 redesign workflow 一致：固定位置来自 `--init_binder_seq`，可变位置从合法 alphabet 的 soft logits 建立多个 root。默认 tuple 同时作为已知对照和 framework 完整序列的构造来源，但不会把所有搜索 root 锁在这个 tuple 附近。

### 7.2 多个独立 seed

计划通过后，推荐每个 target 生成 8 个独立 run：

```bash
python scripts/run_affibody_design.py \
  --target-manifest data_prepared/affibody/targets.csv \
  --target-id T001 \
  --library libb \
  --replicates 8 \
  --seed 43 \
  --output-root outputs/affibody_design_wave0
```

这仍然只生成 8 条命令，不运行模型。

### 7.3 真正执行设计

只有在盲评/closed-set 达到预先设定的 Go 门槛、GPU 环境就绪后才执行：

```bash
python scripts/run_affibody_design.py \
  --target-manifest data_prepared/affibody/targets.csv \
  --target-id T001 \
  --library libb \
  --replicates 8 \
  --seed 43 \
  --output-root outputs/affibody_design_wave0 \
  --device cuda \
  --execute
```

也可以把 `commands.sh` 交给你自己的 Slurm/队列系统，而不使用脚本的串行 `--execute`。

使用启动器串行执行时，每个完成的 run 还会生成：

```text
T001_LibB_seed43/
├── hallucination_result.txt
├── affibody_result.json
├── run.log
└── esmfold2_artifacts/
```

`affibody_result.json` 只有在 backend、profile、任务类型、MCTS 模式、mutable mask、alphabet、framework 和最终序列全部通过复查后才会写出。

### 7.4 指定其他 target

一个或多个 target：

```bash
python scripts/run_affibody_design.py \
  --target-id T002,T003 \
  --target-id T004 \
  --library libb \
  --replicates 8 \
  --output-root outputs/affibody_design_wave1
```

全部 69 个 target：

```bash
python scripts/run_affibody_design.py \
  --target-id all \
  --library libb \
  --replicates 8 \
  --output-root outputs/affibody_design_all
```

不建议一开始运行 `all`。应先完成 T001 和 Wave 0/1。

### 7.5 使用 LibA

```bash
python scripts/run_affibody_design.py \
  --target-id T001 \
  --library liba \
  --replicates 8 \
  --output-root outputs/affibody_design_liba
```

脚本会自动切换到 LibA 的 4 个可变位点和 15-aa 字母表。`design_runs.jsonl` 中的 `library_has_pair_cleavage` 会说明这个 pair 是否存在相应 LibA cleavage 标签。

## 8. 第四阶段：收集并冻结设计候选

设计完成后，从每个 run 的 `candidates.jsonl` 中按 verifier loss 收集去重候选：

```bash
python scripts/collect_affibody_candidates.py \
  --runs outputs/affibody_design_wave0/design_runs.jsonl \
  --top-per-run 10 \
  --output validation/candidates_model_only.csv
```

该脚本会：

1. 每个 run 先按完整 verifier `total_loss` 排序。
2. 同一 target/library/sequence 去重。
3. 验证所有 framework 固定位点没有变化。
4. 验证所有可变残基位于实验字母表中。
5. 写出完整 60-aa 序列和对应 4/5-aa tuple。

此时的 `candidates_model_only.csv` 只来自模型结果。不要手工查看实验标签后删除“不好看”的候选。

如果只想练习候选文件格式，可查看已知阳性对照示例：

```text
configs/affibody/candidates_example.csv
```

它不是模型新生成结果，不能混入正式盲评候选。

## 9. 第五阶段：冻结后查询已有实验证据

运行：

```bash
python scripts/annotate_affibody_candidates.py \
  --candidates validation/candidates_model_only.csv \
  --prepared-dir data_prepared/affibody \
  --data-dir datafolder \
  --output validation/candidates_annotated.csv
```

输出：

```text
validation/
├── candidates_frozen.csv
├── candidates_annotated.csv
└── candidates_annotated.csv.manifest.json
```

manifest 会记录原始候选、冻结候选、标签表和输出的 checksum。

默认查询四个测序轮次：

```text
R000  背景/初始库
R001  负选择
R009  推荐正选择轮次之一
R010  推荐正选择轮次之一
```

并计算：

```text
R009_R010_count
R009_R010_frequency
log2_enrichment_vs_R000
log2_enrichment_vs_R001
```

若还要扫描某个 Affibody tuple 在所有 peptide pair 上的表现：

```bash
python scripts/annotate_affibody_candidates.py \
  --candidates validation/candidates_model_only.csv \
  --prepared-dir data_prepared/affibody \
  --data-dir datafolder \
  --output validation/candidates_annotated.csv \
  --with-sequencing-landscape
```

额外字段包括：

- `landscape_target_percentile`：目标 pair 在该 tuple 全部已观察 pair 中的分位数。
- `landscape_best_off_target_frequency`：最强 off-target pair 的 pooled frequency。
- `landscape_target_minus_best_off_target`：目标减最强 off-target。
- `landscape_pairs_above_target`：富集高于目标 pair 的数量。

### 9.1 如何解释 `validation_evidence_level`

| 输出值 | 正确解释 |
|---|---|
| `exact_existing_assay` | 完整实验 peptide、tuple 和 library 都精确匹配 |
| `pair_transfer` | 只有 P4/P5 二肽和 tuple 匹配，完整 peptide 不同 |
| `sequencing_only` | 在测序数据中出现，但没有 exact cleavage label |
| `computational_only` | 当前数据中没有对应实验记录；不是自动 non-binder |

代码自检使用的已知对照是：

```text
LibB peptide pair MW + tuple NNYYF
cleavage retention = 83.77%
R009 + R010 count = 3361
evidence level = exact_existing_assay（对 SLLMWITQV）
```

## 10. 第六阶段：建立同 HLA 的 peptide decoy panel

所有 target 都含同一个 HLA-A02，因此必须检查候选是否只是 HLA binder。

先用冻结候选生成 panel：

```bash
python scripts/build_affibody_decoy_panel.py \
  --candidates validation/candidates_frozen.csv \
  --targets data_prepared/affibody/targets.csv \
  --max-decoys 12 \
  --output validation/decoy_complexes_blind.csv
```

每个候选包含：

- 1 个原始 on-target peptide。
- 最多 12 个相同 HLA/β2m、不同 peptide 的 decoy。
- 固定 seed 下优先加入 NY-ESO/MART-1 对照，再从 panel 抽取其余 decoy。

也可以显式固定 decoy ID：

```bash
python scripts/build_affibody_decoy_panel.py \
  --candidates validation/candidates_frozen.csv \
  --decoy-target-id T001,T002,T003,T004 \
  --output validation/decoy_complexes_blind.csv
```

用同一个固定序列评分器先生成计划：

```bash
python scripts/score_affibody_complexes.py \
  --complexes validation/decoy_complexes_blind.csv \
  --reference-pdb datafolder/nyeso_xx133_complex.pdb \
  --output validation/decoy_scores.jsonl \
  --verifier-loss-mode aligned_common \
  --seeds 201,202,203
```

在 GPU 环境确认后才添加 `--execute`。

评分完成后分析：

```bash
python scripts/analyze_affibody_specificity.py \
  --panel validation/decoy_complexes_blind.csv \
  --scores validation/decoy_scores.jsonl \
  --output-dir validation/specificity \
  --score-key total_loss \
  --lower-is-better
```

关键输出：

```text
specificity_margin = target_score - best_off_target_score
target_rank_among_panel
```

`specificity_margin > 0` 表示 target 分数高于所有已测试 decoy，但它只针对当前 decoy panel，不能解释为对所有可能 peptide 都绝对特异。

## 11. 推荐实际执行顺序

不要直接从 69 个 target 全量生成开始。建议顺序是：

### Wave 0：系统验证

1. 整理并冻结 228 条 cleavage 数据。
2. NY-ESO/xx133 小规模 preflight。
3. 228 个固定复合物盲评分。
4. 检查 Spearman、PR-AUC、Top-1/Top-3 和 grouped bootstrap。
5. 参数冻结。

### Wave 1：最高可解释性目标

1. `T001 SLLMWITQV`：完整直接验证。
2. MAGE-A3 `KVAELVHFL`：LibB `EL` pair transfer。
3. AFP `FMNKFIYEI`：LibA `KF` pair transfer。
4. hTERT `ILAKFLHWL`：LibA `KF` pair transfer。
5. EBV BMLF1 `GLCTLVAML`：病毒转移对照。

### Wave 2：Tier 1/high-confidence

只有 Wave 0/1 达到预先规定门槛后再进入。

### Wave 3/4：外推

- 含训练肽库未覆盖残基的化学外推。
- P5/P6 shifted register。
- atypical anchor。

这些必须与 Tier 1 分开报告，不能共享同一可信度表述。

## 12. 常见问题

### 12.1 为什么生成的不是任意 60-aa Affibody？

因为任意序列无法与当前 LibA/LibB 实验数据对应。固定 framework 和小范围界面 redesign 是实现可验证性的前提。

### 12.2 为什么 `NNYYF` 不是完整 Affibody？

它只是 LibB 的 5 个可变位点。完整序列是：

```text
VDNKNKFNKEFNNAYYEIFHLPNLNEEQFDAFVQSLFDDPSQSANLLAEAKKLNDAQAPK
```

程序在设计和数据查询之间自动完成完整序列与 tuple 的双向转换。

### 12.3 为什么 MAGE-A3 的 `EL` 标签不是 exact validation？

cleavage 实验使用固定背景 `SLLELITQV`，MAGE-A3 是 `KVAELVHFL`。两者只有上举二肽 `EL` 相同，HLA anchor 和其余 peptide 环境不同。

### 12.4 候选在测序表里没有 reads，是否就是 non-binder？

不是。可能是库中没有充分采样、read depth 不够，或该 tuple 未进入已观察集合。代码把这种情况标为 unknown/`computational_only`，不会自动写成 non-binder。

### 12.5 为什么当前统计是 228，不是旧计划中的 227？

当前 PowerPoint XML 中 LibB 表有 120 个数值单元格，LibA 有 108 个，总计 228。准备脚本按实际非空数值读取，并把每个来源 slide 和 checksum 写入 manifest。旧文档中的 119/227 已同步修正为当前源文件的 120/228。

### 12.6 `framework position changed` 是什么错误？

说明候选修改了实验未随机化的骨架位点。该候选不能合法映射回当前文库，因此脚本会拒绝，而不是悄悄继续。

### 12.7 `outside allowed_aas` 是什么错误？

说明某个可变位点用了该实验文库没有允许的氨基酸。检查 library 是否选错，或是否绕过了新增的 ESMFold2 alphabet 参数。

### 12.8 为什么 score/design 脚本默认不执行？

这是安全设置。盲评分可能是数百到上千次 full verifier 调用，开放生成也会占用大量 GPU。必须明确添加 `--execute` 才会开始。

## 13. 代码检查

只做语法和 CPU 数据工作流检查：

```bash
python -m compileall -q \
  bindcraft/affibody \
  scripts/prepare_affibody_validation.py \
  scripts/run_affibody_design.py \
  scripts/score_affibody_complexes.py
```

如果测试环境安装了 `pytest` 和 `torch`：

```bash
python -m pytest -q \
  bindcraft/tests/test_affibody_profiles.py \
  bindcraft/tests/test_affibody_targets.py \
  bindcraft/tests/test_affibody_data.py \
  bindcraft/tests/test_affibody_workflow.py \
  bindcraft/tests/test_esmfold2_backend.py \
  bindcraft/tests/test_esmfold2_pipeline.py \
  bindcraft/tests/test_run_hallucination_backend.py
```

当前基础 Python 若没有 `pytest` 或 `torch`，不能据此判断代码失败；应切换到项目的模型/测试环境后再执行相应测试。

## 14. 最小安全操作清单

如果暂时只想确认代码能用，又不想运行任何模型，依次执行：

```bash
cd /hai/users/r/y/rycwang/ProteoCraft

python scripts/prepare_affibody_validation.py \
  --data-dir datafolder \
  --output-dir data_prepared/affibody

python scripts/score_affibody_complexes.py \
  --complexes data_prepared/affibody/known_complexes_blind.csv \
  --output validation/known_pair_scores_blind.jsonl \
  --verifier-loss-mode aligned_common \
  --limit 2 \
  --seeds 101

python scripts/run_affibody_design.py \
  --target-id T001 \
  --library libb \
  --replicates 2 \
  --output-root outputs/affibody_design_dry_run
```

只要不加 `--execute`，这三步都不会加载 ESMFold2，也不会运行设计或实验。
