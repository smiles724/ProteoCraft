# ProteoCraft ESMFold2 运行指南

本文档说明当前推荐的 ESMFold2 默认配置，以及 De novo binder design、TCR binder design、TCR CDR redesign 和 VHH all-CDR redesign 四种任务的启动方式。以下命令均需在仓库根目录执行。

运行前先使用 [`environment/esmfold2/`](../environment/esmfold2/README.md)
创建独立环境并下载固定 revision 的 `biohub/ESMFold2`、
`biohub/ESMFold2-Experimental-Fast` 和 `biohub/ESMC-6B`。不要在 Protenix
环境中直接加装 ESMFold2：`fair-esm` 与 Biohub `esm` 使用相同的 Python
namespace，混装可能加载错误实现。

```bash
bash environment/esmfold2/install.sh
conda activate proteocraft_esmfold2_py312
python environment/esmfold2/download_models.py \
  --cache-dir /path/to/proteocraft_hf_cache
```

模型下载脚本会打印 `HF_HOME`、`HF_HUB_OFFLINE`、`TRANSFORMERS_OFFLINE` 和
`ESMCFOLD_CCD_PATH` 的设置。导出这些变量后，可执行以下自检：

```bash
python environment/esmfold2/check_environment.py \
  --require-cuda \
  --require-models
```

## 1. 默认配置

四种任务共用启动 profile：

```text
configs/proteocraft_profiles/esmfold2_d.yaml
```

运行时通过下面的参数加载：

```bash
--profile esmfold2_d
```

该 profile 当前包含以下推荐默认值：

| 项目 | 默认值 |
|---|---|
| Model backend | ESMFold2 |
| Search | Two-stage MCTS |
| Stage A / Stage B | 70 / 40 outer steps |
| Initial roots | 8 |
| Random seed | 43 |
| One-step diffusion | Strict truncated, `tau=1` |
| Stage A proposal | 1 raw argmax + 最多 7 条 positive-score mutation candidates |
| Stage A mutation cap | 最多 50% mutable positions，且仅使用 positive mutation scores |
| Stage B | `no_group`，严格单点突变 |
| Search-time verifier | 4 loops x 50 sampling steps |

命令行中显式提供的参数会覆盖 profile。例如，如果希望整次搜索使用更高精度的 full verifier，可以显式使用 `--esmfold2_verifier_num_loops 20 --esmfold2_verifier_num_sampling_steps 200`。只对最终候选进行高精度复评时，应在搜索结束后单独运行统一的 20 x 200 rescoring，而不是修改搜索期默认值。

加密的 truncated diffusion schedule 是一个可选项，不改变默认 profile：

```bash
--esmfold2_one_step_truncated_schedule_mode dense \
--esmfold2_one_step_truncated_dense_num_sampling_steps 41
```

该设置保持 `max_sigma=256` 和 `tau=1` 不变，在两者之间直接构造 41 个 Karras sigma 点，对应 40 次无梯度去噪和最后 1 次可微去噪。默认的 `official_clipped` 仍使用官方 50-step schedule 并在 `sigma=256` 截断。

## 2. De novo Binder Design

给定固定 target sequence，从头设计一条新的 binder。Binder 的全部位置默认可变。

```bash
TARGET_SEQ="REPLACE_WITH_TARGET_SEQUENCE"

python scripts/run_hallucination.py \
  --profile esmfold2_d \
  --esmfold2_task_type denovo \
  --target "$TARGET_SEQ" \
  --binder_length 120 \
  --output_dir outputs/esmfold2_denovo \
  --device cuda:0
```

如需指定 target hotspot，可增加：

```bash
--target_hotspots "24,35,48-50"
```

## 3. TCR Binder Design

该模式将 TCR alpha 和 beta chains 作为固定 target，重新设计一条与 TCR 结合的 binder。多条 target chain 使用 `|` 分隔。

```bash
TCR_ALPHA_SEQ="REPLACE_WITH_TCR_ALPHA_SEQUENCE"
TCR_BETA_SEQ="REPLACE_WITH_TCR_BETA_SEQUENCE"

python scripts/run_hallucination.py \
  --profile esmfold2_d \
  --esmfold2_task_type tcr \
  --target "${TCR_ALPHA_SEQ}|${TCR_BETA_SEQ}" \
  --esmfold2_target_chain_roles "target,target" \
  --binder_length 120 \
  --output_dir outputs/esmfold2_tcr_binder \
  --device cuda:0
```

Hotspot 编号采用 `--target` 中各条 chain 按顺序拼接后的 1-based residue index。例如 beta chain 第一个 residue 的编号为 `len(alpha) + 1`。

## 4. TCR CDR Redesign

该模式保留一条已有 TCR chain 的 framework，只更新 CDR mutable mask 中标记的位置。抗原 chain 使用 `target` role；未被 redesign 的另一条 TCR chain 使用 `scaffold` role。

```bash
ANTIGEN_SEQ="REPLACE_WITH_ANTIGEN_SEQUENCE"
PARTNER_TCR_SEQ="REPLACE_WITH_FIXED_PARTNER_TCR_SEQUENCE"
REDESIGN_CHAIN_SEQ="REPLACE_WITH_FULL_TCR_CHAIN_SEQUENCE"
CDR_MASK="REPLACE_WITH_EQUAL_LENGTH_0_1_MASK"

python scripts/run_hallucination.py \
  --profile esmfold2_d \
  --esmfold2_task_type cdr_redesign \
  --target "${ANTIGEN_SEQ}|${PARTNER_TCR_SEQ}" \
  --esmfold2_target_chain_roles "target,scaffold" \
  --binder_length "${#REDESIGN_CHAIN_SEQ}" \
  --init_binder_seq "$REDESIGN_CHAIN_SEQ" \
  --esmfold2_mutable_mask "$CDR_MASK" \
  --output_dir outputs/esmfold2_cdr_redesign \
  --device cuda:0
```

`CDR_MASK` 的长度必须与 `REDESIGN_CHAIN_SEQ` 完全一致：`1` 表示允许修改，`0` 表示固定。Mutable mask 保证 framework positions 不参与更新；下面的批量启动器还会在运行结束后检查 framework sequence 是否保持不变。

批量 CDR redesign 可以使用：

```bash
python benchmark/script/run_esmfold2_cdr_redesign.py \
  --config benchmark/yaml/cdr_tcr/esmfold2_cdr_redesign_d_4cases.yaml \
  --case 9NMW_alpha \
  --device cuda:0
```

该 YAML 已直接引用仓库内
`benchmark/datasets/proteocraft_59_v1/cdr/` 的 4 个 benchmark case，无需另行准备输入。当前 ESMFold2 主流程是 sequence-only；template path 只记录数据来源并用于后续评价，不作为模型输入。

### 指定多个重设计区域

同一批量入口也支持单条 Fab 重链或轻链的多区域 redesign，不需要新增搜索模式。
在 case 中用 `redesign_regions` 替代 `cdr_motif`，例如 PRAME HU-38 重链：

```yaml
id: PRAME_HU38_heavy_allcdr
fasta_path: PRAME_HU38.fasta
binder_chain: D
target_chains: [C]
scaffold_chains: [A, B, E]
redesign_regions:
  - [26, 33]
  - [51, 57]
  - [96, 107]
```

上述链名须对应输入 FASTA 的 `|Chain X|`；示例中 A/B/C/D/E 分别是 HLA、beta2m、肽、重链、轻链。
区间是完整输入链的 **1-based、两端包含** 的序列位置，不是 PDB author 或 IMGT 编号。
每个 case 必须且只能填写 `cdr_motif`、`redesign_regions` 之一；旧配置不需要修改。
多个区间取并集，区间之间的 framework 和其他链的序列不变，不支持长度变化或重、轻链同时联合优化。
输出 `case_spec.json` 记录各区间、原始片段和实际 mask；多区间的旧式单段起止字段为 `null`。

默认仍为 sequence-only；序列固定不表示坐标固定。显式选择下述 template 模式时，
批量入口使用上述区间并集作为 structure-regeneration mask，每个连续区间两侧需至少两个固定 framework 位点。
序列 mask 与结构 mask 是不同概念；底层 CLI 保留各自独立的参数，本批量入口默认使用同一组区间。
VHH 的专用入口与 IMGT all-CDR 校验保持不变。

### 4.1 实验性 Template-Anchored 局部 Redesign

`template_anchored_truncated_trajectory` 是独立实验模式，不替换上述默认配置。target 与未重设计 partner 使用 all-atom clamp，被重设计链的 framework 仅固定 `N/CA/C/O`，指定 redesign 区间的结构重新生成。外部模板在截断扩散轨迹中使用 noise-consistent clamp；Full Verifier 仍为 sequence-only 独立预测。

在自己的 CDR YAML 配置中设置 `esmfold2_one_step_diffusion_mode: template_anchored_truncated_trajectory`，并为 case 提供 `template_path`。下面的配置路径和 case ID 需替换为自己的输入：

```bash
python benchmark/script/run_esmfold2_cdr_redesign.py \
  --config /path/to/cdr_cases.yaml \
  --case YOUR_CASE_ID \
  --device cuda:0
```

该模式要求显式提供完整复合物 template、与模型输入顺序一致的 chain IDs，以及独立的 structure regeneration mask。部署到新环境后，正式使用前应先验证单个 case。

### 4.2 Scaffold-consistency Verifier

`scaffold_consistency` 是与 sequence-only Full Verifier 并列的**后处理复评模式**。它不替换默认 `verify()`，也不进入 MCTS 的候选选择。该模式在标准 ESMFold2 的完整 diffusion 轨迹中固定 target/context 的 all-atom 坐标与被 redesign binder 的 framework `N/CA/C/O`，同时允许 CDR 的全部原子重新生成；随后由同一个标准 confidence head 对该 constrained complex 打分。

使用时必须显式构造带 template contract 的 `ESMFold2ProposalConfig`，并调用 `verify_scaffold_consistency()`：

```python
config = ESMFold2ProposalConfig(
    verifier_structure_mode="scaffold_consistency",
    verifier_num_loops=20,
    verifier_num_sampling_steps=200,
    template_structure_path="benchmark/datasets/.../template.cif",
    template_chain_ids=("C", "A", "E", "D"),
    template_chain_sequences=(target, partner_alpha, partner_beta, binder_template),
    binder_structure_regeneration_mask=cdr_mask,
)
result = backend.verify_scaffold_consistency(
    binder_seq=designed_cdr_sequence,
    target_seq=(target, partner_alpha, partner_beta),
    seed=43,
)
```

输出保留正常的 `pair_chain_iptm` 等完整 verifier 字段，并另外记录 `scaffold_consistency_*` 指标、template mapping report 与 `scaffold_consistency_final_fixed_atom_rmsd`。该分数用于回答“在 template scaffold 下该 redesigned CDR 是否仍保持高置信度”，不能替代 sequence-only 的独立复评。

## 5. VHH All-CDR Redesign

该模式保留已有 VHH 的 framework sequence，只允许 IMGT 定义的 CDR1、CDR2 和 CDR3 位点发生替换。IMGT 是抗体/VHH 可变区的标准编号体系；适配器使用其编号结果定位三段 CDR，再将对应的 VHH sequence indices 转成 mutable mask。IMGT 编号不参与模型损失计算。

使用 7EOW Chain B scaffold 模板启动：

```bash
cp benchmark/yaml/vhh/esmfold2_vhh_redesign_7eow_template.yaml \
  /path/to/vhh_case.yaml
# 编辑 vhh_case.yaml，填入 target sequence 和可选 hotspot。

python benchmark/script/run_esmfold2_vhh_redesign.py \
  --config /path/to/vhh_case.yaml \
  --device cuda:0
```

该 runner 只提供一个正式 VHH 模式：`vhh_redesign` + IMGT all-CDR mask。它会校验 CDR1/2/3 均存在、三段位置不重叠、位置位于 VHH sequence 范围内，并在运行结束后检查所有 framework residues 是否保持不变。当前默认仍使用 50% mutation fraction，不为 VHH 修改 Stage A/B 或候选生成逻辑。

当前实现是 sequence-only：7EOW 等参考结构只记录 scaffold 与编号来源，不输入 ESMFold2；模型会重新预测完整 target--VHH complex。因此固定的是 framework amino-acid sequence，而不是 framework coordinates。`official_exact` 初始化会在 CDR mutable positions 上生成多组随机初始 logits；若要量化设计相对 native VHH 的增益，应另外使用相同 verifier 对原始 VHH sequence 打分。

也可以绕过 YAML 直接调用 `scripts/run_hallucination.py`，但此时需自行提供与 VHH 等长且覆盖全部三段 CDR 的 `--esmfold2_mutable_mask`：

```bash
python scripts/run_hallucination.py \
  --profile esmfold2_d \
  --esmfold2_task_type vhh_redesign \
  --target "$TARGET_SEQ" \
  --esmfold2_target_chain_roles target \
  --binder_length "${#VHH_SEQ}" \
  --init_binder_seq "$VHH_SEQ" \
  --esmfold2_mutable_mask "$IMGT_ALL_CDR_MASK" \
  --output_dir outputs/esmfold2_vhh_redesign \
  --device cuda:0
```

## 6. 输入和输出

- `--target` 只接受单字母氨基酸序列；多条 chain 使用 `|` 分隔。
- `--esmfold2_target_chain_roles` 的数量必须与 `--target` 中的 chain 数一致，可取 `target` 或 `scaffold`。
- `--target_hotspots` 使用拼接后 target sequences 的 1-based residue index。
- 默认结果写入 `--output_dir`。
- `esmfold2_artifacts/run_summary.json` 保存最终摘要。
- `esmfold2_artifacts/candidates.jsonl` 和 `trajectory.jsonl` 保存候选与搜索轨迹。
- 显式指定 `--save_checkpoint_dir` 后，搜索中间 checkpoint 会写入对应目录。

如需保存每条 full-verifier candidate 的结构用于可视化，可增加：

```bash
--esmfold2_save_verified_candidate_cif true
```

查看全部可覆盖参数：

```bash
python scripts/run_hallucination.py --help
```

## 7. 完整 59-case Benchmark

仓库内置的可移植 benchmark 包含 14 个 de novo、41 个 TCR binder design
和 4 个 TCR CDR redesign case：

```text
benchmark/datasets/proteocraft_59_v1/
```

先在不加载模型、不占用 GPU 的情况下验证全部输入并查看命令：

```bash
python benchmark/script/run_esmfold2_59.py --prepare-only
```

运行单个 case：

```bash
python benchmark/script/run_esmfold2_59.py \
  --case denovo_BetV1 \
  --device cuda:0
```

运行一种任务或划分多 GPU shard：

```bash
python benchmark/script/run_esmfold2_59.py \
  --task-type tcr \
  --num-shards 8 \
  --shard-index 0 \
  --device cuda:0
```

启动器统一加载 `esmfold2_d`，只从 manifest 注入 target、hotspot、chain
role、binder length 和 CDR mutable mask 等任务输入。详细字段、八卡启动示例
和输出目录见
[`benchmark/datasets/proteocraft_59_v1/README.md`](../benchmark/datasets/proteocraft_59_v1/README.md)。
