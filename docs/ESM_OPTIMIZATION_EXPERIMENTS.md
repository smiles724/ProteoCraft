# ESM 路线优化与实验

ProteoCraft 的 ESM 路线包含三个不同职责的模型。优化源码目录与运行时模型的对应关系如下。

| 优化目录 | ProteoCraft 职责 | 固定模型 | 可选模式 |
|---|---|---|---|
| `ef2inv/` | 可微 proposal、distogram 和 sequence update | `biohub/ESMFold2-Experimental-Fast` @ `04dec820...` | `off/exact/fast/big` |
| `esmfold2/` | full verifier、候选结构与 confidence | `biohub/ESMFold2` @ `1ebf0e34...` | `off/exact/fast/big` |
| `esmc/` | proposal 内部 LM 与独立 pseudoperplexity prior | `biohub/ESMC-6B` @ `45b0fa5d...` | `off/exact` |

这里的实现是显式的 ProteoCraft adapter，composition 名为
`proteocraft-uplifting-v1`。它复用了三个优化目录中与当前 runtime 调用路径兼容的
instance-level levers，但不宣称等同于各目录自带 cookbook/SDK 的完整 mode。每次运行的
`optimization_report.json` 会记录实际加载的源码 hash、启用的 lever、fallback counter、
版本和分阶段时间；没有启用的优化不会静默标成成功。

## 环境

创建单独环境，不要覆盖基线 ESMFold2 环境：

```bash
export PROTEOCRAFT_UPLIFT_ROOT=/absolute/path/to/uplifting-biomolecular-modeling
bash environment/esmfold2/install_optimization.sh /absolute/path/to/proteocraft_esmopt_py312

export HF_HOME=/path/to/proteocraft_hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ESMCFOLD_CCD_PATH=/path/to/ccd.pkl
```

安装脚本固定 ef2inv kit 的 PyTorch `2.11.0+cu128`、Triton `3.6.0`、Biohub
Transformers fork `ef32577f` 和完整
Biohub `esm` checkout `d0207ea3`。模型快照继续使用
`environment/esmfold2/download_models.py` 下载的固定 revisions。

## ProteoCraft 开关

原有 `esmfold2_d` profile 保持 `off`，在实验命令中显式选择模式：

```bash
python scripts/run_hallucination.py \
  --profile esmfold2_d \
  --esmfold2_optimization_root "$PROTEOCRAFT_UPLIFT_ROOT" \
  --esmfold2_proposal_optimization exact \
  --esmfold2_verifier_optimization exact \
  --esmfold2_esmc_optimization exact \
  --esmfold2_optimization_timing \
  ...
```

`exact` 用于先验证数值等价；`fast` 允许 proposal/verifier 的 bf16 reduction 顺序变化；
`big` 使用 fast 数值口径并优先降低显存。ESMC 只有 `exact`。同一 Python process 不得切换
优化源码根目录，baseline 与不同模式应各自在新 process 中运行。

## 实验 1：分阶段 microbenchmark

先对同一 case、seed、GPU、模型 revision 分别测四个阶段：

- `prior`：ESMC pseudoperplexity 的 forward 和 input gradient。
- `proposal`：Experimental-Fast distogram forward 和 input gradient。
- `update`：一次完整 soft sequence update。
- `verifier`：一次 full verifier fold。

基线示例：

```bash
python benchmark/script/run_esm_optimization.py micro \
  --case denovo_PD1 --phase proposal --arm baseline \
  --output outputs/esm_optimization/micro/proposal/baseline \
  --kit-root "$PROTEOCRAFT_UPLIFT_ROOT" --deterministic
```

对应优化 arm 使用新的 process，并指向 baseline：

```bash
python benchmark/script/run_esm_optimization.py micro \
  --case denovo_PD1 --phase proposal --arm proposal_fast \
  --output outputs/esm_optimization/micro/proposal/proposal_fast \
  --reference outputs/esm_optimization/micro/proposal/baseline \
  --kit-root "$PROTEOCRAFT_UPLIFT_ROOT" --deterministic
```

输出包含 warmup 后的 median/p95 latency、模型驻留显存、峰值显存、首调用编译时间、
speedup，以及输出和 input gradient 相对 baseline 的 bitwise、max-abs、relative-L2 和
cosine 差异。正式数字必须在同一种 GPU 上收集；首调用编译时间与 steady-state 时间分开报告。
`--deterministic` 在 baseline 和优化 arm 上同时应用 ef2inv 自带的 parity recipe；exact 的
bitwise 结论只使用这组结果，不能拿普通的非确定性进程互相比。
仓库还提供 `benchmark/script/sbatch_esm_optimization_smoke.sh` 做 GPU 接口和小规模 smoke test。

### 已收集的阶段数据

同一张 B200、case `denovo_PD1`、seed 43 的当前结果如下。每个 latency 只有两次
warmed repeats，属于方向性 pilot；首调/模型载入时间单独保存，不能据此宣称完整流程加速。

| 阶段 | baseline | 优化 | 中位延迟 | 观测差异 |
|---|---:|---:|---:|---|
| ESMC prior | 108.9 ms | exact 61.2 ms | 1.78× faster | input gradient bitwise 相同 |
| proposal | 794.7 ms | exact 690.8 ms | 1.15× faster | distogram 和 input gradient bitwise 相同；同一 parity recipe |
| verifier | 550.7 ms | exact 724.0 ms | 0.76× speedup（latency +31%） | coordinates bitwise 相同；该 smoke 仅 1 loop × 2 steps |

prior 的精确性来自单次 paired run；proposal/verifier 的 parity 使用了 kit 提供的
deterministic recipe。verifier exact 目前短任务较慢，正式 profile 是 4 loops × 50 steps，
需看完整采样是否摊薄 graph capture/cache 开销。

正式 micro gate 使用 ef2inv 固定的 PyTorch `2.11.0+cu128`、Triton `3.6.0`、xformers
`0.0.35`，在同一张 H200 上跑三次 warmed repeats。verifier 使用 4 loops × 50 steps：

| 阶段/模式 | warmed median | 相对 baseline | 数值差异 | 峰值 allocated |
|---|---:|---:|---|---:|
| proposal baseline | 0.862 s | 1.00× | reference | 42.20 GB |
| proposal exact | 1.071 s | 0.81× | distogram/gradient bitwise 相同 | 42.20 GB |
| proposal fast | 0.495 s | 1.74× | distogram rel-L2 2.34%；gradient rel-L2 5.06%，cos 0.99893 | 41.64 GB |
| verifier baseline | 1.777 s | 1.00× | reference | 14.68 GB |
| verifier exact | 1.829 s | 0.97× | coordinates bitwise 相同 | 15.27 GB |
| verifier fast | 0.933 s | 1.91× | coordinates rel-L2 1.35%，cos 0.99991 | 15.40 GB |
| verifier big | 1.179 s | 1.51× | 与 verifier fast 的差异相同 | 14.60 GB |

`fast/big` 的 numerical delta 必须由下面的端到端生成与盲复评决定是否可接受。`big`
在这个 case 上只比 baseline 少约 0.08 GB allocated，暂时没有显示出值得采用的显存收益。

质量 pilot 已固定在 `outputs/esm_optimization/pilot_2case_43/campaign.json`：PD1 与
1QRN TCR 两种任务、共同 search seed 43、baseline/exact/fast/big，均用仓库的完整 profile。
这是只有两个 source target 的 exploratory pilot；search 完成后以未优化 ESMFold2 用
top-5 unique sequences 和 eval seeds 1043/1044 盲复评。不能把它当作确认性非劣效结论。

## 实验 2：端到端成对生成

主要比较 `baseline/exact/fast/big`。同一 case 使用共同 seeds，arm 顺序按
case/seed 确定性打乱，避免 baseline 总在冷缓存上运行。pilot 建议覆盖不同 target 长度和任务：

```bash
python benchmark/script/run_esm_optimization.py prepare \
  --output outputs/esm_optimization/pilot \
  --kit-root "$PROTEOCRAFT_UPLIFT_ROOT" \
  --arms baseline exact fast big \
  --seeds 43 44 45 \
  --cases denovo_BetV1 denovo_HER2 denovo_PD1 denovo_SpCas9 \
          tcr_1qrn_no_hotspot cdr_9NMW_alpha

python benchmark/script/run_esm_optimization.py run \
  --manifest outputs/esm_optimization/pilot/campaign.json
```

多 GPU 时把 `run` 分成固定 shards；每个 shard 仍需要一张 GPU：

```bash
python benchmark/script/run_esm_optimization.py run \
  --manifest outputs/esm_optimization/pilot/campaign.json \
  --num-shards 8 --shard-index 0
```

主要速度指标是相同 case/seed 的端到端 wall-time ratio，包含模型加载和编译；
`optimization_report.timing.jsonl` 同时给出 proposal、prior 和 verifier 的内部时间。
所有失败和未完成 run 保留在 report 分母中。

## 实验 3：盲复评生成质量

搜索本身的 verifier 分数不能作为唯一质量结论。每个 arm 按搜索 loss 选择 top-k unique
sequences，再用完全关闭优化的标准 ESMFold2 verifier、共同但与搜索不同的 seeds 复评：

```bash
python benchmark/script/run_esm_optimization.py rescore \
  --manifest outputs/esm_optimization/pilot/campaign.json \
  --output outputs/esm_optimization/pilot_rescore \
  --top-k 5 --seeds 1043 1044 1045

python benchmark/script/run_esm_optimization.py report \
  --manifest outputs/esm_optimization/pilot/campaign.json \
  --scores outputs/esm_optimization/pilot_rescore \
  --output outputs/esm_optimization/pilot_report.csv \
  --success-threshold 0.8
```

CSV 保存每个 arm/case/seed 的完成状态、候选数、verifier 调用数、top-1 与 best-of-k
blind `pair_chain_iptm`。summary JSON 给出按 source target 等权的 cluster bootstrap：端到端
speedup、top-1 质量差和 best-of-k 质量差。少于 5 个独立 source targets 时结果标为
`exploratory`，不能把多个 hotspot variant 当成独立 target 增加统计把握度。

如需 AF3/AtlasFold-M 独立评价，可把 top-k 候选导出到已有 metric pipeline：

```bash
python benchmark/script/run_esm_optimization.py export \
  --manifest outputs/esm_optimization/pilot/campaign.json \
  --output outputs/esm_optimization/pilot_eval_manifest.csv \
  --top-k 5
```

导出的 reference complex 是搜索预测结构，不是实验真值，因此结构 RMSD 只能视为
refolding consistency；独立模型的 confidence、interface score 和成功率更适合作为主结果。

## 建议判定门槛

先冻结门槛再看完整结果：

- `exact`：microbenchmark 中声称 exact 的 tensors/gradients 必须 bitwise equal；失败率不能增加；
  blind top-1/best-of-k 质量差应接近 0。
- `fast`：建议用 `pair_chain_iptm` 非劣效 margin `-0.02`，并要求完成率下降不超过 5 个百分点；
  同时报告 mean delta 和 95% target-cluster bootstrap interval。
- `big`：使用与 fast 相同的质量门槛，主要接受条件是峰值显存下降；如果显存没有下降则没有采用理由。
- 只有当 paired wall-time speedup 大于 1、质量门槛通过且优化 report 没有未解释 fallback 时，
  才把该模式提升为推荐 profile。默认 profile 在完整实验结束前保持基线行为。
