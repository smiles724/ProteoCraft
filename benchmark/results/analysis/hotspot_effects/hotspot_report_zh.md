# PD1 hotspot B64/B126/B129/B133 影响分析

## 输入结果

- hotspot result：`benchmark/results/pd1_default_loss_hotspot_B64_B126_B129_B133/pd1_default_loss_hotspot_B64_B126_B129_B133/PD1/result.json`
- 配置：`benchmark/yaml/pd1_default_loss_hotspot_B64_B126_B129_B133.yaml`
- hotspot：`B64,B126,B129,B133`，解析为 `B64;B126;B129;B133`，concat indices `39;101;104;108`。
- template：`/data/huangyanjie/binder_design/BindCraft-v2/benchmark/input/bc_v1_benchmark_data/pdb_template_fixed_cif/PD1.cif`
- binder length：105；best step：27；运行时间：33.9 min。

## Hotspot run 指标

| best_loss | final_loss | ranking_score | pLDDT | ipTM | gpDE | i_con loss | i_pae loss | mutations |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.978 | 0.978 | 0.437 | 72.483 | 0.377 | 0.997 | 0.136 | 0.601 | 19 |

## 与 no-hotspot 结果的参考对比

这里列出的是已有 PD1 no-hotspot runs。注意这次 hotspot run 与这些 baseline 不一定只差 hotspot：部分 run 的 loss weights、template、random seed 或初始序列不同，所以差值应作为参考，不应直接解释为纯 hotspot 因果效应。

| Baseline | same init | template | delta best_loss | delta ranking_score | delta pLDDT | delta ipTM | delta i_con loss | delta mutations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| same_template_seed43_default_no_hotspot | no | yes | -0.078 | +0.165 | +5.453 | +0.182 | -0.019 | -6 |
| same_template_seed43_v1loss_no_hotspot | no | yes | -0.307 | +0.123 | +3.886 | +0.133 | -0.038 | +1 |
| custom_loss_template_seed42_no_hotspot | no | yes | +0.495 | +0.074 | +5.354 | +0.073 | +0.012 | +7 |
| custom_loss_no_template_seed42_no_hotspot | no | no | -0.459 | +0.156 | +26.346 | +0.124 | -0.230 | +1 |
| bc_v1_random_seed43_no_template_no_hotspot | no | no | -1.913 | +0.115 | +24.719 | +0.074 | -0.186 | -14 |

## 主要观察

- 相对最接近的 no-hotspot 参考 `same_template_seed43_default_no_hotspot`，hotspot run 的 best_loss 变化为 -0.078，ranking_score 变化为 +0.165，pLDDT 变化为 +5.453，ipTM 变化为 +0.182。
- hotspot run 的接口接触项 i_con loss 为 0.136；相对 `same_template_seed43_default_no_hotspot` 的 i_con loss 变化为 -0.019。在这个 loss 定义下，i_con loss 越低通常表示接口接触目标更好。
- best step 为 27，之后 loss 没有继续刷新；最终序列与 best 序列一致。
- 现有数据不能严格隔离 hotspot 的单独贡献，因为缺少一组与该 hotspot run 完全相同但 `target_hotspot_residues: null` 的 PD1 对照。更严谨的 hotspot effect 需要补跑这一组 only-hotspot-ablation。

## 文件

- `pd1_hotspot_run.csv`：hotspot run 的完整指标。
- `pd1_hotspot_vs_no_hotspot.csv`：hotspot run 与各 no-hotspot PD1 结果的差值。
- `pd1_hotspot_vs_no_hotspot_delta_conf.png`：confidence 指标差值图。
- `pd1_hotspot_vs_no_hotspot_delta_loss.png`：loss 指标差值图。
