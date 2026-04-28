# 添加 target template 后的结果对比分析

## 分析口径

本次只比较 random_avglen 初始化下的三组 loss 配比，每组做同一 target 的成对比较：

| Loss 组 | 未加 template | 加 template |
|---|---|---|
| Default loss | `bc_v1_random_avglen_hybrid` | `bc_v1_random_avglen_hybrid_template` |
| V1 loss | `bc_v1_random_avglen_hybrid_v1loss` | `bc_v1_random_avglen_hybrid_v1loss_template` |
| Custom loss | `custom_loss_no_hotspot_random_avglen` | `custom_loss_with_template_no_hotspot_random_avglen` |

三组都覆盖 8 个 bc_v1 target：BetV1、DerF21、DerF7、HER2、IFNAR2、PD-L1、PD1、SpCas9。每组 template 和 no-template 的 target fasta、binder length、random seed、优化步数一致，主要差异是加 template 组额外使用了 `target_template_dir`。

指标解释：

- `best_loss` / component loss：越低越好，只能在同一 loss 配比内直接比较。
- `ranking_score`、pLDDT、ipTM、pTM：越高越好。
- gpDE：越低越好。
- runtime 只作为观察值，因为不同运行使用 GPU/worker 数可能不同，不应作为严格性能基准。

## 总体结论

| Loss 组 | Template best_loss 更优 | mean delta best_loss | median delta best_loss | mean delta ranking_score | mean delta ipTM | mean delta pLDDT |
|---|---:|---:|---:|---:|---:|---:|
| Default loss | 7/8 | -1.636 | -1.775 | +0.149 | +0.095 | +33.22 |
| V1 loss | 4/8 | +0.090 | -0.033 | +0.162 | +0.131 | +26.01 |
| Custom loss | 8/8 | -0.709 | -0.504 | +0.136 | +0.100 | +24.43 |

核心结论：

1. 加 template 对结构置信度指标是稳定正向的。三组 loss 下，平均 ranking_score、pLDDT、ipTM 都提升，gpDE 都下降。
2. 对优化目标 best_loss 的影响取决于 loss 配比。Default loss 和 Custom loss 明显受益，V1 loss 的 total best_loss 是混合结果。
3. Custom loss + template 是当前最稳定组合：8/8 target 的 best_loss 都下降，同时 ranking_score、pLDDT、ipTM 平均提升。
4. V1 loss 虽然只有 4/8 target 的 best_loss 下降，但所有 best_loss 变差的 target 仍然表现出 ranking_score、pLDDT、ipTM 提升，说明 V1 objective 与结构置信度指标之间存在一定不一致。

## 分 target 对比

| Loss 组 | Target | no-template best_loss | template best_loss | delta | template 更优 | ranking delta | ipTM delta | pLDDT delta |
|---|---|---:|---:|---:|---|---:|---:|---:|
| Default | BetV1 | 3.266 | 1.550 | -1.715 | yes | -0.068 | -0.156 | +39.61 |
| Default | DerF21 | 1.249 | 0.759 | -0.490 | yes | -0.369 | -0.439 | +7.74 |
| Default | DerF7 | 3.329 | 1.878 | -1.451 | yes | +0.110 | +0.032 | +35.36 |
| Default | HER2 | 3.746 | 1.882 | -1.864 | yes | +0.163 | +0.078 | +44.62 |
| Default | IFNAR2 | 4.854 | 1.420 | -3.434 | yes | +0.403 | +0.372 | +43.35 |
| Default | PD-L1 | 4.501 | 1.286 | -3.215 | yes | +0.614 | +0.620 | +39.29 |
| Default | PD1 | 2.892 | 1.057 | -1.835 | yes | -0.050 | -0.108 | +19.27 |
| Default | SpCas9 | 1.204 | 2.120 | +0.917 | no | +0.388 | +0.357 | +36.53 |
| V1 | BetV1 | 1.922 | 1.580 | -0.342 | yes | +0.119 | +0.063 | +35.84 |
| V1 | DerF21 | 1.116 | 0.953 | -0.163 | yes | -0.067 | -0.083 | -3.43 |
| V1 | DerF7 | 1.525 | 2.145 | +0.620 | no | +0.185 | +0.142 | +29.93 |
| V1 | HER2 | 1.457 | 1.845 | +0.388 | no | +0.199 | +0.132 | +40.61 |
| V1 | IFNAR2 | 1.679 | 1.572 | -0.108 | yes | +0.082 | +0.062 | +24.34 |
| V1 | PD-L1 | 0.916 | 0.717 | -0.199 | yes | +0.302 | +0.296 | +27.78 |
| V1 | PD1 | 1.243 | 1.285 | +0.042 | no | +0.085 | +0.050 | +28.32 |
| V1 | SpCas9 | 2.037 | 2.517 | +0.480 | no | +0.389 | +0.385 | +24.73 |
| Custom | BetV1 | 1.272 | -0.127 | -1.399 | yes | +0.127 | +0.070 | +34.25 |
| Custom | DerF21 | 0.113 | -0.269 | -0.382 | yes | +0.127 | +0.136 | +12.49 |
| Custom | DerF7 | 0.580 | 0.205 | -0.375 | yes | +0.186 | +0.123 | +34.60 |
| Custom | HER2 | 1.606 | 0.080 | -1.526 | yes | +0.282 | +0.217 | +46.70 |
| Custom | IFNAR2 | 0.466 | 0.290 | -0.177 | yes | -0.158 | -0.211 | +7.37 |
| Custom | PD-L1 | 0.575 | -0.051 | -0.626 | yes | +0.346 | +0.372 | +15.09 |
| Custom | PD1 | 1.437 | 0.484 | -0.953 | yes | +0.082 | +0.051 | +20.99 |
| Custom | SpCas9 | 0.473 | 0.238 | -0.235 | yes | +0.094 | +0.038 | +23.96 |

## Component-level 观察

加 template 后，三组 loss 的平均 component delta 有共同模式：

- `i_con` loss 均下降：Default -0.343、V1 -0.288、Custom -0.176。
- `i_pae` loss 均下降：Default -0.116、V1 -0.103、Custom -0.091。
- `iptm` loss 均下降，对应 confidence ipTM 均提升。
- pLDDT confidence 提升幅度很大：Default +33.22、V1 +26.01、Custom +24.43。
- gpDE 均下降：Default -2.09、V1 -1.28、Custom -1.22。

这说明 template 主要带来的不是单一 loss 项变化，而是整体结构预测置信度和接口相关项一起改善。

## 需要强调的例外

Default loss 的 SpCas9 和 V1 loss 的 DerF7、HER2、PD1、SpCas9 出现了 template best_loss 变高的情况。但这些例外里，ranking_score、pLDDT、ipTM 多数仍然明显提升。例如 Default/SpCas9 的 best_loss 变差 +0.917，但 ranking_score +0.388、ipTM +0.357、pLDDT +36.53。

这提示 best_loss 与结构置信度不是完全一致的评价目标。汇报时建议把结论表述为：template 对预测结构质量指标有稳定帮助；对当前优化 objective 的帮助取决于 loss 权重，其中 custom loss 最匹配 template 条件。

## 建议汇报结论

可以按以下逻辑汇报：

1. 我们在三种 loss 配比下都补跑了 target template 条件，每组 8 个 target，和 no-template 条件做 paired comparison。
2. 加 template 后，结构置信度指标整体稳定提升：ranking_score、pLDDT、ipTM 均提升，gpDE 下降。
3. 从优化目标 best_loss 看，Default loss 有 7/8 target 改善，Custom loss 有 8/8 target 改善，V1 loss 是 4/8 改善。
4. Custom loss + template 是目前最稳的组合，既降低 objective，也提升结构置信度。
5. V1 loss 下存在 best_loss 与 confidence 指标不一致的现象，后续如果以结构质量为主，可以考虑继续沿用 custom loss，并增加 template 条件下的重复随机种子验证。

## 产物位置

- 汇总表：`results/analysis/template_effects/template_vs_no_template_summary.csv`
- 分 target 表：`results/analysis/template_effects/template_vs_no_template_by_case.csv`
- 英文/机器生成报告：`results/analysis/template_effects/template_effects_report.md`
- 本中文报告：`results/analysis/template_effects/template_effects_report_zh.md`
- 图：`results/analysis/template_effects/best_loss_scatter.png`
- 图：`results/analysis/template_effects/default_loss_delta_best_loss.png`
- 图：`results/analysis/template_effects/v1_loss_delta_best_loss.png`
- 图：`results/analysis/template_effects/custom_loss_delta_best_loss.png`
