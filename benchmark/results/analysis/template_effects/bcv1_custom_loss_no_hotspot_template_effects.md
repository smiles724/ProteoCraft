# BCV1 Custom Loss No-Hotspot Template Effect

## Scope

本记录只分析 BCV1 的 8 个 case，loss 使用 custom loss，不设置 hotspot。比较对象是同一初始化设定下是否添加 target template。

- No-template: `custom_loss_no_hotspot_random_avglen`
- With-template: `custom_loss_with_template_no_hotspot_random_avglen`
- 数据来源: `results/analysis/template_effects/template_vs_no_template_by_case.csv` 中 `loss_family=custom_loss` 的 8 行
- 判定方向: `best_loss` / `final_loss` 越低越好；`ranking_score`、pLDDT、ipTM 越高越好。

## Summary

- 添加 template 后，`best_loss` 改善 8/8。
- 平均 `delta_best_loss = template - no_template = -0.709`，中位数 -0.504。
- 平均 `delta_final_loss = -0.667`。
- 结构置信度平均变化: `ranking_score` +0.136，pLDDT +24.43，ipTM +0.100。

结论：在 custom loss 且不设置 hotspot 的 BCV1 任务中，添加 target template 对优化目标是稳定正向的；8 个 case 的 `best_loss` 全部下降，同时平均 ranking_score、pLDDT、ipTM 均提升。

## Markdown Table

| Case | No-template best | Template best | Delta best | No-template final | Template final | Delta final | Delta ranking | Delta pLDDT | Delta ipTM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BetV1 | 1.272 | -0.127 | -1.399 | 1.309 | -0.127 | -1.436 | +0.127 | +34.25 | +0.070 |
| DerF21 | 0.113 | -0.269 | -0.382 | 0.113 | -0.269 | -0.382 | +0.127 | +12.49 | +0.136 |
| DerF7 | 0.580 | 0.205 | -0.375 | 0.593 | 0.292 | -0.300 | +0.186 | +34.60 | +0.123 |
| HER2 | 1.606 | 0.080 | -1.526 | 1.606 | 0.080 | -1.526 | +0.282 | +46.70 | +0.217 |
| IFNAR2 | 0.466 | 0.290 | -0.177 | 0.466 | 0.290 | -0.177 | -0.158 | +7.37 | -0.211 |
| PD-L1 | 0.575 | -0.051 | -0.626 | 0.575 | 0.118 | -0.457 | +0.346 | +15.09 | +0.372 |
| PD1 | 1.437 | 0.484 | -0.953 | 1.437 | 0.573 | -0.864 | +0.082 | +20.99 | +0.051 |
| SpCas9 | 0.473 | 0.238 | -0.235 | 0.473 | 0.277 | -0.197 | +0.094 | +23.96 | +0.038 |

## LaTeX Table

```latex
\begin{table}[htbp]
\centering
\caption{Effect of target template on BCV1 custom-loss no-hotspot runs. Lower loss is better; positive confidence deltas are better.}
\label{tab:bcv1_custom_loss_template_effect}
\begin{tabular}{lrrrrrr}
\toprule
Case & No-template best & Template best & $\Delta$ best & $\Delta$ final & $\Delta$ ranking & $\Delta$ ipTM \\
\midrule
BetV1 & 1.272 & -0.127 & -1.399 & -1.436 & +0.127 & +0.070 \\
DerF21 & 0.113 & -0.269 & -0.382 & -0.382 & +0.127 & +0.136 \\
DerF7 & 0.580 & 0.205 & -0.375 & -0.300 & +0.186 & +0.123 \\
HER2 & 1.606 & 0.080 & -1.526 & -1.526 & +0.282 & +0.217 \\
IFNAR2 & 0.466 & 0.290 & -0.177 & -0.177 & -0.158 & -0.211 \\
PD-L1 & 0.575 & -0.051 & -0.626 & -0.457 & +0.346 & +0.372 \\
PD1 & 1.437 & 0.484 & -0.953 & -0.864 & +0.082 & +0.051 \\
SpCas9 & 0.473 & 0.238 & -0.235 & -0.197 & +0.094 & +0.038 \\
\midrule
Mean & -- & -- & -0.709 & -0.667 & +0.136 & +0.100 \\
\bottomrule
\end{tabular}
\end{table}
```

## Full LaTeX Table With pLDDT

```latex
\begin{table}[htbp]
\centering
\caption{Per-case BCV1 template ablation under custom loss without hotspot.}
\label{tab:bcv1_custom_loss_template_ablation_full}
\begin{tabular}{lrrrrrrrr}
\toprule
Case & Base best & Tpl. best & $\Delta$ best & Base final & Tpl. final & $\Delta$ final & $\Delta$ pLDDT & $\Delta$ ipTM \\
\midrule
BetV1 & 1.272 & -0.127 & -1.399 & 1.309 & -0.127 & -1.436 & +34.25 & +0.070 \\
DerF21 & 0.113 & -0.269 & -0.382 & 0.113 & -0.269 & -0.382 & +12.49 & +0.136 \\
DerF7 & 0.580 & 0.205 & -0.375 & 0.593 & 0.292 & -0.300 & +34.60 & +0.123 \\
HER2 & 1.606 & 0.080 & -1.526 & 1.606 & 0.080 & -1.526 & +46.70 & +0.217 \\
IFNAR2 & 0.466 & 0.290 & -0.177 & 0.466 & 0.290 & -0.177 & +7.37 & -0.211 \\
PD-L1 & 0.575 & -0.051 & -0.626 & 0.575 & 0.118 & -0.457 & +15.09 & +0.372 \\
PD1 & 1.437 & 0.484 & -0.953 & 1.437 & 0.573 & -0.864 & +20.99 & +0.051 \\
SpCas9 & 0.473 & 0.238 & -0.235 & 0.473 & 0.277 & -0.197 & +23.96 & +0.038 \\
\midrule
Mean & -- & -- & -0.709 & -- & -- & -0.667 & +24.43 & +0.100 \\
\bottomrule
\end{tabular}
\end{table}
```

## Notes

- 这里的 `Delta` 均定义为 `with_template - no_template`。
- `Delta best` 和 `Delta final` 为负数表示 template 条件更好。
- pLDDT、ipTM、ranking score 的 delta 为正数表示 template 条件更好。
