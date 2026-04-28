# Template Effect Analysis

Paired comparison of no-template vs target-template runs on the eight bc_v1 targets.
For loss values, lower is better. For confidence values such as ranking_score, pLDDT and ipTM, higher is better.

## Overall Summary

| Loss family | Cases | Template lower best_loss | Mean delta best_loss | Median delta best_loss | Mean % delta best_loss | Mean delta ranking_score | Mean delta ipTM | Mean delta pLDDT | Mean delta runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Default loss | 8 | 7/8 | -1.636 | -1.775 | -39.3% | 0.149 | 0.095 | 33.220 | -977.9 |
| V1 loss | 8 | 4/8 | 0.090 | -0.033 | 4.2% | 0.162 | 0.131 | 26.015 | -1236.4 |
| Custom loss | 8 | 8/8 | -0.709 | -0.504 | -108.9% | 0.136 | 0.100 | 24.430 | -1401.6 |

## Per-Target Best Loss

| Loss family | Target | No template | Template | Delta | Delta % | Template better | Ranking delta | ipTM delta | pLDDT delta |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| Default loss | BetV1 | 3.266 | 1.550 | -1.715 | -52.5% | yes | -0.068 | -0.156 | 39.613 |
| Default loss | DerF21 | 1.249 | 0.759 | -0.490 | -39.3% | yes | -0.369 | -0.439 | 7.736 |
| Default loss | DerF7 | 3.329 | 1.878 | -1.451 | -43.6% | yes | 0.110 | 0.032 | 35.356 |
| Default loss | HER2 | 3.746 | 1.882 | -1.864 | -49.8% | yes | 0.163 | 0.078 | 44.620 |
| Default loss | IFNAR2 | 4.854 | 1.420 | -3.434 | -70.8% | yes | 0.403 | 0.372 | 43.350 |
| Default loss | PD-L1 | 4.501 | 1.286 | -3.215 | -71.4% | yes | 0.614 | 0.620 | 39.292 |
| Default loss | PD1 | 2.892 | 1.057 | -1.835 | -63.5% | yes | -0.050 | -0.108 | 19.266 |
| Default loss | SpCas9 | 1.204 | 2.120 | 0.917 | 76.2% | no | 0.388 | 0.357 | 36.528 |
| V1 loss | BetV1 | 1.922 | 1.580 | -0.342 | -17.8% | yes | 0.119 | 0.063 | 35.839 |
| V1 loss | DerF21 | 1.116 | 0.953 | -0.163 | -14.6% | yes | -0.067 | -0.083 | -3.425 |
| V1 loss | DerF7 | 1.525 | 2.145 | 0.620 | 40.7% | no | 0.185 | 0.142 | 29.934 |
| V1 loss | HER2 | 1.457 | 1.845 | 0.388 | 26.6% | no | 0.199 | 0.132 | 40.608 |
| V1 loss | IFNAR2 | 1.679 | 1.572 | -0.108 | -6.4% | yes | 0.082 | 0.062 | 24.339 |
| V1 loss | PD-L1 | 0.916 | 0.717 | -0.199 | -21.8% | yes | 0.302 | 0.296 | 27.777 |
| V1 loss | PD1 | 1.243 | 1.285 | 0.042 | 3.4% | no | 0.085 | 0.050 | 28.316 |
| V1 loss | SpCas9 | 2.037 | 2.517 | 0.480 | 23.6% | no | 0.389 | 0.385 | 24.730 |
| Custom loss | BetV1 | 1.272 | -0.127 | -1.399 | -110.0% | yes | 0.127 | 0.070 | 34.250 |
| Custom loss | DerF21 | 0.113 | -0.269 | -0.382 | -338.6% | yes | 0.127 | 0.136 | 12.487 |
| Custom loss | DerF7 | 0.580 | 0.205 | -0.375 | -64.7% | yes | 0.186 | 0.123 | 34.598 |
| Custom loss | HER2 | 1.606 | 0.080 | -1.526 | -95.0% | yes | 0.282 | 0.217 | 46.699 |
| Custom loss | IFNAR2 | 0.466 | 0.290 | -0.177 | -37.9% | yes | -0.158 | -0.211 | 7.367 |
| Custom loss | PD-L1 | 0.575 | -0.051 | -0.626 | -108.9% | yes | 0.346 | 0.372 | 15.087 |
| Custom loss | PD1 | 1.437 | 0.484 | -0.953 | -66.3% | yes | 0.082 | 0.051 | 20.992 |
| Custom loss | SpCas9 | 0.473 | 0.238 | -0.235 | -49.7% | yes | 0.094 | 0.038 | 23.960 |
