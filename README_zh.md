# Protenix: Protein + X

> 📣📣📣 **我们正在招聘！** \
> 岗位位于 **_北京_** 🇨🇳 和 **_西雅图_** 🇺🇸 \
> 对机器学习、计算化学/生物学、结构生物学或药物发现感兴趣？加入我们，共同构建生物领域的前沿 AI！
> 👉 [**加入我们 »**](#join-us)

<div align="center" style="margin: 20px 0;">
  <span style="margin: 0 10px;">⚡ <a href="https://protenix-server.com">Protenix 在线服务器</a></span>
  &bull; <span style="margin: 0 10px;">📄 <a href="docs/PTX_V1_Technical_Report_202602042356.pdf">技术报告</a></span>
</div>

<div align="center">

[![Twitter](https://img.shields.io/badge/Twitter-Follow-blue?logo=x)](https://x.com/ai4s_protenix)
[![Slack](https://img.shields.io/badge/Slack-Join-yellow?logo=slack)](https://join.slack.com/t/protenixworkspace/shared_invite/zt-3drypwagk-zRnDF2VtOQhpWJqMrIveMw)
[![Wechat](https://img.shields.io/badge/Wechat-Join-brightgreen?logo=wechat)](https://github.com/bytedance/Protenix/issues/52)
[![Email](https://img.shields.io/badge/Email-Contact-lightgrey?logo=gmail)](#contact-us)
</div>

我们很高兴向您介绍 **Protenix** —— 面向高精度开源生物分子结构预测。

Protenix 专注于高精度结构预测，是我们为计算生物学社区提供可及、可扩展研究工具的重要一步。

<img src="assets/protenix_predictions.gif" style="width: 100%; height: auto;" alt="Protenix 预测效果">

## 🌟 相关项目

- **[PXDesign](https://protenix.github.io/pxdesign/)** 是基于 Protenix 基础模型的从头蛋白质-配体设计模型套件。PXDesign 在多个靶点上达到 20–73% 的实验成功率，比 AlphaProteo、RFdiffusion 等先前 SOTA 方法高出 2–6 倍。该框架可通过 Protenix 服务器免费使用。

- **[PXMeter](https://github.com/bytedance/PXMeter/)** 是面向结构预测模型可重复评估的开源工具包，附带经人工审核去除实验伪影和非生物相互作用的高质量基准数据集。相关研究对 SOTA 模型进行了深入对比分析，从大量指标数据和案例研究中提炼洞察。Protenix 的评估基于 PXMeter。

- **[Protenix-Dock](https://github.com/bytedance/Protenix-Dock)**：我们实现的经典蛋白质-配体对接框架，采用经验打分函数。在不使用深度神经网络的情况下，Protenix-Dock 在刚性对接任务中表现优异。

## 🎉 最新更新

- **2026-02-05: Protenix-v1 发布** 💪 [[技术报告](docs/PTX_V1_Technical_Report_202602042356.pdf)]
  - 支持模板/RNA MSA 特征，改进训练动态，并进一步优化推理时模型性能。
- **2025-11-05: Protenix-v0.7.0 发布** 🚀
  - 引入扩散推理优化：共享变量缓存、高效内核融合、TF32 加速。参见 [性能分析](./assets/inference_time_vs_ntoken.png)。
- **2025-07-17: Protenix-Mini 与约束特征**
  - 发布轻量级模型变体 [Protenix-Mini](https://arxiv.org/abs/2507.11839)，在几乎不损失精度的前提下大幅降低推理成本。
  - 新增 [原子级接触与口袋约束](docs/infer_json_format.md#constraint) 支持，通过物理先验提升预测精度。
- **2025-01-16: 流程增强**
  - 开源完整 [训练数据流程](./docs/prepare_training_data.md) 和 [MSA 流程](./docs/msa_template_pipeline.md)。
  - 集成本地 [ColabFold 兼容搜索](./docs/colabfold_compatible_msa.md)，简化 MSA 生成。

## 🚀 快速开始

### 🛠 快速安装

```bash
pip install protenix
```

### 🧬 快速预测

```bash
# 使用 JSON 输入进行结构预测
protenix pred -i examples/input.json -o ./output -n protenix_base_default_v1.0.0
```

#### 主要模型说明

| 模型名称 | MSA | RNA MSA | 模板 | 参数量 | 训练数据截止日期 | 模型发布日期 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `protenix_base_default_v1.0.0` | ✅ | ✅ | ✅ | 368 M | 2021-09-30 | 2026-02-05 |
| `protenix_base_20250630_v1.0.0` | ✅ | ✅ | ✅ | 368 M | 2025-06-30 | 2026-02-05 |
| `protenix_base_default_v0.5.0` | ✅ | ❌ | ❌ | 368 M | 2021-09-30 | 2025-05-30 |

- **protenix_base_default_v1.0.0**：默认模型，训练数据截止日期与 AlphaFold3 对齐（2021-09-30）。
  > 💡 这是进行公平、严谨的公开基准测试及与其他 SOTA 方法对比研究的**强烈推荐**模型。
- **protenix_base_20250630_v1.0.0**：应用型模型，采用更新数据截止日期（2025-06-30）以获得更好的实际性能，适用于实际应用场景。
- **protenix_base_default_v0.5.0**：旧版模型，主要为基于 v0.5.0 开发的用户保持向后兼容。

完整支持的模型列表请参阅 [支持的模型](docs/supported_models.md)。

安装、数据预处理、推理和训练的详细说明请参阅 [训练与推理说明](docs/training_inference_instructions.md)。建议用户参考 [inference_demo.sh](inference_demo.sh) 了解详细推理方法和输入说明。

### 📊 基准测试

**Protenix-v1**（即 `protenix_base_default_v1.0.0` 模型）是首个在多种基准集上超越 AlphaFold3 的全开源模型，同时保持与 AlphaFold3 相同的训练数据截止日期、模型规模和推理预算。对于抗原-抗体复合物等困难靶点，Protenix-v1 的预测精度可通过推理时扩展进一步提升——将采样候选数从数个增至数百个，可获得一致的对数线性增益。

<img src="./assets/protenix_base_default_v1.0.0_metrics.png" style="width: 100%; height: auto;" alt="protenix-v1 模型指标">

<img src="./assets/protenix_base_default_v1.0.0_metrics2.png" style="width: 100%; height: auto;" alt="protenix-v1 模型指标 2">

各数据集的详细基准指标请参阅 [docs/model_1.0.0_benchmark.md](docs/model_1.0.0_benchmark.md)。

## 引用 Protenix

若您在研究中使用 Protenix，请引用以下文献：

```
@article {Zhang2026.02.05.703733,
	author = {Zhang, Yuxuan and Gong, Chengyue and Zhang, Hanyu and Ma, Wenzhi and Liu, Zhenyu and Chen, Xinshi and Guan, Jiaqi and Wang, Lan and Yang, Yanping and Xia, Yu and Xiao, Wenzhi},
	title = {Protenix-v1: Toward High-Accuracy Open-Source Biomolecular Structure Prediction},
	elocation-id = {2026.02.05.703733},
	year = {2026},
	doi = {10.64898/2026.02.05.703733},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2026/02/22/2026.02.05.703733.1},
	eprint = {https://www.biorxiv.org/content/early/2026/02/22/2026.02.05.703733.1.full.pdf},
	journal = {bioRxiv}
}
```

### 📚 引用相关工作

Protenix 基于并受多项重要工作启发。若您使用 Protenix，我们也鼓励在适当处引用以下基础工作：

```
@article{abramson2024accurate,
  title={Accurate structure prediction of biomolecular interactions with AlphaFold 3},
  author={Abramson, Josh and Adler, Jonas and Dunger, Jack and Evans, Richard and Green, Tim and Pritzel, Alexander and Ronneberger, Olaf and Willmore, Lindsay and Ballard, Andrew J and Bambrick, Joshua and others},
  journal={Nature},
  volume={630},
  number={8016},
  pages={493--500},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
@article{ahdritz2024openfold,
  title={OpenFold: Retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization},
  author={Ahdritz, Gustaf and Bouatta, Nazim and Floristean, Christina and Kadyan, Sachin and Xia, Qinghui and Gerecke, William and O'Donnell, Timothy J and Berenberg, Daniel and Fisk, Ian and Zanichelli, Niccol{\`o} and others},
  journal={Nature Methods},
  volume={21},
  number={8},
  pages={1514--1524},
  year={2024},
  publisher={Nature Publishing Group US New York}
}
@article{mirdita2022colabfold,
  title={ColabFold: making protein folding accessible to all},
  author={Mirdita, Milot and Sch{\"u}tze, Konstantin and Moriwaki, Yoshitaka and Heo, Lim and Ovchinnikov, Sergey and Steinegger, Martin},
  journal={Nature methods},
  volume={19},
  number={6},
  pages={679--682},
  year={2022},
  publisher={Nature Publishing Group US New York}
}
```

## 参与贡献

我们欢迎社区贡献以帮助改进 Protenix！

📄 查看 [贡献指南](CONTRIBUTING.md) 开始参与。

✅ 代码质量：我们使用 `pre-commit` 钩子确保一致性与代码质量。提交前请安装：

```bash
pip install pre-commit
pre-commit install
```

🐞 发现 bug 或有功能建议？[提交 Issue](https://github.com/bytedance/Protenix/issues)。

## 致谢

LayerNorm 算子的实现参考了 [OneFlow](https://github.com/Oneflow-Inc/oneflow) 和 [FastFold](https://github.com/hpcaitech/FastFold)。我们还采用了 [OpenFold](https://github.com/aqlaboratory/openfold) 的若干[模块](protenix/openfold_local/)实现，但 [`LayerNorm`](protenix/model/layer_norm/) 为独立实现。

## 行为准则

我们致力于营造友好、包容的环境。请查阅 [行为准则](CODE_OF_CONDUCT.md) 了解如何以尊重的方式参与。

## 安全

若您发现本项目的潜在安全问题，或认为可能发现安全问题，请通过 [安全中心](https://security.bytedance.com/src) 或 [漏洞报告邮箱](sec@bytedance.com) 联系字节跳动安全团队。

请**不要**在 GitHub 上公开提交相关 Issue。

## 许可证

Protenix 项目（包括代码和模型参数）采用 [Apache 2.0 许可证](./LICENSE) 发布，学术研究和商业使用均免费。

## 联系我们

我们欢迎关于模型高级应用的咨询与合作，如开发新功能、针对特定场景微调等。欢迎发送邮件至 ai4s-bio@bytedance.com。

## 加入我们

字节跳动 Seed-AI for Science 的 **Protenix 团队**正在扩招！我们寻找机器学习与计算生物学/化学领域的人才（*“计算生物学/化学”涵盖结构生物学、计算生物学、计算化学、药物发现等*）。岗位分布于 **北京** 和 **西雅图**，包括实习、应届生和全职。

优秀候选人将有机会入选 **字节跳动 Top Seed 人才计划**，获得更多支持。

### 📍 北京，中国

| 类型 | 方向 | 申请链接 |
|------------|------------------------------------|------------|
| 全职 | 蛋白质设计科学家 | [社招](https://jobs.bytedance.com/society/position/detail/7550992796392982792) |
| 全职 | 计算生物学/化学 | [社招](https://jobs.bytedance.com/society/position/detail/7505998274429421842)、[校招](https://job.toutiao.com/s/HGwWBs1UGR4) |
| 全职 | 机器学习 | [社招](https://jobs.bytedance.com/society/position/detail/7505999453133015314)、[校招](https://job.toutiao.com/s/upy82CljXlY) |
| 实习 | 计算生物学/化学 | [实习](https://job.toutiao.com/s/Wr3yig1Wet4) |
| 实习 | 机器学习 | [实习](https://job.toutiao.com/s/w2GQQDfQUkc) |

### 📍 西雅图，美国

| 类型 | 方向 | 申请链接 |
|------------|------------------------------------|------------|
| 全职 | 计算生物学/化学 | [Experienced](https://jobs.bytedance.com/en/position/7270666468370614585/detail)、[New Grad](https://job.toutiao.com/s/iH00nSEvrFo) |
| 全职 | 机器学习 | [Experienced](https://jobs.bytedance.com/en/position/7270665658072926521/detail)、[New Grad](https://job.toutiao.com/s/dmU_fbEHGOw) |
| 实习 | 计算生物学/化学 | [Internship](https://job.toutiao.com/s/aiCZz0kJexs) |
| 实习 | 机器学习 | [Internship](https://job.toutiao.com/s/DiGnn5l1QpQ) |
