# ESMFold2 运行环境

本目录用于重建当前 ProteoCraft ESMFold2 实验环境。它与 Protenix 环境分开，原因是
Protenix 使用 `fair-esm`，而 ESMFold2 路径使用 Biohub `esm`；两者提供相同的
Python `esm` namespace，不应安装在同一个环境中。

## 已验证环境

| 项目 | 当前实验环境 |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.7.1，CUDA 12.6 |
| GPU / driver | NVIDIA H200 / 580.126.20 |
| Biohub ESM | commit `82ee35553d39169d678f784c8d3f8712ffd7d2c4` |
| Biohub Transformers | commit `ef32577f55da19a4989cd7b22e004dc43a4998cb` |
| Fused kernels | 未安装 `xformers`、`flash-attn`、`transformer-engine` |

15122 和 15322 的核心包版本、Biohub commits 和模型 revisions 一致。两边完整
环境还包含各自的非 ProteoCraft 工具，因此本目录只固定 ESMFold2 运行所需依赖。

## 1. 安装

在仓库根目录执行：

```bash
bash environment/esmfold2/install.sh
conda activate proteocraft_esmfold2_py312
```

安装脚本不会执行根目录的 `requirements.txt`，也不会安装 `fair-esm`。ProteoCraft
本身使用 editable、`--no-deps` 方式安装，避免把 Protenix 的依赖写入该环境。
该步骤需要访问 PyPI 和 GitHub；无法访问 GitHub 的计算节点应先在联网机器构建同一
Python/CUDA 平台的 wheelhouse，或由管理员复制已经验证的 conda environment。

## 2. 下载模型

三组模型的当前缓存约需 34 GB，建议至少预留 50 GB：

```bash
python environment/esmfold2/download_models.py \
  --cache-dir /path/to/proteocraft_hf_cache
```

脚本固定并下载以下 revisions：

| Hugging Face model | Revision |
|---|---|
| `biohub/ESMFold2` | `1ebf0e3481a5184eb6171d40615c79e384b48796` |
| `biohub/ESMFold2-Experimental-Fast` | `04dec820ede9283c9893e318e5ca5a9ac2ab93bc` |
| `biohub/ESMC-6B` | `45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a` |

下载完成后，脚本会打印本次 shell 需要设置的四个环境变量：`HF_HOME`、
`HF_HUB_OFFLINE`、`TRANSFORMERS_OFFLINE` 和 `ESMCFOLD_CCD_PATH`。模型权重与
`ccd.pkl` 不进入 Git，需要每个运行环境自行缓存或由管理员复制共享缓存。

## 3. 自检

不加载模型，只检查 Python 包、Biohub commits、CUDA、模型 revision 和 CCD：

```bash
python environment/esmfold2/check_environment.py \
  --require-cuda \
  --require-models
```

当前实验为了保持数值口径，没有安装 fused kernels。安装 `xformers`、
`flash-attn` 或 `transformer-engine` 可以提速，但会改变部分 bf16 reduction 的数值
顺序；需要严格复现实验时不要自行增加。

算法默认配置及四种任务的启动命令见
[`docs/ESMFold2_QUICK_START.md`](../../docs/ESMFold2_QUICK_START.md)。
