# AtlasFold-M 独立复评环境

该环境只用于 `atlasfold_m` sequence-only 交叉复评，不进入 ProteoCraft 搜索、
proposal、MCTS 或 loss。AtlasFold 官方源码和模型权重均保存在 ProteoCraft 仓库之外。

## 1. 获取锁定源码

```bash
git clone https://github.com/SeonghwanSeo/atlasfold.git /path/to/atlasfold-v1.0.2
git -C /path/to/atlasfold-v1.0.2 checkout --detach \
  444f376d85b9954a5f2f5f3f8b3cbcae1201ebb1
```

该提交对应 `v1.0.2`。安装和运行前都会检查提交、版本和 tracked diff；不匹配时
直接停止。完整锁定信息见 `source.lock.json`。

## 2. 安装隔离环境

```bash
bash environment/atlasfold/install.sh \
  /path/to/atlasfold-v1.0.2 \
  proteocraft_atlasfold_m_py312 \
  torch
```

第三个参数可用 `torch` 或 `cuequiv`。正式对比必须在所有机器使用同一种 kernel，
并在结果中保留 `atlasfold_kernel`。

如果服务器保留了可用的 Python/CUDA/PyTorch 环境，但没有 conda 主程序，可建立一个
不修改原环境的独立 venv，并只复用原环境的只读 site-packages：

```bash
bash environment/atlasfold/install_venv.sh \
  /path/to/atlasfold-v1.0.2 \
  /path/to/cuda-python \
  /path/to/proteocraft_atlasfold_m_py312 \
  torch
```

该方式适合现有 GPU 节点；每次正式运行仍会记录实际 Python、Torch、CUDA 和依赖版本。

## 3. 下载并固定权重

```bash
conda activate proteocraft_atlasfold_m_py312
python environment/atlasfold/download_models.py \
  --cache-dir /path/to/atlasfold_cache
```

脚本下载 `atlasfold-m-260725@a9f07668` 和 `atlaslm-3b-base@b4e6d8ab`，并写出
包含两个完整 Hub commit 的 `atlasfold_m_weights_receipt.json`。正式运行使用脚本
打印的两个本地权重路径，不要让官方 runner 再按 `main` 动态下载。脚本默认最多
重试 12 次；Hugging Face 的 `.incomplete` 缓存会在连接中断后继续下载。

## 4. 自检

```bash
python environment/atlasfold/check_environment.py \
  --source-root /path/to/atlasfold-v1.0.2 \
  --model-path "$ATLASFOLD_MODEL_PATH" \
  --lm-path "$ATLASFOLD_LM_PATH" \
  --weights-receipt "$ATLASFOLD_WEIGHTS_RECEIPT" \
  --require-cuda \
  --require-models
```

`download_models.py` 会打印 `ATLASFOLD_MODEL_PATH`、`ATLASFOLD_LM_PATH`、
`ATLASFOLD_CACHE_DIR` 和 `ATLASFOLD_WEIGHTS_RECEIPT`。正式运行必须同时保留这四个
变量；evaluator 会拒绝未绑定到 40 位 Hub commit 的 receipt。

ProteoCraft evaluator 的启动示例见
`evaluation/metric_pipeline/README.md` 的 AtlasFold-M 小节。
