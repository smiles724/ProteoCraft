# ProteoCraft Fast 部署与运行指南

本文档说明如何在新机器上部署 ProteoCraft 的 ESMFold2 Fast 运行时。
Fast 不是一个独立的权重文件，而是以下组件的固定组合：

- ProteoCraft 的 `proteocraft-uplifting-v1` 适配层；
- `biohub/ESMFold2-Experimental-Fast` proposal 模型；
- `biohub/ESMFold2` full verifier；
- `biohub/ESMC-6B` sequence prior 及模型内部 LM；
- `uplifting-biomolecular-modeling` 中的 CUDA/Triton 优化内核。

所有命令默认在 ProteoCraft 仓库根目录中执行。

## 1. 硬件与系统要求

- Linux x86-64；
- Python 3.12；
- 兼容 CUDA 12.8 PyTorch wheel 的 NVIDIA driver；
- NVIDIA GPU compute capability 8.0 或 9.0，例如 A100、H100 或 H200；
- 推荐 H200。80 GB GPU 可能只适合较短的输入；
- 至少 50 GB 可用磁盘空间存放三个模型的 Hugging Face cache。

当前 Fast adapter 会拒绝 compute capability 8.0/9.0 以外的 GPU。其他
GPU 必须先完成 kernel 兼容性和数值验证，不应直接删除该检查。

## 2. 获取固定源码

克隆本分支：

```bash
git clone --branch feature/proteocraft-fast <PROTEOCRAFT_GIT_URL> ProteoCraft
cd ProteoCraft
```

克隆并固定优化内核仓库：

```bash
git clone https://github.com/anthropics/uplifting-biomolecular-modeling.git \
  ../uplifting-biomolecular-modeling

git -C ../uplifting-biomolecular-modeling checkout --detach \
  f4f62fa6592ae4938d49b1757bea0cfeff9f468e

export PROTEOCRAFT_UPLIFT_ROOT="$(cd ../uplifting-biomolecular-modeling && pwd)"
```

不要删除该 checkout 的 `.git` 目录。ProteoCraft 会读取 Git revision，
并对实际加载的优化源码计算 hash。

## 3. 创建独立 Fast 环境

Fast 运行时必须与 Protenix/基线 ESMFold2 环境分开。`fair-esm` 与 Biohub
`esm` 提供相同的 Python namespace，不能混装。

```bash
export FAST_ENV="$(pwd)/../proteocraft_esmopt_py312"
export ESMOPT_BASE_PYTHON=python3.12

bash environment/esmfold2/install_optimization.sh "$FAST_ENV"

"$FAST_ENV/bin/python" -m pip install --editable "$(pwd)" --no-deps
```

安装脚本固定了以下核心运行时：

| 组件 | 版本/Revision |
|---|---|
| Python | 3.12 |
| PyTorch | `2.11.0+cu128` |
| Triton | `3.6.0` |
| xFormers | `0.0.35` |
| Biohub Transformers fork | `ef32577f` |
| Biohub ESM | `d0207ea3c8cbf072679ece3bb332787d25e06852` |

`install_optimization.sh` 要求目标环境路径不存在，防止覆盖已有环境。

## 4. 下载固定模型

```bash
export HF_HOME="$(pwd)/../proteocraft_hf_cache"

"$FAST_ENV/bin/python" environment/esmfold2/download_models.py \
  --cache-dir "$HF_HOME"
```

下载脚本使用下列固定 revision：

| 模型 | Revision |
|---|---|
| `biohub/ESMFold2` | `1ebf0e3481a5184eb6171d40615c79e384b48796` |
| `biohub/ESMFold2-Experimental-Fast` | `04dec820ede9283c9893e318e5ca5a9ac2ab93bc` |
| `biohub/ESMC-6B` | `45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a` |

建议用户从官方 Hugging Face 仓库下载权重，而不是从二次打包的
ProteoCraft 源码包中获取。离线交付时应整体复制 `$HF_HOME/hub/`，
不要只复制 `snapshots/`，因为 snapshot 文件通常指向同一 cache 中的
`blobs/`。

## 5. 配置运行环境

每个新 shell 在运行前执行：

```bash
export FAST_ENV="/absolute/path/to/proteocraft_esmopt_py312"
export PROTEOCRAFT_UPLIFT_ROOT="/absolute/path/to/uplifting-biomolecular-modeling"
export HF_HOME="/absolute/path/to/proteocraft_hf_cache"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ESMCFOLD_CCD_PATH="$HF_HOME/hub/models--biohub--ESMFold2/snapshots/1ebf0e3481a5184eb6171d40615c79e384b48796/ccd.pkl"

# ProteoCraft 使用显式 adapter，禁止外部 kit 的隐式 autoload 开关污染对照实验。
unset ESMFOLD2_OPT ESMC_OPT EF2_FAST_KIT ESMC_KIT

# 将 JIT cache 放到可写的持久目录。
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/proteocraft/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/proteocraft/torchinductor}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
```

检查模型和 CCD 文件：

```bash
test -f "$ESMCFOLD_CCD_PATH"
test -d "$HF_HOME/hub/models--biohub--ESMFold2-Experimental-Fast/snapshots/04dec820ede9283c9893e318e5ca5a9ac2ab93bc"
test -d "$HF_HOME/hub/models--biohub--ESMC-6B/snapshots/45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a"
```

Fast 环境不应使用当前的 `environment/esmfold2/check_environment.py`；该脚本
检查的是 PyTorch 2.7.1 基线环境。Fast adapter 及上游 pin 可用下面的测试检查：

```bash
"$FAST_ENV/bin/python" -m pytest \
  bindcraft/tests/test_esmfold2_optimizations.py -q
```

## 6. 运行 Fast 设计

下面是最小 de novo binder design 示例：

```bash
TARGET_SEQ="REPLACE_WITH_TARGET_SEQUENCE"

"$FAST_ENV/bin/python" scripts/run_hallucination.py \
  --profile esmfold2_d \
  --esmfold2_task_type denovo \
  --target "$TARGET_SEQ" \
  --binder_length 120 \
  --output_dir outputs/esmfold2_fast_denovo \
  --device cuda:0 \
  --esmfold2_optimization_root "$PROTEOCRAFT_UPLIFT_ROOT" \
  --esmfold2_proposal_optimization fast \
  --esmfold2_verifier_optimization fast \
  --esmfold2_esmc_optimization exact \
  --esmfold2_optimization_timing
```

ProteoCraft 实验中的 `fast` arm 精确定义为：

```text
proposal = fast
verifier = fast
ESMC = exact
```

首次运行会编译 Triton/TorchInductor kernels，因此启动时间显著长于后续运行。
运行结束后，检查：

```text
outputs/esmfold2_fast_denovo/optimization_report.json
outputs/esmfold2_fast_denovo/optimization_report.timing.jsonl
outputs/esmfold2_fast_denovo/esmfold2_artifacts/
```

`optimization_report.json` 记录了实际使用的模式、优化 lever、源码 revision/hash
及 fallback counter。分享生成结果时应一并保留该文件。

TCR binder、TCR CDR redesign 和 VHH all-CDR redesign 的输入方式见
[`docs/ESMFold2_QUICK_START.md`](docs/ESMFold2_QUICK_START.md)；在其命令末尾加入上述
四个 `--esmfold2_*optimization*` 参数即可启用 Fast。

## 7. GPU smoke test

用一个短 PD1 case 分别检查 baseline 和 Fast proposal：

```bash
PROPOSAL_SNAPSHOT="$HF_HOME/hub/models--biohub--ESMFold2-Experimental-Fast/snapshots/04dec820ede9283c9893e318e5ca5a9ac2ab93bc"
VERIFIER_SNAPSHOT="$HF_HOME/hub/models--biohub--ESMFold2/snapshots/1ebf0e3481a5184eb6171d40615c79e384b48796"

"$FAST_ENV/bin/python" benchmark/script/run_esm_optimization.py micro \
  --case denovo_PD1 \
  --phase proposal \
  --arm baseline \
  --output outputs/esm_optimization/smoke/proposal/baseline \
  --kit-root "$PROTEOCRAFT_UPLIFT_ROOT" \
  --proposal-model "$PROPOSAL_SNAPSHOT" \
  --verifier-model "$VERIFIER_SNAPSHOT" \
  --warmup 1 \
  --repeats 2 \
  --deterministic

"$FAST_ENV/bin/python" benchmark/script/run_esm_optimization.py micro \
  --case denovo_PD1 \
  --phase proposal \
  --arm proposal_fast \
  --output outputs/esm_optimization/smoke/proposal/proposal_fast \
  --reference outputs/esm_optimization/smoke/proposal/baseline \
  --kit-root "$PROTEOCRAFT_UPLIFT_ROOT" \
  --proposal-model "$PROPOSAL_SNAPSHOT" \
  --verifier-model "$VERIFIER_SNAPSHOT" \
  --warmup 1 \
  --repeats 2 \
  --deterministic
```

Slurm 环境也可使用
[`benchmark/script/sbatch_esm_optimization_smoke.sh`](benchmark/script/sbatch_esm_optimization_smoke.sh)。

## 8. 常见问题

### `Upstream source mismatch`

Biohub Transformers fork 或 uplifting checkout 与固定版本不匹配。不要跳过检查；
重新创建一个空环境并确认 uplifting commit。

### `Missing ... ccd.pkl` / `ESMCFOLD_CCD_PATH`

确认执行了权重下载脚本，且该变量指向固定 ESMFold2 snapshot 内的
`ccd.pkl`。

### GPU capability 被拒绝

当前 Fast kernels 只验证了 SM80/SM90。请切换到 A100/H100/H200，或使用
`off`/`exact` 进行新硬件的兼容性验证。

### OOM

尝试更短的 target/binder，或将 proposal/verifier 设为 `big`。`big` 允许与
Fast 相同级别的数值变化，不应当作 exact 模式。当前实验并未确认
`big` 能在所有任务上降低进程累计峰值显存。

### 同一 Python process 切换优化源码或 mode

不支持。baseline、exact、fast 和 big 应分别在新 Python process 中运行。

## 9. 许可与分发

ProteoCraft 仓库使用 Apache-2.0。`uplifting-biomolecular-modeling` 的原始代码
使用 Apache-2.0，并包含需要保留的第三方 notices。模型权重有独立的
Hugging Face model card 和 third-party terms。发布源码或离线包时请保留所有
`LICENSE`、`NOTICE`、`THIRD_PARTY_NOTICES.md` 和 model card，并在二次分发权重前
单独确认其条款。

优化 kit 的 `pyproject.toml` 标记了 `Private :: Do Not Upload`；不要将其作为
PyPI package 上传。应通过固定 Git checkout、内部源码归档或容器镜像分发。
