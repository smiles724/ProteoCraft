# ESMFold2 长复合物与显存管理

公共实现：`bindcraft/hallucination/esmfold2_memory.py`；加速缓存适配位于
`esmfold2_optimizations.py`，调用边界位于 `esmfold2_runtime.py`。
适用于使用该运行层的单靶标、双靶标、抗体和融合蛋白任务，不包含具体靶点或位点规则。

## 配置

| 字段 | 默认 | 含义 |
|---|---|---|
| `memory_mode` | `resident` | 保留历史驻留方式；`auto` 条件触发；`phase_offload` 始终卸载当前阶段不使用的模型 |
| `memory_token_threshold` | `1024` | 自动模式的完整复合物残基数触发阈值，包括全部靶标链及 binder |
| `memory_min_free_gib` | `null` | 可选的可用显存触发阈值，按驱动空闲加 allocator 可复用缓存计算 |
| `memory_saved_tensor_pin_memory` | `true` | 卸载反向张量时使用锁页内存；`false` 使用普通 CPU 内存，避免锁页分配池长期保留大块内存 |
| `memory_prior_residency` | `per_step` | 独立 ESMC prior：逐步卸载；`proposal` 在提案阶段保留；`between_diffusion` 在普通梯度步之间保留、扩散梯度前卸载 |

1024 是针对已观察到的约 1368–1525 残基长复合物 OOM 采用的保守工程触发点，
不是模型长度上限，也不是所有 GPU 都适用的内存保证。触发后本次运行保持启用，避免振荡。

双靶标 JSON 的 `runtime` 可加入：

```json
{
  "memory_mode": "auto",
  "memory_token_threshold": 1024,
  "memory_min_free_gib": 24.0
}
```

通用 CLI 对应：

```bash
--esmfold2_memory_mode auto \
--esmfold2_memory_token_threshold 1024 \
--esmfold2_memory_min_free_gib 24
```

VHH YAML 启动器透传同名 CLI 字段。默认 `resident` 不进行模型搬移或显存探测。

## 执行方式

1. 进入 verifier 阶段前，把 proposal 和独立 ESMC prior 移到 CPU；使用 verifier。
2. 进入 proposal 阶段前，把 verifier 和独立 ESMC prior 移到 CPU；使用 proposal。
   只有计算 ESMC prior 梯度时才恢复独立 prior，梯度完成后再次移出。
   proposal 保持在 GPU，因为后续 one-step diffusion 仍可能使用其已捕获的梯度图。
3. 搬移前同步设备，按加速库协议清理整个 CUDA Graph 池世代及其静态缓冲，
   同时清除与模型关联的 transition/triangle GPU 权重副本缓存。
4. 释放已不用的 allocator 缓存，再恢复当前阶段的模型。同阶段连续的人/鼠计算不重复搬移。
5. 激活时将 verifier 的有效图捕获预算限制在 `memory_token_threshold` 内；
   用户显式设置的更小正预算继续保留。请求值与实际值都记录在优化报告中。
6. proposal、ESMC prior 及 one-step diffusion 的可微前向计算使用 PyTorch
   锁页 CPU 快照（默认）或等布局的普通 CPU 快照，把反向传播需要保存的张量
   放在主机内存，在反向时恢复原 dtype 和设备。这也覆盖长 pair trunk 的 checkpoint 输入。

模型在 CPU 内存中保留，不重复下载或重读权重。dtype、权重值、随机种子、损失、
目标序列、可变位点、Stage A/B 步数和候选数量均不调整。没有将 FP32 prior 换成 BF16。
需要足够主机内存，并增加 CPU/GPU 数据搬运时间；这是显存和吞吐之间的取舍。
共享主机内存不足时，可设置 `memory_saved_tensor_pin_memory: false`；通用 CLI 为
`--no-esmfold2_memory_saved_tensor_pin_memory`。这不压缩张量或降低精度，代价是普通
CPU 内存的传输不能使用同样的异步锁页路径。它只减少分配池留存，不承诺消除仍被
活动梯度图引用的张量；并发数必须依据实测峰值决定。历史默认行为保持使用锁页内存。
Pinned tensor 保存可能改变非连续视图的布局，从而改变 GEMM 的浮点归约舍入。
因此保持计算精度不等于所有结果逐位相等；保留图 CUDA 对照的梯度最大绝对误差为
1.19e-7，按 FP32 浮点容差检查。完整设计的质量仍使用原有指标。

模型只在其自身计算/反向已经结束的边界移动；另一个仍有活动梯度图的模型留在 GPU。
反向中间量卸载采用 PyTorch 的 saved-tensor hooks，不做有损压缩或更换计算精度。
它不承诺任意长度都能运行，单个激活张量仍可能过大。发生 OOM 时继续显式报错，
不自动重试、截短靶标、减少预测步骤或切换评分。

## 验证与记录

`optimization_report.json` 包含 `memory` 决策、触发原因、实际图预算和累计统计；
同目录 `optimization_report.memory.jsonl` 记录准备、卸载、恢复和就绪时的显存快照。
模型/梯度的 CUDA 往返一致性测试位于 `bindcraft/tests/test_esmfold2_memory.py`。
单元测试同时覆盖历史默认行为、完整复合物计数、低显存触发、缓存清理顺序和失败传播。

首次应用于新 GPU 或更长输入时，检查实际完整设计步骤、梯度有限性和峰值显存，
不能仅凭父本预测成功或进程已启动判定修复完成。

参考：[PyTorch save_on_cpu](https://docs.pytorch.org/docs/2.8/autograd.html#torch.autograd.graph.save_on_cpu)。

普通 CPU 快照也逐次重新拷贝，并保持锁页路径的连续布局；不复用相同地址的保存张量。
实验性的快照去重未通过完整 M16 的梯度对照，已撤回，未用于生产搜索。

## 可选的 ESMC prior 驻留策略

`memory_prior_residency` 只在 `auto` 已触发或 `phase_offload` 时改变模型摆放；
`resident` 沿用全部模型驻留的历史行为。默认 `per_step` 不改变已有任务。
双靶标 JSON 可在 `runtime` 中配置；通用 CLI / VHH YAML 使用
`--esmfold2_memory_prior_residency between_diffusion` / 同名 YAML 字段。

- `per_step`：先验梯度结束后立即把 prior 搬到 CPU。
- `proposal`：整个提案阶段保留 proposal 和 prior；进入完整验证时卸载两者。
- `between_diffusion`：普通梯度步之间保留 prior；独立 prior 的反向完成后、
  高峰 one-step diffusion 梯度开始前卸载 prior。proposal 和活动梯度图始终保留。
  同时覆盖定期注入和仅末端注入，适用于不同任务、靶标链数和 binder 类型。

新策略增加提案阶段的显存驻留量。这里不根据任务名自动选择策略，也不在 OOM 后
自动重试或降级；用完整输入实测决定使用哪档。权重、dtype、RNG、损失和预算均不改动。
首次使用前，按相同输入和种子对照完整梯度更新及扩散注入，检查梯度、logits、候选
序列、验证指标和峰值内存。单元测试不代表完整模型 GPU 对照已经通过。
