# Hallucination Trajectory Visualization Tools

Yanchao, 2026-04-03

三个独立脚本，读取 hallucination checkpoint 目录，生成可视化结果。不依赖 pipeline 代码

所有代码在 `visualization/` 目录下，与主 pipeline 完全隔离。

---

## 快速使用

### 脚本 1：生成 4 张分析图（RMSD / Loss / Confidence / 分项 Loss）

```bash
python visualization/visualize_trajectory.py \
    --checkpoint_dir path/to/checkpoints \
    --output_dir path/to/figures \
    --dpi 150       # 默认 150，论文用 300
```

**输出 4 张 PNG：**

| 文件名                     | 内容                                                  | 方哥关心的点                           |
| -------------------------- | ----------------------------------------------------- | -------------------------------------- |
| `rmsd_evolution.png`     | Binder RMSD vs step 0 + 相邻帧 RMSD + Target RMSD     | **核心图**：结构差异是否不断累积 |
| `total_loss.png`         | Total loss 曲线，标注 best step / accepted / rejected | Loss 下降趋势                          |
| `confidence_metrics.png` | iPTM / mean pLDDT / Rg loss 三面板                    | 是否有 plateau / 边际递减              |
| `all_losses.png`         | 12 项 loss 各自趋势（3x4 网格）                       | 诊断用                                 |

**RMSD 计算方式：** 先在 target CA 上做 Kabsch 对齐（把所有帧对齐到 step 0 的 target），然后度量 binder CA 的位移。这样同时捕获了 binder 内部折叠变化和 binder 相对于 target 的重定位。

### 脚本 2：生成 PyMOL 渲染脚本

```bash
python visualization/generate_pymol_script.py \
    --checkpoint_dir path/to/checkpoints \
    --output trajectory_render.pml \
    --n_frames 8    # 采样帧数，默认 8
```

生成 `.pml` 文件后，在有 PyMOL 的机器上执行：

```bash
pymol -c trajectory_render.pml
```

**渲染内容：**

1. **Overlay 图**：8 帧叠加，binder 从蓝→红渐变（按 step），target 灰色
2. **逐帧 Gallery**：每帧单独渲染，binder 按 pLDDT（B-factor）着色，统一视角

渲染图输出到 `pymol_renders/` 目录。

### 脚本 3：快速 3D 结构查看（浏览器）

```bash
python visualization/quick_view_structures.py \
    --checkpoint_dir path/to/checkpoints \
    --output structure_view.html \
    --steps 0 25 49
```

生成 HTML 文件，在浏览器中打开即可交互式查看 3D 结构（支持鼠标旋转/缩放）。不需要安装 PyMOL。

---

## 输入数据格式

脚本读取 `CheckpointManager` 输出的标准 checkpoint 目录：

```
checkpoints/
  best.json              # 可选，标注 best step
  step_0000/
    step_0000_beam_0_complex.pdb
    step_0000_beam_0_meta.json
  step_0001/
    ...
```

至少需要 **2 个 step** 才能计算 RMSD。

---

## 依赖

- Python 3.10+
- numpy
- matplotlib

不需要 BioPython、protenix 或任何 pipeline 依赖。PyMOL 仅在渲染 `.pml` 时需要。

---

## 测试

```bash
cd BindCraft-v2
python -m pytest visualization/tests/ -v
# 32 tests, ~1s
```

---

## 已知限制

- 单次运行分析（不支持多 run 对比，后续可扩展）
- PDB B-factor 列是 logit 空间的 pLDDT（约 -8.5 到 +4.2），不是 0-100 标准值
- 当前只处理 beam_0，未来 beam search (B>=2) 需要扩展
