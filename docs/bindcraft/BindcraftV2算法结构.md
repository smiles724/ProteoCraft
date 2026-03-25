# BindCraft v2 算法架构文档

# BindCraft v2 核心算法架构设计：基于 AlphaFold 3 的逆向梯度生成

在基于深度学习大模型进行蛋白质序列从头设计（De novo design）时，BindCraft 系列算法的核心思想是 **网络求逆与推理期优化 (Network Inversion & Inference-time Optimization)**。具体而言，该方法冻结蛋白质大模型的网络权重，将输入的氨基酸序列设定为连续可导的变量，通过在模型输出的 3D 结构上计算物理约束目标（如结合距离、空间位阻、置信度等），利用梯度的反向传播直接更新输入序列。

从 AlphaFold 2 (AF2) 升级至 AlphaFold 3 / Protenix (AF3) 时，由于底层物理表征与生成范式的根本性调整，原有的连续梯度传播通道被彻底阻断。BindCraft v2 必须在**输入端离散化重构**与**输出端扩散截断**两个方向进行深度的架构级调整。

## 一、 核心理论映射（CV 领域到蛋白质 3D 生成）

在冻结网络权重的前提下，纯靠回传梯度修改输入序列的“幻觉生成”，在计算机视觉 (CV) 领域已有成熟的数学先例。BindCraft v2 的方案严格映射了以下三大前沿范式：

1. **解决“离散氨基酸不可导”问题**：类似于 CV 中的 **PEZ 算法 (ICLR 2023)**（用于反推离散文本提示词），通过连续的 Logits 构建软概率分布，结合预计算特征与直通估计器 (STE)，使连续的梯度能够安全穿过完全离散、不可导的 Token 查表映射层。
2. **解决“多步扩散计算限制与梯度衰减”**：参考 **Universal Guidance (ICCV 2023)** 或 **DPS** 等算法，截断长步骤的 ODE (Ordinary Differential Equation) 求解过程。在特定的微小噪声时间步，利用单次前向传播外推出无噪声的三维结构，并在其上计算物理约束损失，规避显存爆炸（OOM）并保证梯度的有效性。
3. **利用生成模型反向优化输入**：参考 **SDS / DreamFusion (ICLR 2023)** 的思路，将冻结参数的生成大模型视为一个单步可导的“物理打分器（Frozen Critic）”，其单步回传的切线梯度在数学期望上等效于驱动一维序列演化的最速下降方向。

## 二、 底层设计动机：AF3 的死锁与改造必要性

为了彻底理解架构重构的必要性，我们需要明确 AF3 给逆向优化带来的两座“物理死锁”：

### 1. 输入端死锁：AF3 的双重“物理离散化”操作

在 AF2 时代，系统为每种氨基酸分配固定的 37 个原子坐标槽位，总原子数恒定，可以直接对序列概率进行加权混合。但 AF3 追求极致的物理真实性，改用了**展平的、变长的真实原子列表**。这导致：

- **序列** $\rightarrow$ **变长物理原子**：系统在 CPU 端查表，将一维序列转化为真实的重原子。由于甘氨酸仅有 4 个原子，色氨酸有 14 个，**系统的总重原子数是一个随氨基酸种类动态变化的标量**。若输入连续混合概率，系统物理上无法分配“非整数”的原子数，直接导致 3D 张量初始化崩溃（Shape Mismatch）。
    
    <aside>
    💡
    
    这里引入的第一个问题是从embedding倒推回去的时候token的数量是未知的
    
    </aside>
    
- **原子** $\rightarrow$ **Token 初始特征**：生成的变长原子需经过底层的 `AtomAttentionEncoder` 聚合为一维残基级特征。该过程的底层图拓扑完全由前一步离散查表定死，反向传播的梯度在此会被彻底截断。
    
    <aside>
    💡
    
    这里的第二个问题就是查表的问题，embedding对应的可能是现实中不存在的原子种类，而原子排列出来的也可能是现实中不存在的氨基酸（但这个bindcraft解决了，我们可以参考）
    
    </aside>
    

因此，我们必须在输入端使用“硬拓扑建图，软特征回传（STE）”的策略跨越此鸿沟。

### 2. 输出端死锁：为什么要将 Diffusion 变成 One-Step？

- **规避 OOM 与梯度粉碎**：如果在 PyTorch 中保留数十步流匹配 ODE 迭代的完整计算图，会瞬间耗尽显存；即便显存足够，梯度经过数千层网络的链式相乘，也会衰减为纯粹的高频数值噪音，导致优化失效。
- **单步优化的理论合理性**：逆向优化需要的是“方向导数”而非“完美的生成样本”。利用单次前向传播预测出速度场并外推三维结构，如果该结构中 Binder 侧链与靶点发生碰撞，此时产生的瞬时一阶梯度已足够明确地指示底层特征去“缩小该处侧链体积”。这种未经成千上万层网络污染的单步切线梯度，能为优化器提供信噪比最高的更新信号。

## 三、 AF3 原始前向计算流 (The Baseline Forward Pass)

在模型所有网络权重参数（统称为 $\Phi$）完全冻结的情况下，AF3 原始处理离散序列的物理映射过程分为三步。这三步也是我们优化时必须面对的底层约束：

### 1. 物理展开与初始编码 (Input Embedder - 离散不可导)

给定包含 Binder 和 Target 的一维离散输入氨基酸序列 $\mathbf{a}$（序列总长度定义为标量 $N_{token}$），系统查表确定本轮动态实例化的**真实物理重原子总数** $N_{atom}$（随氨基酸种类动态变化的标量）。随后提取初始的单体特征 $\mathbf{s}^{init}$（维度为 $N_{token} \times c_s$，$c_s$ 为单体特征通道数）与对特征 $\mathbf{z}^{init}$（维度为 $N_{token} \times N_{token} \times c_z$，$c_z$ 为对通道数）：

$$
\mathbf{s}^{init}, \mathbf{z}^{init}, N_{atom} = \text{InputEmbedder}(\mathbf{a})
$$

### 2. 主干网络交互 (Pairformer - 连续可导)

利用 48 层 Transformer 块，结合冻结的主干网络权重 $\Phi_{pair}$，更新全局 3D 拓扑关联的深层单体特征 $\mathbf{s}$ 和深层对特征 $\mathbf{z}$（输出维度与初始特征保持不变）：

$$
\mathbf{s}, \mathbf{z} = \text{Pairformer}(\mathbf{s}^{init}, \mathbf{z}^{init} \mid \Phi_{pair})
$$

### 3. 扩散生成模块 (Diffusion Module - 连续可导)

在给定的连续标量时间步 $t \in (0,1]$（$t=1$ 为纯噪声）和此时刻对应的带噪 3D 原子坐标 $\mathbf{x}_t  \in \mathbb{R}^{N_{atom} \times 3}$ 下，网络以 $\mathbf{s}, \mathbf{z}$ 为条件，输出拟合的去噪速度场 $\mathbf{v}_{pred}$*（维度为* $N{atom} \times 3$，代表原子的运动方向，$\Phi_{diff}$ 为冻结的扩散网络权重）：

$$
\mathbf{v}_{pred} = \text{Diffusion}(\mathbf{x}_t, t, \mathbf{s}, \mathbf{z} \mid \Phi_{diff}) \quad  \in \mathbb{R}^{N_{atom} \times 3}
$$

## 四、 BindCraft v2 的逆向幻觉重构 (The Hallucination Pipeline)

为了将外部的三维物理约束转化为指导一维序列突变的梯度，我们对计算图进行如下底层重构：

### 模块 A：连续重参数化与 STE 桥接 (对应 CV: PEZ 算法)

*(解决问题：跨越 InputEmbedder 中由于动态原子数分配与离散查表带来的绝对不可导鸿沟)*

1. **从连续空间映射概率分布**:

定义待优化的连续序列变量为 $\mathbf{\Theta} \in \mathbb{R}^{L_{binder} \times 20}$（即我们要幻觉生成的 Binder 区域的 Logits，设置 `requires_grad=True`，$L_{binder}$ 为 Binder 序列长度，20 为氨基酸种类）。引入平滑温度超参数 $\tau_{temp}$，转化为连续的软概率矩阵 $\mathbf{P}$：

$$
\mathbf{P} = \text{Softmax}(\mathbf{\Theta} / \tau_{temp}) \quad  \in \mathbb{R}^{L_{binder} \times 20}
$$

1. **硬采样 (Hard Pass)**：

取概率最大值对应的离散序列 $\mathbf{a}_{hard} = \argmax(\mathbf{P})$*，将其输入给官方的* $text_{InputEmbedder}$*。此举是为了获取本轮绝对合法（但无计算图梯度）的真实单体特征* $\mathbf{s}^{init}_{hard}  \in \mathbb{R}^{L_{binder} \times c_s}$ 和合法的系统变长原子总数 $N_{atom}$。

1. **软特征构建 (Soft Pass)**：

离线预计算一个常量特征字典 $\mathbf{W}_{AAE}  \in \mathbb{R}^{20 \times c_s}$*（由 20 种单体氨基酸通过 AtomAttentionEncoder 提取的基础特征）。利用连续概率与字典相乘，构建带连续梯度的平滑特征* $\mathbf{s}^{init}_{soft}$：

$$
\mathbf{s}^{init}_{soft} = \mathbf{P} \cdot \mathbf{W}_{AAE} \quad  \in \mathbb{R}^{L_{binder} \times c_s}
$$

1. **直通估计器 (STE) 截流**:

利用数值截断拼接出送入主干网络的 Binder 特征 $\tilde{\mathbf{s}}^{init}  \in \mathbb{R}^{L_{binder} \times c_s}$：

$$
\tilde{\mathbf{s}}^{init} = \mathbf{s}^{init}_{hard}.\text{detach}() + \mathbf{s}^{init}_{soft} - \mathbf{s}^{init}_{soft}.\text{detach}()
$$

*(机制解析：此操作使前向传播的数值完美等于合法的* $\mathbf{s}^{init}_{hard}$*，骗过了 AF3 变长原子的严苛数据校验；而反向求导时梯度则被强制绕道进入* $\mathbf{s}^{init}_{soft}$*，接通了网络到底层 Logits 的血脉)。*

将融合后的 Binder 特征与 Target 特征拼接（总长度恢复为 $N_{token}$），送入连续可导的主干网络：

$$
\mathbf{s}, \mathbf{z} = \text{Pairformer}(\tilde{\mathbf{s}}^{init}, \mathbf{z}^{init})
$$

### 模块 B：截断式单步 $\hat{\mathbf{x}}_0$ 外推预测 (对应 CV: Universal Guidance)

*(解决问题：规避数十步 Diffusion ODE 循环迭代带来的 OOM 与梯度粉碎)*

在选定的特定微小噪声时间步标量 $\tau$（如 $\tau=0.1$），将对应的带噪 3D 骨架坐标 $\mathbf{x}_\tau  \in \mathbb{R}^{N_{atom} \times 3}$ 输入网络（该粗略骨架可由无梯度的前向 ODE 采样低成本获得）。**我们不再执行循环，仅执行唯一一次前向传播**，预测出当前的去噪速度场 $\mathbf{v}_{pred}$：

$$
\mathbf{v}_{pred} = \text{Diffusion}(\mathbf{x}_\tau, \tau, \mathbf{s}, \mathbf{z})
$$

依据流匹配 (Flow Matching) 的常微分方程几何切线原理，代数外推出完全可导的、无噪声的清晰全原子坐标 $\hat{\mathbf{x}}_0$：

$$
\hat{\mathbf{x}}_0 = \mathbf{x}_\tau - \tau \cdot \mathbf{v}_{pred} \quad  \in \mathbb{R}^{N_{atom} \times 3}
$$

### 模块 C：分数蒸馏与目标 Loss 构建 (对应 CV: Score Distillation/SDS)

*(解决问题：冻结大模型权重，将其转化为单步可导的 3D 物理环境打分器)*

在具备侧链结构且全链路可导的坐标 $\hat{\mathbf{x}}_0$ 上，直接施加我们的结构学设计目标，计算标量总损失 $\mathcal{L}_{total}$*（其中* $\mathbf{x}_{target}$ 为给定的靶点热点坐标，$\lambda$ 为各项权重标量系数）：

$$
\mathcal{L}_{total} = \lambda_1 \underbrace{\text{Distance}(\hat{\mathbf{x}}_0^{(binder)}, \mathbf{x}_{target})}_{\text{Binder 距靶点热点的距离}} + \lambda_2 \underbrace{\text{Clash}(\hat{\mathbf{x}}_0)}_{\text{避免空间物理位阻}} - \lambda_3 \underbrace{\text{iPTM}(\hat{\mathbf{x}}_0, \mathbf{s}, \mathbf{z})}_{\text{最大化网络复合物置信度}}
$$

![cb70a3df24c1251e53c257bed53e3e0.png](attachment:ab9139e3-9c95-4b91-91f1-0d45a5162bff:cb70a3df24c1251e53c257bed53e3e0.png)

## 五、 逆向梯度的数学全推演 (Backward Gradient Flow)

当调用 `loss.backward()` 时，底层微积分的雅可比链式法则（Chain Rule）是如何精准穿越复杂的物理引擎，最终更新前端连续变量 $\mathbf{\Theta}$ 的？我们将全链路雅可比矩阵相乘的过程拆解如下：

**完整雅可比链条 (The Jacobian Chain)：**

$$
\frac{\partial \mathcal{L}_{total}}{\partial \mathbf{\Theta}} = \underbrace{ \frac{\partial \mathcal{L}_{total}}{\partial \hat{\mathbf{x}}_0} }_{\text{物理误差层}} \cdot \underbrace{ \frac{\partial \hat{\mathbf{x}}_0}{\partial \mathbf{v}_{pred}} }_{\text{外推切线层}} \cdot \underbrace{ \frac{\partial \mathbf{v}_{pred}}{\partial \mathbf{s}} }_{\text{Diffusion 坍缩层}} \cdot \underbrace{ \frac{\partial \mathbf{s}}{\partial \tilde{\mathbf{s}}^{init}} }_{\text{Pairformer 层}} \cdot \underbrace{ \frac{\partial \tilde{\mathbf{s}}^{init}}{\partial \mathbf{P}} }_{\text{STE 旁路}} \cdot \underbrace{ \frac{\partial \mathbf{P}}{\partial \mathbf{\Theta}} }_{\text{Softmax 映射}}
$$

### 1. 物理误差产生层 (3D Space):

误差首先在单步外推出的 $\hat{\mathbf{x}}_0$ 的 $N_{atom}$ 个空间坐标上产生，这一步反映了原子的三维空间受力（如向靶点热点的引力、消除 Clash 的斥力）：

$$
\nabla_{\hat{\mathbf{x}}_0} \mathcal{L} = \frac{\partial \mathcal{L}_{total}}{\partial \hat{\mathbf{x}}_0} \quad  \in \mathbb{R}^{N_{atom} \times 3}
$$

### 2. 切线误差回传 (Extrapolation Layer):

根据代数外推方程 $\hat{\mathbf{x}}_0 = \mathbf{x}_\tau - \tau \cdot \mathbf{v}_{pred}$*，对速度场* $\mathbf{v}_{pred}$ 求偏导会产生一个 $-\tau$ 的标量缩放系数，将绝对坐标误差转化为网络直接输出的速度场误差：

$$
\nabla_{\mathbf{v}_{pred}} \mathcal{L} = -\tau \cdot \nabla_{\hat{\mathbf{x}}_0} \mathcal{L} \quad  \in \mathbb{R}^{N_{atom} \times 3}
$$

### 3. 跨维度的空间物理坍缩 (Diffusion: Atom $\rightarrow$ Token):

**【团队需重点关注底层自动对齐机制】** AF3 的 Diffusion 模块内部核心包含 `Atom-to-Token Attention` 算子。在计算雅可比矩阵转置乘法时，PyTorch 底层的 Autograd 引擎会根据原子本身的物理氨基酸归属索引，自动调用 **Scatter-Add（散点求和）** 操作。

系统会将 $N_{atom}$ 维（变长原子级）的 3D 空间受力梯度，在数学上绝对精确地坍缩 (Collapse) 并累加为 $N_{token}$ 维（残基级、总序列长度）的一维特征梯度：

$$
\nabla_{\mathbf{s}} \mathcal{L} = \left( \frac{\partial \mathbf{v}_{pred}}{\partial \mathbf{s}} \right)^T \nabla_{\mathbf{v}_{pred}} \mathcal{L} \quad  \in \mathbb{R}^{N_{token} \times c_s}
$$

### 4. 穿透主干网络 (Pairformer: Token $\rightarrow$ Token):

坍缩后的残基级特征梯度，安全穿越 48 层连续的 Pairformer 矩阵网络，到达我们设置的 STE 融合节点口：

$$
\nabla_{\tilde{\mathbf{s}}^{init}} \mathcal{L} = \left( \frac{\partial \mathbf{s}}{\partial \tilde{\mathbf{s}}^{init}} \right)^T \nabla_{\mathbf{s}} \mathcal{L} \quad  \in \mathbb{R}^{N_{token} \times c_s}
$$

### 5. 穿透 STE 旁路并更新 Logits (Token $\rightarrow$ Probabilities):

由于 `.detach()` 彻底阻断了硬编码（$\mathbf{s}^{init}_{hard}$*）的不可导死路，梯度的* $100\%$ *必须流向我们构建的软特征分支* $\mathbf{s}^{init}_{soft} = \mathbf{P} \cdot \mathbf{W}_{AAE}$：

*(注意：我们需要从长度为* $N_{token}$ *的全局特征梯度中，切片提取出仅属于待设计 Binder 的前* $L_{binder}$ *个残基的特征梯度* $\nabla_{\tilde{\mathbf{s}}^{init}}^{binder} \mathcal{L}  \in \mathbb{R}^{L_{binder} \times c_s}$*，再参与运算)*

$$
\nabla_{\mathbf{P}} \mathcal{L} = \nabla_{\tilde{\mathbf{s}}^{init}}^{binder} \mathcal{L} \cdot \mathbf{W}_{AAE}^T \quad  \in \mathbb{R}^{L_{binder} \times 20}
$$

最后，乘上 Softmax 函数的雅可比矩阵，成功获得并更新我们的源头幻觉种子 Logits $\mathbf{\Theta}$：

$$
\nabla_{\mathbf{\Theta}} \mathcal{L} = \left( \frac{\partial \mathbf{P}}{\partial \mathbf{\Theta}} \right)^T \nabla_{\mathbf{P}} \mathcal{L} \quad  \in\mathbb{R}^{L_{binder} \times 20}
$$

### 6. 参数优化闭环 (Optimization Step)：

利用学习率为 $\eta$ 的 Adam 优化器，执行一维序列连续分布的迭代闭环：

$$
\mathbf{\Theta}_{new} \leftarrow \mathbf{\Theta}_{old} - \eta \cdot \text{Adam}(\nabla_{\mathbf{\Theta}} \mathcal{L})
$$

---

## 总结

<aside>
💡

**BindCraft v2 架构核心总结**

针对 AlphaFold 3 (AF3) 底层物理架构的根本性变化，BindCraft v2 成功打通了从三维物理约束到一维氨基酸序列的逆向优化闭环，核心突破如下：

1. **解决离散不可导问题（输入端重构）**：创新性地利用直通估计器（STE）和连续概率矩阵，成功跨越了 AF3 查表动态生成变长物理原子导致的绝对不可导鸿沟，使前向数据合法，反向梯度畅通。
2. **解决显存与梯度衰减问题（输出端截断）**：截断了漫长的扩散（Diffusion）计算流程，仅在特定微小噪声步进行单次外推预测计算损失，极大提高了优化效率并有效防止了梯度爆炸与高频干扰。
3. **全链路微积分打通**：通过精确的数学推演，验证了梯度从物理空间的原子级误差、经特征坍缩至残基级，并穿越 48 层主干网络，最终成功更新输入序列 Logits 的可行性。

总而言之，BindCraft v2 巧妙地将冻结权重的 AF3 大模型转化为了一个单步可导的三维“物理打分器”，为结构学主导的蛋白质序列幻觉生成提供了一条高效、连续的演化路径。

</aside>

## 六、一阶梯度引导的零阶离散搜索 (Gradient-Guided Zeroth-Order Search)

在前述章节中，我们使用 One-Step 扩散截断与 STE 旁路回传来近似获得梯度 $\nabla_{\mathbf{\Theta}}\,\mathcal{L}$。但在蛋白质序列设计这一 **离散搜索空间（规模约** $20^{L_{binder}}$**）** 中，直接依赖一阶优化器（如 Adam）在连续 Logits 空间更新，通常需要额外机制来提升稳定性与可解释性。常见的两类风险包括：

1. **梯度截断误差 (Gradient Noise)**：单步外推带来的近似误差可能使连续空间更新偏离真实物理可行集合。
2. **连续-离散鸿沟 (The STE Gap)**：连续概率空间的下降趋势，不必然在投影回离散序列（如 `argmax`）后保持一致。

因此，这里采用 **“一阶提议 + 零阶验证” (1st-Order Proposal, 0th-Order Verification)** 的混合框架：

- 一阶梯度用于提出少量更有希望的候选突变。
- 零阶前向推断用于在离散物理空间内做更严格的筛选与接受。

### 1. 核心动机

- **一阶梯度（First-Order）：降维与定向**
    - 在长序列上，纯随机突变的效率通常较低。
    - 梯度可用于估计位点敏感性，以及候选氨基酸替换对损失的影响趋势。
- **零阶验证（Zeroth-Order）：离散物理空间验收**
    - 将 AF3 的原始前向传播视为黑盒评测器（Oracle）。
    - 通过对离散候选序列做完整前向计算，得到更接近真实物理评价的损失，并据此决定是否接受。

### 2. 混合优化流程 (The Hybrid Execution Pipeline)

**Step 1：一阶反向传播得到“突变潜力图” (First-Order Proposal)**

- 得到梯度矩阵：$\mathbf{G}=\nabla_{\mathbf{\Theta}}\,\mathcal{L}\in\mathbb{R}^{L_{binder}\times 20}$。
- 将其作为“突变潜力”的启发式指标，用于缩小候选空间。

**Step 2：梯度引导的离散候选生成 (Gradient-Guided Candidate Generation)**

- 选取 Top-$K$ 个 ([位置, 氨基酸]) 候选突变。
- 基于当前离散序列组合这些突变，生成一小批候选序列（例如 $B=8$ 或 16）：
    - ${\mathbf{A}_{cand}^{(1)}, \mathbf{A}_{cand}^{(2)}, \dots, \mathbf{A}_{cand}^{(B)}}$。

**Step 3：零阶前向物理验证 (Zeroth-Order Verification - Low VRAM Cost)**

- 在 `with torch.no_grad():` 下并行评估 $B$ 个离散候选序列（此阶段不构建反向计算图）。
- 得到验证损失：

$$
\mathcal{L}_{val}^{(b)} = \text{Exact\_AF3\_Forward\_Loss}\left(\mathbf{A}_{cand}^{(b)}\right)
$$

**Step 4：状态更新与接受准则 (State Update & MCMC)**

- 将候选验证损失与当前损失 $\mathcal{L}_{current}$ 比较。
- 可选策略：
    - **贪心 (Greedy)**：若 $\min(\mathcal{L}_{val}^{(b)}) < \mathcal{L}_{current}$，则接受最佳候选并更新当前序列与 Logits。
    - **模拟退火 / Metropolis-Hastings**：以
        - $P_{accept} = \min\left(1, \exp\left(\frac{\mathcal{L}_{current} - \mathcal{L}_{val}}{T_{anneal}}\right)\right)$
        
        概率接受略差候选，用于增强探索并缓解局部最优。
        

### 3. 工程含义（简要）

- **提升对梯度近似误差的鲁棒性**：梯度用于提议候选，而非直接决定最终序列更新。
- **降低连续-离散投影不一致带来的风险**：接受与保留对象始终是合法离散序列。
- **资源使用更灵活**：每轮做一次带反传的方向估计，并利用无梯度前向并行验证多个候选。

## 七、时间步的物理频域特性与动态调度策略 (Timestep Scheduling & Escaping Local Minima)

在单步扩散外推（$\hat{\mathbf{x}}0 = \mathbf{x}_\tau - \tau \cdot \mathbf{v}_{pred}$）中，需要注意：单步回传梯度通常是近似方向信息，且不同时间步 $\tau$ 对应的梯度在尺度与物理含义上可能不同。

### 1. 误差来源与频域直觉 (Frequency Intuition)

- **较大时间步（高噪声区，如**$\tau \in [0.6, 1.0]$**）**
    - 预测结构更粗略。
    - 梯度往往更偏向全局拓扑与宏观形状约束。
    - 若在此阶段加入原子级 Clash 等高频项，可能引入较多噪声。
- **较小时间步（低噪声区，如** $\tau \in [0.01, 0.2]$**）**
    - 预测结构更清晰。
    - 梯度更偏向局部几何细节与侧链相互作用。

### 2. 分阶段时间步调度 (Coarse-to-Fine Annealing)

- **阶段 1：宏观拓扑探索（Epoch 0% - 30%）**
    - 建议 $\tau \in [0.6, 0.9]$。
    - 以距离类的粗粒度约束为主，减少原子级高频项。
- **阶段 2：界面雏形塑造（Epoch 30% - 70%）**
    - 建议 $\tau \in [0.2, 0.6]$。
    - 逐步引入 iPTM 与接触面等约束。
- **阶段 3：原子级细化（Epoch 70% - 100%）**
    - 建议 $\tau \in [0.01, 0.2]$。
    - 引入 Clash、氢键等细粒度项，对局部结构进行修正。

![843aeb8de312fd7206dfb7e3fc57556.png](attachment:e679291b-b393-4c87-a32b-9a0204acf390:843aeb8de312fd7206dfb7e3fc57556.png)

### 3. 缓解局部最优的常用策略

- **时间步抖动与多频段融合 (Timestep Jittering & Ensembling)**
    - 在每一阶段内避免固定单一 $\tau$。
    - 可在同一轮同时采样一个较大 $\tau$ 与一个较小 $\tau$，分别计算单步梯度并加权融合。
- **零阶验证中的模拟退火 (Simulated Annealing in Oracle Verification)**
    - 在候选接受环节引入温度参数 $T$。
    - 早期较高 $T$ 增强探索，后期逐步降低 $T$ 以稳定收敛。

## Appendix：AF2 vs AF3（输入端与输出端差异对照）

<aside>
📌

这一附录把 AF2 与 AF3 在**输入表示**与**输出生成范式**上的关键差异做成对照表，便于快速定位“为什么 AF3 会让逆向梯度生成变难”。

</aside>

| **核心对比维度** | **AlphaFold 2 (AF2)** | **AlphaFold 3 (AF3)** | **本质上的设计代差** |
| --- | --- | --- | --- |
| **基础定位** | 仅限蛋白质（高度特化） | 蛋白质、核酸、配体、离子等（更通用） | 从“专精蛋白质”到“通用生物体系” |
| **一｜输入端：从离散序列到连续空间** |  |  |  |
| **输入的原始形态** | 氨基酸序列字符串（如 `A-R-N-D...`） | 序列字符串 + 小分子 SMILES 表达式 |  |
| **底层的物理视角** | **氨基酸残基级**（Residue-level） | **全原子级**（All-atom level） | AF2 的最小单位是“残基”。AF3 将对象还原为更底层的“原子”。 |
| **离散化操作次数**
*(进入 Embedding 前)* | **1 次**（单维粗粒度映射） | **1 次**（多维图谱展开映射） | 两者进入连续空间前都只有 1 次离散化，但离散化对象的粒度完全不同。 |
| **离散化的具体内容** | 查阅氨基酸表，将序列映射为 `0~21` 的离散残基 ID。 | 查阅化学组件字典（CCD），显式展开为原子图谱，并提取元素类型 ID、电荷、所属残基 ID 等离散属性。 | AF2 提取“词汇（残基）”。AF3 提取“字母（物理化学属性）”。 |
| **进入连续空间后的操作**
*(Embedding 后)* | 残基 ID Embedding 为连续残基特征后，直接进入主干网络（Evoformer）。 | 原子特征 Embedding 为连续原子向量后，还需要通过 `AtomAttentionEncoder` 池化聚合为 Token 向量，再进入主干网络（Pairformer）。 | AF3 在连续空间增加了 “Atom-to-Token” 的降维，以控制原子规模带来的算力压力。 |
| **二｜输出端：预测属性与 3D 坐标生成** |  |  |  |
| **3D 生成核心模块** | 结构模块（Structure Module） | 扩散模块（Diffusion Module） | 结构生成范式从“几何学拼接”切换到“生成式去噪”。 |
| **直接预测的原子属性**
*(核心区别)* | **局部刚体框架 + 偏转角**
预测主链刚体框架（旋转与平移）与侧链扭转角（$\chi$ angles）。
*(不直接预测坐标)* | **绝对三维坐标**（$x,y,z$）
将 Token 特征广播回每个原子后，直接预测各原子的绝对坐标。 | 这是输出端最大的代差：AF2 优化角度参数。AF3 直接生成位置点云。 |
| **最终坐标生成机制** | **硬编码公式推导（搭积木）**：用理想键长、键角等规则，通过几何运动学拼接得到坐标。 | **扩散模型直接生成（捏泥人）**：从高斯噪声出发，迭代去噪得到坐标。 | AF2 依赖人类规则补全。AF3 依赖网络从数据中学习隐式化学约束。 |
| **对先验化学规则依赖** | 强依赖：遇到超出预设规则的小分子或修饰会受限。 | 低依赖：不需要为每一种分子显式写出几何拼接规则。 |  |

### 核心研究逻辑提炼

1. **关于“离散化”次数（输入端数据流）**
    - **AF2：** `字符` →（1 次映射，离散）→ `残基 ID` →（Embedding，连续）→ `残基向量` → 进入主干网络
    - **AF3：** `字符/SMILES` →（1 次查字典，离散）→ `原子级离散属性` →（Embedding，连续）→ `原子向量` →（聚合降维，连续）→ `Token 向量` → 进入主干网络
2. **关于“预测属性”（输出端生成范式）**
    - **AF2 更像“填空题”：** 预测偏转角，坐标由人类规则补全。
    - **AF3 更像“生成题”：** 直接生成三维坐标点云，适配更广泛的生物与化学对象。