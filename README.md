# BindCraft v2: Inference-Time Optimization for Protein Binder Design

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-green.svg)]()

**BindCraft v2** is a protein binder design framework built upon [Protenix](https://github.com/bytedance/Protenix), introducing a hybrid **1st-order + 0th-order** optimization pipeline for hallucinating target-binding protein sequences. By treating Protenix as a frozen differentiable critic, BindCraft v2 enables gradient-based sequence optimization without any fine-tuning of the base model.

---

## Table of Contents

- [Background & Motivation](#background--motivation)
- [Core Idea: Frozen Critic + Inference-Time Optimization](#core-idea-frozen-critic--inference-time-optimization)
- [Architecture: Three-Module Pipeline](#architecture-three-module-pipeline)
  - [Module A — STE Bridge (Input Discretization Bridge)](#module-a--ste-bridge-input-discretization-bridge)
  - [Module B — One-Step Diffusion Extrapolation (Output Truncation)](#module-b--one-step-diffusion-extrapolation-output-truncation)
  - [Module C — Physical Constraint Losses (Frozen Critic)](#module-c--physical-constraint-losses-frozen-critic)
- [Hybrid Search: 1st-Order Proposal + 0th-Order Verification](#hybrid-search-1st-order-proposal--0th-order-verification)
- [Timestep Scheduling: Coarse-to-Fine Annealing](#timestep-scheduling-coarse-to-fine-annealing)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Algorithm Documents](#algorithm-documents)
- [Citing](#citing)
- [License](#license)

---

## Background & Motivation

Designing a protein that binds to a specific target (e.g., an antigen, enzyme pocket, or signaling interface) is a central challenge in therapeutic protein engineering. Traditional methods rely on library screening or physics-based docking — both expensive and slow.

The key insight of **network-inversion design** is: **treat the forward structure-prediction network as a differentiable black box**. Freeze the model weights, treat the amino-acid sequence as a continuous optimizable variable, and backpropagate physical design objectives (binding distance, steric clash, structural confidence) directly through the network to update the input sequence.

This paradigm works naturally with AlphaFold 2 (AF2), which uses a residue-level representation with fixed atom counts. However, when upgrading to **AlphaFold 3 / Protenix** (AF3), two fundamental deadlocks emerge:

| | AF2 | AF3 / Protenix |
|---|---|---|
| **Input representation** | Fixed-residue slots (one ID per residue, constant atom count) | Dynamic atom lists — glycine has 4 atoms, tryptophan has 14. Total atom count is a *scalar that varies with amino-acid composition*. |
| **Output generation** | Geometry拼接 (angles → coordinates) | Diffusion-based (Gaussian noise → all-atom coordinates). |

These changes **break the gradient path** from 3D coordinates back to 1D sequence:
1. **Input deadlock**: You cannot naively backprop through a variable-size atom table.
2. **Output deadlock**: Running 50+ diffusion ODE steps in a full computation graph causes OOM and gradient degradation.

BindCraft v2 resolves both deadlocks architecturally.

---

## Core Idea: Frozen Critic + Inference-Time Optimization

BindCraft v2 inherits the **"generate → score → revise"** loop from score-distillation / SDS methods (DreamFusion, ICLR 2023), adapted for the AF3 physics model:

```
┌──────────────────────────────────────────────────────────────────────┐
│  Current Binder Sequence (logits Θ)                                  │
│       ↓                                                               │
│  ┌─────────────┐   hard sample    ┌──────────────────────────┐       │
│  │  STE Bridge  │ ───────────────→ │  Protenix Pairformer     │       │
│  │  (Module A) │  soft features   │  (48-layer, frozen)      │       │
│  └─────────────┘ ←───────────────  └──────────────────────────┘       │
│                                              ↓                         │
│                                    ┌──────────────────────┐            │
│                                    │  One-Step Diffusion   │            │
│                                    │  Extrapolation        │            │
│                                    │  (Module B, τ = 0.1)   │            │
│                                    └──────────────────────┘            │
│                                              ↓                         │
│                               ┌───────────────────────────┐           │
│                               │  Physical Constraint Loss  │           │
│                               │  Distance + Clash + iPTM   │           │
│                               │  (Module C — frozen critic)│           │
│                               └───────────────────────────┘           │
│                                              ↓                         │
│                                  ┌────────────────────┐                │
│                                  │  backward(∂L/∂Θ)   │                │
│                                  │  Adam update Θ     │                │
│                                  └────────────────────┘                │
└──────────────────────────────────────────────────────────────────────┘
```

The frozen Protenix model acts as a **single-step differentiable physics critic**. One forward + backward pass yields a tangent gradient that, in expectation, is mathematically equivalent to the steepest-descent direction for the 1D sequence.

---

## Architecture: Three-Module Pipeline

### Module A — STE Bridge (Input Discretization Bridge)

*(Maps CV: PEZ algorithm, ICLR 2023 — "soft features forward, hard tokens validated")*

**Problem**: AF3's `InputEmbedder` performs a discrete lookup that produces a variable-length atom list. Any continuous gradient is cut off here.

**Solution — Straight-Through Estimator (STE) with soft-reweighted monomer features**:

1. Define a continuous logits variable $\mathbf{\Theta} \in \mathbb{R}^{L_{binder} \times 20}$ (requires_grad=True), where 20 is the number of amino-acid types.

2. Convert to a soft probability matrix with temperature $\tau_{temp}$:
   $$\mathbf{P} = \text{Softmax}\left(\mathbf{\Theta} / \tau_{temp}\right) \in \mathbb{R}^{L_{binder} \times 20}$$

3. **Hard pass** (forward validation): `argmax(P)` gives a discrete amino-acid sequence $\mathbf{a}_{hard}$. Pass it through the official `InputEmbedder` to obtain a legitimate system atom count $N_{atom}$ and hard monomer features $\mathbf{s}^{init}_{hard}$. No gradient flows here.

4. **Soft pass** (gradient highway): Pre-compute an offline feature dictionary $\mathbf{W}_{AAE} \in \mathbb{R}^{20 \times c_s}$ (atom-attention-encoded monomer features for all 20 amino acids). Build a smooth feature:
   $$\mathbf{s}^{init}_{soft} = \mathbf{P} \cdot \mathbf{W}_{AAE}$$

5. **STE fusion** — forward uses hard features (legit shape/size), backward routes gradient through soft features:
   $$\tilde{\mathbf{s}}^{init} = \mathbf{s}^{init}_{hard}.\text{detach()} + \mathbf{s}^{init}_{soft} - \mathbf{s}^{init}_{soft}.\text{detach()}$$

The fused binder features are concatenated with the target features (total length $N_{token}$) and fed into the Pairformer.

---

### Module B — One-Step Diffusion Extrapolation (Output Truncation)

*(Maps CV: Universal Guidance, ICCV 2023 — "truncate ODE, extrapolate clean structure from noise")*

**Problem**: Running 50+ diffusion ODE steps inside a full computation graph causes OOM and gradient shattering.

**Solution — Single-step $\hat{\mathbf{x}}_0$ extrapolation**:

At a chosen small noise timestep $\tau$ (e.g., $\tau = 0.1$), take the corresponding noisy 3D backbone $\mathbf{x}_\tau$ (obtainable via a cheap no-grad forward ODE solve). Execute **exactly one** forward diffusion pass to predict the denoising velocity:

$$\mathbf{v}_{pred} = \text{Diffusion}\left(\mathbf{x}_\tau, \tau, \mathbf{s}, \mathbf{z}\right) \in \mathbb{R}^{N_{atom} \times 3}$$

Per flow-matching ODE geometry, algebraically extrapolate back to the clean all-atom structure:

$$\hat{\mathbf{x}}_0 = \mathbf{x}_\tau - \tau \cdot \mathbf{v}_{pred} \in \mathbb{R}^{N_{atom} \times 3}$$

This $\hat{\mathbf{x}}_0$ is **fully differentiable** and avoids the multi-step ODE loop entirely. The gradient is routed through $\tau$ (scaling) and $\mathbf{v}_{pred}$ back into the Pairformer.

**Why this works**: Inverse optimization needs a *direction derivative*, not a perfect sample. A single-step tangent gradient — uncontaminated by thousands of network layers — provides the highest signal-to-noise update signal.

---

### Module C — Physical Constraint Losses (Frozen Critic)

*(Maps CV: SDS / DreamFusion — "frozen model as differentiable critic")*

On the extrapolated all-atom structure $\hat{\mathbf{x}}_0$, we directly impose structural design objectives:

$$\mathcal{L}_{total} = \lambda_1 \underbrace{\text{Distance}\left(\hat{\mathbf{x}}_0^{(binder)}, \mathbf{x}_{target}\right)}_{\text{Binder-to-hotspot distance}} + \lambda_2 \underbrace{\text{Clash}\left(\hat{\mathbf{x}}_0\right)}_{\text{Steric clash penalty}} - \lambda_3 \underbrace{\text{iPTM}\left(\hat{\mathbf{x}}_0, \mathbf{s}, \mathbf{z}\right)}_{\text{Complex confidence (maximize)}}$$

This makes the frozen Protenix model behave as a **single-step differentiable 3D physics oracle**.

---

## Hybrid Search: 1st-Order Proposal + 0th-Order Verification

Pure 1st-order optimization in a discrete sequence space ($20^{L_{binder}}$ scale) has two failure modes:
1. **Gradient truncation error**: Single-step extrapolation is approximate.
2. **STE gap**: Continuous-space descent may not be preserved after `argmax` projection.

BindCraft v2 resolves this with a **1st + 0th hybrid framework**:

```
┌───────────────────────────────────────────────────────────────┐
│  1st-Order Proposal                                          │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Backprop → get ∇_Θ L                                    │  │
│  │  Use Taylor approx to rank (position, AA) mutations     │  │
│  │  Generate B candidate sequences (e.g., B = 8–16)       │  │
│  └─────────────────────────────────────────────────────────┘  │
│                          ↓                                     │
│  0th-Order Verification (no grad)                             │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Full AF3 forward for each candidate (parallel)         │  │
│  │  Evaluate exact physical loss on generated structures    │  │
│  │  Sort by real loss → keep top B                          │  │
│  └─────────────────────────────────────────────────────────┘  │
│                          ↓                                     │
│  State Update (Greedy or Metropolis-Hastings SA)            │
└───────────────────────────────────────────────────────────────┘
```

- **1st-order**: fast gradient-based proposal of promising mutations.
- **0th-order**: exact AF3 evaluation as oracle for rigorous screening.
- **Beam search**: maintains B parallel branches for implicit backtracking.

---

## Timestep Scheduling: Coarse-to-Fine Annealing

Different noise levels carry different physical frequency information:

| Phase | Epoch Range | τ Range | Focus |
|-------|-------------|---------|-------|
| 1 — Global topology | 0–30% | [0.6, 0.9] | Macroscopic shape, binder orientation |
| 2 — Interface shaping | 30–70% | [0.2, 0.6] | Contact surface, iPTM |
| 3 — Atomic refinement | 70–100% | [0.01, 0.2] | Clash, H-bonds, side-chain packing |

**Timestep jittering**: within each phase, sample multiple τ values and fuse their gradients to reduce bias.

**Simulated annealing**: Metropolis-Hastings acceptance at temperature $T$ enables controlled escape from local minima in early epochs.

---

## Project Structure

```
BindCraft/
├── bindcraft/                  # Core binder design algorithms
│   ├── hallucination/          # Optimization pipeline
│   │   ├── pipeline.py         # 3-stage design pipeline
│   │   ├── ste_bridge.py      # STE input discretization bridge
│   │   ├── diffusion_one_step.py  # One-step x̂₀ extrapolation
│   │   ├── zeroth_order.py     # 0th-order refinement
│   │   ├── mutation_proposer.py # Gradient-guided mutation proposer
│   │   └── scheduler.py        # Timestep & parameter scheduler
│   └── losses/                 # Physical constraint losses
│       ├── geometry.py         # Distance, clash, H-bond
│       ├── confidence.py       # iPTM, pLDDT
│       └── total_loss.py       # Weighted total loss
│
├── protenix/                   # Protenix foundation model
│   ├── model/                 # Model architecture
│   │   ├── protenix.py        # Full model
│   │   ├── modules/           # Transformer / Pairformer
│   │   └── layer_norm/        # Custom CUDA LayerNorm
│   ├── data/                  # Data pipeline
│   │   ├── inference/         # JSON input parser
│   │   └── msa/               # MSA processing
│   └── metrics/               # Evaluation metrics
│
├── runner/                     # Training & inference entry points
│   ├── train.py               # Training / fine-tuning
│   └── inference.py           # Single-structure inference
│
├── scripts/                    # Design & evaluation scripts
│   ├── run_hallucination.py   # Main binder design script
│   └── run_full_eval.py       # Full evaluation pipeline
│
├── configs/                    # Configuration files
│   ├── configs_model_type.py  # Model selection
│   └── configs_inference.py   # Inference config
│
├── examples/                   # Example input JSON files
├── docs/bindcraft/            # Algorithm documentation (Chinese)
│   ├── BindcraftV2算法结构.md  # Full algorithm architecture
│   └── Bindcraft v2零阶更新.md  # 0th-order optimization details
│
├── inference_demo.sh           # Inference demo
├── finetune_demo.sh           # Fine-tuning demo
└── train_demo.sh              # Training demo
```

---

## Quick Start

### 1. Installation

```bash
pip install protenix
```

### 2. Run Protein Binder Design

```bash
# Fine-tune (optional, recommended for specific targets)
bash finetune_demo.sh

# Design a binder
bash inference_demo.sh
```

### 3. Configuration

Key parameters in `configs/configs_inference.py` and `runner/train.py`:

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `tau` | Diffusion noise timestep for extrapolation | 0.05–0.2 |
| `design_steps` | Number of optimization iterations | 200–500 |
| `lambda_distance` | Weight for target-binding distance loss | 1.0–5.0 |
| `lambda_clash` | Weight for steric clash penalty | 0.5–2.0 |
| `lambda_iptm` | Weight for complex confidence (negative = maximize) | −0.5–−2.0 |
| `temperature` | Softmax temperature for STE | 0.1–0.5 |
| `n_candidates` | Number of 0th-order candidates per iteration | 8–16 |
| `beam_width` | Number of parallel beam branches | 4–8 |

---

## Algorithm Documents

Detailed algorithm documentation (in Chinese):

- **[BindcraftV2算法结构.md](docs/bindcraft/BindcraftV2%E7%AE%97%E6%B3%95%E7%BB%93%E6%9E%84.md)** — Full algorithm architecture, Jacobian chain derivation, AF2 vs AF3 comparison table.
- **[Bindcraft v2零阶更新.md](docs/bindcraft/Bindcraft%20v2%E9%9B%B6%E9%98%B6%E6%9B%B4%E6%96%B0.md)** — 0th-order refinement, beam search, simulated annealing, phase annealing scheduler.

---

## Citing

If you use BindCraft v2 in your research, please cite both BindCraft v2 and the underlying Protenix model:

```
@article{bindcraftv2,
  title  = {BindCraft v2: Hybrid 1st-Order + 0th-Order Optimization for Protein Binder Design},
  author = {Your Name and Co-Authors},
  year   = {2026}
}

@article{zhang2026protenix,
  author = {Zhang, Yuxuan and Gong, Chengyue and Zhang, Hanyu and Ma, Wenzhi and Liu, Zhenyu and others},
  title  = {Protenix-v1: Toward High-Accuracy Open-Source Biomolecular Structure Prediction},
  journal = {bioRxiv},
  year   = {2026},
  doi    = {10.64898/2026.02.05.703733}
}
```

---

## License

BindCraft v2 is released under the [Apache 2.0 License](./LICENSE). Both code and model parameters are freely available for academic research and commercial use.
