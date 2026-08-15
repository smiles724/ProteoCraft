# BindCraft v2: Protein Binder Design via Inference-Time Optimization

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

**BindCraft v2** is a protein binder design framework built on [Protenix](https://github.com/bytedance/Protenix). It freezes the structure-prediction model and directly backpropagates 3D physical objectives (interface contact, steric clash, complex confidence) through the network to optimize input amino-acid sequences — no fine-tuning of the base model required.

---

## Core Approach

BindCraft v2 addresses the two fundamental deadlocks that make AlphaFold 3 / Protenix incompatible with naive gradient-based sequence optimization:

1. **Input deadlock** — AF3 uses variable-length atom lists (glycine: 4 atoms, tryptophan: 14 atoms), which breaks the gradient path at the atom-lookup table. BindCraft v2 bridges this with a **Straight-Through Estimator (STE)**: forward pass uses hard discrete features (legit atom counts), backward pass routes gradients through continuous soft features.

2. **Output deadlock** — AF3's diffusion module runs 50+ ODE steps; keeping the full computation graph causes OOM and gradient shattering. BindCraft v2 truncates the diffusion to a **single-step extrapolation**: at noise timestep τ, predict one velocity field and algebraically extrapolate back to a clean structure $\hat{x}_0 = x_\tau - \tau \cdot v_{pred}$. This structure is fully differentiable and sidesteps the ODE loop entirely.

On top of the 1st-order gradient, BindCraft v2 adds a **0th-order verification layer**: candidate sequences are re-evaluated with a full AF3 forward pass (exact physics, no gradient), and accepted via greedy or Metropolis-Hastings selection. This hybrid scheme mitigates STE discretization error and improves robustness.

> See **[docs/bindcraft/BindcraftV2算法结构.md](docs/bindcraft/BindcraftV2%E7%AE%97%E6%B3%95%E7%BB%93%E6%9E%84.md)** for the full algorithm derivation (Jacobian chain, AF2 vs AF3 comparison, STE mechanism, one-step extrapolation math).

> See **[docs/bindcraft/Bindcraft v2零阶更新.md](docs/bindcraft/Bindcraft%20v2%E9%9B%B6%E9%98%B6%E6%9B%B4%E6%96%B0.md)** for the 0th-order search, beam search, simulated annealing, and timestep scheduling details.

---

## Quick Start

### ESMFold2 Three-Mode Workflow

The current recommended ESMFold2 workflow supports de novo binder design, TCR binder design, and TCR CDR redesign through the shared `esmfold2_d` launch profile. See the Chinese **[ESMFold2 quick-start guide](docs/ESMFold2_QUICK_START.md)** for the default configuration and runnable examples for all three task modes.

### Installation

```bash
pip install protenix
```

### Design a Binder

Two modes are supported via `scripts/run_hallucination.py`:

**Mode A — Pure 1st-Order (STE + One-Step Diffusion)**

```bash
python scripts/run_hallucination.py \
    --target GIVEQCCTSICSLYQLENYCN \
    --binder_length 40 \
    --n_steps 200 \
    --learning_rate 0.05 \
    --tau_noise 4.0 \
    --w_i_con 3.0
```

**Mode B — Hybrid 1st + 0th (Recommended)**

Adds 0th-order verification + beam search on top of the 1st-order gradient proposals:

```bash
python scripts/run_hallucination.py \
    --target GIVEQCCTSICSLYQLENYCN \
    --binder_length 40 \
    --mode hybrid \
    --n_steps 100 \
    --learning_rate 0.05 \
    --tau_noise 4.0 \
    --k_top 10 \
    --b_candidates 8 \
    --max_mutations_per_seq 3 \
    --use_annealing true \
    --t_init 1.0 \
    --t_min 0.01 \
    --use_tau_scheduling true \
    --tau_jitter true \
    --w_i_pae 0.1 \
    --w_i_con 3.0
```

**Optional: Fine-tune for a specific target**

```bash
bash finetune_demo.sh
```

### Key Parameters

Full list: `python scripts/run_hallucination.py --help`

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--target` | Target antigen sequence (single-letter AA code) | **required** |
| `--mode` | `first_order` or `hybrid` | `first_order` |
| `--binder_length` | Binder sequence length | `40` |
| `--n_steps` | Optimization iterations | `200` |
| `--learning_rate` | Adam learning rate | `0.05` |
| `--tau_noise` | Diffusion noise timestep | `4.0` |
| `--w_i_pae` | Inter-chain PAE weight; with hotspot enabled, scores binder-hotspot pairs only | `0.1` |
| `--w_i_con` | Inter-chain contact weight; with hotspot enabled, scores binder-hotspot contacts only | `3.0` |
| `--w_clash` | Intra-binder clash penalty weight | `0.5` |
| `--w_iptm` | iPTM weight | `0.05` |
| `--w_plddt` | pLDDT weight | `0.1` |
| `--k_top` | Top-K mutations per branch (hybrid only) | `10` |
| `--b_candidates` | 0th-order candidates (hybrid only) | `8` |
| `--use_tau_scheduling` | Coarse-to-fine τ annealing | `true` |

---

## Project Structure

```
bindcraft/
├── hallucination/             # Optimization pipeline
│   ├── pipeline.py            # Main loop (1st-order + hybrid modes)
│   ├── ste_bridge.py          # STE input discretization
│   ├── diffusion_one_step.py  # One-step x̂₀ extrapolation
│   ├── zeroth_order.py        # 0th-order verification
│   ├── mutation_proposer.py   # Gradient-guided mutation proposer
│   └── scheduler.py           # Timestep & parameter scheduler
└── losses/                    # Physical constraint losses
    ├── geometry.py            # Rg, helix, clash
    ├── confidence.py          # iPTM, pLDDT
    └── total_loss.py          # Weighted total loss

protenix/                      # Protenix foundation model

scripts/
└── run_hallucination.py      # Main entry script

configs/                       # Configuration files
```

---

## Citing

```bibtex
@article{bindcraftv2,
  title  = {BindCraft v2: Hybrid 1st-Order + 0th-Order Optimization for Protein Binder Design},
  author = {Your Name and Co-Authors},
  year   = {2026}
}

@article{zhang2026protenix,
  author = {Zhang, Yuxuan and Gong, Chengyue and others},
  title  = {Protenix-v1: Toward High-Accuracy Open-Source Biomolecular Structure Prediction},
  journal = {bioRxiv},
  year   = {2026},
  doi    = {10.64898/2026.02.05.703733}
}
```

## License

Apache 2.0 — free for academic and commercial use.
