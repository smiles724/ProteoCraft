"""
BindCraft v2 - Hallucination 模块

基于 AlphaFold 3 / Protenix 的蛋白质序列逆向设计
"""

from .ste_bridge import (
    precompute_waa_embeddings,
    compute_soft_features,
    compute_hard_features,
    apply_ste,
    ste_bridge,
)

from .diffusion_one_step import (
    get_coarse_structure,
    one_step_denoise,
    diffusion_one_step,
    diffusion_one_step_fused,
)

from .pipeline import run_hallucination, run_hybrid_hallucination, run_mcts_hallucination
from .pipeline_cdr import run_hybrid_hallucination_cdr, run_mcts_hallucination_cdr
from .mcts_controller import MCTSNode, MCTSTree
from .mutation_proposer import propose_mutations
from .scheduler import AnnealingScheduler, TauScheduler, MacroBreathTracker

__all__ = [
    # STE Bridge
    "precompute_waa_embeddings",
    "compute_soft_features",
    "compute_hard_features",
    "apply_ste",
    "ste_bridge",
    # Diffusion One-Step
    "get_coarse_structure",
    "one_step_denoise",
    "diffusion_one_step",
    "diffusion_one_step_fused",
    # Pipeline
    "run_hallucination",
    "run_hybrid_hallucination",
    "run_mcts_hallucination",
    "run_hybrid_hallucination_cdr",
    "run_mcts_hallucination_cdr",
    "MCTSNode",
    "MCTSTree",
    # Mutation Proposer
    "propose_mutations",
    # Schedulers
    "AnnealingScheduler",
    "TauScheduler",
    "MacroBreathTracker",
]
