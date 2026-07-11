"""
BindCraft v2 - Hallucination 模块

基于 AlphaFold 3 / Protenix 的蛋白质序列逆向设计
"""

# Lazy re-exports (PEP 562). Importing this package (e.g. for the lightweight
# ``bindcraft.hallucination.tcr_prior`` backend) no longer eagerly imports the
# heavy protenix/diffusion modules. Each public name is resolved from its
# submodule on first access, preserving the original public API.
import importlib

# name -> submodule that provides it.
_LAZY_EXPORTS = {
    # STE Bridge
    "precompute_waa_embeddings": "ste_bridge",
    "compute_soft_features": "ste_bridge",
    "compute_hard_features": "ste_bridge",
    "apply_ste": "ste_bridge",
    "ste_bridge": "ste_bridge",
    # Diffusion One-Step
    "get_coarse_structure": "diffusion_one_step",
    "one_step_denoise": "diffusion_one_step",
    "diffusion_one_step": "diffusion_one_step",
    "diffusion_one_step_fused": "diffusion_one_step",
    # Pipeline
    "run_hallucination": "pipeline",
    "run_hybrid_hallucination": "pipeline",
    "run_mcts_hallucination": "pipeline",
    "run_hybrid_hallucination_cdr": "pipeline_cdr",
    "run_mcts_hallucination_cdr": "pipeline_cdr",
    # MCTS
    "MCTSNode": "mcts_controller",
    "MCTSTree": "mcts_controller",
    # Mutation Proposer
    "propose_mutations": "mutation_proposer",
    # Schedulers
    "AnnealingScheduler": "scheduler",
    "TauScheduler": "scheduler",
    "MacroBreathTracker": "scheduler",
}


def __getattr__(name):
    submodule = _LAZY_EXPORTS.get(name)
    if submodule is not None:
        module = importlib.import_module(f"{__name__}.{submodule}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))


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
