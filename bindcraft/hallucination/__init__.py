"""
BindCraft v2 - Hallucination module.

Exports are loaded lazily so optional non-Protenix backends can import their
helpers without requiring Protenix runtime modules.
"""

from __future__ import annotations

import importlib


_EXPORTS = {
    # STE Bridge
    "precompute_waa_embeddings": ".ste_bridge",
    "compute_soft_features": ".ste_bridge",
    "compute_hard_features": ".ste_bridge",
    "apply_ste": ".ste_bridge",
    "ste_bridge": ".ste_bridge",
    # Diffusion One-Step
    "get_coarse_structure": ".diffusion_one_step",
    "one_step_denoise": ".diffusion_one_step",
    "diffusion_one_step": ".diffusion_one_step",
    "diffusion_one_step_fused": ".diffusion_one_step",
    # Pipeline
    "run_hallucination": ".pipeline",
    "run_hybrid_hallucination": ".pipeline",
    "run_mcts_hallucination": ".pipeline",
    "run_hybrid_hallucination_cdr": ".pipeline_cdr",
    "run_mcts_hallucination_cdr": ".pipeline_cdr",
    # MCTS
    "MCTSNode": ".mcts_controller",
    "MCTSTree": ".mcts_controller",
    # Mutation Proposer
    "propose_mutations": ".mutation_proposer",
    # Schedulers
    "AnnealingScheduler": ".scheduler",
    "TauScheduler": ".scheduler",
    "MacroBreathTracker": ".scheduler",
}


__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(_EXPORTS[name], package=__name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
