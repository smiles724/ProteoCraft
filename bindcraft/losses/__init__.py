"""
BindCraft v2 - Loss 函数模块（对标原版 BindCraft）
"""

from .geometry import (
    compute_rg_loss,
    compute_helix_loss,
    compute_soft_clash_loss,
)
from .confidence import (
    compute_plddt_loss,
    compute_pae_loss,
    compute_contact_loss,
    compute_iptm_loss,
    compute_pde_loss,
    run_confidence_heads,
)

def __getattr__(name):
    # ESMFold2 uses geometry losses without the optional Protenix backend.
    if name == "compute_all_losses":
        from .total_loss import compute_all_losses

        return compute_all_losses
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "compute_rg_loss",
    "compute_helix_loss",
    "compute_soft_clash_loss",
    "compute_plddt_loss",
    "compute_pae_loss",
    "compute_contact_loss",
    "compute_iptm_loss",
    "compute_pde_loss",
    "run_confidence_heads",
    "compute_all_losses",
]
