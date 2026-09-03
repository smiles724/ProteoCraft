"""Affibody preparation and validation helpers for the ESMFold2 workflow.

The package is intentionally lightweight: importing it does not load ESMFold2,
so manifests and experimental tables can be prepared on a CPU login node.
"""

from bindcraft.affibody.profiles import (
    AFFIBODY_PROFILES,
    AffibodyProfile,
    get_affibody_profile,
)
from bindcraft.affibody.targets import (
    PMHC_TARGET_CHAIN_ROLES,
    PMHC_TARGET_CHAIN_ROLES_ARGUMENT,
    PMHCTarget,
    build_pmhc_target,
)

__all__ = [
    "AFFIBODY_PROFILES",
    "AffibodyProfile",
    "PMHC_TARGET_CHAIN_ROLES",
    "PMHC_TARGET_CHAIN_ROLES_ARGUMENT",
    "PMHCTarget",
    "build_pmhc_target",
    "get_affibody_profile",
]
