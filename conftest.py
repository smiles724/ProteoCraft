"""
Root conftest: stub out heavy optional dependencies so unit tests for
bindcraft.losses can be collected without needing protenix / ml_collections.
"""
import sys
import types


def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = []  # mark as package
    sys.modules[name] = mod
    return mod


for _pkg in [
    "ml_collections",
    "ml_collections.config_dict",
    "protenix",
    "protenix.model",
    "protenix.model.sample_confidence",
]:
    if _pkg not in sys.modules:
        _make_stub(_pkg)

# Provide the specific symbol imported as `from ml_collections.config_dict import ConfigDict`
sys.modules["ml_collections.config_dict"].ConfigDict = dict

# Provide logits_to_score used by confidence loss functions
import torch
import torch.nn.functional as _F


def _logits_to_score(logits, min_bin, max_bin, no_bins, return_prob=False):
    prob = _F.softmax(logits, dim=-1)
    bin_width = (max_bin - min_bin) / no_bins
    centers = torch.linspace(
        min_bin + 0.5 * bin_width,
        max_bin - 0.5 * bin_width,
        no_bins,
        device=logits.device,
        dtype=logits.dtype,
    )
    score = prob @ centers
    if return_prob:
        return score, prob
    return score


sys.modules["protenix.model.sample_confidence"].logits_to_score = _logits_to_score
