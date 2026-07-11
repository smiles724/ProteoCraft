"""Smoke tests for the TCR sequence prior interface (bindcraft.hallucination.tcr_prior).

The bulk of the suite uses lightweight dummy/fake backends and needs only torch.
A set of opt-in *real-backend* tests (ESM-C / TCRLM / LigandMPNN) run only when
``RUN_REAL_TCR_PRIOR=1`` and the relevant repos/weights are available; these are
skipped by default so the suite stays fast and network-free.
"""

import os
import types

import pytest
import torch

from bindcraft.hallucination.tcr_prior import (
    AA_1_TO_IDX,
    BackendPriorScorer,
    ESMCBackend,
    LigandMPNNBackend,
    TCRLMBackend,
    TCRPriorBackend,
    TCRPriorStrictError,
    TCRSequencePriorScorerHF,
    apply_tcr_verifier_score,
    build_tcr_prior_scorer,
    maybe_build_tcr_prior_scorer,
    resolve_region_mask,
)
from bindcraft.utils import NUM_AMINO_ACIDS

_RUN_REAL = os.environ.get("RUN_REAL_TCR_PRIOR") == "1"
_real_only = pytest.mark.skipif(
    not _RUN_REAL, reason="set RUN_REAL_TCR_PRIOR=1 (and provide weights) to run real-backend tests"
)


def _make_logits(length=10, seed=0):
    torch.manual_seed(seed)
    return torch.randn(length, NUM_AMINO_ACIDS, requires_grad=True)


def test_disabled_by_default_returns_none():
    assert maybe_build_tcr_prior_scorer(None) is None
    assert maybe_build_tcr_prior_scorer({}) is None
    assert maybe_build_tcr_prior_scorer(
        {"use_tcr_prior_gradient": False, "use_tcr_prior_verifier": False}
    ) is None


def test_build_scorer_when_enabled():
    scorer = maybe_build_tcr_prior_scorer(
        {
            "use_tcr_prior_gradient": True,
            "tcr_prior_gradient_weight": 0.2,
            "use_tcr_prior_verifier": True,
            "tcr_prior_verifier_weight": 0.3,
            "tcr_prior_model_name": "dummy",
        }
    )
    assert isinstance(scorer, TCRSequencePriorScorerHF)
    assert scorer.gradient_weight == 0.2
    assert scorer.verifier_weight == 0.3


def test_soft_prior_loss_is_differentiable_and_scaled():
    scorer = build_tcr_prior_scorer(
        {"use_tcr_prior_gradient": True, "tcr_prior_gradient_weight": 0.5, "tcr_prior_model_name": "dummy"}
    )
    logits = _make_logits()
    region_mask = resolve_region_mask("all", None, logits.shape[0])
    loss = scorer.soft_prior_loss(logits, region_mask)
    assert loss.requires_grad
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_soft_prior_loss_zero_when_weight_zero():
    scorer = TCRSequencePriorScorerHF(model_name="dummy", gradient_weight=0.0)
    logits = _make_logits()
    loss = scorer.soft_prior_loss(logits, None)
    assert float(loss.detach()) == 0.0


def test_region_mask_none_zeros_all_gradient():
    scorer = build_tcr_prior_scorer(
        {"use_tcr_prior_gradient": True, "tcr_prior_gradient_weight": 1.0, "tcr_prior_model_name": "dummy"}
    )
    logits = _make_logits()
    mask = resolve_region_mask("none", None, logits.shape[0])
    loss = scorer.soft_prior_loss(logits, mask)
    assert float(loss.detach()) == 0.0


def test_score_hard_sequence_returns_float():
    scorer = build_tcr_prior_scorer(
        {"use_tcr_prior_verifier": True, "tcr_prior_verifier_weight": 1.0, "tcr_prior_model_name": "dummy"}
    )
    seq = "ACDEFGHIKL"
    mask = resolve_region_mask("all", None, len(seq))
    score = scorer.score_hard_sequence(seq, mask)
    assert isinstance(score, float)
    assert score < 0.0  # log-prob of uniform placeholder is negative


def test_resolve_region_mask_cdr_uses_cdr_mask():
    cdr_mask = [False, True, True, False, True]
    assert resolve_region_mask("cdr", cdr_mask, len(cdr_mask)) == cdr_mask
    assert resolve_region_mask("all", cdr_mask, len(cdr_mask)) == [True] * len(cdr_mask)
    assert resolve_region_mask("none", cdr_mask, len(cdr_mask)) == [False] * len(cdr_mask)


def test_resolve_region_mask_invalid_raises():
    with pytest.raises(ValueError):
        resolve_region_mask("bogus", None, 5)


def test_apply_tcr_verifier_score_updates_result():
    scorer = build_tcr_prior_scorer(
        {"use_tcr_prior_verifier": True, "tcr_prior_verifier_weight": 1.0, "tcr_prior_model_name": "dummy"}
    )
    result = types.SimpleNamespace(binder_seq="ACDEFGHIKL", total_loss=1.0, loss_dict={})
    mask = resolve_region_mask("all", None, len(result.binder_seq))
    before = result.total_loss
    out = apply_tcr_verifier_score(result, scorer, region_mask=mask)
    assert "tcr_prior_hard" in out.loss_dict
    # score is negative log-prob => loss = -score > 0 => total_loss increases
    assert out.total_loss != before
    assert out.total_loss == pytest.approx(before + out.loss_dict["tcr_prior_hard"])


def test_apply_tcr_verifier_score_noop_when_disabled():
    scorer = TCRSequencePriorScorerHF(model_name="dummy", verifier_weight=0.0)
    result = types.SimpleNamespace(binder_seq="ACDEFG", total_loss=2.0, loss_dict={})
    out = apply_tcr_verifier_score(result, scorer, region_mask=[True] * 6)
    assert out.total_loss == 2.0
    assert "tcr_prior_hard" not in out.loss_dict


# ======================================================================
# Pluggable backend tests
# ======================================================================


class _FakeBackend(TCRPriorBackend):
    """Deterministic backend: strong preference for a fixed target AA per position."""

    name = "fake"
    supports_soft = True

    def __init__(self, favored="A", available=True):
        self.favored = favored
        self._available = available

    def per_position_logprobs(self, sequence, region_indices, metadata=None):
        if not self._available:
            return None
        n = len(region_indices)
        logits = torch.zeros(n, NUM_AMINO_ACIDS)
        fav = AA_1_TO_IDX[self.favored]
        logits[:, fav] = 5.0  # peaked distribution toward favored AA
        return torch.log_softmax(logits, dim=-1)


def test_backend_soft_prior_pulls_toward_favored_aa():
    scorer = BackendPriorScorer(_FakeBackend("A"), gradient_weight=1.0)
    logits = torch.zeros(6, NUM_AMINO_ACIDS, requires_grad=True)
    loss = scorer.soft_prior_loss(logits, [True] * 6, metadata={"current_seq": "CCCCCC"})
    assert loss.requires_grad
    loss.backward()
    # gradient descent on CE should increase the favored-AA logit
    fav = AA_1_TO_IDX["A"]
    assert logits.grad[:, fav].mean() < 0  # d(loss)/d(logit_fav) negative => push up


def test_backend_soft_prior_noop_when_backend_returns_none():
    scorer = BackendPriorScorer(_FakeBackend(available=False), gradient_weight=1.0)
    logits = torch.zeros(4, NUM_AMINO_ACIDS, requires_grad=True)
    loss = scorer.soft_prior_loss(logits, [True] * 4, metadata={"current_seq": "CCCC"})
    assert float(loss.detach()) == 0.0


def test_backend_hard_score_prefers_natural_sequence():
    scorer = BackendPriorScorer(_FakeBackend("A"), verifier_weight=1.0)
    mask = [True] * 6
    # sequence of all favored AA => high naturalness => score near 0 (small NLL)
    score_good = scorer.score_hard_sequence("AAAAAA", mask)
    score_bad = scorer.score_hard_sequence("CCCCCC", mask)
    assert score_good > score_bad  # more natural => higher (less negative) score


def test_backend_region_nll_matches_manual():
    backend = _FakeBackend("A")
    idx = [0, 1, 2]
    logp = backend.per_position_logprobs("AAA", idx)
    manual = float(-logp[torch.arange(3), [AA_1_TO_IDX["A"]] * 3].mean())
    assert abs(backend.region_nll("AAA", idx) - manual) < 1e-6


def test_build_dispatch_to_esmc_falls_back_gracefully_without_esm():
    # This test targets the no-``esm`` case: the esmc dispatch must yield a
    # BackendPriorScorer that safely no-ops. When ``esm`` IS installed, building
    # the backend would download real weights, so we skip here (real ESM-C
    # behavior is validated by scripts/validate_tcr_prior_backends.py).
    import importlib.util

    if importlib.util.find_spec("esm") is not None:
        pytest.skip("esm installed; real ESM-C validated via validate_tcr_prior_backends.py")

    scorer = build_tcr_prior_scorer(
        {
            "use_tcr_prior_verifier": True,
            "tcr_prior_verifier_weight": 1.0,
            "tcr_prior_model_name": "esmc:esmc_300m",
        }
    )
    assert isinstance(scorer, BackendPriorScorer)
    assert scorer.model_name == "esmc"
    # no weights/deps => region_nll returns 0 => score 0 (no-op, pipeline safe)
    assert scorer.score_hard_sequence("ACDEFGHIKL", [True] * 10) == 0.0


def test_build_dispatch_to_tcrlm_missing_repo_is_noop():
    scorer = build_tcr_prior_scorer(
        {
            "use_tcr_prior_verifier": True,
            "tcr_prior_verifier_weight": 1.0,
            "tcr_prior_model_name": "tcrlm:/nonexistent/path/tcrLM",
        }
    )
    assert isinstance(scorer, BackendPriorScorer)
    assert scorer.model_name == "tcrlm"
    assert scorer.score_hard_sequence("ACDEFGHIKL", [True] * 10) == 0.0


def test_build_dispatch_to_ligandmpnn_missing_repo_is_noop():
    scorer = build_tcr_prior_scorer(
        {
            "use_tcr_prior_gradient": True,
            "tcr_prior_gradient_weight": 1.0,
            "tcr_prior_model_name": "ligandmpnn:/nonexistent/LigandMPNN",
        }
    )
    assert isinstance(scorer, BackendPriorScorer)
    logits = torch.zeros(5, NUM_AMINO_ACIDS, requires_grad=True)
    loss = scorer.soft_prior_loss(logits, [True] * 5)
    assert float(loss.detach()) == 0.0


def test_dummy_names_still_use_hf_placeholder_scorer():
    # "dummy"/""/"none" must keep the original placeholder scorer (no network).
    for name in ["dummy", "", "none", "null"]:
        scorer = build_tcr_prior_scorer(
            {"use_tcr_prior_verifier": True, "tcr_prior_model_name": name}
        )
        assert isinstance(scorer, TCRSequencePriorScorerHF)


# ======================================================================
# Real-backend tests (opt-in: RUN_REAL_TCR_PRIOR=1 + weights available)
# ======================================================================


@_real_only
def test_tcrlm_real_loads_and_discriminates_natural_cdr3():
    repo = os.environ.get("TCRLM_HOME")
    ckpt = os.environ.get("TCRLM_CKPT")
    if not repo or not ckpt or not os.path.isfile(ckpt):
        pytest.skip("TCRLM_HOME/TCRLM_CKPT not available")
    be = TCRLMBackend(repo_path=repo, checkpoint=ckpt, device=torch.device("cpu"))
    status = be.status()
    assert status["loaded"] and status["real_weights"], status
    # Per-position log-probs are a real distribution over the 20 canonical AAs.
    logp = be.per_position_logprobs("CASSLAPGATNEKLFF", list(range(16)))
    assert logp is not None and logp.shape == (16, NUM_AMINO_ACIDS)
    assert torch.allclose(logp.exp().sum(-1), torch.ones(16), atol=1e-3)
    # Natural CDR3 should be scored as more natural (lower NLL) than a degenerate one.
    natural = be.region_nll("CASSLAPGATNEKLFF", list(range(16)))
    unnatural = be.region_nll("MHMHMHMHMHMHMH", list(range(13)))
    assert natural < unnatural


@_real_only
def test_esmc_real_loads_and_scores():
    import importlib.util

    if importlib.util.find_spec("esm") is None:
        pytest.skip("esm package not installed")
    be = ESMCBackend(model_id=os.environ.get("ESMC_MODEL", "esmc_300m"), device=torch.device("cpu"))
    status = be.status()
    assert status["loaded"] and status["real_weights"], status
    seq = "EVQLVESGGGLVQPGGSLRLSCAAS"
    logp = be.per_position_logprobs(seq, list(range(len(seq))))
    assert logp is not None and logp.shape == (len(seq), NUM_AMINO_ACIDS)
    assert torch.allclose(logp.exp().sum(-1), torch.ones(len(seq)), atol=1e-3)


@_real_only
def test_ligandmpnn_real_soft_gradient_and_hard_score():
    repo = os.environ.get("LIGANDMPNN_HOME")
    ckpt = os.environ.get("LIGANDMPNN_CKPT")
    pdb = os.environ.get("TCR_PRIOR_PDB")
    if not (repo and ckpt and pdb) or not all(os.path.exists(p) for p in (repo, ckpt, pdb)):
        pytest.skip("LigandMPNN repo/ckpt/PDB not available")
    be = LigandMPNNBackend(repo_path=repo, checkpoint=ckpt, device=torch.device("cpu"))
    status = be.status()
    assert status["loaded"] and status["real_weights"], status
    meta = {"pdb_path": pdb}
    logp = be.per_position_logprobs("A" * 10, list(range(10)), metadata=meta)
    assert logp is not None
    assert torch.allclose(logp.exp().sum(-1), torch.ones(logp.shape[0]), atol=1e-3)


@_real_only
def test_tcrlm_strict_raises_without_checkpoint():
    repo = os.environ.get("TCRLM_HOME")
    if not repo or not os.path.isdir(repo):
        pytest.skip("TCRLM_HOME not available")
    with pytest.raises(TCRPriorStrictError):
        build_tcr_prior_scorer(
            {
                "use_tcr_prior_verifier": True,
                "tcr_prior_model_name": f"tcrlm:{repo}",
                "tcr_prior_checkpoint": "/nonexistent/ckpt.bin",
                "require_real_tcr_prior": True,
            },
            device=torch.device("cpu"),
        )
