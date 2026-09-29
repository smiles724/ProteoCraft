import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from bindcraft.hallucination.esmfold2_backend import ESMFold2ProposalConfig
from bindcraft.hallucination.esmfold2_optimizations import OptimizationManager


@pytest.mark.parametrize("field,value", [
    ("proposal_optimization", "auto"), ("verifier_optimization", "unknown"),
    ("esmc_optimization", "fast"),
])
def test_invalid_optimization_config(field, value):
    with pytest.raises(ValueError):
        ESMFold2ProposalConfig(**{field: value})


def test_off_does_not_import_external_kit(monkeypatch):
    manager = OptimizationManager(ESMFold2ProposalConfig(), "cpu")
    monkeypatch.setattr(manager, "_source", lambda: pytest.fail("off imported kit"))
    model = torch.nn.Linear(2, 2)
    manager.proposal(model)
    manager.verifier(model, None)
    manager.esmc(model, "prior")
    assert manager.records == {}


def test_off_rejects_autoload_contamination(monkeypatch):
    monkeypatch.setenv("ESMFOLD2_OPT", "fast")
    with pytest.raises(RuntimeError, match="Unset ESMFOLD2_OPT"):
        OptimizationManager(ESMFold2ProposalConfig(), "cpu")


def test_timing_records_failure_without_claiming_success(tmp_path):
    path = tmp_path / "report.json"
    manager = OptimizationManager(ESMFold2ProposalConfig(
        optimization_timing=True, optimization_report_path=str(path)), "cpu")
    with pytest.raises(ValueError):
        with manager.measure("verifier"):
            raise ValueError("failed")
    event = json.loads(path.with_suffix(".timing.jsonl").read_text())
    assert event["status"] == "error"
    assert event["elapsed_s"] >= 0
    assert not path.exists()
    manager.finalize()
    assert json.loads(path.read_text())["counters"]["verifier.calls"] == 1


def test_source_refresh_resolves_each_module_filename_once(tmp_path, monkeypatch):
    source = tmp_path / "cached_source.py"
    source.write_text("VALUE = 1\n")
    manager = OptimizationManager(ESMFold2ProposalConfig(), "cpu")
    manager.root = tmp_path.resolve()
    for dependency in tuple(sys.modules.values()):
        filename = getattr(dependency, "__file__", None)
        if filename:
            manager._source_files_scanned.add(os.fspath(filename))
    monkeypatch.setitem(
        sys.modules, "_proteocraft_cached_source_test",
        SimpleNamespace(__file__=str(source)),
    )
    original = manager._resolve_source_path
    calls = []

    def resolve(filename):
        calls.append(os.fspath(filename))
        return original(filename)

    monkeypatch.setattr(manager, "_resolve_source_path", resolve)
    manager._refresh_source_hashes()
    manager._refresh_source_hashes()

    assert calls == [str(source)]
    assert "cached_source.py" in manager.source_hashes


def test_verifier_installs_explicitly_without_builder_fold(monkeypatch):
    manager = OptimizationManager(ESMFold2ProposalConfig(verifier_optimization="exact"), "cpu")
    calls = []
    module = SimpleNamespace(
        CFG=SimpleNamespace(
            graph_budget_tokens=512,
            graph_budget_tokens_trunk=-1,
            graph_budget_tokens_encoder=-1,
            graph_budget_tokens_sampler=-1,
        ),
        install=lambda model, **kw: calls.append((model, kw)) or {"trunk_graphs": 1},
    )
    monkeypatch.setattr(manager, "_require_cuda", lambda model: None)
    monkeypatch.setattr(manager, "_check_fork", lambda *names: None)
    monkeypatch.setattr(manager, "_check_core_pin", lambda kit: {"version": "test"})
    monkeypatch.setattr(manager, "_import", lambda name: module)
    model = object()
    manager.verifier(model, object())
    manager.verifier(model, object())
    assert len(calls) == 1
    assert calls[0][1]["builder"] is None
    assert calls[0][1]["esmc_cache"] is False
    assert manager.records["verifier"]["levers"]["trunk_graphs"] == 1
    assert module.CFG.graph_budget_tokens == 0
    assert module.CFG.graph_budget_tokens_trunk == 0
    assert module.CFG.graph_budget_tokens_encoder == 0
    assert module.CFG.graph_budget_tokens_sampler == 0


def test_proposal_exact_does_not_replace_gradient_trunk(monkeypatch):
    manager = OptimizationManager(ESMFold2ProposalConfig(proposal_optimization="exact"), "cpu")
    calls = []
    module = SimpleNamespace(
        enable=lambda *args, **kwargs: pytest.fail("exact replaced the gradient trunk"),
        enable_skip_unused_confidence=lambda model: calls.append(model),
    )
    monkeypatch.setattr(manager, "_require_cuda", lambda model: None)
    monkeypatch.setattr(manager, "_check_fork", lambda *names: None)
    monkeypatch.setattr(manager, "_import", lambda name: module)
    model = object()
    manager.proposal(model)
    assert calls == [model]
    assert manager.records["proposal"]["levers"] == ["skip_unused_confidence"]


def kit_root():
    root = os.environ.get("PROTEOCRAFT_UPLIFT_ROOT")
    if not root:
        pytest.skip("Set PROTEOCRAFT_UPLIFT_ROOT for real upstream integration tests")
    return root


def test_real_upstream_source_pins():
    manager = OptimizationManager(ESMFold2ProposalConfig(optimization_root=kit_root()), "cpu")
    manager._check_fork("transformers.models.esmc.modeling_esmc",
                        "transformers.models.esmfold2.modeling_esmfold2_common",
                        "transformers.models.esmfold2.modeling_esmfold2_experimental",
                        "transformers.models.esmfold2.modeling_esmfold2")
    assert len(manager._verified) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA integration test")
def test_real_esmc_attention_forward_and_input_gradient():
    from transformers.models.esmc.modeling_esmc import MultiHeadAttention
    torch.manual_seed(13)
    original = MultiHeadAttention(2560, 40).cuda().eval().requires_grad_(False)
    optimized = copy.deepcopy(original)
    manager = OptimizationManager(ESMFold2ProposalConfig(
        optimization_root=kit_root(), esmc_optimization="exact"), "cuda")
    manager.esmc(optimized, "test_esmc")
    for length in (8, 13, 8):
        x = torch.randn(1, length, 2560, device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            expected = original(x, None)[0]
            actual = optimized(x, None)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    x = torch.randn(1, 8, 2560, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a = original(x, None)[0].float().square().mean()
        b = optimized(x, None)[0].float().square().mean()
    ga = torch.autograd.grad(a, x)[0]
    gb = torch.autograd.grad(b, x)[0]
    torch.testing.assert_close(gb, ga, rtol=0, atol=0)
    assert manager.events["test_esmc.fused_qk"] == 3
    assert manager.events["test_esmc.stock_attention"] == 1
