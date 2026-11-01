"""Validate the parity gate itself without loading scientific model weights."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/check_esmfold2_prior_residency.py"
spec = importlib.util.spec_from_file_location("residency_check", SCRIPT)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def test_strict_comparison_rejects_even_small_gradient_changes():
    base = {"gradient": torch.tensor([1., 2.])}
    other = {"gradient": torch.tensor([1., 2.000001])}
    assert check.differences(base, base) == []
    assert "gradient" in check.differences(base, other)[0]
    assert check.differences(torch.tensor([float("nan")]), torch.tensor([float("nan")]))
    assert check.differences(torch.ones(2), torch.ones(2, dtype=torch.float64))


def arm(directory, policy):
    directory.mkdir()
    (directory / "search").mkdir()
    (directory / "search/optimization_report.json").write_text(json.dumps({
        "source_root": "same_source", "source_revision": "same_revision",
        "source_hashes": {"kernel": "same_hash"}, "versions": {"torch": "same_version"},
        "requested": {"proposal": "fast", "verifier": "fast", "esmc": "exact"}}))
    (directory / "CAPTURE.json").write_text(json.dumps({"status": "completed"}))
    torch.save({"runtime": {"memory_prior_residency": policy,
                           "optimization_report_path": str(directory / "report.json"),
                           "learning_rate": .05},
                "raw_gradients": [torch.ones(3)], "updates": [torch.ones(3)]}, directory / "trace.pt")
    (directory / "search/verifications.jsonl").write_text(json.dumps({
        "context": policy, "side": "human", "sequence": "AAAA", "status": "ok", "loss": .2,
        "summary_confidence": {"pair_chain_iptm": .8, "structure_path": str(directory / "s.cif"),
                               "structure_sha256": "identical_coordinates"}}) + "\n")


def test_gate_ignores_only_expected_placement_and_output_identifiers(tmp_path):
    base, other = tmp_path / "base", tmp_path / "other"
    arm(base, "per_step")
    arm(other, "proposal")
    args = SimpleNamespace(baseline=base, candidate=other, output=tmp_path / "result.json")
    check.compare(args)
    assert json.loads(args.output.read_text())["status"] == "passed"
    data = torch.load(other / "trace.pt", weights_only=True)
    data["runtime"]["learning_rate"] = .06
    torch.save(data, other / "trace.pt")
    with pytest.raises(RuntimeError, match="parity failed"):
        check.compare(args)
    assert json.loads(args.output.read_text())["status"] == "failed"


def test_read_only_hooks_preserve_updates_and_rng(monkeypatch, tmp_path):
    from bindcraft.hallucination.esmfold2_backend import ESMFold2ProposalConfig
    from bindcraft.hallucination.esmfold2_runtime import ESMFold2RuntimeBackend as Backend
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(Backend, "_init_current_sequence_logits", lambda self: torch.randn(1, 3, 20, requires_grad=True))

    def toy_propose(self, **kwargs):
        logits = self._init_current_sequence_logits()
        before = logits.detach().clone()
        optimizer = torch.optim.SGD([logits], lr=.05)
        for _ in range(3):
            loss = torch.softmax(logits, -1).square().sum()
            logits.grad = torch.autograd.grad(loss, logits)[0]
            optimizer.step()
        return SimpleNamespace(logits_before=before, logits_after=logits.detach().clone(),
            sequences=["AAA"], soft_step_metrics=[], candidate_metadata=[], proposal_diagnostics={})
    monkeypatch.setattr(Backend, "propose", toy_propose)
    torch.manual_seed(43)
    plain = Backend(config=ESMFold2ProposalConfig(), device="cpu").propose()
    expected_rng = torch.get_rng_state().clone()
    results = []
    def entrypoint(*values, **kwargs):
        results.append(Backend(config=ESMFold2ProposalConfig(), device="cpu").propose())
        directory = tmp_path / "capture/search"
        directory.mkdir()
        (directory / "manifest.json").write_text('{"status":"completed"}')
    monkeypatch.setattr(check.runpy, "run_path", entrypoint)
    before_step = torch.optim.SGD.step
    check.capture(SimpleNamespace(trace_dir=tmp_path / "capture", search_args=[], rng_seed=43,
                                  entrypoint=Path("toy.py")))
    assert torch.equal(plain.logits_after, results[0].logits_after)
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert torch.optim.SGD.step is before_step
    trace = torch.load(tmp_path / "capture/trace.pt", weights_only=True)
    assert len(trace["updates"]) == len(trace["raw_gradients"]) == 3
    assert torch.equal(trace["updates"][-1]["after"], plain.logits_after)
