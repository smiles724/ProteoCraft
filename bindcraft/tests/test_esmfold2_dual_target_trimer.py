"""Homotrimer routing/identity gates; no model weights or GPU are loaded."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from bindcraft.hallucination import esmfold2_dual_target as dual
from bindcraft.hallucination import esmfold2_dual_target_stageb as stageb
from bindcraft.hallucination.esmfold2_backend import ESMFold2ProposalConfig
from bindcraft.tests.test_esmfold2_dual_target_stageb import FullBackend, PARENT, TARGETS, full_config

NEW_DEFAULTS = {"target_assembly": "monomer", "target_interface_mode": "all",
                "target_interface_contact_cutoff": 5.0, "target_interface_min_contacts": 5}


def trimer_contexts():
    return {side: {"sequence": "|".join([target["sequence"]] * 3),
                   "assembly": "homotrimer", "chain_ids": [f"{side}_{i}" for i in (1, 2, 3)]}
            for side, target in TARGETS.items()}


def trimer_runtime():
    return replace(FullBackend().config, target_assembly="homotrimer", target_interface_mode="contacted",
                   target_chain_roles=("target", "target", "target"))


def test_monomer_context_shape_and_alphabet_remain_unchanged():
    for target in TARGETS.values():
        assert dual.validate_target_context(target) == target
    with pytest.raises(ValueError, match="canonical"):
        dual.validate_target_context({"sequence": "AAA|AAA|AAA"})
    with pytest.raises(ValueError, match="metadata"):
        dual.validate_target_context({"sequence": "AAA", "assembly": "monomer"})


def test_explicit_homotrimer_is_validated_without_replicating_or_relabelling():
    runtime = SimpleNamespace(target_assembly="homotrimer", target_interface_mode="contacted")
    target = trimer_contexts()["human"]
    assert dual.validate_target_context(target, runtime) == target
    assert target["chain_ids"] == ["human_1", "human_2", "human_3"]


@pytest.mark.parametrize("change,reason", [
    ({"assembly": "monomer"}, "explicitly declare"),
    ({"sequence": "GGG|GGG"}, "exactly three"),
    ({"sequence": "GGG|GGG|SSS"}, "identical"),
    ({"sequence": "GGG||GGG"}, "canonical"),
    ({"sequence": ["GGG", "GGG", "GGG"]}, "pipe-delimited"),
    ({"chain_ids": ["a", "a", "c"]}, "unique chain_ids"),
    ({"chain_ids": ["a", "b", " "]}, "unique chain_ids"),
    ({"chain_ids": None}, "unique chain_ids"),
    ({"surprise": True}, "Unknown target context"),
])
def test_bad_trimer_contexts_fail_closed(change, reason):
    runtime = SimpleNamespace(target_assembly="homotrimer", target_interface_mode="contacted")
    with pytest.raises(ValueError, match=reason):
        dual.validate_target_context({**trimer_contexts()["human"], **change}, runtime)


def test_homotrimer_requires_contacted_policy_and_all_target_roles():
    target = trimer_contexts()["human"]
    with pytest.raises(ValueError, match="target_interface_mode=contacted"):
        dual.validate_target_context(target, {"target_assembly": "homotrimer"})
    with pytest.raises(ValueError, match="target chain role"):
        dual.validate_target_context(target, {"target_assembly": "homotrimer",
            "target_interface_mode": "contacted", "target_chain_roles": ["target", "scaffold", "target"]})


def test_trimer_contexts_reach_both_stages_preview_and_verifier_without_dropping_chains(tmp_path):
    backend = FullBackend()
    backend.config = trimer_runtime()
    targets = trimer_contexts()
    before = deepcopy(targets)
    ranked = dual.run_dual_target(backend=backend, targets=targets, seeds={"one": PARENT},
                                 config=full_config(), output_dir=tmp_path)
    expected = {targets[side]["sequence"] for side in targets}
    assert ranked and targets == before
    assert {call["target_seq"] for call in backend.propose_calls} == expected
    assert {target for _, target, _ in backend.verify_calls} == expected
    assert {target for _, target, _ in backend.preview_calls} == expected
    assert all(target.count("|") == 2 for _, target, _ in backend.verify_calls)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["targets"] == targets
    assert "esmfold2_interface.py" in manifest["source_sha256"]
    assert manifest["completed_stagea_steps"] == 2 and manifest["completed_stageb_steps"] == 3
    handoff = torch.load(tmp_path / "stagea_handoff.pt", weights_only=True)
    assert handoff["targets"] == targets
    assert set(handoff["roots"][0]["states"]) == {"human", "mouse"}


@pytest.mark.parametrize("legacy", [False, True])
def test_trimer_stageb_handoff_keeps_identity_and_rejects_relabelled_ids_or_changed_policy(tmp_path, legacy):
    backend = FullBackend()
    backend.config = trimer_runtime()
    targets = trimer_contexts()
    source = tmp_path / "source"
    dual.run_dual_target(backend=backend, targets=targets, seeds={"one": PARENT},
                        config=dual.DualTargetConfig(outer_steps=1), output_dir=source)
    if legacy:
        (source / "stagea_handoff.pt").unlink()
    config = dual.DualTargetConfig(strategy="stage_b_only", outer_steps=1)
    loaded = stageb.load_stagea_source(source, config, targets, backend.config)
    assert loaded["state_origin"] == ("sequence_only_reinitialized" if legacy else "paired_soft_logits")
    altered = deepcopy(targets)
    altered["human"]["chain_ids"] = list(reversed(altered["human"]["chain_ids"]))
    with pytest.raises(ValueError, match="target/hotspot contexts"):
        stageb.load_stagea_source(source, config, altered, backend.config)
    with pytest.raises(ValueError, match="target interface contract"):
        stageb.load_stagea_source(source, config, targets,
                                 replace(backend.config, target_interface_contact_cutoff=6.0))
    with pytest.raises(ValueError, match="target interface contract"):
        stageb.load_stagea_source(source, config, targets,
                                 replace(backend.config, target_interface_min_contacts=6))


def test_old_monomer_paired_handoff_resolves_only_new_fields_to_legacy_defaults(tmp_path):
    backend = FullBackend()
    dual.run_dual_target(backend=backend, targets=TARGETS, seeds={"one": PARENT},
                        config=dual.DualTargetConfig(outer_steps=1), output_dir=tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    checkpoint = torch.load(tmp_path / "stagea_handoff.pt", weights_only=True)
    for key in NEW_DEFAULTS:
        manifest["runtime"].pop(key)
        checkpoint["runtime"].pop(key)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    torch.save(checkpoint, tmp_path / "stagea_handoff.pt")
    receipt = json.loads((tmp_path / "stagea_handoff.json").read_text())
    receipt["checkpoint_sha256"] = hashlib.sha256((tmp_path / "stagea_handoff.pt").read_bytes()).hexdigest()
    (tmp_path / "stagea_handoff.json").write_text(json.dumps(receipt))
    config = dual.DualTargetConfig(strategy="stage_b_only", outer_steps=1)
    assert stageb.load_stagea_source(tmp_path, config, TARGETS, backend.config)["state_origin"] == "paired_soft_logits"
    manifest["runtime"].pop("verifier_num_loops")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="verifier_num_loops"):
        stageb.load_stagea_source(tmp_path, config, TARGETS, backend.config)


def cli(tmp_path, *, assembly="homotrimer", human_records=None, mouse_records=None):
    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / "configs/esmfold2_dual_target.json").read_text())
    if assembly == "homotrimer":
        payload["runtime"].update(target_assembly="homotrimer", target_interface_mode="contacted")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(payload))
    for side, records in (("human", human_records), ("mouse", mouse_records)):
        records = records if records is not None else [(f"{side}_{i}", TARGETS[side]["sequence"])
                                                     for i in (1, 2, 3)]
        (tmp_path / f"{side}.fasta").write_text("".join(f">{name}\n{sequence}\n" for name, sequence in records))
    (tmp_path / "seeds.fasta").write_text(f">seed\n{PARENT}\n")
    return subprocess.run([sys.executable, str(root / "scripts/run_esmfold2_dual_target.py"),
        "--config", str(config), "--human-fasta", str(tmp_path / "human.fasta"),
        "--mouse-fasta", str(tmp_path / "mouse.fasta"), "--seeds-fasta", str(tmp_path / "seeds.fasta"),
        "--output-dir", str(tmp_path / "output"), "--validate-only"], capture_output=True,
        text=True, timeout=30)


def test_cli_trimer_preflight_reports_chain_counts_and_policy_without_model_loading(tmp_path):
    process = cli(tmp_path)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result["target_chain_counts"] == {"human": 3, "mouse": 3}
    assert result["target_lengths"] == {"human": 150, "mouse": 150}
    assert result["target_chain_ids"]["human"] == ["human_1", "human_2", "human_3"]
    assert result["target_assembly"] == "homotrimer"
    assert result["target_interface_policy"] == {
        "mode": "contacted", "contact_cutoff_angstrom": 5.0, "min_residue_contacts": 5}
    assert not result["models_loaded"] and not (tmp_path / "output").exists()


@pytest.mark.parametrize("human,reason", [
    ([("a", "GGG")], "exactly three separate"),
    ([("a", "GGG"), ("a", "GGG"), ("c", "GGG")], "unique"),
    ([("a", "GGG"), ("b", "GGG"), ("c", "SSS")], "identical"),
    ([("a", "GGG|GGG|GGG"), ("b", "GGG"), ("c", "GGG")], "canonical"),
])
def test_cli_does_not_silently_expand_or_repair_bad_target_fastas(tmp_path, human, reason):
    process = cli(tmp_path, human_records=human)
    assert process.returncode != 0 and reason in process.stderr
    assert not (tmp_path / "output").exists()


def test_cli_default_monomer_rejects_multirecord_targets(tmp_path):
    process = cli(tmp_path, assembly="monomer")
    assert process.returncode != 0 and "exactly one record" in process.stderr
