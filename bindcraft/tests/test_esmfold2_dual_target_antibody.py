"""Contract tests use synthetic masks, not an unapproved G532 CDR policy."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from bindcraft.hallucination import esmfold2_dual_target as dual
from bindcraft.hallucination.esmfold2_dual_target_constraints import validate_design, check_sequence
from bindcraft.hallucination.esmfold2_backend import ESMFold2ProposalConfig, AA_TO_ESM_IDX
from bindcraft.hallucination.esmfold2_runtime import ESMFold2RuntimeBackend
from bindcraft.tests.test_esmfold2_dual_target_stageb import FullBackend, TARGETS, events

SEQUENCE = "C" + "A" * 38 + "C"
MASK = tuple(i in range(5, 10) or i in range(20, 24) for i in range(40))


def config(**kwargs):
    return dual.DualTargetConfig(design_mode="masked_antibody", seed_cysteine_policy="preserve_fixed", **kwargs)


def backend():
    calls = []
    def update(current_seq, mutable_mask, proposal_logits, target_seq, **kwargs):
        assert tuple(mutable_mask) == MASK
        calls.append((current_seq, proposal_logits.clone(), kwargs))
        before, after = proposal_logits.clone(), proposal_logits.clone()
        # Fixed residues have the strongest misleading updates. The actual
        # runtime candidate sampler must still leave every one unchanged.
        for i in range(len(current_seq)):
            aa = "Y" if target_seq.startswith("G") else "F"
            aa = "W" if aa == current_seq[i] else aa
            after[i, AA_TO_ESM_IDX[aa]] += 100 if not MASK[i] else 4
        return before, after
    runtime = ESMFold2ProposalConfig(verifier_loss_mode="aligned_common", binder_mutable_mask=MASK)
    actual = ESMFold2RuntimeBackend(config=runtime, device="cpu", task_type="antibody", soft_update_fn=update)
    fake = FullBackend()
    actual.verify = fake.verify
    actual.preview_loss_dict = fake.preview_loss_dict
    return actual, fake, calls


def test_mask_survives_actual_sampler_both_stages_handoff_and_both_verifiers(tmp_path):
    actual, fake, calls = backend()
    ranked = dual.run_dual_target(backend=actual, targets=TARGETS, seeds={"seed": SEQUENCE},
        config=config(strategy="two_stage", outer_steps=5, stagea_steps=2), output_dir=tmp_path)
    assert len(calls) == 10
    observed = {seq for seq, _, _ in fake.verify_calls} | {seq for seq, _, _ in fake.preview_calls}
    assert len(observed) > 2
    for sequence in observed:
        check_sequence(sequence, SEQUENCE, MASK)
        assert [i for i, aa in enumerate(sequence) if aa == "C"] == [0, 39]
    for sequence, target, _ in fake.verify_calls:
        assert len({t for seq, t, _ in fake.verify_calls if seq == sequence}) == 2
    for sequence, logits, _ in calls:
        for i, mutable in enumerate(MASK):
            if not mutable:
                assert logits[i].argmax().item() == AA_TO_ESM_IDX[SEQUENCE[i]]
    # A saturated synthetic reward need not admit a child every step, but
    # exercise actual child-state propagation in each stage.
    for stage in ["stage_a", "stage_b"]:
        assert any(event["admitted"] for event in events(tmp_path) if event["stage"] == stage)
    assert all(event["preview_mode"] == "dual_worst_preview" for event in events(tmp_path)[2:])
    manifest = json.loads((tmp_path/"manifest.json").read_text())
    assert manifest["sequence_constraints"]["fixed_cysteines_1based"] == [1, 40]
    assert manifest["seed_initialization"]["seed"]["replacements"] == []
    assert ranked and actual.verifier_model is None


def test_fixed_cysteines_survive_stagea_to_separate_stageb_resume(tmp_path):
    actual, _, _ = backend()
    a = tmp_path/"a"
    dual.run_dual_target(backend=actual, targets=TARGETS, seeds={"seed": SEQUENCE},
                        config=config(outer_steps=2), output_dir=a)
    second, fake, _ = backend()
    dual.run_dual_target(backend=second, targets=TARGETS, seeds={}, stagea_results=a,
                        config=config(strategy="stage_b_only", outer_steps=2), output_dir=tmp_path/"b")
    for seq, _, _ in fake.verify_calls: check_sequence(seq, SEQUENCE, MASK)
    manifest = json.loads((tmp_path/"b/manifest.json").read_text())
    assert manifest["stagea_source"]["state_origin"] == "paired_soft_logits"
    # A different protection mask cannot inherit old logits or scores.
    changed, _, _ = backend()
    changed.config = replace(changed.config, binder_mutable_mask=tuple(i == 4 or x for i, x in enumerate(MASK)))
    with pytest.raises(ValueError, match="mask differs"):
        dual.run_dual_target(backend=changed, targets=TARGETS, seeds={}, stagea_results=a,
                            config=config(strategy="stage_b_only", outer_steps=2), output_dir=tmp_path/"bad")


@pytest.mark.parametrize("mask,reason", [
    (None,"requires runtime"), ([True]*40,"immutable positions"),
    ([False]*5+[True]*5,"length"), ([True]+list(MASK[1:]),"Cys at mutable"),
    (["false"]+list(MASK[1:]),"booleans"),
])
def test_invalid_antibody_masks_fail_before_model_calls(tmp_path, mask, reason):
    actual, fake, calls = backend()
    actual.config = replace(actual.config, binder_mutable_mask=mask)
    with pytest.raises(ValueError, match=reason):
        dual.run_dual_target(backend=actual, targets=TARGETS, seeds={"seed":SEQUENCE},
                            config=config(outer_steps=1), output_dir=tmp_path)
    assert not calls and not fake.verify_calls


def test_implicit_mask_and_cys_policy_are_rejected():
    runtime = ESMFold2ProposalConfig(verifier_loss_mode="aligned_common", binder_mutable_mask=MASK)
    with pytest.raises(ValueError, match="explicit masked_antibody"):
        validate_design(dual.DualTargetConfig(), runtime, {"seed":SEQUENCE})
    with pytest.raises(ValueError, match="preserve_fixed"):
        dual.DualTargetConfig(design_mode="masked_antibody")
    with pytest.raises(ValueError, match="Immutable position"):
        validate_design(config(), runtime, {"one":SEQUENCE,"two":SEQUENCE[:1]+"G"+SEQUENCE[2:]})


def test_api_cannot_silently_use_denovo_prior_for_antibody(tmp_path):
    actual, fake, calls = backend()
    actual.task_type = "denovo"
    with pytest.raises(ValueError, match="antibody runtime"):
        dual.run_dual_target(backend=actual, targets=TARGETS, seeds={"seed":SEQUENCE},
                            config=config(outer_steps=1), output_dir=tmp_path)
    assert not calls and not fake.verify_calls


def test_stageb_rejects_backend_mutating_immutable_residue_before_verification(tmp_path):
    actual, fake, _ = backend()
    original = actual.propose
    def bad(**kwargs):
        batch = original(**kwargs)
        if kwargs.get("action_group"):
            batch.sequences[:] = ["Y"+kwargs["current_seq"][1:]]
        return batch
    actual.propose = bad
    with pytest.raises(ValueError, match="Immutable position"):
        dual.run_dual_target(backend=actual, targets=TARGETS, seeds={"seed":SEQUENCE},
            config=config(strategy="two_stage",outer_steps=2,stagea_steps=1), output_dir=tmp_path)
    assert all(seq.startswith("C") for seq, _, _ in fake.verify_calls)
    assert json.loads((tmp_path/"manifest.json").read_text())["status"] == "failed"


def test_antibody_cli_validation_requires_explicit_mode_and_keeps_all_fixed_cys(tmp_path):
    root = Path(__file__).resolve().parents[2]
    paths = {}
    for name, seq in {"seed":SEQUENCE,"human":"G"*50,"mouse":"S"*51}.items():
        paths[name] = tmp_path/f"{name}.fasta"
        paths[name].write_text(f">{name}\n{seq}\n")
    payload = {"search":asdict(config(outer_steps=1)),"runtime":{"verifier_loss_mode":"aligned_common","binder_mutable_mask":list(MASK)}}
    conf = tmp_path/"config.json"; conf.write_text(json.dumps(payload))
    argv = [sys.executable,str(root/"scripts/run_esmfold2_dual_target_antibody.py"),"--config",str(conf),
            "--seeds-fasta",str(paths["seed"]),"--human-fasta",str(paths["human"]),"--mouse-fasta",str(paths["mouse"]),
            "--output-dir",str(tmp_path/"output"),"--validate-only"]
    run = subprocess.run(argv,text=True,capture_output=True)
    assert run.returncode == 0, run.stderr
    receipt = json.loads(run.stdout)
    assert receipt["cysteines_replaced"] == 0 and receipt["mutable_positions"] == sum(MASK)
    assert receipt["optimization_requested"] == {"proposal":"fast","verifier":"fast","esmc":"exact"}
    assert not (tmp_path/"output").exists()
