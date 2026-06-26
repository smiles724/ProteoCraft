import pytest
import torch

from bindcraft.hallucination.esmfold2_backend import ESMFold2ProposalConfig
from bindcraft.hallucination.esmfold2_runtime import (
    ESMFold2RuntimeBackend,
    _stageb_structure_loss,
)


class FakeComplex:
    def to_mmcif_string(self):
        return "data_fake\n"


class FakeFoldResult:
    def __init__(self):
        self.complex = FakeComplex()
        self.plddt = torch.tensor([0.8, 0.9])
        self.ptm = 0.72
        self.iptm = 0.61
        self.pair_chains_iptm = torch.tensor(
            [
                [1.0, 0.83],
                [0.41, 1.0],
            ]
        )


class FakeInputBuilder:
    def __init__(self):
        self.calls = []

    def fold(self, model, input_spec, **kwargs):
        self.calls.append((model, input_spec, kwargs))
        return FakeFoldResult()


class FakeProteinInput:
    def __init__(self, *, id, sequence, msa=None):
        self.id = id
        self.sequence = sequence
        self.msa = msa


class FakeStructurePredictionInput:
    def __init__(self, *, sequences):
        self.sequences = sequences


def test_runtime_verifier_extracts_pair_chain_metrics_and_mmcif():
    builder = FakeInputBuilder()
    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(),
        device="cpu",
        verifier_model=object(),
        input_builder=builder,
        protein_input_cls=FakeProteinInput,
        structure_prediction_input_cls=FakeStructurePredictionInput,
    )

    result = backend.verify(binder_seq="YYY", target_seq="AAAA", seed=7)

    assert result.total_loss == pytest.approx(0.17)
    assert result.summary_confidence["pair_chain_iptm"] == pytest.approx(0.83)
    assert result.summary_confidence["iptm"] == pytest.approx(0.61)
    assert result.summary_confidence["ptm"] == pytest.approx(0.72)
    assert result.summary_confidence["plddt"] == pytest.approx(0.85)
    assert result.summary_confidence["mmcif"] == "data_fake\n"

    _, input_spec, kwargs = builder.calls[0]
    assert [seq.id for seq in input_spec.sequences] == ["A", "B"]
    assert [seq.sequence for seq in input_spec.sequences] == ["AAAA", "YYY"]
    assert kwargs["seed"] == 7
    assert kwargs["num_loops"] == backend.verifier_num_loops
    assert kwargs["num_sampling_steps"] == backend.verifier_num_sampling_steps


def test_runtime_propose_uses_injected_soft_update_and_logit_delta_candidates():
    aa_to_idx = {aa: idx for idx, aa in enumerate("ARNDCQEGHILKMFPSTWYV")}

    seen = {}

    def fake_soft_update(*, current_seq, target_seq, global_step, seed, action_group, mutable_mask):
        seen["action_group"] = action_group
        seen["mutable_mask"] = mutable_mask
        before = torch.zeros(len(current_seq), 20)
        after = before.clone()
        after[1, aa_to_idx["Y"]] = 3.0
        after[2, aa_to_idx["F"]] = 2.0
        return before, after

    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(),
        device="cpu",
        verifier_model=object(),
        input_builder=FakeInputBuilder(),
        protein_input_cls=FakeProteinInput,
        structure_prediction_input_cls=FakeStructurePredictionInput,
        soft_update_fn=fake_soft_update,
    )

    candidates = backend.propose(
        current_seq="ALD",
        target_seq="AAAA",
        global_step=0,
        K_top=2,
        B_candidates=3,
        T_sample=0.01,
        max_mutations_per_seq=1,
        seed=5,
        action_group="interface_formation",
        mutable_mask=[False, True, True],
    )

    assert len(candidates) == 3
    assert "AYD" in candidates
    assert all(seq[0] == "A" for seq in candidates)
    assert seen["action_group"] == "interface_formation"
    assert seen["mutable_mask"] == [False, True, True]


def test_stageb_structure_loss_keeps_base_total_when_group_is_selected():
    losses = {
        "total_loss": torch.tensor(10.0),
        "inter_contact_loss": torch.tensor(3.0),
    }

    proposal_loss, source = _stageb_structure_loss(
        losses,
        "interface_formation",
        group_weight=1.0,
    )

    assert proposal_loss.item() == pytest.approx(13.0)
    assert source == "total_loss+inter_contact_loss"
