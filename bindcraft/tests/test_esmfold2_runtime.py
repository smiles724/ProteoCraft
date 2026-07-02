import sys
import types
from contextlib import contextmanager

import pytest
import torch

from bindcraft.hallucination.esmfold2_backend import (
    AA_TO_ESM_IDX,
    ESMFold2ProposalConfig,
    init_binder_logits,
    sequence_from_logits_argmax,
)
from bindcraft.hallucination.esmfold2_runtime import (
    ESMFold2RuntimeBackend,
    _stageb_structure_loss,
)
import bindcraft.hallucination.esmfold2_runtime as esmfold2_runtime_module


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


class FakeFoldResultBatchedPairIptm(FakeFoldResult):
    def __init__(self):
        super().__init__()
        self.pair_chains_iptm = torch.tensor(
            [
                [
                    [1.0, 0.74],
                    [0.31, 1.0],
                ]
            ]
        )


class FakeInputBuilder:
    def __init__(self):
        self.calls = []

    def fold(self, model, input_spec, **kwargs):
        self.calls.append((model, input_spec, kwargs))
        return FakeFoldResult()


class FakeBatchedPairInputBuilder(FakeInputBuilder):
    def fold(self, model, input_spec, **kwargs):
        self.calls.append((model, input_spec, kwargs))
        return FakeFoldResultBatchedPairIptm()


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


def test_runtime_verifier_extracts_batched_pair_chain_metrics():
    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(),
        device="cpu",
        verifier_model=object(),
        input_builder=FakeBatchedPairInputBuilder(),
        protein_input_cls=FakeProteinInput,
        structure_prediction_input_cls=FakeStructurePredictionInput,
    )

    result = backend.verify(binder_seq="YYY", target_seq="AAAA", seed=7)

    assert result.total_loss == pytest.approx(0.26)
    assert result.summary_confidence["pair_chain_iptm"] == pytest.approx(0.74)
    assert result.summary_confidence["esm_verify_source"] == "pair_chain_iptm"


def test_runtime_preserves_matching_free_proposal_logits_without_hard_anchor():
    generator = torch.Generator().manual_seed(11)
    proposal_logits = init_binder_logits(
        "AAA",
        mutable_mask=[True, True, True],
        random_scale=0.01,
        requires_grad=False,
        generator=generator,
    )[0]
    current_seq = sequence_from_logits_argmax(
        proposal_logits,
        current_seq="AAA",
        mutable_mask=[True, True, True],
    )

    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(),
        device="cpu",
    )
    initialized = backend._init_current_sequence_logits(
        current_seq,
        mutable_mask=[True, True, True],
        proposal_logits=proposal_logits,
    )

    assert initialized.shape == (1, 3, len(AA_TO_ESM_IDX))
    for pos, aa in enumerate(current_seq):
        aa_idx = AA_TO_ESM_IDX[aa]
        assert initialized[0, pos, aa_idx].item() == pytest.approx(
            proposal_logits[pos, aa_idx].item()
        )
        assert initialized[0, pos, aa_idx].item() < 0.5


def test_runtime_configures_official_proposal_lm_dropout(monkeypatch):
    class FakeExperimentalModel:
        last_instance = None

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.last_instance = cls()
            cls.last_instance.from_pretrained_args = args
            cls.last_instance.from_pretrained_kwargs = kwargs
            cls.last_instance.config = types.SimpleNamespace(esmc_id="fake-esmc")
            return cls.last_instance

        def load_esmc(self, esmc_id):
            self.load_esmc_call = esmc_id

        def configure_lm_dropout(self, lm_dropout, *, force_lm_dropout_during_inference):
            self.dropout_call = {
                "lm_dropout": lm_dropout,
                "force_lm_dropout_during_inference": force_lm_dropout_during_inference,
            }

        def set_kernel_backend(self, backend):
            self.kernel_backend = backend

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.eval_called = True
            return self

        def requires_grad_(self, requires_grad):
            self.requires_grad = requires_grad
            return self

    module = types.ModuleType("transformers.models.esmfold2.modeling_esmfold2_experimental")
    module.ESMFold2ExperimentalModel = FakeExperimentalModel
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    monkeypatch.setitem(sys.modules, "transformers.models", types.ModuleType("transformers.models"))
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.esmfold2",
        types.ModuleType("transformers.models.esmfold2"),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.esmfold2.modeling_esmfold2_experimental",
        module,
    )

    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(
            experimental_model_name="biohub/ESMFold2-Experimental-Fast",
            lambda_esmc=0.0,
            proposal_lm_dropout=0.5,
            force_proposal_lm_dropout=True,
        ),
        device="cpu",
    )

    backend._ensure_proposal_models_loaded()

    instance = FakeExperimentalModel.last_instance
    assert instance.from_pretrained_args == ("biohub/ESMFold2-Experimental-Fast",)
    assert instance.from_pretrained_kwargs["load_esmc"] is False
    assert instance.load_esmc_call == "fake-esmc"
    assert instance.dropout_call == {
        "lm_dropout": 0.5,
        "force_lm_dropout_during_inference": True,
    }


def test_runtime_propose_uses_injected_soft_update_and_logit_delta_candidates():
    aa_to_idx = {aa: idx for idx, aa in enumerate("ARNDCQEGHILKMFPSTWYV")}

    seen = {}

    def fake_soft_update(
        *,
        current_seq,
        target_seq,
        global_step,
        seed,
        action_group,
        mutable_mask,
        proposal_logits=None,
    ):
        seen["action_group"] = action_group
        seen["mutable_mask"] = mutable_mask
        seen["proposal_logits"] = proposal_logits
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

    assert "AYD" in candidates
    assert all(seq[0] == "A" for seq in candidates)
    assert seen["action_group"] == "interface_formation"
    assert seen["mutable_mask"] == [False, True, True]
    assert seen["proposal_logits"] is None


def test_runtime_propose_can_continue_from_saved_proposal_logits():
    aa_to_idx = {aa: idx for idx, aa in enumerate("ARNDCQEGHILKMFPSTWYV")}
    saved_logits = torch.zeros(3, 20)
    saved_logits[1, aa_to_idx["F"]] = 4.0
    seen = {}

    def fake_soft_update(
        *,
        current_seq,
        target_seq,
        global_step,
        seed,
        action_group,
        mutable_mask,
        proposal_logits=None,
    ):
        seen["proposal_logits"] = proposal_logits
        before = proposal_logits.detach().clone()
        after = before.clone()
        after[2, aa_to_idx["Y"]] = 5.0
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

    batch = backend.propose(
        current_seq="AAA",
        target_seq="TTTT",
        global_step=0,
        K_top=2,
        B_candidates=1,
        T_sample=0.01,
        max_mutations_per_seq=1,
        proposal_logits=saved_logits,
    )

    assert seen["proposal_logits"] is not None
    assert seen["proposal_logits"][1, aa_to_idx["F"]].item() == pytest.approx(4.0)
    assert batch.sequences == ["AFY"]
    assert batch.logits_after[1, aa_to_idx["F"]].item() == pytest.approx(4.0)
    assert batch.logits_after[2, aa_to_idx["Y"]].item() == pytest.approx(5.0)


def test_runtime_continuous_design_records_soft_step_metrics(monkeypatch):
    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(
            soft_steps=2,
            lambda_esmc=0.0,
            num_loops=1,
            num_sampling_steps=1,
        ),
        device="cpu",
        experimental_model=object(),
    )
    monkeypatch.setattr(backend, "_ensure_proposal_models_loaded", lambda: None)

    def fake_fold_distogram(**kwargs):
        design = kwargs["design"]
        full_len = len(kwargs["target_seq"]) + design.shape[1]
        bins = torch.arange(128, dtype=design.dtype, device=design.device)
        signal = design[..., 0].sum()
        distogram = signal * bins.reshape(1, 1, 1, -1).expand(1, full_len, full_len, 128)
        return {"distogram_logits": distogram}

    monkeypatch.setattr(backend, "_fold_distogram", fake_fold_distogram)

    batch = backend.continuous_design(
        current_seq="AAA",
        target_seq="TTTT",
        global_step=0,
        seed=5,
        mutable_mask=[True, True, True],
        proposal_logits=torch.zeros(3, 20),
    )

    assert len(batch.soft_step_metrics) == 2
    assert batch.soft_step_metrics[0]["local_step"] == 0
    assert batch.soft_step_metrics[0]["global_step"] == 0
    assert "intra_contact_loss" in batch.soft_step_metrics[0]
    assert "inter_contact_loss" in batch.soft_step_metrics[0]
    assert "glob_loss" in batch.soft_step_metrics[0]
    assert "structure_total_loss" in batch.soft_step_metrics[0]
    assert "backprop_loss" in batch.soft_step_metrics[0]
    assert "temperature" in batch.soft_step_metrics[0]


def test_fold_distogram_uses_seed_context(monkeypatch):
    entered = []

    @contextmanager
    def fake_seed_context(seed):
        entered.append(seed)
        yield

    monkeypatch.setattr(esmfold2_runtime_module, "_seed_context", fake_seed_context)

    class FakeProteinInput:
        def __init__(self, *, id, sequence, msa=None):
            self.id = id
            self.sequence = sequence
            self.msa = msa

    class FakeStructurePredictionInput:
        def __init__(self, *, sequences):
            self.sequences = sequences

    def fake_prepare(input_raw, seed=None):
        return {"dummy": torch.tensor([1.0])}, None

    module = types.ModuleType("esm.models.esmfold2")
    module.ProteinInput = FakeProteinInput
    module.StructurePredictionInput = FakeStructurePredictionInput
    module.prepare_esmfold2_input = fake_prepare
    monkeypatch.setitem(sys.modules, "esm", types.ModuleType("esm"))
    monkeypatch.setitem(sys.modules, "esm.models", types.ModuleType("esm.models"))
    monkeypatch.setitem(sys.modules, "esm.models.esmfold2", module)

    class FakeExperimentalModel:
        def __call__(self, **kwargs):
            return {"distogram_logits": torch.zeros(1, 7, 7, 128)}

    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(),
        device="cpu",
        experimental_model=FakeExperimentalModel(),
    )
    monkeypatch.setattr(
        backend,
        "_sequence_to_folding_one_hot",
        lambda sequence: torch.zeros(1, len(sequence), 33),
    )
    monkeypatch.setattr(
        backend,
        "_folding_tokens_to_aa",
        lambda: {idx: "A" for idx in range(33)},
    )

    backend._fold_distogram(
        target_seq="TTTT",
        design=torch.zeros(1, 3, 20),
        seed=123,
    )

    assert entered == [123]


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


def test_proposal_confidence_tail_steps_override_temperature_gate():
    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(
            soft_steps=30,
            track_proposal_confidence=True,
            proposal_confidence_temperature=0.01,
            proposal_confidence_tail_steps=20,
        ),
        device="cpu",
        verifier_model=object(),
        input_builder=FakeInputBuilder(),
        protein_input_cls=FakeProteinInput,
        structure_prediction_input_cls=FakeStructurePredictionInput,
    )

    tracked_steps = [
        local_step
        for local_step in range(30)
        if backend._should_track_proposal_confidence(
            track_confidence=True,
            local_step=local_step,
            temperature=1.0,
        )
    ]

    assert tracked_steps == list(range(10, 30))
