import pytest
import torch

from bindcraft.hallucination.esmfold2_one_step import (
    ESMFOLD2_COMMON_LOSS_WEIGHTS,
    compute_esmfold2_common_losses,
    scatter_reference_coordinates,
)


def _features() -> dict[str, torch.Tensor]:
    return {
        "token_attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "asym_id": torch.tensor([[0, 0, 1, 1]]),
        "atom_attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "atom_to_token": torch.tensor([[0, 1, 2, 3]]),
        "distogram_atom_idx": torch.tensor([[0, 1, 2, 3]]),
        "residue_index": torch.tensor([[0, 1, 0, 1]]),
    }


def test_common_loss_weights_match_protenix_subset_without_pde():
    assert set(ESMFOLD2_COMMON_LOSS_WEIGHTS) == {
        "plddt",
        "pae",
        "i_pae",
        "con",
        "i_con",
        "iptm",
        "rg",
        "helix",
        "clash",
        "i_clash",
    }
    assert ESMFOLD2_COMMON_LOSS_WEIGHTS["i_con"] == 3.0
    assert "pde" not in ESMFOLD2_COMMON_LOSS_WEIGHTS
    assert "i_pde" not in ESMFOLD2_COMMON_LOSS_WEIGHTS


def test_common_losses_are_finite_and_differentiable():
    coordinates = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
            ]
        ],
        requires_grad=True,
    )
    plddt_logits = torch.zeros(1, 4, 50, requires_grad=True)
    pae_logits = torch.zeros(1, 4, 4, 64, requires_grad=True)
    distogram_logits = torch.zeros(1, 4, 4, 128, requires_grad=True)
    distogram_losses = {
        "intra_contact_loss": distogram_logits[..., :8].mean(),
        "inter_contact_loss": distogram_logits[..., 8:16].mean(),
    }

    total, losses = compute_esmfold2_common_losses(
        coordinates=coordinates,
        plddt_logits=plddt_logits,
        pae_logits=pae_logits,
        distogram_logits=distogram_logits,
        features=_features(),
        binder_length=2,
        distogram_losses=distogram_losses,
    )
    assert torch.isfinite(total)
    assert {
        "plddt",
        "pae",
        "i_pae",
        "con",
        "i_con",
        "iptm",
        "rg",
        "helix",
        "clash",
        "i_clash",
        "total",
    } == set(losses)

    total.backward()
    assert coordinates.grad is not None
    assert plddt_logits.grad is not None
    assert pae_logits.grad is not None
    assert distogram_logits.grad is not None


def test_common_losses_respect_binder_mutable_mask():
    coordinates = torch.randn(1, 4, 3, requires_grad=True)
    plddt_logits = torch.randn(1, 4, 50, requires_grad=True)
    pae_logits = torch.randn(1, 4, 4, 64, requires_grad=True)
    distogram_logits = torch.randn(1, 4, 4, 128, requires_grad=True)

    total, _ = compute_esmfold2_common_losses(
        coordinates=coordinates,
        plddt_logits=plddt_logits,
        pae_logits=pae_logits,
        distogram_logits=distogram_logits,
        features=_features(),
        binder_length=2,
        binder_mutable_mask=[False, True],
        weights={"con": 0.0, "i_con": 0.0, "helix": 0.0},
    )
    total.backward()
    assert torch.isfinite(total)
    assert coordinates.grad is not None


def test_common_losses_reject_bad_mutable_mask():
    with pytest.raises(ValueError, match="binder_mutable_mask length"):
        compute_esmfold2_common_losses(
            coordinates=torch.zeros(1, 4, 3),
            plddt_logits=torch.zeros(1, 4, 50),
            pae_logits=torch.zeros(1, 4, 4, 64),
            distogram_logits=torch.zeros(1, 4, 4, 128),
            features=_features(),
            binder_length=2,
            binder_mutable_mask=[True],
        )


def test_scatter_reference_coordinates_handles_valid_atom_layout():
    atom_mask = torch.tensor([[True, True, False, True, False]])
    source = torch.tensor(
        [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]
    )
    padded = scatter_reference_coordinates(
        source,
        atom_attention_mask=atom_mask,
    )
    assert padded.shape == (1, 5, 3)
    assert torch.equal(padded[0, atom_mask[0]], source[0])
    assert torch.equal(padded[0, ~atom_mask[0]], torch.zeros(2, 3))
