import pytest
import torch

from bindcraft.hallucination.esmfold2_one_step import (
    ESMFOLD2_COMMON_LOSS_WEIGHTS,
    ESMFOLD2_VERIFIER_LOSS_WEIGHTS,
    compute_esmfold2_common_losses,
    remap_reference_coordinates,
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


def test_verifier_weights_recalibrate_interface_selection_only():
    expected = {
        **ESMFOLD2_COMMON_LOSS_WEIGHTS,
        "i_con": 0.5,
        "iptm": 1.0,
    }
    assert ESMFOLD2_VERIFIER_LOSS_WEIGHTS == expected
    assert ESMFOLD2_COMMON_LOSS_WEIGHTS["i_con"] == 3.0
    assert ESMFOLD2_COMMON_LOSS_WEIGHTS["iptm"] == 0.3


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


def test_common_losses_support_full_verifier_64_bin_distogram():
    coordinates = torch.randn(1, 4, 3, requires_grad=True)
    plddt_logits = torch.randn(1, 4, 50, requires_grad=True)
    pae_logits = torch.randn(1, 4, 4, 64, requires_grad=True)
    distogram_logits = torch.randn(1, 4, 4, 64, requires_grad=True)
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
    assert torch.isfinite(losses["helix"])


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


def test_common_losses_localize_one_step_ipae_to_hotspot():
    coordinates = torch.randn(1, 4, 3, requires_grad=True)
    plddt_logits = torch.zeros(1, 4, 50, requires_grad=True)
    pae_logits = torch.full((1, 4, 4, 64), -20.0, requires_grad=True)
    with torch.no_grad():
        # Tokens 0-1 are target and 2-3 are binder.  Make target token 0
        # confident and target token 1 uncertain for both binder residues.
        for binder_index in (2, 3):
            pae_logits[0, binder_index, 0, 0] = 20.0
            pae_logits[0, 0, binder_index, 0] = 20.0
            pae_logits[0, binder_index, 1, 63] = 20.0
            pae_logits[0, 1, binder_index, 63] = 20.0
    distogram_logits = torch.zeros(1, 4, 4, 128, requires_grad=True)
    weights = {name: 0.0 for name in ESMFOLD2_COMMON_LOSS_WEIGHTS}
    weights["i_pae"] = 1.0

    total, losses = compute_esmfold2_common_losses(
        coordinates=coordinates,
        plddt_logits=plddt_logits,
        pae_logits=pae_logits,
        distogram_logits=distogram_logits,
        features=_features(),
        binder_length=2,
        weights=weights,
        hotspot_params={"target_hotspot_concat_indices": [0]},
        hotspot_weight=0.5,
        hotspot_interface_mode="localized",
    )

    assert losses["i_pae_hotspot"] < losses["i_pae_global"]
    assert losses["i_pae"].item() == pytest.approx(
        0.5 * (
            losses["i_pae_global"].item()
            + losses["i_pae_hotspot"].item()
        )
    )
    assert total.item() == pytest.approx(losses["i_pae"].item())
    total.backward()
    assert pae_logits.grad is not None
    assert torch.isfinite(pae_logits.grad).all()


def test_common_losses_reject_out_of_range_hotspot():
    with pytest.raises(ValueError, match="hotspot target concat indices"):
        compute_esmfold2_common_losses(
            coordinates=torch.zeros(1, 4, 3),
            plddt_logits=torch.zeros(1, 4, 50),
            pae_logits=torch.zeros(1, 4, 4, 64),
            distogram_logits=torch.zeros(1, 4, 4, 128),
            features=_features(),
            binder_length=2,
            weights={"i_pae": 1.0},
            hotspot_params={"target_hotspot_concat_indices": [2]},
            hotspot_weight=0.5,
        )


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


def _encoded_atom_name(name: str) -> list[int]:
    return [ord(char) - 32 if char != " " else 0 for char in name.ljust(4)]


def _atom_layout(
    names: list[str],
    positions: list[list[float]],
    *,
    mask: list[bool],
) -> dict[str, torch.Tensor]:
    return {
        "atom_attention_mask": torch.tensor([mask]),
        "atom_to_token": torch.zeros(1, len(names), dtype=torch.long),
        "ref_atom_name_chars": torch.tensor(
            [[_encoded_atom_name(name) for name in names]]
        ),
        "ref_element": torch.tensor(
            [[7 if name == "N" else 8 if name == "O" else 6 for name in names]]
        ),
        "ref_pos": torch.tensor([positions], dtype=torch.float32),
    }


def test_remap_reference_coordinates_rebuilds_new_side_chain_from_backbone():
    source_features = _atom_layout(
        ["N", "CA", "PAD", "C", "O"],
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
        ],
        mask=[True, True, False, True, True],
    )
    destination_features = _atom_layout(
        ["N", "CA", "C", "O", "CB", "CG"],
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
        ],
        mask=[True] * 6,
    )
    translation = torch.tensor([10.0, 20.0, 30.0])
    source_coordinates = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 2.0, 0.0],
            ]
        ]
    ) + translation

    remapped = remap_reference_coordinates(
        source_coordinates,
        source_features=source_features,
        destination_features=destination_features,
    )

    assert remapped.shape == (1, 6, 3)
    assert torch.allclose(
        remapped[0, :4],
        source_coordinates[0],
        atol=1e-5,
    )
    assert torch.allclose(
        remapped[0, 4],
        torch.tensor([1.0, 0.0, 1.0]) + translation,
        atol=1e-5,
    )
    assert torch.allclose(
        remapped[0, 5],
        torch.tensor([1.0, 0.0, 2.0]) + translation,
        atol=1e-5,
    )
