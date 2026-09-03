import pytest
import torch
from torch import nn

from bindcraft.hallucination.esmfold2_one_step import (
    ESMFOLD2_COMMON_LOSS_WEIGHTS,
    ESMFOLD2_LEGACY_VERIFIER_LOSS_WEIGHTS,
    ESMFOLD2_VERIFIER_LOSS_WEIGHTS,
    compute_esmfold2_bounded_verifier_losses,
    compute_esmfold2_common_losses,
    remap_reference_coordinates,
    run_esmfold2_truncated_diffusion,
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


def _three_chain_features() -> dict[str, torch.Tensor]:
    return {
        "token_attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "asym_id": torch.tensor([[0, 1, 2, 2]]),
        "atom_attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "atom_to_token": torch.tensor([[0, 1, 2, 3]]),
        "distogram_atom_idx": torch.tensor([[0, 1, 2, 3]]),
        "residue_index": torch.tensor([[0, 0, 0, 1]]),
    }


def _four_chain_features() -> dict[str, torch.Tensor]:
    return {
        "token_attention_mask": torch.ones(1, 5, dtype=torch.bool),
        "asym_id": torch.tensor([[0, 1, 2, 3, 3]]),
        "atom_attention_mask": torch.ones(1, 5, dtype=torch.bool),
        "atom_to_token": torch.tensor([[0, 1, 2, 3, 4]]),
        "distogram_atom_idx": torch.tensor([[0, 1, 2, 3, 4]]),
        "residue_index": torch.tensor([[0, 0, 0, 0, 1]]),
    }


def _context_uncertain_pae_logits() -> torch.Tensor:
    logits = torch.full((1, 4, 4, 64), -20.0)
    logits[..., 0] = 20.0
    with torch.no_grad():
        for binder_index in (2, 3):
            logits[0, binder_index, 1, :] = -20.0
            logits[0, binder_index, 1, 63] = 20.0
            logits[0, 1, binder_index, :] = -20.0
            logits[0, 1, binder_index, 63] = 20.0
    return logits.requires_grad_(True)


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


def test_verifier_weights_use_bounded_full_prediction_metrics():
    expected_legacy = {
        **ESMFOLD2_COMMON_LOSS_WEIGHTS,
        "i_con": 0.5,
        "iptm": 1.0,
    }
    assert ESMFOLD2_LEGACY_VERIFIER_LOSS_WEIGHTS == expected_legacy
    assert ESMFOLD2_VERIFIER_LOSS_WEIGHTS == {
        "pair_chain_iptm_loss": 1.0,
        "binder_confidence_loss": 0.2,
        "global_confidence_loss": 0.1,
        "binder_rg_excess_loss": 0.1,
    }
    assert ESMFOLD2_COMMON_LOSS_WEIGHTS["i_con"] == 3.0
    assert ESMFOLD2_COMMON_LOSS_WEIGHTS["iptm"] == 0.3


def test_bounded_verifier_loss_is_not_affected_by_raw_helix_or_rg_scale():
    total, losses = compute_esmfold2_bounded_verifier_losses(
        {
            "pair_chain_iptm": 0.7,
            "binder_plddt": 0.8,
            "ptm": 0.9,
            "binder_rg_excess": 4.0,
            "helix": 100.0,
            "rg": 100.0,
        }
    )

    assert losses == pytest.approx(
        {
            "pair_chain_iptm_loss": 0.3,
            "binder_confidence_loss": 0.2,
            "global_confidence_loss": 0.1,
            "binder_rg_excess_loss": 1.0,
            "total": 0.45,
        }
    )
    assert total == pytest.approx(0.45)


def test_bounded_verifier_loss_accepts_percent_binder_plddt():
    total, losses = compute_esmfold2_bounded_verifier_losses(
        {
            "pair_chain_iptm": 0.75,
            "binder_plddt": 80.0,
            "ptm": 0.9,
            "binder_rg_excess": 0.5,
        }
    )

    assert losses["binder_confidence_loss"] == pytest.approx(0.2)
    assert total == pytest.approx(0.35)


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


def test_tcr_common_pae_excludes_non_target_context_chain():
    weights = {name: 0.0 for name in ESMFOLD2_COMMON_LOSS_WEIGHTS}
    weights["pae"] = 1.0
    common = {
        "coordinates": torch.zeros(1, 4, 3),
        "plddt_logits": torch.zeros(1, 4, 50),
        "distogram_logits": torch.zeros(1, 4, 4, 128),
        "features": _three_chain_features(),
        "binder_length": 2,
        "target_chain_roles": ("target", "scaffold"),
        "weights": weights,
    }

    tcr_total, tcr_losses = compute_esmfold2_common_losses(
        **common,
        pae_logits=_context_uncertain_pae_logits(),
        task_type="tcr",
    )
    cdr_total, cdr_losses = compute_esmfold2_common_losses(
        **common,
        pae_logits=_context_uncertain_pae_logits(),
        task_type="cdr_redesign",
    )
    legacy_total, legacy_losses = compute_esmfold2_common_losses(
        **common,
        pae_logits=_context_uncertain_pae_logits(),
    )

    assert tcr_losses["pae"] < cdr_losses["pae"]
    assert tcr_total.item() < 0.05
    assert cdr_total.item() > 0.25
    assert cdr_total.item() == pytest.approx(legacy_total.item())
    assert cdr_losses["pae"].item() == pytest.approx(
        legacy_losses["pae"].item()
    )


def test_denovo_common_pae_preserves_default_behavior():
    weights = {name: 0.0 for name in ESMFOLD2_COMMON_LOSS_WEIGHTS}
    weights["pae"] = 1.0
    common = {
        "coordinates": torch.zeros(1, 4, 3),
        "plddt_logits": torch.zeros(1, 4, 50),
        "pae_logits": torch.randn(1, 4, 4, 64),
        "distogram_logits": torch.zeros(1, 4, 4, 128),
        "features": _features(),
        "binder_length": 2,
        "weights": weights,
    }

    default_total, default_losses = compute_esmfold2_common_losses(**common)
    denovo_total, denovo_losses = compute_esmfold2_common_losses(
        **common,
        task_type="denovo",
    )

    assert denovo_total.item() == pytest.approx(default_total.item())
    assert denovo_losses["pae"].item() == pytest.approx(
        default_losses["pae"].item()
    )


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


def test_tcr_hotspot_indices_use_all_target_chain_offsets():
    weights = {name: 0.0 for name in ESMFOLD2_COMMON_LOSS_WEIGHTS}
    weights["i_pae"] = 1.0
    common = {
        "coordinates": torch.zeros(1, 4, 3),
        "plddt_logits": torch.zeros(1, 4, 50),
        "pae_logits": torch.zeros(1, 4, 4, 64),
        "distogram_logits": torch.zeros(1, 4, 4, 128),
        "features": _three_chain_features(),
        "binder_length": 2,
        "task_type": "tcr",
        "target_chain_roles": ("scaffold", "target"),
        "weights": weights,
        "hotspot_weight": 0.5,
    }

    total, losses = compute_esmfold2_common_losses(
        **common,
        hotspot_params={"target_hotspot_concat_indices": [1]},
    )

    assert torch.isfinite(total)
    assert torch.isfinite(losses["i_pae_hotspot"])
    with pytest.raises(ValueError, match="role=target"):
        compute_esmfold2_common_losses(
            **common,
            hotspot_params={"target_hotspot_concat_indices": [0]},
        )


def test_cdr_hotspot_indices_use_full_non_binder_chain_offsets():
    weights = {name: 0.0 for name in ESMFOLD2_COMMON_LOSS_WEIGHTS}
    weights["i_pae"] = 1.0
    common = {
        "coordinates": torch.zeros(1, 4, 3),
        "plddt_logits": torch.zeros(1, 4, 50),
        "pae_logits": torch.zeros(1, 4, 4, 64),
        "distogram_logits": torch.zeros(1, 4, 4, 128),
        "features": _three_chain_features(),
        "binder_length": 2,
        "task_type": "cdr_redesign",
        "target_chain_roles": ("scaffold", "target"),
        "weights": weights,
        "hotspot_weight": 0.5,
    }

    total, losses = compute_esmfold2_common_losses(
        **common,
        hotspot_params={"target_hotspot_concat_indices": [1]},
    )

    assert torch.isfinite(total)
    assert torch.isfinite(losses["i_pae_hotspot"])
    with pytest.raises(ValueError, match="role=target"):
        compute_esmfold2_common_losses(
            **common,
            hotspot_params={"target_hotspot_concat_indices": [0]},
        )


def test_affibody_hotspot_uses_hla_and_beta2m_offsets_but_targets_peptide():
    weights = {name: 0.0 for name in ESMFOLD2_COMMON_LOSS_WEIGHTS}
    weights["i_pae"] = 1.0
    common = {
        "coordinates": torch.zeros(1, 5, 3),
        "plddt_logits": torch.zeros(1, 5, 50),
        "pae_logits": torch.zeros(1, 5, 5, 64),
        "distogram_logits": torch.zeros(1, 5, 5, 128),
        "features": _four_chain_features(),
        "binder_length": 2,
        "task_type": "affibody_redesign",
        "target_chain_roles": ("scaffold", "scaffold", "target"),
        "weights": weights,
        "hotspot_weight": 0.5,
    }

    total, losses = compute_esmfold2_common_losses(
        **common,
        hotspot_params={"target_hotspot_concat_indices": [2]},
    )

    assert torch.isfinite(total)
    assert torch.isfinite(losses["i_pae_hotspot"])
    for scaffold_index in (0, 1):
        with pytest.raises(ValueError, match="role=target"):
            compute_esmfold2_common_losses(
                **common,
                hotspot_params={
                    "target_hotspot_concat_indices": [scaffold_index]
                },
            )


def test_tcr_interface_clash_includes_non_selected_target_chain():
    weights = {name: 0.0 for name in ESMFOLD2_COMMON_LOSS_WEIGHTS}
    weights["i_clash"] = 1.0
    common = {
        "coordinates": torch.tensor(
            [
                [
                    [100.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [20.0, 0.0, 0.0],
                ]
            ]
        ),
        "plddt_logits": torch.zeros(1, 4, 50),
        "pae_logits": torch.zeros(1, 4, 4, 64),
        "distogram_logits": torch.zeros(1, 4, 4, 128),
        "features": _three_chain_features(),
        "binder_length": 2,
        "target_chain_roles": ("target", "scaffold"),
        "weights": weights,
    }

    tcr_total, tcr_losses = compute_esmfold2_common_losses(
        **common,
        task_type="tcr",
    )
    cdr_total, cdr_losses = compute_esmfold2_common_losses(
        **common,
        task_type="cdr_redesign",
    )

    assert tcr_losses["i_clash"] > 0.0
    assert cdr_losses["i_clash"] == pytest.approx(0.0)
    assert tcr_total > cdr_total


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_remap_reference_coordinates_accepts_cuda_features_and_cpu_coordinates():
    source_features = {
        key: value.cuda()
        for key, value in _atom_layout(
            ["N", "CA", "PAD", "C", "O"],
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 2.0, 0.0],
            ],
            mask=[True, True, False, True, True],
        ).items()
    }
    destination_features = {
        key: value.cuda()
        for key, value in _atom_layout(
            ["N", "CA", "C", "O"],
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 2.0, 0.0],
            ],
            mask=[True] * 4,
        ).items()
    }
    source_coordinates = torch.tensor(
        [
            [
                [10.0, 20.0, 30.0],
                [11.0, 20.0, 30.0],
                [11.0, 21.0, 30.0],
                [11.0, 22.0, 30.0],
            ]
        ],
        device="cpu",
    )

    remapped = remap_reference_coordinates(
        source_coordinates,
        source_features=source_features,
        destination_features=destination_features,
    )

    assert remapped.device.type == "cpu"
    assert torch.equal(remapped, source_coordinates)


class _FakeDiffusionModule(nn.Module):
    def forward(self, *, x_noisy, z_trunk, **kwargs):
        signal = z_trunk.mean(dim=(1, 2, 3))[:, None, None]
        return {
            "x_denoised": 0.5 * x_noisy + signal,
            "token_repr": z_trunk.mean(dim=-2),
        }


class _FakeStructureHead:
    def __init__(self):
        self.diffusion_module = _FakeDiffusionModule()
        self.noise_scale = 1.0
        self.step_scale = 1.0
        self.gamma_min = 1000.0
        self.gamma_0 = 0.0

    def inference_noise_schedule(self, num_steps, device):
        assert num_steps == 4
        return torch.tensor([10.0, 4.0, 1.0, 0.0], device=device)

    def _center_random_augmentation(self, x, atom_mask, second_coords=None):
        return x, second_coords

    def _weighted_rigid_align(self, x, x_gt, w, mask):
        return x


class _FakeConfidenceHead(nn.Module):
    def forward(self, *, s_inputs, x_pred, **kwargs):
        n_tokens = s_inputs.shape[1]
        signal = x_pred.mean() + s_inputs.mean()
        return {
            "plddt_logits": signal.expand(1, n_tokens, 50),
            "pae_logits": signal.expand(1, n_tokens, n_tokens, 64),
        }


class _FakeExperimentalModel:
    def __init__(self):
        self.structure_head = _FakeStructureHead()
        self.confidence_head = _FakeConfidenceHead()


def _truncated_features() -> dict[str, torch.Tensor]:
    return {
        "atom_attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "ref_element": torch.tensor([[6, 6, 8]]),
        "ref_atom_name_chars": torch.zeros(1, 3, 4, dtype=torch.long),
        "atom_to_token": torch.tensor([[0, 1, 1]]),
        "ref_pos": torch.zeros(1, 3, 3),
        "ref_charge": torch.zeros(1, 3),
        "ref_space_uid": torch.zeros(1, 3, dtype=torch.long),
        "asym_id": torch.tensor([[0, 1]]),
        "residue_index": torch.tensor([[0, 0]]),
        "entity_id": torch.tensor([[0, 1]]),
        "token_index": torch.tensor([[0, 1]]),
        "sym_id": torch.zeros(1, 2, dtype=torch.long),
        "token_attention_mask": torch.ones(1, 2, dtype=torch.bool),
        "distogram_atom_idx": torch.tensor([[0, 1]]),
        "mol_type": torch.zeros(1, 2, dtype=torch.long),
    }


def test_truncated_diffusion_reaches_tau_without_tracking_trajectory_gradients():
    s_inputs = torch.randn(1, 2, 4, requires_grad=True)
    z_trunk = torch.randn(1, 2, 2, 3, requires_grad=True)
    captured = {
        "s_inputs": s_inputs,
        "z_trunk": z_trunk,
        "relative_position_encoding": torch.zeros(1, 2, 2, 3),
    }

    result = run_esmfold2_truncated_diffusion(
        model=_FakeExperimentalModel(),
        captured=captured,
        features=_truncated_features(),
        tau=2.0,
        seed=7,
        num_sampling_steps=4,
        max_inference_sigma=10.0,
    )

    assert result["truncated_steps"] == 3
    assert result["source_coordinates"].requires_grad is False
    assert result["coordinates"].requires_grad is True
    result["coordinates"].sum().backward()
    assert z_trunk.grad is not None
    assert torch.isfinite(z_trunk.grad).all()
