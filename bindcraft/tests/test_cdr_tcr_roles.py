import pytest
import torch


def _minimal_feature_dict():
    return {
        "atom_to_token_idx": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "asym_id": torch.tensor([0, 0, 1, 2], dtype=torch.long),
        "has_frame": torch.ones(4, dtype=torch.bool),
        "distogram_rep_atom_mask": torch.ones(4, dtype=torch.bool),
        "residue_index": torch.tensor([0, 1, 0, 0], dtype=torch.long),
    }


def test_type1_chain_roles_route_interface_to_target_only(monkeypatch):
    from bindcraft.losses import total_loss as total_loss_mod

    captured = {}

    def fake_contact_loss(
        distogram_logits,
        binder_token_mask,
        target_token_mask,
        distogram_params,
        mode,
        **kwargs,
    ):
        captured["mode"] = mode
        captured["target_token_mask"] = target_token_mask.detach().cpu().tolist()
        captured["row_token_mask"] = kwargs["row_token_mask"].detach().cpu().tolist()
        captured["col_token_mask"] = kwargs["col_token_mask"].detach().cpu().tolist()
        return torch.tensor(2.0)

    monkeypatch.setattr(total_loss_mod, "compute_contact_loss", fake_contact_loss)

    weights = {key: 0.0 for key in ("plddt", "pae", "i_pae", "con", "i_con", "iptm", "rg", "helix", "clash", "i_clash", "pde", "i_pde")}
    weights["i_con"] = 1.0

    loss, losses = total_loss_mod.compute_all_losses(
        x0_flat=torch.zeros(4, 3),
        plddt_logits=torch.zeros(1, 4, 10),
        pae_logits=torch.zeros(1, 4, 4, 10),
        distogram_logits=torch.zeros(4, 4, 10),
        input_feature_dict=_minimal_feature_dict(),
        L_binder=2,
        weights=weights,
        contact_params={"inter_cutoff": 8.0, "inter_num": 1, "inter_num_pos": 1, "intra_cutoff": 14.0, "intra_num": 1, "seqsep": 3},
        plddt_params={"min_bin": 0.0, "max_bin": 1.0, "no_bins": 10},
        pae_params={"min_bin": 0.0, "max_bin": 1.0, "no_bins": 10},
        distogram_params={"min_bin": 0.0, "max_bin": 1.0, "no_bins": 10},
        opt_token_mask=[False, True],
        target_chain_roles=["target", "scaffold"],
    )

    assert loss.item() == pytest.approx(2.0)
    assert losses["i_con"].item() == pytest.approx(2.0)
    assert captured["mode"] == "inter"
    assert captured["target_token_mask"] == [False, False, True, False]
    assert captured["row_token_mask"] == [False, True, False, False]
    assert captured["col_token_mask"] == [False, False, True, False]


def test_type1_chain_roles_validate_role_count():
    from bindcraft.losses.total_loss import compute_all_losses

    weights = {key: 0.0 for key in ("plddt", "pae", "i_pae", "con", "i_con", "iptm", "rg", "helix", "clash", "i_clash", "pde", "i_pde")}
    with pytest.raises(ValueError, match="target_chain_roles length"):
        compute_all_losses(
            x0_flat=torch.zeros(4, 3),
            plddt_logits=torch.zeros(1, 4, 10),
            pae_logits=torch.zeros(1, 4, 4, 10),
            distogram_logits=torch.zeros(4, 4, 10),
            input_feature_dict=_minimal_feature_dict(),
            L_binder=2,
            weights=weights,
            contact_params={},
            plddt_params={},
            pae_params={},
            distogram_params={},
            target_chain_roles=["target"],
        )
