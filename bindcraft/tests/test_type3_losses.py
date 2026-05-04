import torch
import pytest


def _pmhc_feature_dict():
    return {
        "atom_to_token_idx": torch.arange(6),
        "asym_id": torch.tensor([0, 0, 1, 1, 2, 2]),
        "residue_index": torch.arange(6),
        "has_frame": torch.ones(6, dtype=torch.bool),
        "distogram_rep_atom_mask": torch.ones(6, dtype=torch.bool),
    }


def _weights(**enabled):
    keys = [
        "plddt", "pae", "i_pae", "con", "i_con", "iptm",
        "rg", "helix", "clash", "i_clash", "pde", "i_pde",
    ]
    weights = {key: 0.0 for key in keys}
    weights.update(enabled)
    return weights


def test_type3_positive_interface_uses_peptide_not_hla(monkeypatch):
    from protenix.model import sample_confidence
    import bindcraft.losses.total_loss as total_loss_mod
    from bindcraft.losses.total_loss import compute_all_losses

    def fake_logits_to_score(logits, *args, return_prob=False, **kwargs):
        score = logits.squeeze(-1)
        if return_prob:
            return score, torch.ones_like(logits)
        return score

    def fake_contact_prob(distogram_logits, **kwargs):
        return torch.ones(distogram_logits.shape[:2], device=distogram_logits.device)

    pair_vals = torch.full((6, 6), 100.0)
    for i in [0, 1]:
        for j in [2, 3]:
            pair_vals[i, j] = 2.0
            pair_vals[j, i] = 2.0
        for j in [4, 5]:
            pair_vals[i, j] = 9.0
            pair_vals[j, i] = 9.0

    def fake_contact_loss(
        distogram_logits,
        binder_token_mask,
        target_token_mask,
        distogram_params,
        mode="intra",
        row_token_mask=None,
        col_token_mask=None,
        hotspot_token_mask=None,
        **kwargs,
    ):
        if row_token_mask is not None and col_token_mask is not None:
            mask = row_token_mask.unsqueeze(1) & col_token_mask.unsqueeze(0)
        elif mode == "inter" and hotspot_token_mask is not None:
            mask = binder_token_mask.unsqueeze(1) & hotspot_token_mask.unsqueeze(0)
        elif mode == "inter":
            mask = binder_token_mask.unsqueeze(1) & target_token_mask.unsqueeze(0)
        else:
            mask = binder_token_mask.unsqueeze(1) & binder_token_mask.unsqueeze(0)
        return pair_vals[mask].mean()

    monkeypatch.setattr(sample_confidence, "logits_to_score", fake_logits_to_score)
    monkeypatch.setattr(sample_confidence, "compute_contact_prob", fake_contact_prob)
    monkeypatch.setattr(total_loss_mod, "compute_contact_loss", fake_contact_loss)

    _, losses = compute_all_losses(
        x0_flat=torch.zeros(6, 3),
        plddt_logits=torch.zeros(1, 6, 1),
        pae_logits=pair_vals.view(1, 6, 6, 1),
        distogram_logits=torch.zeros(6, 6, 2),
        input_feature_dict=_pmhc_feature_dict(),
        L_binder=2,
        weights=_weights(i_pae=1.0, i_con=1.0, i_pde=1.0),
        contact_params={"intra_cutoff": 8.0, "inter_cutoff": 8.0, "intra_num": 1, "inter_num": 1, "seqsep": 1},
        plddt_params={},
        pae_params={},
        distogram_params={"min_bin": 0.0, "max_bin": 2.0, "no_bins": 2},
        pde_logits=pair_vals.view(1, 6, 6, 1),
        pde_params={"max_bin": 10.0},
        task_type="type3",
        target_chain_roles=["peptide", "hla"],
    )

    assert losses["i_pae"].item() == pytest.approx(2.0 / 31.0)
    assert losses["i_pde"].item() == pytest.approx(2.0 / 10.0)
    assert losses["i_con"].item() == pytest.approx(2.0)
    assert losses["i_pae_hla"].item() == pytest.approx(9.0 / 31.0)
    assert losses["i_pde_hla"].item() == pytest.approx(9.0 / 10.0)
    assert losses["i_con_hla"].item() == pytest.approx(9.0)


def test_type3_rejects_hla_hotspot():
    from bindcraft.losses.total_loss import compute_all_losses

    with pytest.raises(ValueError, match="hotspot indices must fall on peptide"):
        compute_all_losses(
            x0_flat=torch.zeros(6, 3),
            plddt_logits=torch.zeros(1, 6, 1),
            pae_logits=torch.zeros(1, 6, 6, 1),
            distogram_logits=torch.zeros(6, 6, 2),
            input_feature_dict=_pmhc_feature_dict(),
            L_binder=2,
            weights=_weights(i_pae=1.0),
            contact_params={"intra_cutoff": 8.0, "inter_cutoff": 8.0, "intra_num": 1, "inter_num": 1, "seqsep": 1},
            plddt_params={},
            pae_params={},
            distogram_params={"min_bin": 0.0, "max_bin": 2.0, "no_bins": 2},
            hotspot_params={"target_hotspot_concat_indices": [2]},
            task_type="type3",
            target_chain_roles=["peptide", "hla"],
        )


def test_type3_iclash_detaches_target_side(monkeypatch):
    from protenix.model import sample_confidence
    from bindcraft.losses.total_loss import compute_all_losses

    monkeypatch.setattr(sample_confidence, "logits_to_score", lambda logits, *args, **kwargs: logits.squeeze(-1))

    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [5.5, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [5.4, 0.0, 0.0],
        ],
        requires_grad=True,
    )
    total, _ = compute_all_losses(
        x0_flat=coords,
        plddt_logits=torch.zeros(1, 6, 1),
        pae_logits=torch.zeros(1, 6, 6, 1),
        distogram_logits=torch.zeros(6, 6, 2),
        input_feature_dict=_pmhc_feature_dict(),
        L_binder=2,
        weights=_weights(i_clash=1.0),
        contact_params={"intra_cutoff": 8.0, "inter_cutoff": 8.0, "intra_num": 1, "inter_num": 1, "seqsep": 1},
        plddt_params={},
        pae_params={},
        distogram_params={"min_bin": 0.0, "max_bin": 2.0, "no_bins": 2},
        clash_params={"r_clash": 2.0, "bond_sep": 1},
        task_type="type3",
        target_chain_roles=["peptide", "hla"],
    )
    total.backward()

    assert coords.grad[:2].abs().sum().item() > 0
    assert coords.grad[2:].abs().sum().item() == pytest.approx(0.0)


def test_type3_requires_peptide_and_hla_roles():
    from bindcraft.losses.total_loss import compute_all_losses

    with pytest.raises(ValueError, match="requires at least one HLA"):
        compute_all_losses(
            x0_flat=torch.zeros(6, 3),
            plddt_logits=torch.zeros(1, 6, 1),
            pae_logits=torch.zeros(1, 6, 6, 1),
            distogram_logits=torch.zeros(6, 6, 2),
            input_feature_dict=_pmhc_feature_dict(),
            L_binder=2,
            weights=_weights(i_pae=1.0),
            contact_params={"intra_cutoff": 8.0, "inter_cutoff": 8.0, "intra_num": 1, "inter_num": 1, "seqsep": 1},
            plddt_params={},
            pae_params={},
            distogram_params={"min_bin": 0.0, "max_bin": 2.0, "no_bins": 2},
            task_type="type3",
            target_chain_roles=["peptide", "peptide"],
        )
