import torch
import pytest


def _feature_dict():
    return {
        "atom_to_token_idx": torch.arange(5),
        "asym_id": torch.tensor([0, 0, 0, 1, 1]),
        "residue_index": torch.arange(5),
        "has_frame": torch.ones(5, dtype=torch.bool),
        "distogram_rep_atom_mask": torch.ones(5, dtype=torch.bool),
    }


def test_compute_all_losses_local_masks_confidence(monkeypatch):
    from protenix.model import sample_confidence
    from bindcraft.losses.total_loss import compute_all_losses

    def fake_logits_to_score(logits, *args, return_prob=False, **kwargs):
        score = logits.squeeze(-1)
        if return_prob:
            return score, torch.ones_like(logits)
        return score

    monkeypatch.setattr(sample_confidence, "logits_to_score", fake_logits_to_score)
    monkeypatch.setattr(sample_confidence, "compute_contact_prob", lambda distogram_logits, **kwargs: torch.ones(5, 5))

    plddt_logits = torch.tensor([[[0.2], [0.9], [0.4], [0.8], [0.8]]])
    pair_vals = torch.full((5, 5), 100.0)
    # O = binder tokens 0 and 2; S = binder token 1; A = target tokens 3 and 4.
    for i, j in [(0, 0), (0, 2), (2, 0), (2, 2), (0, 1), (2, 1)]:
        pair_vals[i, j] = 1.0
        pair_vals[j, i] = 1.0
    for i in [0, 2]:
        for j in [3, 4]:
            pair_vals[i, j] = 2.0
            pair_vals[j, i] = 2.0

    weights = {
        "plddt": 1.0,
        "pae": 1.0,
        "i_pae": 1.0,
        "con": 0.0,
        "i_con": 0.0,
        "iptm": 0.0,
        "rg": 0.0,
        "helix": 0.0,
        "clash": 0.0,
        "i_clash": 0.0,
        "pde": 1.0,
        "i_pde": 1.0,
    }
    _, losses = compute_all_losses(
        x0_flat=torch.zeros(5, 3),
        plddt_logits=plddt_logits,
        pae_logits=pair_vals.view(1, 5, 5, 1),
        distogram_logits=torch.zeros(5, 5, 2),
        input_feature_dict=_feature_dict(),
        L_binder=3,
        weights=weights,
        contact_params={"intra_cutoff": 8.0, "inter_cutoff": 8.0, "intra_num": 1, "inter_num": 1, "seqsep": 1},
        plddt_params={},
        pae_params={},
        distogram_params={"min_bin": 0.0, "max_bin": 2.0, "no_bins": 2},
        pde_logits=pair_vals.view(1, 5, 5, 1),
        pde_params={"max_bin": 10.0},
        opt_token_mask=torch.tensor([True, False, True]),
    )

    assert losses["plddt"].item() == pytest.approx(1.0 - 0.3)
    assert losses["pae"].item() == pytest.approx(1.0 / 31.0)
    assert losses["i_pae"].item() == pytest.approx(2.0 / 31.0)
    assert losses["pde"].item() == pytest.approx(1.0 / 10.0)
    assert losses["i_pde"].item() == pytest.approx(2.0 / 10.0)


def test_compute_all_losses_type1_with_external_scaffold_roles(monkeypatch):
    from protenix.model import sample_confidence
    import bindcraft.losses.total_loss as total_loss_mod
    from bindcraft.losses.total_loss import compute_all_losses

    feature_dict = {
        "atom_to_token_idx": torch.arange(7),
        "asym_id": torch.tensor([0, 0, 0, 1, 1, 2, 2]),
        "residue_index": torch.arange(7),
        "has_frame": torch.ones(7, dtype=torch.bool),
        "distogram_rep_atom_mask": torch.ones(7, dtype=torch.bool),
    }
    opt_token_mask = torch.tensor([True, False, True])
    target_chain_roles = ["target", "scaffold"]

    design_token_mask = torch.tensor([True, False, True, False, False, False, False])
    scaffold_token_mask = torch.tensor([False, True, False, False, False, True, True])
    target_token_mask = torch.tensor([False, False, False, True, True, False, False])
    design_atom_mask = design_token_mask.clone()
    scaffold_atom_mask = scaffold_token_mask.clone()
    target_atom_mask = target_token_mask.clone()

    captured = {"pae": [], "pde": [], "contact": [], "clash": []}

    def fake_logits_to_score(logits, *args, return_prob=False, **kwargs):
        score = logits.squeeze(-1)
        if return_prob:
            return score, torch.ones_like(logits)
        return score

    def fake_compute_contact_prob(distogram_logits, **kwargs):
        return torch.ones(distogram_logits.shape[:2], device=distogram_logits.device)

    def fake_pae_loss(pae_logits, asym_id, pae_params, mode="intra", hotspot_token_mask=None, pair_mask=None):
        captured["pae"].append(pair_mask.clone())
        return torch.tensor(1.0, device=pae_logits.device)

    def fake_pde_loss(pde_logits, asym_id, pde_params, mode="intra", contact_probs=None, pair_mask=None):
        captured["pde"].append(pair_mask.clone())
        return torch.tensor(2.0, device=pde_logits.device)

    def fake_contact_loss(
        distogram_logits,
        binder_token_mask,
        target_token_mask,
        distogram_params,
        mode="intra",
        cutoff=14.0,
        num=2,
        num_pos=2,
        seqsep=9,
        binary=False,
        residue_index=None,
        hotspot_token_mask=None,
        row_token_mask=None,
        col_token_mask=None,
    ):
        captured["contact"].append(
            {
                "mode": mode,
                "row": None if row_token_mask is None else row_token_mask.clone(),
                "col": None if col_token_mask is None else col_token_mask.clone(),
                "hotspot": None if hotspot_token_mask is None else hotspot_token_mask.clone(),
            }
        )
        return torch.tensor(3.0, device=distogram_logits.device)

    def fake_clash_loss(
        x0_flat,
        binder_atom_mask,
        target_atom_mask,
        atom_to_token_idx,
        residue_index,
        mode="intra",
        r_clash=2.0,
        bond_sep=1,
        atom_mask_a=None,
        atom_mask_b=None,
        detach_b=False,
        exclude_bonded=None,
    ):
        captured["clash"].append(
            {
                "mode": mode,
                "a": None if atom_mask_a is None else atom_mask_a.clone(),
                "b": None if atom_mask_b is None else atom_mask_b.clone(),
                "detach_b": detach_b,
                "exclude_bonded": exclude_bonded,
            }
        )
        if atom_mask_a is not None and atom_mask_b is not None:
            if torch.equal(atom_mask_b, design_atom_mask):
                return torch.tensor(4.0, device=x0_flat.device)
            if torch.equal(atom_mask_b, scaffold_atom_mask):
                return torch.tensor(5.0, device=x0_flat.device)
            if torch.equal(atom_mask_b, target_atom_mask):
                return torch.tensor(6.0, device=x0_flat.device)
        return torch.tensor(0.0, device=x0_flat.device)

    monkeypatch.setattr(sample_confidence, "logits_to_score", fake_logits_to_score)
    monkeypatch.setattr(sample_confidence, "compute_contact_prob", fake_compute_contact_prob)
    monkeypatch.setattr(total_loss_mod, "compute_pae_loss", fake_pae_loss)
    monkeypatch.setattr(total_loss_mod, "compute_pde_loss", fake_pde_loss)
    monkeypatch.setattr(total_loss_mod, "compute_contact_loss", fake_contact_loss)
    monkeypatch.setattr(total_loss_mod, "compute_soft_clash_loss", fake_clash_loss)

    plddt_logits = torch.tensor([[[0.2], [0.9], [0.4], [0.1], [0.1], [0.1], [0.1]]])
    weights = {
        "plddt": 1.0,
        "pae": 1.0,
        "i_pae": 1.0,
        "con": 1.0,
        "i_con": 1.0,
        "iptm": 0.0,
        "rg": 0.0,
        "helix": 0.0,
        "clash": 1.0,
        "i_clash": 1.0,
        "pde": 1.0,
        "i_pde": 1.0,
    }

    total_loss, losses = compute_all_losses(
        x0_flat=torch.zeros(7, 3),
        plddt_logits=plddt_logits,
        pae_logits=torch.zeros(1, 7, 7, 1),
        distogram_logits=torch.zeros(7, 7, 2),
        input_feature_dict=feature_dict,
        L_binder=3,
        weights=weights,
        contact_params={"intra_cutoff": 8.0, "inter_cutoff": 8.0, "intra_num": 1, "inter_num": 1, "seqsep": 1},
        plddt_params={},
        pae_params={},
        distogram_params={"min_bin": 0.0, "max_bin": 2.0, "no_bins": 2},
        pde_logits=torch.zeros(1, 7, 7, 1),
        pde_params={"max_bin": 10.0},
        clash_params={"r_clash": 2.0, "bond_sep": 1},
        opt_token_mask=opt_token_mask,
        target_chain_roles=target_chain_roles,
    )

    assert losses["plddt"].item() == pytest.approx(1.0 - 0.3)
    assert losses["pae"].item() == pytest.approx(1.0)
    assert losses["i_pae"].item() == pytest.approx(1.0)
    assert losses["con"].item() == pytest.approx(3.0)
    assert losses["i_con"].item() == pytest.approx(3.0)
    assert losses["clash"].item() == pytest.approx(9.0)
    assert losses["i_clash"].item() == pytest.approx(6.0)
    assert losses["pde"].item() == pytest.approx(2.0)
    assert losses["i_pde"].item() == pytest.approx(2.0)

    expected_pae_mask = (design_token_mask.unsqueeze(0) & design_token_mask.unsqueeze(1)) | (
        design_token_mask.unsqueeze(1) & scaffold_token_mask.unsqueeze(0)
    )
    expected_ipae_mask = design_token_mask.unsqueeze(1) & target_token_mask.unsqueeze(0)
    assert torch.equal(captured["pae"][0], expected_pae_mask)
    assert torch.equal(captured["pae"][1], expected_ipae_mask)
    assert torch.equal(captured["pde"][0], expected_pae_mask)
    assert torch.equal(captured["pde"][1], expected_ipae_mask)
    assert captured["contact"][0]["mode"] == "intra"
    assert torch.equal(captured["contact"][0]["row"], design_token_mask)
    assert torch.equal(captured["contact"][0]["col"], design_token_mask)
    assert captured["contact"][1]["mode"] == "inter"
    assert torch.equal(captured["contact"][1]["row"], design_token_mask)
    assert torch.equal(captured["contact"][1]["col"], target_token_mask)
    assert len(captured["clash"]) == 3
    assert torch.equal(captured["clash"][0]["a"], design_atom_mask)
    assert torch.equal(captured["clash"][0]["b"], design_atom_mask)
    assert torch.equal(captured["clash"][1]["a"], design_atom_mask)
    assert torch.equal(captured["clash"][1]["b"], scaffold_atom_mask)
    assert torch.equal(captured["clash"][2]["a"], design_atom_mask)
    assert torch.equal(captured["clash"][2]["b"], target_atom_mask)
    assert total_loss.item() == pytest.approx(sum(losses[k].item() for k in ("plddt", "pae", "i_pae", "con", "i_con", "clash", "i_clash", "pde", "i_pde")))
    assert torch.equal(expected_ipae_mask, design_token_mask.unsqueeze(1) & target_token_mask.unsqueeze(0))


def test_local_clash_detaches_fixed_side():
    from bindcraft.losses.geometry import compute_soft_clash_loss

    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # O
            [0.5, 0.0, 0.0],  # S
            [0.4, 0.0, 0.0],  # A
        ],
        requires_grad=True,
    )
    atom_to_token = torch.arange(3)
    residue_index = torch.tensor([0, 5, 10])
    o_mask = torch.tensor([True, False, False])
    s_mask = torch.tensor([False, True, False])
    a_mask = torch.tensor([False, False, True])
    empty = torch.zeros(3, dtype=torch.bool)

    loss = compute_soft_clash_loss(
        coords, o_mask, empty, atom_to_token, residue_index,
        mode="intra", atom_mask_a=o_mask, atom_mask_b=s_mask,
        detach_b=True, exclude_bonded=True, r_clash=2.0,
    )
    loss = loss + compute_soft_clash_loss(
        coords, o_mask, a_mask, atom_to_token, residue_index,
        mode="inter", atom_mask_a=o_mask, atom_mask_b=a_mask,
        detach_b=True, exclude_bonded=False, r_clash=2.0,
    )
    loss.backward()

    assert coords.grad[0].abs().sum().item() > 0
    assert coords.grad[1].abs().sum().item() == pytest.approx(0.0)
    assert coords.grad[2].abs().sum().item() == pytest.approx(0.0)


def test_empty_local_mask_is_rejected():
    from bindcraft.losses.total_loss import compute_all_losses

    with pytest.raises(ValueError, match="selects no binder tokens"):
        compute_all_losses(
            x0_flat=torch.zeros(5, 3),
            plddt_logits=torch.zeros(1, 5, 1),
            pae_logits=torch.zeros(1, 5, 5, 1),
            distogram_logits=torch.zeros(5, 5, 2),
            input_feature_dict=_feature_dict(),
            L_binder=3,
            weights={"plddt": 1.0},
            contact_params={"intra_cutoff": 8.0, "inter_cutoff": 8.0, "intra_num": 1, "inter_num": 1, "seqsep": 1},
            plddt_params={},
            pae_params={},
            distogram_params={"min_bin": 0.0, "max_bin": 2.0, "no_bins": 2},
            opt_token_mask=torch.tensor([False, False, False]),
        )
