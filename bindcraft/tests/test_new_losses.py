"""Tests for new AF3-specific losses: Soft Clash and PDE."""

import torch
import pytest


class TestSoftClashLoss:

    def _make_clash_inputs(self, n_binder=4, n_target=3, device="cpu"):
        n_total = n_binder + n_target
        binder_atom_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
        binder_atom_mask[:n_binder] = True
        target_atom_mask = ~binder_atom_mask
        atom_to_token_idx = torch.arange(n_total, device=device)
        residue_index = torch.arange(n_total, device=device)
        return binder_atom_mask, target_atom_mask, atom_to_token_idx, residue_index

    def test_no_clash_returns_zero(self):
        from bindcraft.losses.geometry import compute_soft_clash_loss
        coords = torch.tensor([
            [0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [15.0, 0.0, 0.0],
            [20.0, 0.0, 0.0], [25.0, 0.0, 0.0], [30.0, 0.0, 0.0],
        ])
        bm, tm, a2t, ri = self._make_clash_inputs(n_binder=4, n_target=3)
        loss_intra = compute_soft_clash_loss(coords, bm, tm, a2t, ri, mode="intra", r_clash=2.0, bond_sep=1)
        loss_inter = compute_soft_clash_loss(coords, bm, tm, a2t, ri, mode="inter", r_clash=2.0, bond_sep=1)
        assert loss_intra.item() == 0.0
        assert loss_inter.item() == 0.0

    def test_inter_clash_detected(self):
        from bindcraft.losses.geometry import compute_soft_clash_loss
        coords = torch.tensor([
            [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0],
            [1.0, 0.0, 0.0], [50.0, 0.0, 0.0], [60.0, 0.0, 0.0],
        ])
        bm, tm, a2t, ri = self._make_clash_inputs(n_binder=4, n_target=3)
        loss = compute_soft_clash_loss(coords, bm, tm, a2t, ri, mode="inter", r_clash=2.0, bond_sep=1)
        assert loss.item() == pytest.approx(0.25, abs=1e-5)

    def test_intra_bond_exclusion(self):
        from bindcraft.losses.geometry import compute_soft_clash_loss
        coords = torch.tensor([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [5.0, 0.0, 0.0], [50.0, 0.0, 0.0],
        ])
        bm = torch.tensor([True, True, True, True, False])
        tm = ~bm
        a2t = torch.tensor([0, 0, 1, 1, 2])
        ri = torch.tensor([0, 0, 1, 1, 5])
        loss = compute_soft_clash_loss(coords, bm, tm, a2t, ri, mode="intra", r_clash=2.0, bond_sep=1)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_inter_clash_target_size_invariant(self):
        from bindcraft.losses.geometry import compute_soft_clash_loss
        coords_small = torch.tensor([
            [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [1.0, 0.0, 0.0], [50.0, 0.0, 0.0],
        ])
        bm_s = torch.tensor([True, True, False, False])
        tm_s = ~bm_s
        far_atoms = torch.tensor([[100.0 + i * 10.0, 0.0, 0.0] for i in range(10)])
        coords_large = torch.cat([coords_small[:3], far_atoms], dim=0)
        bm_l = torch.zeros(13, dtype=torch.bool); bm_l[:2] = True
        tm_l = ~bm_l
        a2t_s, ri_s = torch.arange(4), torch.arange(4)
        a2t_l, ri_l = torch.arange(13), torch.arange(13)
        loss_small = compute_soft_clash_loss(coords_small, bm_s, tm_s, a2t_s, ri_s, mode="inter", r_clash=2.0)
        loss_large = compute_soft_clash_loss(coords_large, bm_l, tm_l, a2t_l, ri_l, mode="inter", r_clash=2.0)
        assert loss_small.item() == pytest.approx(loss_large.item(), abs=1e-5)

    def test_gradient_pushes_apart(self):
        from bindcraft.losses.geometry import compute_soft_clash_loss
        coords = torch.tensor([
            [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.5, 0.0, 0.0],
        ], requires_grad=True)
        bm = torch.tensor([True, True, False])
        tm = ~bm
        a2t, ri = torch.arange(3), torch.arange(3)
        loss = compute_soft_clash_loss(coords, bm, tm, a2t, ri, mode="inter", r_clash=2.0)
        loss.backward()
        assert coords.grad[0, 0] > 0  # gradient > 0 → gradient descent moves x_0 negative (away from target at +0.5)


class TestPDELoss:

    def test_low_pde_gives_low_loss(self):
        from bindcraft.losses.confidence import compute_pde_loss
        N_token, no_bins = 5, 64
        pde_params = {"min_bin": 0, "max_bin": 32, "no_bins": no_bins}
        asym_id = torch.tensor([0, 0, 0, 1, 1])
        logits = torch.full((1, N_token, N_token, no_bins), -10.0)
        logits[..., 0] = 10.0
        loss = compute_pde_loss(logits, asym_id, pde_params, mode="intra")
        assert loss.item() < 0.05

    def test_high_pde_gives_high_loss(self):
        from bindcraft.losses.confidence import compute_pde_loss
        N_token, no_bins = 5, 64
        pde_params = {"min_bin": 0, "max_bin": 32, "no_bins": no_bins}
        asym_id = torch.tensor([0, 0, 0, 1, 1])
        logits = torch.full((1, N_token, N_token, no_bins), -10.0)
        logits[..., -1] = 10.0
        loss = compute_pde_loss(logits, asym_id, pde_params, mode="intra")
        assert loss.item() > 0.8

    def test_inter_pde_requires_contact_probs(self):
        from bindcraft.losses.confidence import compute_pde_loss
        logits = torch.randn(1, 5, 5, 64)
        asym_id = torch.tensor([0, 0, 0, 1, 1])
        pde_params = {"min_bin": 0, "max_bin": 32, "no_bins": 64}
        with pytest.raises(AssertionError):
            compute_pde_loss(logits, asym_id, pde_params, mode="inter")

    def test_inter_pde_contact_weighted(self):
        from bindcraft.losses.confidence import compute_pde_loss
        N_token, no_bins = 5, 64
        pde_params = {"min_bin": 0, "max_bin": 32, "no_bins": no_bins}
        asym_id = torch.tensor([0, 0, 0, 1, 1])
        logits = torch.full((1, N_token, N_token, no_bins), -10.0)
        logits[..., -1] = 10.0
        contact_probs = torch.zeros(N_token, N_token)
        contact_probs[0, 3] = 1.0
        contact_probs[3, 0] = 1.0
        loss = compute_pde_loss(logits, asym_id, pde_params, mode="inter", contact_probs=contact_probs)
        assert loss.item() > 0.5
