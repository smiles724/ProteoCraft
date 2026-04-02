"""
Unit tests for AF3-specific losses: Soft Clash Penalty and PDE Loss.
"""
import torch
import pytest


class TestSoftClashLoss:
    """compute_soft_clash_loss tests."""

    def test_intra_no_clash(self):
        """Well-separated atoms should produce zero clash."""
        from bindcraft.losses.geometry import compute_soft_clash_loss
        coords = torch.tensor([
            [0.0, 0.0, 0.0],   # atom 0 (res 0)
            [5.0, 0.0, 0.0],   # atom 1 (res 1)
            [10.0, 0.0, 0.0],  # atom 2 (res 2)
        ])
        binder_mask = torch.tensor([True, True, True])
        target_mask = torch.tensor([False, False, False])
        atom_to_token = torch.tensor([0, 1, 2])
        residue_index = torch.tensor([0, 1, 2])

        loss = compute_soft_clash_loss(
            coords, binder_mask, target_mask, atom_to_token, residue_index,
            mode="intra", r_clash=2.0, bond_sep=1,
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_intra_with_clash(self):
        """Atoms closer than r_clash should produce positive loss."""
        from bindcraft.losses.geometry import compute_soft_clash_loss
        coords = torch.tensor([
            [0.0, 0.0, 0.0],   # res 0
            [5.0, 0.0, 0.0],   # res 2
            [10.0, 0.0, 0.0],  # res 4
            [10.5, 0.0, 0.0],  # res 6 — clashes with res 4 (dist=0.5, sep=2)
        ])
        binder_mask = torch.ones(4, dtype=torch.bool)
        target_mask = torch.zeros(4, dtype=torch.bool)
        atom_to_token = torch.tensor([0, 1, 2, 3])
        residue_index = torch.tensor([0, 2, 4, 6])

        loss = compute_soft_clash_loss(
            coords, binder_mask, target_mask, atom_to_token, residue_index,
            mode="intra", r_clash=2.0, bond_sep=1,
        )
        assert loss.item() > 0.0

    def test_intra_bonded_exclusion(self):
        """Adjacent residues (bond_sep=1) should be excluded from clash."""
        from bindcraft.losses.geometry import compute_soft_clash_loss
        coords = torch.tensor([
            [0.0, 0.0, 0.0],   # res 0
            [1.3, 0.0, 0.0],   # res 1 — 1.3Å from res 0 (peptide bond)
        ])
        binder_mask = torch.ones(2, dtype=torch.bool)
        target_mask = torch.zeros(2, dtype=torch.bool)
        atom_to_token = torch.tensor([0, 1])
        residue_index = torch.tensor([0, 1])

        loss = compute_soft_clash_loss(
            coords, binder_mask, target_mask, atom_to_token, residue_index,
            mode="intra", r_clash=2.0, bond_sep=1,
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_inter_clash(self):
        """Binder-target clash should produce positive loss."""
        from bindcraft.losses.geometry import compute_soft_clash_loss
        binder_coords = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        target_coords = torch.tensor([[0.5, 0.0, 0.0]])
        coords = torch.cat([binder_coords, target_coords], dim=0)
        binder_mask = torch.tensor([True, True, False])
        target_mask = torch.tensor([False, False, True])
        atom_to_token = torch.tensor([0, 1, 2])
        residue_index = torch.tensor([0, 1, 2])

        loss = compute_soft_clash_loss(
            coords, binder_mask, target_mask, atom_to_token, residue_index,
            mode="inter", r_clash=2.0,
        )
        assert loss.item() > 0.0

    def test_inter_normalized_by_binder_atoms(self):
        """Inter clash should be normalized by N_binder_atoms, not N_pairs."""
        from bindcraft.losses.geometry import compute_soft_clash_loss
        binder_coords = torch.tensor([[0.0, 0.0, 0.0]])
        target_coords = torch.tensor([[0.5, 0.0, 0.0]])
        coords = torch.cat([binder_coords, target_coords], dim=0)
        binder_mask = torch.tensor([True, False])
        target_mask = torch.tensor([False, True])
        atom_to_token = torch.tensor([0, 1])
        residue_index = torch.tensor([0, 1])

        loss = compute_soft_clash_loss(
            coords, binder_mask, target_mask, atom_to_token, residue_index,
            mode="inter", r_clash=2.0,
        )
        # 1 binder atom, dist=0.5, penalty = (2.0 - 0.5)^2 = 2.25, / 1 atom
        assert loss.item() == pytest.approx(2.25, abs=1e-4)

    def test_differentiable(self):
        """Loss must support backward() for one-step gradient optimization."""
        from bindcraft.losses.geometry import compute_soft_clash_loss
        coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ], requires_grad=True)
        binder_mask = torch.tensor([True, True, False])
        target_mask = torch.tensor([False, False, True])
        atom_to_token = torch.tensor([0, 1, 2])
        residue_index = torch.tensor([0, 5, 10])

        loss = compute_soft_clash_loss(
            coords, binder_mask, target_mask, atom_to_token, residue_index,
            mode="intra", r_clash=2.0, bond_sep=1,
        )
        loss.backward()
        assert coords.grad is not None


class TestPDELoss:
    """compute_pde_loss tests."""

    def _make_pde_logits(self, n_token, n_bins=64):
        return torch.randn(1, n_token, n_token, n_bins)

    def test_intra_pde(self):
        """Intra PDE should return a scalar."""
        from bindcraft.losses.confidence import compute_pde_loss
        pde_logits = self._make_pde_logits(10)
        asym_id = torch.tensor([0]*5 + [1]*5)
        pde_params = {"min_bin": 0.0, "max_bin": 32.0, "no_bins": 64}
        loss = compute_pde_loss(pde_logits, asym_id, pde_params, mode="intra")
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_inter_pde_contact_weighted(self):
        """Inter PDE should use contact_probs weighting."""
        from bindcraft.losses.confidence import compute_pde_loss
        n = 10
        pde_logits = self._make_pde_logits(n)
        asym_id = torch.tensor([0]*5 + [1]*5)
        pde_params = {"min_bin": 0.0, "max_bin": 32.0, "no_bins": 64}
        contact_probs = torch.rand(n, n)
        loss = compute_pde_loss(
            pde_logits, asym_id, pde_params, mode="inter",
            contact_probs=contact_probs,
        )
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_inter_pde_zero_contact(self):
        """Zero contact_probs should give zero loss."""
        from bindcraft.losses.confidence import compute_pde_loss
        n = 10
        pde_logits = self._make_pde_logits(n)
        asym_id = torch.tensor([0]*5 + [1]*5)
        pde_params = {"min_bin": 0.0, "max_bin": 32.0, "no_bins": 64}
        contact_probs = torch.zeros(n, n)
        loss = compute_pde_loss(
            pde_logits, asym_id, pde_params, mode="inter",
            contact_probs=contact_probs,
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_differentiable(self):
        """PDE loss must be differentiable."""
        from bindcraft.losses.confidence import compute_pde_loss
        pde_logits = torch.randn(1, 6, 6, 64, requires_grad=True)
        asym_id = torch.tensor([0]*3 + [1]*3)
        pde_params = {"min_bin": 0.0, "max_bin": 32.0, "no_bins": 64}
        loss = compute_pde_loss(pde_logits, asym_id, pde_params, mode="intra")
        loss.backward()
        assert pde_logits.grad is not None


class TestDefaultWeights:
    """Verify DEFAULT_LOSS_WEIGHTS includes new losses."""

    def test_twelve_loss_items(self):
        from bindcraft.utils import DEFAULT_LOSS_WEIGHTS
        assert "clash" in DEFAULT_LOSS_WEIGHTS
        assert "i_clash" in DEFAULT_LOSS_WEIGHTS
        assert "pde" in DEFAULT_LOSS_WEIGHTS
        assert "i_pde" in DEFAULT_LOSS_WEIGHTS
        assert len(DEFAULT_LOSS_WEIGHTS) == 12

    def test_clash_params_exist(self):
        from bindcraft.utils import DEFAULT_CLASH_PARAMS
        assert "r_clash" in DEFAULT_CLASH_PARAMS
        assert "bond_sep" in DEFAULT_CLASH_PARAMS
