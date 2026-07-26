"""Differentiable one-step diffusion support for the optional ESMFold2 backend.

This module is deliberately ESMFold2-specific.  It does not import the ESM or
Transformers packages at module import time, and it does not alter the existing
Protenix one-step diffusion implementation.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from bindcraft.losses.geometry import compute_rg_loss, compute_soft_clash_loss


# Common subset of the tuned Protenix verifier weights.  ESMFold2 does not
# expose PDE logits, so pde/i_pde are intentionally absent.
ESMFOLD2_COMMON_LOSS_WEIGHTS: Dict[str, float] = {
    "plddt": 0.1,
    "pae": 0.4,
    "i_pae": 0.2,
    "con": 0.2,
    "i_con": 3.0,
    "iptm": 0.3,
    "rg": 0.3,
    "helix": -0.3,
    "clash": 0.1,
    "i_clash": 0.1,
}


def _batch0(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor
    return tensor[0]


def _sample0_coordinates(coordinates: torch.Tensor) -> torch.Tensor:
    if coordinates.ndim == 4:
        return coordinates[0, 0]
    if coordinates.ndim == 3:
        return coordinates[0]
    if coordinates.ndim == 2:
        return coordinates
    raise ValueError(
        "coordinates must have shape [A,3], [B,A,3], or [B,S,A,3]"
    )


def _categorical_mean(
    logits: torch.Tensor,
    *,
    start: float,
    end: float,
) -> torch.Tensor:
    bins = torch.linspace(
        float(start),
        float(end),
        logits.shape[-1],
        device=logits.device,
        dtype=logits.dtype,
    )
    return (torch.softmax(logits, dim=-1) * bins).sum(dim=-1)


def _normalize_target_roles(
    roles: Optional[Sequence[str]],
    *,
    n_target_chains: int,
) -> tuple[str, ...]:
    if roles is None:
        return ("target",) * n_target_chains
    normalized = tuple(str(role).strip().lower() for role in roles)
    if len(normalized) != n_target_chains:
        raise ValueError(
            "target_chain_roles length must match the number of target chains"
        )
    if set(normalized) - {"target", "scaffold"}:
        raise ValueError("target_chain_roles only supports target/scaffold")
    if "target" not in normalized:
        raise ValueError("target_chain_roles must contain at least one target")
    return normalized


def _token_masks(
    features: Mapping[str, torch.Tensor],
    *,
    binder_length: int,
    binder_mutable_mask: Optional[Sequence[bool]],
    target_chain_roles: Optional[Sequence[str]],
) -> Dict[str, torch.Tensor]:
    token_mask = _batch0(features["token_attention_mask"]).bool()
    asym_id = _batch0(features["asym_id"]).long()
    valid_asym = asym_id[token_mask]
    if valid_asym.numel() == 0:
        raise ValueError("ESMFold2 features contain no valid tokens")

    binder_chain_id = valid_asym[-1]
    binder_mask = token_mask & (asym_id == binder_chain_id)
    if int(binder_mask.sum().item()) != int(binder_length):
        raise ValueError(
            "ESMFold2 binder token count does not match binder sequence length"
        )

    chain_ids = [
        int(chain_id)
        for chain_id in torch.unique(valid_asym, sorted=True).tolist()
        if int(chain_id) != int(binder_chain_id.item())
    ]
    roles = _normalize_target_roles(
        target_chain_roles,
        n_target_chains=len(chain_ids),
    )
    target_mask = torch.zeros_like(token_mask)
    external_scaffold_mask = torch.zeros_like(token_mask)
    for chain_id, role in zip(chain_ids, roles):
        if role == "target":
            target_mask |= token_mask & (asym_id == chain_id)
        else:
            external_scaffold_mask |= token_mask & (asym_id == chain_id)

    design_mask = binder_mask.clone()
    if binder_mutable_mask is not None:
        mutable = torch.as_tensor(
            binder_mutable_mask,
            dtype=torch.bool,
            device=binder_mask.device,
        )
        if mutable.numel() != binder_length:
            raise ValueError(
                "binder_mutable_mask length must match binder sequence length"
            )
        design_mask.zero_()
        design_mask[binder_mask.nonzero(as_tuple=True)[0]] = mutable
        if not design_mask.any():
            raise ValueError("binder_mutable_mask selects no residues")

    binder_scaffold_mask = binder_mask & ~design_mask
    scaffold_mask = external_scaffold_mask | binder_scaffold_mask
    return {
        "token": token_mask,
        "asym_id": asym_id,
        "binder": binder_mask,
        "design": design_mask,
        "target": target_mask,
        "scaffold": scaffold_mask,
    }


def _differentiable_pair_chain_iptm(
    pae_logits: torch.Tensor,
    *,
    masks: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Recompute target-binder pair-chain iPTM without detached summaries."""

    logits = pae_logits
    if logits.ndim == 3:
        logits = logits.unsqueeze(0)
    valid = masks["token"].to(logits.device)
    target = masks["target"].to(logits.device)
    binder = masks["binder"].to(logits.device)
    asym_id = masks["asym_id"].to(logits.device)

    n_bins = logits.shape[-1]
    bin_width = 32.0 / n_bins
    bin_centers = torch.arange(
        0.5 * bin_width,
        32.0,
        bin_width,
        device=logits.device,
        dtype=logits.dtype,
    )
    n_res = valid.sum().clamp(min=19).to(logits.dtype)
    d0 = 1.24 * (n_res - 15.0).pow(1.0 / 3.0) - 1.8
    tm_per_bin = 1.0 / (1.0 + (bin_centers / d0) ** 2)
    tm_expected = (torch.softmax(logits, dim=-1) * tm_per_bin).sum(dim=-1)

    scores = []
    target_chain_ids = torch.unique(asym_id[target], sorted=True)
    for chain_id in target_chain_ids:
        chain_mask = target & (asym_id == chain_id)
        pair_mask = chain_mask.unsqueeze(-1) & binder.unsqueeze(-2)
        denom = pair_mask.sum().clamp(min=1).to(tm_expected.dtype)
        scores.append(
            (tm_expected * pair_mask.unsqueeze(0).to(tm_expected.dtype)).sum(
                dim=(-1, -2)
            )
            / denom
        )
    if not scores:
        raise ValueError("No target-binder chain pair is available for iPTM")
    return torch.stack(scores, dim=0).mean()


def _helix_loss(
    distogram_logits: torch.Tensor,
    *,
    design_mask: torch.Tensor,
    residue_index: torch.Tensor,
) -> torch.Tensor:
    logits = distogram_logits[0] if distogram_logits.ndim == 4 else distogram_logits
    boundaries = torch.linspace(
        2.0,
        52.0,
        logits.shape[-1] - 1,
        device=logits.device,
        dtype=logits.dtype,
    )
    expanded = torch.cat(
        (
            torch.tensor([1.0], device=logits.device, dtype=logits.dtype),
            boundaries,
            torch.tensor([57.0], device=logits.device, dtype=logits.dtype),
        )
    )
    bin_centers = (expanded[:-1] + expanded[1:]) / 2.0
    contact_probability = (
        torch.softmax(logits, dim=-1)
        * (bin_centers < 6.0).to(logits.dtype)
    ).sum(dim=-1)
    pair_loss = -torch.log(contact_probability + 1e-8)

    indices = design_mask.nonzero(as_tuple=True)[0]
    if indices.numel() < 4:
        return logits.sum() * 0.0
    design_residue_index = residue_index[indices]
    offset = design_residue_index.unsqueeze(1) - design_residue_index.unsqueeze(0)
    pairs = (offset == 3).nonzero(as_tuple=False)
    if pairs.numel() == 0:
        return logits.sum() * 0.0
    return pair_loss[indices[pairs[:, 1]], indices[pairs[:, 0]]].mean()


def compute_esmfold2_common_losses(
    *,
    coordinates: torch.Tensor,
    plddt_logits: torch.Tensor,
    pae_logits: torch.Tensor,
    distogram_logits: torch.Tensor,
    features: Mapping[str, torch.Tensor],
    binder_length: int,
    target_chain_roles: Optional[Sequence[str]] = None,
    binder_mutable_mask: Optional[Sequence[bool]] = None,
    distogram_losses: Optional[Mapping[str, torch.Tensor]] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the Protenix-aligned loss subset available from ESMFold2."""

    resolved_weights = dict(ESMFOLD2_COMMON_LOSS_WEIGHTS)
    if weights is not None:
        resolved_weights.update({str(k): float(v) for k, v in weights.items()})

    masks = _token_masks(
        features,
        binder_length=binder_length,
        binder_mutable_mask=binder_mutable_mask,
        target_chain_roles=target_chain_roles,
    )
    device = coordinates.device
    masks = {key: value.to(device) for key, value in masks.items()}

    atom_mask = _batch0(features["atom_attention_mask"]).bool().to(device)
    atom_to_token = _batch0(features["atom_to_token"]).long().to(device)
    residue_index = _batch0(features["residue_index"]).long().to(device)
    rep_idx = _batch0(features["distogram_atom_idx"]).long().to(device)
    coords = _sample0_coordinates(coordinates)
    if coords.shape[0] != atom_mask.shape[0]:
        raise ValueError("coordinates and atom_attention_mask have different sizes")

    design_atom_mask = atom_mask & masks["design"][atom_to_token]
    binder_atom_mask = atom_mask & masks["binder"][atom_to_token]
    target_atom_mask = atom_mask & masks["target"][atom_to_token]
    scaffold_atom_mask = atom_mask & masks["scaffold"][atom_to_token]
    local_redesign = bool((masks["design"] != masks["binder"]).any().item())
    has_scaffold = bool(masks["scaffold"].any().item())

    losses: Dict[str, torch.Tensor] = {}
    plddt = _categorical_mean(plddt_logits, start=0.0, end=1.0)
    plddt0 = _batch0(plddt)
    plddt_indices = rep_idx[masks["design"] if local_redesign else masks["binder"]]
    losses["plddt"] = 1.0 - plddt0[plddt_indices].mean()

    pae = _categorical_mean(pae_logits, start=0.0, end=32.0) / 32.0
    pae = 0.5 * (pae + pae.transpose(-1, -2))
    design = masks["design"]
    binder = masks["binder"]
    intra_pair_mask = (
        design.unsqueeze(-1) & design.unsqueeze(-2)
        if not (local_redesign or has_scaffold)
        else (
            (design.unsqueeze(-1) & design.unsqueeze(-2))
            | (design.unsqueeze(-1) & masks["scaffold"].unsqueeze(-2))
        )
    )
    inter_pair_mask = design.unsqueeze(-1) & masks["target"].unsqueeze(-2)

    def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(0).to(value.dtype)
        return (value * mask_f).sum() / mask_f.sum().clamp(min=1.0)

    losses["pae"] = masked_mean(pae, intra_pair_mask)
    losses["i_pae"] = masked_mean(pae, inter_pair_mask)
    losses["iptm"] = 1.0 - _differentiable_pair_chain_iptm(
        pae_logits,
        masks=masks,
    )

    if distogram_losses is not None:
        if "intra_contact_loss" in distogram_losses:
            losses["con"] = distogram_losses["intra_contact_loss"].mean()
        if "inter_contact_loss" in distogram_losses:
            losses["i_con"] = distogram_losses["inter_contact_loss"].mean()

    rep_coords = coords[rep_idx]
    rg_mask = design if local_redesign else binder
    losses["rg"] = compute_rg_loss(
        rep_coords[rg_mask],
        int(rg_mask.sum().item()),
    )
    losses["helix"] = _helix_loss(
        distogram_logits,
        design_mask=design if local_redesign else binder,
        residue_index=residue_index,
    )

    clash_a = design_atom_mask if (local_redesign or has_scaffold) else binder_atom_mask
    clash_b = (
        design_atom_mask | scaffold_atom_mask
        if (local_redesign or has_scaffold)
        else binder_atom_mask
    )
    losses["clash"] = compute_soft_clash_loss(
        coords,
        binder_atom_mask,
        target_atom_mask,
        atom_to_token,
        residue_index,
        mode="intra",
        atom_mask_a=clash_a,
        atom_mask_b=clash_b,
        exclude_bonded=True,
    )
    losses["i_clash"] = compute_soft_clash_loss(
        coords,
        binder_atom_mask,
        target_atom_mask,
        atom_to_token,
        residue_index,
        mode="inter",
        atom_mask_a=design_atom_mask if (local_redesign or has_scaffold) else binder_atom_mask,
        atom_mask_b=target_atom_mask,
        detach_b=True,
        exclude_bonded=False,
    )

    total = coords.sum() * 0.0
    for name, weight in resolved_weights.items():
        if name in losses and weight != 0.0:
            total = total + float(weight) * losses[name]
    losses["total"] = total
    return total, losses


@contextmanager
def capture_experimental_trunk_state(model: Any) -> Iterator[Dict[str, torch.Tensor]]:
    """Capture differentiable intermediates from an unmodified Experimental model."""

    captured: Dict[str, torch.Tensor] = {}

    def save(name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor, torch.Tensor):
                captured[name] = tensor

        return hook

    handles = [
        model.inputs_embedder.register_forward_hook(save("s_inputs")),
        model.rel_pos.register_forward_hook(save("relative_position_encoding")),
        model.token_bonds.register_forward_hook(save("token_bonds_encoding")),
        model.folding_trunk.register_forward_hook(save("z_trunk")),
    ]
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def scatter_reference_coordinates(
    coordinates: torch.Tensor,
    *,
    atom_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Scatter decoded real-atom coordinates into ESMFold2's padded atom layout."""

    atom_mask = atom_attention_mask.bool()
    if atom_mask.ndim == 1:
        atom_mask = atom_mask.unsqueeze(0)
    source = coordinates
    if source.ndim == 2:
        source = source.unsqueeze(0)
    if source.ndim != 3 or source.shape[0] != atom_mask.shape[0]:
        raise ValueError("reference coordinates must have shape [B,A,3] or [A,3]")

    padded = torch.zeros(
        atom_mask.shape[0],
        atom_mask.shape[1],
        3,
        device=source.device,
        dtype=source.dtype,
    )
    for batch_index in range(atom_mask.shape[0]):
        expected = int(atom_mask[batch_index].sum().item())
        if source.shape[1] == atom_mask.shape[1]:
            padded[batch_index] = source[batch_index]
        elif source.shape[1] == expected:
            padded[batch_index, atom_mask[batch_index]] = source[batch_index]
        else:
            raise ValueError(
                "reference coordinate count does not match padded or valid atom "
                f"count: source={source.shape[1]}, "
                f"proposal_padded={atom_mask.shape[1]}, proposal_valid={expected}"
            )
    return padded


def remap_reference_coordinates(
    coordinates: torch.Tensor,
    *,
    source_features: Mapping[str, torch.Tensor],
    destination_features: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Map parent coordinates into a child layout and rebuild new side chains."""

    source_coords = _sample0_coordinates(coordinates)

    def feature0(
        features: Mapping[str, torch.Tensor],
        name: str,
    ) -> torch.Tensor:
        value = features[name]
        return value[0] if value.ndim > 1 and value.shape[0] == 1 else value

    def atom_keys(
        features: Mapping[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        list[tuple[int, tuple[int, ...], int]],
        torch.Tensor,
    ]:
        mask = feature0(features, "atom_attention_mask").bool()
        atom_to_token = feature0(features, "atom_to_token").long()
        atom_names = feature0(features, "ref_atom_name_chars").long()
        elements = feature0(features, "ref_element").long()
        ref_pos = feature0(features, "ref_pos").float()
        if not (
            mask.shape[0]
            == atom_to_token.shape[0]
            == atom_names.shape[0]
            == elements.shape[0]
            == ref_pos.shape[0]
        ):
            raise ValueError("ESMFold2 atom identity features have inconsistent sizes")
        keys = [
            (
                int(atom_to_token[index].item()),
                tuple(int(value) for value in atom_names[index].tolist()),
                int(elements[index].item()),
            )
            for index in mask.nonzero(as_tuple=True)[0].tolist()
        ]
        return mask, keys, ref_pos[mask]

    source_mask, source_keys, _ = atom_keys(source_features)
    destination_mask, destination_keys, destination_ref_pos = atom_keys(
        destination_features
    )
    if source_coords.shape[0] == source_mask.shape[0]:
        source_valid_coords = source_coords[source_mask]
    elif source_coords.shape[0] == int(source_mask.sum().item()):
        source_valid_coords = source_coords
    else:
        raise ValueError(
            "source coordinate count does not match verifier atom layout: "
            f"source={source_coords.shape[0]}, padded={source_mask.shape[0]}, "
            f"valid={int(source_mask.sum().item())}"
        )

    coordinate_by_key: dict[tuple[int, tuple[int, ...], int], torch.Tensor] = {}
    for key, coordinate in zip(source_keys, source_valid_coords, strict=True):
        if key in coordinate_by_key:
            raise ValueError(f"duplicate verifier atom identity: {key}")
        coordinate_by_key[key] = coordinate

    def atom_name(key: tuple[int, tuple[int, ...], int]) -> str:
        return "".join(
            chr(value + 32) if value > 0 else " " for value in key[1]
        ).strip()

    source_by_token_and_name = {
        (key[0], atom_name(key)): coordinate
        for key, coordinate in coordinate_by_key.items()
    }
    destination_ref_by_token_and_name = {
        (key[0], atom_name(key)): ref_pos
        for key, ref_pos in zip(
            destination_keys,
            destination_ref_pos,
            strict=True,
        )
    }

    def rigid_transform(
        moving: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        moving_center = moving.mean(dim=0)
        target_center = target.mean(dim=0)
        covariance = (moving - moving_center).transpose(0, 1) @ (
            target - target_center
        )
        u, _, vh = torch.linalg.svd(covariance)
        correction = torch.eye(3, device=moving.device, dtype=moving.dtype)
        correction[-1, -1] = torch.sign(torch.det(u @ vh))
        rotation = u @ correction @ vh
        translation = target_center - moving_center @ rotation
        return rotation, translation

    rebuilt_by_key: dict[tuple[int, tuple[int, ...], int], torch.Tensor] = {}
    missing_by_token: dict[int, list[tuple[int, tuple[int, ...], int]]] = {}
    for key in destination_keys:
        if key in coordinate_by_key:
            rebuilt_by_key[key] = coordinate_by_key[key]
        else:
            missing_by_token.setdefault(key[0], []).append(key)

    for token_index, missing_keys in missing_by_token.items():
        anchors = []
        for name in ("N", "CA", "C"):
            source_anchor = source_by_token_and_name.get((token_index, name))
            destination_anchor = destination_ref_by_token_and_name.get(
                (token_index, name)
            )
            if source_anchor is not None and destination_anchor is not None:
                anchors.append((destination_anchor, source_anchor))
        if len(anchors) != 3:
            missing_names = ", ".join(atom_name(key) for key in missing_keys[:3])
            raise ValueError(
                "cannot rebuild proposal side-chain coordinates without N/CA/C "
                f"anchors for token={token_index}; missing_atoms=[{missing_names}]"
            )
        moving = torch.stack([item[0] for item in anchors]).to(
            device=source_coords.device,
            dtype=source_coords.dtype,
        )
        target = torch.stack([item[1] for item in anchors]).to(
            device=source_coords.device,
            dtype=source_coords.dtype,
        )
        rotation, translation = rigid_transform(moving, target)
        for key in missing_keys:
            idealized = destination_ref_by_token_and_name[
                (token_index, atom_name(key))
            ].to(device=source_coords.device, dtype=source_coords.dtype)
            rebuilt_by_key[key] = idealized @ rotation + translation

    destination = torch.zeros(
        1,
        destination_mask.shape[0],
        3,
        dtype=source_coords.dtype,
        device=source_coords.device,
    )
    destination[0, destination_mask.to(source_coords.device)] = torch.stack(
        [rebuilt_by_key[key] for key in destination_keys],
        dim=0,
    )
    return destination


def run_esmfold2_one_step_diffusion(
    *,
    model: Any,
    captured: Mapping[str, torch.Tensor],
    features: Mapping[str, torch.Tensor],
    reference_coordinates: torch.Tensor,
    tau: float,
    seed: Optional[int],
) -> Dict[str, torch.Tensor]:
    """Run one differentiable ESMFold2 denoiser call at a fixed noise level."""

    if float(tau) <= 0.0:
        raise ValueError("ESMFold2 one-step diffusion tau must be positive")
    required = {"s_inputs", "relative_position_encoding", "z_trunk"}
    missing = sorted(required - set(captured))
    if missing:
        raise RuntimeError(
            f"Failed to capture ESMFold2 proposal intermediates: {missing}"
        )

    from transformers.models.esmfold2.modeling_esmfold2_common import (
        CHAR_VOCAB_SIZE,
        MAX_ATOMIC_NUMBER,
    )

    atom_mask = features["atom_attention_mask"].bool()
    x_clean = scatter_reference_coordinates(
        reference_coordinates.to(
            device=atom_mask.device,
            dtype=torch.float32,
        ),
        atom_attention_mask=atom_mask,
    )
    devices = []
    if x_clean.is_cuda:
        devices = [x_clean.device.index if x_clean.device.index is not None else 0]
    with torch.random.fork_rng(devices=devices, enabled=True):
        if seed is not None:
            torch.manual_seed(int(seed))
            if x_clean.is_cuda:
                torch.cuda.manual_seed_all(int(seed))
        noise = torch.randn_like(x_clean)
    x_noisy = x_clean.detach() + float(tau) * noise

    ref_element = F.one_hot(
        features["ref_element"].long(),
        num_classes=MAX_ATOMIC_NUMBER,
    ).float()
    ref_atom_name_chars = F.one_hot(
        features["ref_atom_name_chars"].long(),
        num_classes=CHAR_VOCAB_SIZE,
    ).float()
    atom_mask_f = atom_mask.float()
    ref_element = ref_element * atom_mask_f.unsqueeze(-1)
    ref_atom_name_chars = (
        ref_atom_name_chars
        * atom_mask_f.unsqueeze(-1).unsqueeze(-1)
    )
    atom_to_token = features["atom_to_token"].long() * atom_mask.long()

    diffusion_module = model.structure_head.diffusion_module
    diffusion_output = diffusion_module(
        x_noisy=x_noisy,
        t_hat=torch.full(
            (x_clean.shape[0],),
            float(tau),
            device=x_clean.device,
            dtype=torch.float32,
        ),
        ref_pos=features["ref_pos"],
        ref_charge=features["ref_charge"],
        ref_mask=atom_mask,
        ref_element=ref_element,
        ref_atom_name_chars=ref_atom_name_chars,
        ref_space_uid=features["ref_space_uid"],
        tok_idx=atom_to_token,
        s_inputs=captured["s_inputs"],
        s_trunk=None,
        z_trunk=captured["z_trunk"].float(),
        relative_position_encoding=captured["relative_position_encoding"],
        asym_id=features["asym_id"],
        residue_index=features["residue_index"],
        entity_id=features["entity_id"],
        token_index=features["token_index"],
        sym_id=features["sym_id"],
        token_attention_mask=features["token_attention_mask"],
        num_diffusion_samples=1,
    )
    x_denoised = diffusion_output["x_denoised"]
    if x_denoised is None:
        raise RuntimeError("ESMFold2 diffusion module returned no denoised coordinates")

    confidence_head = getattr(model, "confidence_head", None)
    if confidence_head is None:
        raise RuntimeError(
            "ESMFold2 one-step diffusion requires an Experimental model with a confidence head"
        )
    confidence = confidence_head(
        s_inputs=captured["s_inputs"],
        z=captured["z_trunk"].float(),
        x_pred=x_denoised,
        distogram_atom_idx=features["distogram_atom_idx"],
        token_attention_mask=features["token_attention_mask"],
        atom_to_token=atom_to_token,
        atom_attention_mask=atom_mask,
        asym_id=features["asym_id"],
        mol_type=features["mol_type"],
        num_diffusion_samples=1,
        relative_position_encoding=captured["relative_position_encoding"],
        token_bonds_encoding=captured.get("token_bonds_encoding"),
    )
    return {
        "coordinates": x_denoised,
        "plddt_logits": confidence["plddt_logits"],
        "pae_logits": confidence["pae_logits"],
    }
