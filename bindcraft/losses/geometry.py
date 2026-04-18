"""
BindCraft v2 - 几何 Loss 函数

对标原版 BindCraft 的 Rg loss 和 Helix loss。
Rg 使用动态目标值 2.38 * N^0.365 + elu 惩罚。
Helix 使用 distogram 计算 i-(i+3) 接触概率。
"""

import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def compute_rg_loss(
    binder_coords: torch.Tensor,
    binder_len: int,
) -> torch.Tensor:
    """
    Radius of gyration loss（对标原版 BindCraft add_rg_loss）。

    rg_th = 2.38 * N^0.365（理论紧凑球蛋白的 Rg）
    loss = elu(rg - rg_th)

    Args:
        binder_coords: Binder 原子坐标 [N_binder_atoms, 3]
        binder_len: Binder 残基数（用于计算理论 Rg）
    """
    centroid = binder_coords.mean(dim=0, keepdim=True)
    rg = torch.sqrt(((binder_coords - centroid) ** 2).sum(dim=-1).mean() + 1e-8)
    rg_th = 2.38 * (binder_len ** 0.365)
    return F.elu(rg - rg_th)


def compute_helix_loss(
    distogram_logits: torch.Tensor,
    binder_token_mask: torch.Tensor,
    distogram_params: dict,
    cutoff: float = 6.0,
) -> torch.Tensor:
    """
    Helix loss（对标原版 BindCraft add_helix_loss）。

    计算 binder 区域内 i-(i+3) 残基对的接触概率。
    接触概率越高说明越倾向于形成 alpha-helix。
    返回值越小表示越多螺旋（loss = 1 - contact_prob）。

    Args:
        distogram_logits: [N_token, N_token, no_bins]
        binder_token_mask: [N_token] bool, True for binder tokens
        distogram_params: dict with min_bin, max_bin, no_bins
        cutoff: 接触距离阈值 (Angstrom)
    """
    prob = F.softmax(distogram_logits, dim=-1)
    min_bin = distogram_params["min_bin"]
    max_bin = distogram_params["max_bin"]
    no_bins = distogram_params["no_bins"]
    bin_width = (max_bin - min_bin) / no_bins
    boundaries = torch.linspace(min_bin, max_bin - bin_width, no_bins,
                                device=prob.device)
    bin_centers = boundaries + 0.5 * bin_width
    thres_idx = int((bin_centers < cutoff).sum().item())
    bins = (bin_centers < cutoff).to(distogram_logits.dtype)
    # 对齐 ColabDesign _get_con_loss(binary=True): -log(sum_{d<cutoff} p(d))
    helix_pair_loss = -torch.log((prob * bins).sum(dim=-1) + 1e-8)  # [N_token, N_token]

    N = binder_token_mask.sum().item()
    if N < 4:
        return torch.tensor(0.0, device=prob.device, requires_grad=True)

    binder_idx = binder_token_mask.nonzero(as_tuple=True)[0]
    i_idx = binder_idx[:-3]
    j_idx = binder_idx[3:]
    helix_vals = helix_pair_loss[i_idx, j_idx]
    return helix_vals.mean()


def compute_hotspot_distance_loss(
    x0_flat: torch.Tensor,
    binder_atom_mask: torch.Tensor,
    hotspot_target_atom_mask: torch.Tensor,
    cutoff: float = 8.0,
    top_k: int = 3,
) -> tuple[torch.Tensor, dict]:
    """
    hotspot 距离约束：鼓励 binder 靠近指定 target hotspot 残基。

    对每个 hotspot representative atom 计算其到 binder representative atoms 的最小距离，
    再对最靠近的 top-k 个 hotspot 距离取均值，最后用 relu(distance-cutoff) 作为惩罚。
    """
    binder_coords = x0_flat[binder_atom_mask]
    hotspot_coords = x0_flat[hotspot_target_atom_mask]

    if binder_coords.shape[0] == 0 or hotspot_coords.shape[0] == 0:
        zero = torch.tensor(0.0, device=x0_flat.device, requires_grad=True)
        return zero, {
            "hotspot_min_distance": zero,
            "hotspot_mean_min_distance": zero,
        }

    dist = torch.cdist(hotspot_coords, binder_coords)
    min_dist_per_hotspot = dist.min(dim=1).values
    hotspot_min_distance = min_dist_per_hotspot.min()

    if top_k is None or top_k <= 0 or top_k >= min_dist_per_hotspot.shape[0]:
        hotspot_mean_min_distance = min_dist_per_hotspot.mean()
    else:
        hotspot_mean_min_distance = torch.topk(
            min_dist_per_hotspot,
            k=top_k,
            largest=False,
        ).values.mean()

    hotspot_loss = torch.relu(hotspot_mean_min_distance - cutoff)
    return hotspot_loss, {
        "hotspot_min_distance": hotspot_min_distance,
        "hotspot_mean_min_distance": hotspot_mean_min_distance,
    }


def compute_soft_clash_loss(
    x0_flat: torch.Tensor,
    binder_atom_mask: torch.Tensor,
    target_atom_mask: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    residue_index: torch.Tensor,
    mode: str = "intra",
    r_clash: float = 2.0,
    bond_sep: int = 1,
) -> torch.Tensor:
    """
    Soft clash penalty: quadratic repulsion for atom pairs closer than r_clash.

    Intra: penalizes binder internal clashes (excludes bonded atoms via
        residue_index separation <= bond_sep to avoid penalizing peptide bonds).
    Inter: penalizes binder-target clashes, normalized by N_binder_atoms
        (target-size invariant).

    Args:
        x0_flat: [N_atoms, 3] all-atom coordinates
        binder_atom_mask: [N_atoms] bool
        target_atom_mask: [N_atoms] bool
        atom_to_token_idx: [N_atoms] token index per atom
        residue_index: [N_tokens] residue index per token
        mode: "intra" or "inter"
        r_clash: clash distance threshold (Angstrom)
        bond_sep: minimum residue separation to consider (intra only)
    """
    if mode == "inter":
        coords_a = x0_flat[binder_atom_mask]
        coords_b = x0_flat[target_atom_mask]
        n_binder = coords_a.shape[0]
        if n_binder == 0:
            return torch.tensor(0.0, device=x0_flat.device, requires_grad=True)
        dist = torch.cdist(coords_a, coords_b)
        clash = torch.clamp(r_clash - dist, min=0) ** 2
        return clash.sum() / n_binder
    else:
        coords = x0_flat[binder_atom_mask]
        if coords.shape[0] < 2:
            return torch.tensor(0.0, device=x0_flat.device, requires_grad=True)
        dist = torch.cdist(coords, coords)

        token_idx = atom_to_token_idx[binder_atom_mask]
        res_idx = residue_index[token_idx]
        sep = (res_idx.unsqueeze(1) - res_idx.unsqueeze(0)).abs()
        bonded = sep <= bond_sep

        mask_valid = ~bonded
        mask_valid.fill_diagonal_(False)

        clash = torch.clamp(r_clash - dist, min=0) ** 2
        clash = clash * mask_valid.float()
        n_pairs = mask_valid.sum().clamp(min=1)
        return clash.sum() / n_pairs
