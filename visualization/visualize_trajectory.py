"""
visualize_trajectory.py — Data loading, parsing, RMSD computation,
plotting, and CLI for BindCraft checkpoint trajectories.
"""

import argparse
import json
import math
import os
from glob import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Step discovery
# ---------------------------------------------------------------------------

def discover_steps(checkpoint_dir):
    """Glob step_*/step_*_beam_0_meta.json, load meta, find PDB, return list of dicts.

    Each dict has keys: step (int), pdb_path (str), meta (dict).

    Raises ValueError if fewer than 2 usable steps exist (checked both before
    and after PDB-file filtering).
    """
    pattern = os.path.join(checkpoint_dir, "step_*", "step_*_beam_0_meta.json")
    meta_files = sorted(glob(pattern))

    if len(meta_files) < 2:
        raise ValueError(
            f"discover_steps: need at least 2 meta.json files, found {len(meta_files)} "
            f"in {checkpoint_dir}"
        )

    steps = []
    for meta_path in meta_files:
        with open(meta_path) as f:
            meta = json.load(f)

        step_dir = os.path.dirname(meta_path)
        pdb_filename = meta.get("pdb_file", "")
        pdb_path = os.path.join(step_dir, pdb_filename)

        if not os.path.isfile(pdb_path):
            # Skip steps whose PDB is missing but keep counting for the
            # pre-filter check (already passed above).
            continue

        steps.append({
            "step": meta["step"],
            "pdb_path": pdb_path,
            "meta": meta,
        })

    if len(steps) < 2:
        raise ValueError(
            f"discover_steps: fewer than 2 steps have a valid PDB file "
            f"(found {len(steps)}) in {checkpoint_dir}"
        )

    # Sort by step index so callers can rely on ordering.
    steps.sort(key=lambda d: d["step"])
    return steps


# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------

def parse_ca_atoms(pdb_path):
    """Parse CA atoms from a PDB file using split-based (non-fixed-width) parsing.

    Returns (binder_ca, target_ca) as numpy float64 arrays of shape (N, 3).
    Chain A = binder, chain B = target.

    Raises ValueError if either chain has no CA atoms.
    """
    binder_coords = []
    target_coords = []

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            parts = line.split()
            # PDB ATOM line fields (split-based):
            # 0:ATOM  1:serial  2:name  3:resName  4:chainID  5:resSeq
            # 6:x  7:y  8:z  9:occupancy  10:tempFactor  ...
            if len(parts) < 9:
                continue
            atom_name = parts[2]
            chain_id = parts[4]
            if atom_name != "CA":
                continue
            try:
                x, y, z = float(parts[6]), float(parts[7]), float(parts[8])
            except ValueError:
                continue

            if chain_id == "A":
                binder_coords.append([x, y, z])
            elif chain_id == "B":
                target_coords.append([x, y, z])

    if not binder_coords:
        raise ValueError(f"parse_ca_atoms: no chain-A CA atoms found in {pdb_path}")
    if not target_coords:
        raise ValueError(f"parse_ca_atoms: no chain-B CA atoms found in {pdb_path}")

    return np.array(binder_coords, dtype=np.float64), np.array(target_coords, dtype=np.float64)


# ---------------------------------------------------------------------------
# Kabsch RMSD
# ---------------------------------------------------------------------------

def kabsch_rmsd(P, Q):
    """Compute Kabsch-aligned RMSD between two (N, 3) coordinate arrays.

    Centers both arrays, uses SVD to find the optimal rotation (with
    reflection correction via the determinant sign), then computes RMSD.

    Args:
        P: (N, 3) numpy array — the array to rotate/align.
        Q: (N, 3) numpy array — the reference (fixed).

    Returns:
        float RMSD after optimal superposition.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    # Centre
    P_c = P - P.mean(axis=0)
    Q_c = Q - Q.mean(axis=0)

    # Covariance matrix
    H = P_c.T @ Q_c

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Correct reflection (improper rotation)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])

    # Optimal rotation
    R = Vt.T @ D @ U.T

    # Rotate P_c
    P_rot = P_c @ R.T

    # RMSD
    diff = P_rot - Q_c
    rmsd = math.sqrt((diff ** 2).sum() / len(P))
    return rmsd


# ---------------------------------------------------------------------------
# Target-alignment helper
# ---------------------------------------------------------------------------

def _apply_target_alignment(binder, target, ref_target):
    """Superpose *target* onto *ref_target* via Kabsch, apply the same rigid
    transform to *binder*.

    Args:
        binder:     (N_b, 3) numpy array — binder CA coords to be transformed.
        target:     (N_t, 3) numpy array — target CA coords (moving frame).
        ref_target: (N_t, 3) numpy array — target CA coords (reference frame).

    Returns:
        transformed_binder: (N_b, 3) numpy array in the reference frame.
    """
    binder = np.asarray(binder, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    ref_target = np.asarray(ref_target, dtype=np.float64)

    t_cent = target.mean(axis=0)
    r_cent = ref_target.mean(axis=0)

    target_c = target - t_cent
    ref_c = ref_target - r_cent

    H = target_c.T @ ref_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    # Apply same transform to binder: translate by t_cent, rotate, translate to r_cent
    binder_transformed = (binder - t_cent) @ R.T + r_cent
    return binder_transformed


# ---------------------------------------------------------------------------
# Full RMSD computation over trajectory
# ---------------------------------------------------------------------------

def _positional_rmsd(A, B):
    """Direct positional RMSD between (N, 3) arrays — no centering, no rotation.

    Used for binder displacement after target alignment, because the target
    superposition already places both frames in a common coordinate system;
    applying a second Kabsch fit on the binder would remove any remaining
    rigid-body displacement.
    """
    diff = np.asarray(A, dtype=np.float64) - np.asarray(B, dtype=np.float64)
    return math.sqrt((diff ** 2).sum() / len(A))


def compute_all_rmsds(steps):
    """For each step in *steps*, align on target CAs and measure binder RMSDs.

    After target superposition all binder frames share a common coordinate
    system, so binder displacement is measured as a direct positional RMSD
    (no further Kabsch fit on the binder).

    Returns a dict with:
        binder_vs_0:    list of float — binder RMSD vs step 0, after target alignment.
        binder_adjacent: list of float — binder RMSD vs previous step (NaN for step 0).
        target_vs_0:    list of float — target self-aligned RMSD vs step 0.

    Raises ValueError if CA counts are inconsistent across steps.
    """
    # Parse all steps
    all_binder = []
    all_target = []
    for s in steps:
        b, t = parse_ca_atoms(s["pdb_path"])
        all_binder.append(b)
        all_target.append(t)

    # Validate count consistency
    n_b0 = len(all_binder[0])
    n_t0 = len(all_target[0])
    for i, (b, t) in enumerate(zip(all_binder, all_target)):
        if len(b) != n_b0:
            raise ValueError(
                f"compute_all_rmsds: binder CA count mismatch at step index {i}: "
                f"expected {n_b0}, got {len(b)}"
            )
        if len(t) != n_t0:
            raise ValueError(
                f"compute_all_rmsds: target CA count mismatch at step index {i}: "
                f"expected {n_t0}, got {len(t)}"
            )

    ref_target = all_target[0]

    # Pre-compute all binder coords aligned into the reference (step 0 target) frame.
    # This ensures all comparisons use a consistent coordinate frame.
    aligned_binders = [
        _apply_target_alignment(b, t, ref_target)
        for b, t in zip(all_binder, all_target)
    ]

    # Step-0 aligned binder is the reference for binder_vs_0.
    ref_binder_aligned = aligned_binders[0]

    binder_vs_0 = []
    binder_adjacent = []
    target_vs_0 = []

    for i, (b_aligned, t) in enumerate(zip(aligned_binders, all_target)):
        # Target RMSD vs step 0 (target aligning itself to ref)
        t_rmsd = kabsch_rmsd(t, ref_target)
        target_vs_0.append(t_rmsd)

        # Binder displacement vs step 0, both in the reference frame.
        # Direct positional RMSD — no further Kabsch on binder because
        # the target superposition already defines a common frame.
        b_rmsd_0 = _positional_rmsd(b_aligned, ref_binder_aligned)
        binder_vs_0.append(b_rmsd_0)

        # Adjacent displacement
        if i == 0:
            binder_adjacent.append(float("nan"))
        else:
            adj_rmsd = _positional_rmsd(b_aligned, aligned_binders[i - 1])
            binder_adjacent.append(adj_rmsd)

    return {
        "binder_vs_0": binder_vs_0,
        "binder_adjacent": binder_adjacent,
        "target_vs_0": target_vs_0,
    }


# ---------------------------------------------------------------------------
# Style helper
# ---------------------------------------------------------------------------

def _get_style():
    """Return seaborn-v0_8-whitegrid if available, else ggplot."""
    available = plt.style.available
    if "seaborn-v0_8-whitegrid" in available:
        return "seaborn-v0_8-whitegrid"
    return "ggplot"


# ---------------------------------------------------------------------------
# Fig 1: RMSD evolution
# ---------------------------------------------------------------------------

def plot_rmsd(steps, rmsds, best_step, output_path, dpi=150):
    """Plot RMSD evolution over trajectory steps.

    Args:
        steps:       list of step dicts (from discover_steps).
        rmsds:       dict from compute_all_rmsds.
        best_step:   int — step index of the best structure.
        output_path: str — path to write PNG.
        dpi:         int — figure DPI.
    """
    with plt.style.context(_get_style()):
        fig, ax = plt.subplots(figsize=(10, 6))

        step_nums = np.array([s["step"] for s in steps])
        bv0 = np.array(rmsds["binder_vs_0"])
        badj = np.array(rmsds["binder_adjacent"])
        tv0 = np.array(rmsds["target_vs_0"])

        # Build accepted mask from meta
        accepted = np.array([bool(s["meta"].get("accepted", False)) for s in steps])

        # Continuous line for binder_vs_0 (no markers)
        ax.plot(step_nums, bv0, color="tab:blue", linewidth=1.2, zorder=1)

        # Accepted scatter (filled circles)
        ax.scatter(
            step_nums[accepted], bv0[accepted],
            color="tab:blue", s=40, zorder=3,
            label="Binder vs s0 (accepted)",
        )
        # Rejected scatter (hollow circles)
        ax.scatter(
            step_nums[~accepted], bv0[~accepted],
            facecolors="none", edgecolors="tab:blue", s=40, zorder=3,
            label="Binder vs s0 (rejected)",
        )

        # Adjacent RMSD (orange dashed); skip NaN at step 0
        ax.plot(step_nums, badj, color="tab:orange", linestyle="--",
                linewidth=1.2, label="Binder adjacent", zorder=1)

        # Target vs step 0 (gray dotted)
        ax.plot(step_nums, tv0, color="gray", linestyle=":",
                linewidth=1.2, label="Target vs step 0", zorder=1)

        # Best step marker
        best_idx = np.where(step_nums == best_step)[0]
        if len(best_idx):
            idx = best_idx[0]
            ax.scatter([step_nums[idx]], [bv0[idx]],
                       marker="*", s=200, color="red", zorder=4, label=f"Best (step {best_step})")

        ax.set_xlabel("Step")
        ax.set_ylabel("RMSD (Å)")
        ax.set_title("RMSD Evolution")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2: Total loss
# ---------------------------------------------------------------------------

def plot_total_loss(steps, best_step, output_path, dpi=150):
    """Plot total loss over trajectory steps.

    Args:
        steps:       list of step dicts (from discover_steps).
        best_step:   int — step index of the best structure.
        output_path: str — path to write PNG.
        dpi:         int — figure DPI.
    """
    with plt.style.context(_get_style()):
        fig, ax = plt.subplots(figsize=(10, 6))

        step_nums = np.array([s["step"] for s in steps])
        total_loss = np.array([s["meta"]["loss_dict"]["total"] for s in steps])
        accepted = np.array([bool(s["meta"].get("accepted", False)) for s in steps])

        # Continuous line (no markers)
        ax.plot(step_nums, total_loss, color="tab:blue", linewidth=1.2, zorder=1)

        # Accepted scatter (blue circles)
        ax.scatter(
            step_nums[accepted], total_loss[accepted],
            marker="o", color="tab:blue", s=40, zorder=3,
            label="Accepted",
        )
        # Rejected scatter (red crosses)
        ax.scatter(
            step_nums[~accepted], total_loss[~accepted],
            marker="x", color="red", s=40, zorder=3,
            label="Rejected",
        )

        # Best step marker (gold star with black edge)
        best_idx = np.where(step_nums == best_step)[0]
        if len(best_idx):
            idx = best_idx[0]
            ax.scatter([step_nums[idx]], [total_loss[idx]],
                       marker="*", s=200, color="gold",
                       edgecolors="black", linewidths=0.8,
                       zorder=4, label=f"Best (step {best_step})")

        ax.set_xlabel("Step")
        ax.set_ylabel("Total Loss")
        ax.set_title("Total Loss Evolution")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3: Confidence metrics (3 subplots sharing X)
# ---------------------------------------------------------------------------

def plot_confidence_metrics(steps, output_path, dpi=150):
    """Plot iPTM, mean pLDDT, and Rg Loss in 3 vertically stacked subplots.

    Args:
        steps:       list of step dicts (from discover_steps).
        output_path: str — path to write PNG.
        dpi:         int — figure DPI.
    """
    with plt.style.context(_get_style()):
        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

        step_nums = [s["step"] for s in steps]

        iptm_vals = [s["meta"]["summary_confidence"]["iptm"] for s in steps]
        plddt_vals = [s["meta"]["summary_confidence"]["plddt"] for s in steps]
        rg_vals = [s["meta"]["loss_dict"]["rg"] for s in steps]

        axes[0].plot(step_nums, iptm_vals, color="green", linewidth=1.2)
        axes[0].set_ylabel("iPTM")
        axes[0].set_title("Confidence Metrics")

        axes[1].plot(step_nums, plddt_vals, color="purple", linewidth=1.2)
        axes[1].set_ylabel("Mean pLDDT")

        axes[2].plot(step_nums, rg_vals, color="brown", linewidth=1.2)
        axes[2].set_ylabel("Rg Loss")
        axes[2].set_xlabel("Step")

        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4: All individual losses (3x4 grid)
# ---------------------------------------------------------------------------

_LOSS_KEYS = [
    "plddt", "pae", "i_pae", "con", "i_con", "iptm",
    "rg", "helix", "clash", "i_clash", "pde", "i_pde",
]


def plot_all_losses(steps, output_path, dpi=150):
    """Plot 12 individual loss components in a 3x4 grid.

    Args:
        steps:       list of step dicts (from discover_steps).
        output_path: str — path to write PNG.
        dpi:         int — figure DPI.
    """
    with plt.style.context(_get_style()):
        fig, axes = plt.subplots(3, 4, figsize=(14, 10))

        step_nums = [s["step"] for s in steps]

        for ax, key in zip(axes.flat, _LOSS_KEYS):
            vals = [s["meta"]["loss_dict"].get(key, float("nan")) for s in steps]
            ax.plot(step_nums, vals, linewidth=1.0)
            ax.set_title(key, fontsize=9)
            ax.set_xlabel("Step", fontsize=7)
            ax.set_ylabel("Loss", fontsize=7)
            ax.tick_params(labelsize=6)

        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize BindCraft checkpoint trajectory."
    )
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Directory containing step_* subdirectories.")
    parser.add_argument("--output_dir", default="figures",
                        help="Directory to write PNG figures (default: figures).")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Figure DPI (default: 150).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading steps from: {args.checkpoint_dir}")
    steps = discover_steps(args.checkpoint_dir)
    print(f"  Found {len(steps)} usable steps.")

    # Load best step index from best.json if present
    best_json = os.path.join(args.checkpoint_dir, "best.json")
    if os.path.isfile(best_json):
        with open(best_json) as f:
            best_data = json.load(f)
        best_step = best_data.get("step", steps[0]["step"])
    else:
        # Fall back to step with lowest total loss
        best_step = min(steps, key=lambda s: s["meta"]["loss_dict"]["total"])["step"]
    print(f"  Best step: {best_step}")

    print("Computing RMSDs...")
    rmsds = compute_all_rmsds(steps)

    print("Generating figures...")

    out1 = os.path.join(args.output_dir, "rmsd_evolution.png")
    plot_rmsd(steps, rmsds, best_step, out1, dpi=args.dpi)
    print(f"  Saved: {out1}")

    out2 = os.path.join(args.output_dir, "total_loss.png")
    plot_total_loss(steps, best_step, out2, dpi=args.dpi)
    print(f"  Saved: {out2}")

    out3 = os.path.join(args.output_dir, "confidence_metrics.png")
    plot_confidence_metrics(steps, out3, dpi=args.dpi)
    print(f"  Saved: {out3}")

    out4 = os.path.join(args.output_dir, "all_losses.png")
    plot_all_losses(steps, out4, dpi=args.dpi)
    print(f"  Saved: {out4}")

    print("Done.")


if __name__ == "__main__":
    main()
