#!/usr/bin/env python3
import argparse
import json
import os
from glob import glob

import numpy as np


DEFAULT_TARGET_CHAIN = None
DEFAULT_BINDER_CHAIN = None


def _extract_step_num(path):
    step_dir = os.path.basename(os.path.dirname(path))
    try:
        return int(step_dir.split("_")[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Could not parse step number from {path}") from exc


def _checkpoint_roots(run_dir):
    roots = []
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    if os.path.isdir(checkpoint_dir):
        roots.append(checkpoint_dir)
    if os.path.isdir(run_dir):
        roots.append(run_dir)
    return list(dict.fromkeys(roots))


def discover_steps(run_dir, beam=0, accepted_only=False):
    checkpoint_roots = _checkpoint_roots(run_dir)
    meta_files = []
    for checkpoint_dir in checkpoint_roots:
        patterns = [
            os.path.join(checkpoint_dir, "step_*", f"step_*_beam_{beam}_meta.json"),
            os.path.join(checkpoint_dir, "*", "step_*", f"step_*_beam_{beam}_meta.json"),
        ]
        for pattern in patterns:
            meta_files.extend(glob(pattern))
    meta_files = sorted(set(meta_files))

    if not meta_files:
        searched = ", ".join(checkpoint_roots) or run_dir
        raise ValueError(f"No meta files found for beam {beam} in {searched}")

    steps = []
    skipped = []
    for meta_path in meta_files:
        with open(meta_path) as f:
            meta = json.load(f)

        if accepted_only and not meta.get("accepted", False):
            continue

        step_dir = os.path.dirname(meta_path)
        step_num = meta.get("step", _extract_step_num(meta_path))
        pdb_file = meta.get("pdb_file") or f"step_{step_num:04d}_beam_{beam}_complex.pdb"
        pdb_path = os.path.join(step_dir, pdb_file)

        if not os.path.isfile(pdb_path):
            skipped.append((step_num, pdb_path))
            continue

        steps.append({
            "step": int(step_num),
            "meta": meta,
            "pdb_path": pdb_path,
        })

    for step_num, pdb_path in skipped:
        print(f"Warning: skipping step {step_num} because PDB is missing: {pdb_path}")

    if not steps:
        searched = ", ".join(checkpoint_roots) or run_dir
        raise ValueError(f"No usable steps found in {searched}")

    steps.sort(key=lambda item: item["step"])
    return steps


def _evenly_sample(items, max_frames):
    if max_frames is None or max_frames <= 0 or len(items) <= max_frames:
        return list(items)
    if max_frames == 1:
        return [items[0]]

    indices = {0, len(items) - 1}
    for i in range(1, max_frames - 1):
        idx = round(i * (len(items) - 1) / (max_frames - 1))
        indices.add(idx)

    sampled = [items[i] for i in sorted(indices)]
    while len(sampled) < max_frames:
        positions = sorted(indices)
        gaps = [(positions[i + 1] - positions[i], i) for i in range(len(positions) - 1)]
        if not gaps:
            break
        _, gap_index = max(gaps)
        new_idx = (positions[gap_index] + positions[gap_index + 1]) // 2
        if new_idx in indices:
            break
        indices.add(new_idx)
        sampled = [items[i] for i in sorted(indices)]

    return sampled


def sample_steps(steps, stride=None, max_frames=None):
    sampled = list(steps)

    if stride is not None:
        if stride <= 0:
            raise ValueError("--stride must be a positive integer")
        if stride > 1:
            sampled = sampled[::stride]
            if sampled[-1]["step"] != steps[-1]["step"]:
                sampled.append(steps[-1])

    return _evenly_sample(sampled, max_frames)


def _parse_atom_line(line):
    if not line.startswith(("ATOM", "HETATM")):
        return None

    parts = line.split()
    atom_name = line[12:16].strip() or (parts[2] if len(parts) >= 3 else "")
    chain_id = line[21].strip() or (parts[4] if len(parts) >= 5 else "")

    try:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except ValueError:
        if len(parts) < 9:
            return None
        try:
            x, y, z = float(parts[6]), float(parts[7]), float(parts[8])
        except ValueError:
            return None

    return {
        "atom_name": atom_name,
        "chain_id": chain_id,
        "coords": np.array([x, y, z], dtype=np.float64),
    }


def extract_chain_ca_coords(pdb_text, chain_id):
    coords = []
    for line in pdb_text.splitlines():
        parsed = _parse_atom_line(line)
        if parsed is None:
            continue
        if parsed["atom_name"] == "CA" and parsed["chain_id"] == chain_id:
            coords.append(parsed["coords"])

    if not coords:
        raise ValueError(f"No CA atoms found for chain {chain_id}")

    return np.array(coords, dtype=np.float64)


def list_chain_ids(pdb_text):
    chain_ids = []
    seen = set()
    for line in pdb_text.splitlines():
        parsed = _parse_atom_line(line)
        if parsed is None:
            continue
        chain_id = parsed["chain_id"]
        if not chain_id or chain_id in seen:
            continue
        seen.add(chain_id)
        chain_ids.append(chain_id)
    return chain_ids


def infer_chain_roles(pdb_text):
    chain_ids = list_chain_ids(pdb_text)
    if len(chain_ids) < 2:
        raise ValueError(f"Expected at least 2 chains, found {chain_ids}")
    binder_chain = chain_ids[0]
    target_chains = chain_ids[1:]
    return binder_chain, target_chains


def extract_target_ca_coords(pdb_text, target_chains):
    arrays = [extract_chain_ca_coords(pdb_text, chain_id) for chain_id in target_chains]
    return np.concatenate(arrays, axis=0)


def fit_rigid_transform(moving, reference):
    moving = np.asarray(moving, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    if moving.shape != reference.shape:
        raise ValueError(
            f"Target CA count mismatch: moving shape {moving.shape}, reference shape {reference.shape}"
        )
    if moving.ndim != 2 or moving.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) coordinates, got {moving.shape}")

    moving_center = moving.mean(axis=0)
    reference_center = reference.mean(axis=0)
    moving_centered = moving - moving_center
    reference_centered = reference - reference_center

    covariance = moving_centered.T @ reference_centered
    U, _, Vt = np.linalg.svd(covariance)
    sign = -1.0 if np.linalg.det(Vt.T @ U.T) < 0 else 1.0
    correction = np.diag([1.0, 1.0, sign])
    rotation = Vt.T @ correction @ U.T

    return moving_center, rotation, reference_center


def transform_pdb_text(pdb_text, moving_center, rotation, reference_center):
    transformed_lines = []

    for line in pdb_text.splitlines(keepends=True):
        parsed = _parse_atom_line(line)
        if parsed is None:
            transformed_lines.append(line)
            continue

        transformed = (parsed["coords"] - moving_center) @ rotation.T + reference_center
        raw_line = line[:-1] if line.endswith("\n") else line
        newline = "\n" if line.endswith("\n") else ""
        raw_line = raw_line.ljust(54)
        raw_line = (
            raw_line[:30]
            + f"{transformed[0]:8.3f}{transformed[1]:8.3f}{transformed[2]:8.3f}"
            + raw_line[54:]
        )
        transformed_lines.append(raw_line + newline)

    return "".join(transformed_lines)


def extract_pdb_subset(pdb_text, chain_ids):
    chain_ids = set(chain_ids)
    selected_lines = []

    for line in pdb_text.splitlines():
        parsed = _parse_atom_line(line)
        if parsed is None:
            continue
        if parsed["chain_id"] in chain_ids:
            selected_lines.append(line.rstrip())

    if not selected_lines:
        raise ValueError(f"No atoms found for chains {sorted(chain_ids)}")

    return "\n".join(selected_lines) + "\nEND\n"


def build_aligned_frames(steps):
    with open(steps[0]["pdb_path"]) as f:
        reference_pdb = f.read()

    binder_chain, target_chains = infer_chain_roles(reference_pdb)
    reference_target = extract_target_ca_coords(reference_pdb, target_chains)
    reference_count = len(reference_target)
    static_target_pdb = extract_pdb_subset(reference_pdb, target_chains)
    frames = []

    for step in steps:
        with open(step["pdb_path"]) as f:
            pdb_text = f.read()

        moving_target = extract_target_ca_coords(pdb_text, target_chains)
        if len(moving_target) != reference_count:
            raise ValueError(
                f"Step {step['step']} has {len(moving_target)} target CA atoms; expected {reference_count}"
            )

        moving_center, rotation, reference_center = fit_rigid_transform(moving_target, reference_target)
        aligned_pdb = transform_pdb_text(pdb_text, moving_center, rotation, reference_center)
        binder_pdb = extract_pdb_subset(aligned_pdb, [binder_chain])
        frames.append({
            "step": step["step"],
            "meta": step["meta"],
            "binder_pdb": binder_pdb,
        })

    return frames, static_target_pdb, binder_chain, target_chains


def _meta_value(meta, *path):
    value = meta
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def build_3dmol_style():
    """Enhanced 3Dmol style configuration for beautiful protein visualization."""
    return {
        "cartoon": {
            "style": "trace",
            "opacity": 0.9,
            "color": "spectrum",
            "resolution": 3,
            "linewidth": 0.5,
        }
    }


def build_html(frames, output_title, fps, binder_chain, target_chains, static_target_pdb):
    payload = []
    chain_colors = {
        binder_chain: "#2563eb",
    }
    palette = ["#f59e0b", "#8b5cf6", "#10b981", "#ef4444", "#14b8a6"]
    for idx, chain in enumerate(target_chains):
        chain_colors[chain] = palette[idx % len(palette)]

    for frame in frames:
        meta = frame["meta"]
        payload.append({
            "step": frame["step"],
            "binder_pdb": frame["binder_pdb"],
            "accepted": bool(meta.get("accepted", False)),
            "is_global_best": bool(meta.get("is_global_best", False)),
            "total_loss": _meta_value(meta, "loss_dict", "total") if _meta_value(meta, "loss_dict", "total") is not None else meta.get("total_loss"),
            "ranking_score": _meta_value(meta, "summary_confidence", "ranking_score"),
        })

    legend_items = [
        {"chain": chain, "color": color}
        for chain, color in chain_colors.items()
    ]

    frame_json = json.dumps(payload, ensure_ascii=False)
    title_json = json.dumps(output_title, ensure_ascii=False)
    chain_colors_json = json.dumps(chain_colors, ensure_ascii=False)
    legend_items_json = json.dumps(legend_items, ensure_ascii=False)
    static_target_json = json.dumps(static_target_pdb, ensure_ascii=False)

    return f'''<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{output_title}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
  <style>
    :root {{
      color-scheme: light;
      --bg: #0f172a;
      --panel: #ffffff;
      --panel-2: #f8fafc;
      --line: #dbe4f0;
      --text: #0f172a;
      --muted: #64748b;
      --accent: #2563eb;
      --accent-2: #1d4ed8;
      --ok-bg: #dcfce7;
      --ok-text: #166534;
      --bad-bg: #fee2e2;
      --bad-text: #991b1b;
      --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
      --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -2px rgba(0,0,0,0.1);
      --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -4px rgba(0,0,0,0.1);
      --shadow-xl: 0 20px 25px -5px rgba(0,0,0,0.1), 0 8px 10px -6px rgba(0,0,0,0.1);
      --radius: 18px;
      --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
      color: var(--text);
      min-height: 100vh;
      overflow-x: hidden;
    }}
    .app-container {{
      display: flex;
      min-height: 100vh;
    }}
    .main-content {{
      flex: 1;
      padding: 20px;
      display: flex;
      flex-direction: column;
      transition: var(--transition);
    }}
    .main-content.expanded {{
      padding: 12px;
    }}
    .hero {{
      margin-bottom: 16px;
      padding: 0 8px;
    }}
    .hero h1 {{
      margin: 0 0 6px;
      font-size: 24px;
      font-weight: 700;
      color: white;
      letter-spacing: -0.02em;
    }}
    .hero p {{
      margin: 0;
      color: rgba(255,255,255,0.6);
      font-size: 13px;
    }}
    .viewer-wrapper {{
      flex: 1;
      background: rgba(255,255,255,0.95);
      border-radius: var(--radius);
      box-shadow: var(--shadow-xl);
      overflow: hidden;
      display: flex;
      flex-direction: column;
      transition: var(--transition);
      position: relative;
    }}
    .viewer-wrapper.expanded {{
      border-radius: 12px;
    }}
    .controls {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 12px 16px;
      background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
      border-bottom: 1px solid var(--line);
      flex-wrap: wrap;
    }}
    .btn-group {{
      display: flex;
      gap: 6px;
    }}
    .btn {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      padding: 10px 16px;
      border: none;
      border-radius: 12px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: var(--transition);
      font-family: inherit;
    }}
    .btn-primary {{
      background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
      color: white;
      box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
    }}
    .btn-primary:hover {{
      background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
    }}
    .btn-ghost {{
      background: rgba(255,255,255,0.8);
      color: #475569;
      border: 1px solid var(--line);
    }}
    .btn-ghost:hover {{
      background: white;
      border-color: #cbd5e1;
    }}
    .btn-icon {{
      width: 40px;
      padding: 10px;
    }}
    .view-mode-group {{
      display: flex;
      gap: 4px;
      margin-left: auto;
      background: rgba(0,0,0,0.05);
      padding: 4px;
      border-radius: 10px;
    }}
    .view-mode-btn {{
      padding: 8px 14px;
      font-size: 12px;
      font-weight: 500;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      transition: var(--transition);
      background: transparent;
      color: var(--muted);
      font-family: inherit;
    }}
    .view-mode-btn.active {{
      background: white;
      color: var(--accent);
      box-shadow: var(--shadow-sm);
    }}
    .view-mode-btn:hover:not(.active) {{
      color: var(--text);
    }}
    .slider-section {{
      flex: 1;
      min-width: 200px;
      max-width: 400px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .slider-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .slider-header strong {{
      font-size: 14px;
      color: #1e293b;
      font-weight: 600;
    }}
    .slider-header span {{
      font-size: 13px;
      color: var(--muted);
      font-family: 'JetBrains Mono', monospace;
    }}
    .slider-wrap {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .slider-wrap input[type=range] {{
      flex: 1;
      height: 6px;
      border-radius: 3px;
      background: linear-gradient(90deg, var(--accent), #60a5fa);
      -webkit-appearance: none;
      cursor: pointer;
    }}
    .slider-wrap input[type=range]::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: white;
      border: 3px solid var(--accent);
      box-shadow: 0 2px 6px rgba(0,0,0,0.2);
      cursor: pointer;
      transition: var(--transition);
    }}
    .slider-wrap input[type=range]::-webkit-slider-thumb:hover {{
      transform: scale(1.1);
    }}
    .fps-control {{
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 0 8px;
      border-left: 1px solid var(--line);
      margin-left: 8px;
    }}
    .fps-control label {{
      font-size: 12px;
      color: var(--muted);
      font-weight: 500;
    }}
    .fps-control input {{
      width: 50px;
      padding: 6px 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      font-size: 13px;
      font-family: 'JetBrains Mono', monospace;
      text-align: center;
    }}
    .viewer {{
      flex: 1;
      min-height: 500px;
      background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
      position: relative;
    }}
    .viewer-hint {{
      position: absolute;
      bottom: 12px;
      left: 12px;
      background: rgba(255,255,255,0.9);
      backdrop-filter: blur(8px);
      padding: 8px 12px;
      border-radius: 10px;
      font-size: 11px;
      color: var(--muted);
      display: flex;
      gap: 8px;
      box-shadow: var(--shadow-md);
    }}
    .viewer-hint kbd {{
      display: inline-block;
      padding: 2px 6px;
      border-radius: 4px;
      background: white;
      border: 1px solid var(--line);
      font-family: 'JetBrains Mono', monospace;
      font-size: 10px;
    }}
    .sidebar {{
      width: 340px;
      background: rgba(255,255,255,0.98);
      backdrop-filter: blur(20px);
      border-left: 1px solid rgba(255,255,255,0.1);
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      transition: var(--transition);
      overflow-y: auto;
    }}
    .sidebar.collapsed {{
      width: 0;
      padding: 0;
      opacity: 0;
      overflow: hidden;
    }}
    .sidebar-toggle {{
      position: fixed;
      right: 16px;
      top: 20px;
      z-index: 100;
      width: 44px;
      height: 44px;
      border-radius: 12px;
      background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
      color: white;
      border: none;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: var(--shadow-lg);
      transition: var(--transition);
      font-size: 18px;
    }}
    .sidebar-toggle:hover {{
      transform: scale(1.05);
      box-shadow: 0 8px 20px rgba(37, 99, 235, 0.4);
    }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
    }}
    .stat-card {{
      background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      transition: var(--transition);
    }}
    .stat-card:hover {{
      border-color: #cbd5e1;
      transform: translateY(-2px);
      box-shadow: var(--shadow-md);
    }}
    .stat-label {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
      margin-bottom: 6px;
      font-weight: 600;
    }}
    .stat-value {{
      font-size: 22px;
      font-weight: 700;
      color: #1e293b;
      font-family: 'JetBrains Mono', monospace;
    }}
    .stat-card.status .stat-value {{
      font-size: 14px;
    }}
    .status-pill {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 28px;
      border-radius: 999px;
      padding: 6px 12px;
      font-size: 12px;
      font-weight: 700;
      width: 100%;
    }}
    .status-true {{ background: linear-gradient(135deg, #86efac 0%, #4ade80 100%); color: #166534; }}
    .status-false {{ background: linear-gradient(135deg, #fecaca 0%, #f87171 100%); color: #991b1b; }}
    .section-title {{
      font-size: 14px;
      font-weight: 700;
      color: #1e293b;
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .section-title::before {{
      content: '';
      width: 4px;
      height: 16px;
      background: linear-gradient(180deg, var(--accent), #60a5fa);
      border-radius: 2px;
    }}
    .legend {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px;
      background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
      border: 1px solid var(--line);
      border-radius: 12px;
      transition: var(--transition);
    }}
    .legend-item:hover {{
      border-color: #cbd5e1;
      transform: translateX(4px);
    }}
    .swatch {{
      width: 24px;
      height: 24px;
      border-radius: 8px;
      flex-shrink: 0;
      box-shadow: inset 0 -2px 4px rgba(0,0,0,0.1);
    }}
    .chain-name {{
      font-weight: 700;
      font-size: 14px;
      min-width: 70px;
    }}
    .chain-role {{
      color: var(--muted);
      font-size: 12px;
      padding: 2px 8px;
      background: rgba(0,0,0,0.05);
      border-radius: 6px;
    }}
    .download-btn {{
      width: 100%;
      padding: 14px;
      background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
      color: white;
      border: none;
      border-radius: 12px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: var(--transition);
      font-family: inherit;
    }}
    .download-btn:hover {{
      background: linear-gradient(135deg, #334155 0%, #475569 100%);
      transform: translateY(-2px);
    }}
    .progress-track {{
      height: 8px;
      background: #e2e8f0;
      border-radius: 4px;
      overflow: hidden;
      position: relative;
    }}
    .progress-fill {{
      height: 100%;
      background: linear-gradient(90deg, var(--accent), #60a5fa);
      border-radius: 4px;
      transition: width 0.3s ease;
    }}
    @media (max-width: 1024px) {{
      .app-container {{ flex-direction: column; }}
      .sidebar {{
        width: 100%;
        border-left: none;
        border-top: 1px solid var(--line);
      }}
      .sidebar.collapsed {{
        height: 0;
        padding: 0;
      }}
    }}
  </style>
</head>
<body>
  <button class="sidebar-toggle" id="sidebar-toggle" title="Toggle sidebar (S)">
    ☰
  </button>
  <div class="app-container">
    <div class="main-content" id="main-content">
      <div class="hero">
        <h1 id="title"></h1>
        <p>Target 结构固定为 step 0，仅更新 binder 轨迹。拖动时间轴或按空格键播放。</p>
      </div>
      <div class="viewer-wrapper" id="viewer-wrapper">
        <div class="controls">
          <div class="btn-group">
            <button id="play-btn" class="btn btn-primary">▶ Play</button>
            <button id="pause-btn" class="btn btn-ghost">⏸</button>
            <button id="prev-btn" class="btn btn-ghost btn-icon">◀</button>
            <button id="next-btn" class="btn btn-ghost btn-icon">▶</button>
          </div>
          <div class="slider-section">
            <div class="slider-header">
              <strong id="frame-label">Frame 1 / {len(frames)}</strong>
              <span id="step-label">Step 0</span>
            </div>
            <div class="slider-wrap">
              <input id="frame-slider" type="range" min="0" max="{max(len(frames) - 1, 0)}" value="0" step="1">
              <div class="progress-track" style="width: 60px;">
                <div class="progress-fill" id="progress-fill" style="width: {100/len(frames) if len(frames) > 0 else 100}%;"></div>
              </div>
            </div>
          </div>
          <div class="fps-control">
            <label>FPS</label>
            <input id="fps-input" type="number" min="1" max="60" value="{int(fps)}">
          </div>
          <div class="view-mode-group">
            <button class="view-mode-btn active" data-mode="cartoon">Cartoon</button>
            <button class="view-mode-btn" data-mode="trace">Trace</button>
            <button class="view-mode-btn" data-mode="ribbon">Ribbon</button>
            <button class="view-mode-btn" data-mode="stick">Stick</button>
            <button class="view-mode-btn" data-mode="sphere">Sphere</button>
            <button class="view-mode-btn" data-mode="surface">Surface</button>
          </div>
        </div>
        <div id="viewer" class="viewer"></div>
        <div class="viewer-hint">
          <span><kbd>Space</kbd> 播放/暂停</span>
          <span><kbd>←</kbd><kbd>→</kbd> 切换帧</span>
          <span><kbd>S</kbd> 侧边栏</span>
          <span><kbd>F</kbd> 全屏</span>
        </div>
      </div>
    </div>
    <div class="sidebar" id="sidebar">
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-label">Step</div>
          <div class="stat-value" id="step-value">0</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Total Loss</div>
          <div class="stat-value" id="loss-value">-</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Ranking Score</div>
          <div class="stat-value" id="ranking-value">-</div>
        </div>
        <div class="stat-card status">
          <div class="stat-label">Accepted</div>
          <div class="stat-value"><span class="status-pill" id="accepted-value">-</span></div>
        </div>
        <div class="stat-card status">
          <div class="stat-label">Global Best</div>
          <div class="stat-value"><span class="status-pill" id="best-value">-</span></div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Frames</div>
          <div class="stat-value">{len(frames)}</div>
        </div>
      </div>
      <div class="section-title">Chain Colors</div>
      <div class="legend" id="legend"></div>
      <button class="download-btn" id="download-btn">
        ⬇ Download Current PDB
      </button>
    </div>
  </div>
  <script>
    const title = {title_json};
    const chainColors = {chain_colors_json};
    const legendItems = {legend_items_json};
    const staticTargetPdb = {static_target_json};
    const frames = {frame_json};

    document.getElementById('title').textContent = title;

    const viewer = $3Dmol.createViewer(document.getElementById('viewer'), {{
      backgroundColor: 'linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%)',
      antialias: true,
      id: 'mainViewer'
    }});
    const slider = document.getElementById('frame-slider');
    const frameLabel = document.getElementById('frame-label');
    const stepLabel = document.getElementById('step-label');
    const fpsInput = document.getElementById('fps-input');
    const stepValue = document.getElementById('step-value');
    const lossValue = document.getElementById('loss-value');
    const rankingValue = document.getElementById('ranking-value');
    const acceptedValue = document.getElementById('accepted-value');
    const bestValue = document.getElementById('best-value');
    const legend = document.getElementById('legend');
    const sidebar = document.getElementById('sidebar');
    const mainContent = document.getElementById('main-content');
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const viewerWrapper = document.getElementById('viewer-wrapper');
    const progressFill = document.getElementById('progress-fill');

    let currentIndex = 0;
    let playTimer = null;
    let savedView = null;
    let targetModel = null;
    let binderModel = null;
    let currentViewMode = 'cartoon';

    // Build color mapping per chain for 3Dmol colorscheme
    function getChainColorMap() {{
      const map = {{}};
      legendItems.forEach(item => {{ map[item.chain] = item.color; }});
      return map;
    }}

    // Apply view mode to a model with chain-color mapping
    function applyStyle(model, chains) {{
      const chainMap = getChainColorMap();
      const filtered = {{}};
      chains.forEach(c => {{ filtered[c] = chainMap[c]; }});
      const style = {{
        cartoon: {{ colorscheme: {{ prop: 'chain', map: filtered }}, opacity: 1.0 }},
        ribbon: {{ cartoon: {{ style: "oval", colorscheme: {{ prop: 'chain', map: filtered }}, opacity: 0.9, thickness: 0.5 }} }},
        trace: {{ cartoon: {{ style: "trace", colorscheme: {{ prop: 'chain', map: filtered }}, opacity: 0.95, linewidth: 1.0 }} }},
        surface: {{ surface: {{ colorscheme: {{ prop: 'chain', map: filtered }}, opacity: 0.75 }} }},
        stick: {{ stick: {{ colorscheme: {{ prop: 'chain', map: filtered }}, radius: 0.25 }} }},
        sphere: {{ sphere: {{ colorscheme: {{ prop: 'chain', map: filtered }}, scale: 0.5 }} }}
      }};
      model.setStyle({{}}, style[currentViewMode]);
    }}

    // Initialize legend
    legendItems.forEach((item, index) => {{
      const div = document.createElement('div');
      div.className = 'legend-item';
      const role = index === 0 ? 'Binder' : 'Target';
      div.innerHTML = `<span class="swatch" style="background:${{item.color}}"></span><span class="chain-name">Chain ${{item.chain}}</span><span class="chain-role">${{role}}</span>`;
      legend.appendChild(div);
    }});

    function formatNumber(value) {{
      if (value === null || value === undefined || Number.isNaN(value)) return '-';
      return Number(value).toFixed(4);
    }}

    function setPill(element, value) {{
      element.textContent = value ? '✓ Accepted' : '✗ Rejected';
      element.className = 'status-pill ' + (value ? 'status-true' : 'status-false');
    }}

    function ensureBaseModels() {{
      if (targetModel !== null) return;
      targetModel = viewer.addModel(staticTargetPdb, 'pdb');
      applyStyle(targetModel, legendItems.slice(1).map(i => i.chain));
    }}

    function updateMeta(frame) {{
      stepValue.textContent = frame.step;
      lossValue.textContent = formatNumber(frame.total_loss);
      rankingValue.textContent = formatNumber(frame.ranking_score);
      setPill(acceptedValue, frame.accepted);
      setPill(bestValue, frame.is_global_best);
      stepLabel.textContent = `Step ${{frame.step}}`;
    }}

    function renderFrame(index) {{
      currentIndex = index;
      const frame = frames[index];
      ensureBaseModels();
      if (binderModel !== null) {{
        viewer.removeModel(binderModel);
        binderModel = null;
      }}
      binderModel = viewer.addModel(frame.binder_pdb, 'pdb');
      applyStyle(binderModel, [legendItems[0].chain]);
      if (savedView === null) {{
        viewer.zoomTo();
        savedView = viewer.getView();
      }} else {{
        viewer.setView(savedView);
      }}
      viewer.render();
      slider.value = index;
      frameLabel.textContent = `Frame ${{index + 1}} / ${{frames.length}}`;
      progressFill.style.width = `${{((index + 1) / frames.length) * 100}}%`;
      updateMeta(frame);
    }}

    function pausePlayback() {{
      if (playTimer !== null) {{
        clearInterval(playTimer);
        playTimer = null;
      }}
      document.getElementById('play-btn').textContent = '▶ Play';
    }}

    function playPlayback() {{
      if (frames.length <= 1) return;
      pausePlayback();
      const fps = Math.max(1, Number(fpsInput.value) || 1);
      document.getElementById('play-btn').textContent = '⏸ Playing...';
      playTimer = window.setInterval(() => {{
        renderFrame((currentIndex + 1) % frames.length);
      }}, 1000 / fps);
    }}

    function toggleSidebar() {{
      sidebar.classList.toggle('collapsed');
      mainContent.classList.toggle('expanded');
      viewerWrapper.classList.toggle('expanded');
      sidebarToggle.textContent = sidebar.classList.contains('collapsed') ? '☰' : '✕';
      setTimeout(() => viewer.resize(), 300);
    }}

    function toggleFullscreen() {{
      if (!document.fullscreenElement) {{
        viewerWrapper.requestFullscreen().catch(err => {{
          console.log('Fullscreen error:', err);
        }});
      }} else {{
        document.exitFullscreen();
      }}
    }}

    function refreshViewMode() {{
      if (targetModel) {{
        applyStyle(targetModel, legendItems.slice(1).map(i => i.chain));
      }}
      if (binderModel) {{
        applyStyle(binderModel, [legendItems[0].chain]);
      }}
      viewer.render();
    }}

    function downloadCurrentPDB() {{
      const frame = frames[currentIndex];
      const blob = new Blob([frame.binder_pdb], {{ type: 'text/plain' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `step_${{String(frame.step).padStart(4, '0')}}_binder.pdb`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }}

    // Event listeners
    document.getElementById('prev-btn').addEventListener('click', () => {{
      pausePlayback();
      renderFrame((currentIndex - 1 + frames.length) % frames.length);
    }});
    document.getElementById('next-btn').addEventListener('click', () => {{
      pausePlayback();
      renderFrame((currentIndex + 1) % frames.length);
    }});
    document.getElementById('play-btn').addEventListener('click', () => {{
      if (playTimer === null) {{
        playPlayback();
      }} else {{
        pausePlayback();
      }}
    }});
    document.getElementById('pause-btn').addEventListener('click', pausePlayback);
    slider.addEventListener('input', () => {{
      pausePlayback();
      renderFrame(Number(slider.value));
    }});
    fpsInput.addEventListener('change', () => {{
      if (playTimer !== null) playPlayback();
    }});
    sidebarToggle.addEventListener('click', toggleSidebar);
    document.getElementById('download-btn').addEventListener('click', downloadCurrentPDB);

    // View mode buttons
    document.querySelectorAll('.view-mode-btn').forEach(btn => {{
      btn.addEventListener('click', () => {{
        document.querySelectorAll('.view-mode-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentViewMode = btn.dataset.mode;
        refreshViewMode();
      }});
    }});

    // Keyboard shortcuts
    window.addEventListener('keydown', (event) => {{
      if (event.code === 'Space') {{
        event.preventDefault();
        if (playTimer === null) playPlayback(); else pausePlayback();
      }} else if (event.code === 'ArrowLeft') {{
        event.preventDefault();
        pausePlayback();
        renderFrame((currentIndex - 1 + frames.length) % frames.length);
      }} else if (event.code === 'ArrowRight') {{
        event.preventDefault();
        pausePlayback();
        renderFrame((currentIndex + 1) % frames.length);
      }} else if (event.code === 'KeyS') {{
        event.preventDefault();
        toggleSidebar();
      }} else if (event.code === 'KeyF') {{
        event.preventDefault();
        toggleFullscreen();
      }}
    }});

    renderFrame(0);
  </script>
</body>
</html>
'''


def parse_args():
    parser = argparse.ArgumentParser(description="Generate an HTML animation for BindCraft step structures")
    parser.add_argument("--run_dir", required=True, help="Result directory containing checkpoints/step_*/")
    parser.add_argument("--output", required=True, help="Output HTML path")
    parser.add_argument("--beam", type=int, default=0, help="Beam id to use (default: 0)")
    parser.add_argument("--stride", type=int, default=None, help="Keep every Nth step before rendering")
    parser.add_argument("--max_frames", type=int, default=None, help="Evenly sample down to at most this many frames")
    parser.add_argument("--fps", type=int, default=4, help="Default playback speed (default: 4)")
    parser.add_argument("--accepted-only", action="store_true", default=True, help="Only include accepted steps (default: True)")
    parser.add_argument("--all-steps", dest="accepted_only", action="store_false", help="Include all steps including rejected")
    return parser.parse_args()


def main():
    args = parse_args()

    steps = discover_steps(args.run_dir, beam=args.beam, accepted_only=args.accepted_only)
    sampled_steps = sample_steps(steps, stride=args.stride, max_frames=args.max_frames)
    frames, static_target_pdb, binder_chain, target_chains = build_aligned_frames(sampled_steps)

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    title = f"BindCraft step animation: {os.path.basename(os.path.abspath(args.run_dir))}"
    html = build_html(
        frames,
        title,
        fps=args.fps,
        binder_chain=binder_chain,
        target_chains=target_chains,
        static_target_pdb=static_target_pdb,
    )

    with open(output_path, "w") as f:
        f.write(html)

    print(f"Discovered steps: {len(steps)}")
    print(f"Rendered frames:  {len(frames)}")
    print(f"Binder chain:     {binder_chain}")
    print(f"Target chains:    {','.join(target_chains)}")
    print(f"Output HTML:      {output_path}")
    print("Open the HTML in a browser to play, pause, and scrub through aligned structures.")


if __name__ == "__main__":
    main()
