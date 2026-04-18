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


def discover_steps(run_dir, beam=0):
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    pattern = os.path.join(checkpoint_dir, "step_*", f"step_*_beam_{beam}_meta.json")
    meta_files = sorted(glob(pattern))

    if not meta_files:
        raise ValueError(f"No meta files found for beam {beam} in {checkpoint_dir}")

    steps = []
    skipped = []
    for meta_path in meta_files:
        with open(meta_path) as f:
            meta = json.load(f)

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
        raise ValueError(f"No usable steps found in {checkpoint_dir}")

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
    atom_name = parts[2] if len(parts) >= 3 else ""
    chain_id = parts[4] if len(parts) >= 5 else ""

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
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, Arial, sans-serif; background: linear-gradient(180deg, #e2e8f0 0%, #f8fafc 260px); color: var(--text); }}
    .page {{ max-width: 1440px; margin: 0 auto; padding: 24px; }}
    .hero {{ margin-bottom: 18px; }}
    .hero h1 {{ margin: 0 0 8px; font-size: 28px; line-height: 1.2; }}
    .hero p {{ margin: 0; color: var(--muted); font-size: 15px; }}
    .layout {{ display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 18px; align-items: start; }}
    .viewer-card, .side-card {{ background: rgba(255,255,255,0.96); border: 1px solid var(--line); border-radius: 18px; box-shadow: 0 14px 40px rgba(15, 23, 42, 0.08); }}
    .viewer-card {{ padding: 18px; }}
    .side-card {{ padding: 18px; }}
    .controls {{ display: grid; gap: 14px; margin-bottom: 16px; }}
    .control-row {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
    .primary-btn, .ghost-btn {{ border: 0; border-radius: 12px; padding: 11px 16px; font-size: 14px; font-weight: 600; cursor: pointer; }}
    .primary-btn {{ background: var(--accent); color: white; }}
    .primary-btn:hover {{ background: var(--accent-2); }}
    .ghost-btn {{ background: #e2e8f0; color: #0f172a; }}
    .ghost-btn:hover {{ background: #cbd5e1; }}
    .slider-wrap {{ display: grid; gap: 8px; }}
    .slider-header {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; }}
    .slider-header strong {{ font-size: 15px; }}
    .slider-header span {{ color: var(--muted); font-size: 14px; }}
    input[type=range] {{ width: 100%; accent-color: var(--accent); }}
    .viewer {{ width: 100%; height: 780px; border: 1px solid var(--line); border-radius: 16px; background: white; overflow: hidden; }}
    .field {{ display: inline-flex; align-items: center; gap: 8px; color: var(--muted); font-size: 14px; }}
    .field input {{ width: 84px; padding: 8px 10px; border-radius: 10px; border: 1px solid var(--line); font-size: 14px; }}
    .stats {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .stat {{ background: var(--panel-2); border: 1px solid var(--line); border-radius: 14px; padding: 14px; }}
    .stat-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }}
    .stat-value {{ font-size: 24px; font-weight: 700; }}
    .legend {{ display: grid; gap: 10px; margin-bottom: 16px; }}
    .legend-item {{ display: flex; align-items: center; gap: 10px; padding: 10px 12px; background: var(--panel-2); border: 1px solid var(--line); border-radius: 12px; }}
    .swatch {{ width: 14px; height: 14px; border-radius: 999px; flex: 0 0 auto; }}
    .chain-name {{ font-weight: 700; min-width: 58px; }}
    .chain-role {{ color: var(--muted); font-size: 13px; }}
    .status-pill {{ display: inline-flex; align-items: center; justify-content: center; min-height: 32px; border-radius: 999px; padding: 6px 12px; font-size: 14px; font-weight: 700; }}
    .status-true {{ background: var(--ok-bg); color: var(--ok-text); }}
    .status-false {{ background: var(--bad-bg); color: var(--bad-text); }}
    .hint {{ color: var(--muted); font-size: 13px; line-height: 1.5; }}
    .kbd {{ display: inline-block; padding: 2px 8px; border-radius: 8px; border: 1px solid var(--line); background: white; font-size: 12px; color: var(--muted); }}
    @media (max-width: 1180px) {{ .layout {{ grid-template-columns: 1fr; }} .viewer {{ height: 640px; }} }}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <h1 id="title"></h1>
      <p>target 固定为 step 0，仅更新 binder 轨迹。拖动时间轴切换 step，点击播放按钮自动播放。</p>
    </div>
    <div class="layout">
      <div class="viewer-card">
        <div class="controls">
          <div class="control-row">
            <button id="play-btn" class="primary-btn">▶ Play</button>
            <button id="pause-btn" class="ghost-btn">⏸ Pause</button>
            <button id="prev-btn" class="ghost-btn">← Prev</button>
            <button id="next-btn" class="ghost-btn">Next →</button>
            <label class="field">FPS <input id="fps-input" type="number" min="1" max="60" value="{int(fps)}"></label>
          </div>
          <div class="slider-wrap">
            <div class="slider-header">
              <strong id="frame-label">Frame 1 / {len(frames)}</strong>
              <span id="step-label">Step 0</span>
            </div>
            <input id="frame-slider" type="range" min="0" max="{max(len(frames) - 1, 0)}" value="0" step="1">
          </div>
        </div>
        <div id="viewer" class="viewer"></div>
      </div>
      <div class="side-card">
        <div class="stats">
          <div class="stat"><div class="stat-label">Step</div><div class="stat-value" id="step-value"></div></div>
          <div class="stat"><div class="stat-label">Total loss</div><div class="stat-value" id="loss-value"></div></div>
          <div class="stat"><div class="stat-label">Ranking score</div><div class="stat-value" id="ranking-value"></div></div>
          <div class="stat"><div class="stat-label">Accepted</div><div class="stat-value"><span class="status-pill" id="accepted-value"></span></div></div>
          <div class="stat"><div class="stat-label">Global best</div><div class="stat-value"><span class="status-pill" id="best-value"></span></div></div>
          <div class="stat"><div class="stat-label">Frames</div><div class="stat-value">{len(frames)}</div></div>
        </div>
        <h3 style="margin:0 0 12px; font-size:16px;">Chain colors</h3>
        <div class="legend" id="legend"></div>
        <div class="hint">
          target 链始终显示 step 0 的结构，只有 binder 会随 frame 变化。<br>
          快捷方式：<span class="kbd">Space</span> 播放/暂停，<span class="kbd">←</span>/<span class="kbd">→</span> 切换 frame。
        </div>
      </div>
    </div>
  </div>
  <script>
    const title = {title_json};
    const chainColors = {chain_colors_json};
    const legendItems = {legend_items_json};
    const staticTargetPdb = {static_target_json};
    const frames = {frame_json};

    document.getElementById('title').textContent = title;

    const viewer = $3Dmol.createViewer(document.getElementById('viewer'), {{ backgroundColor: 'white' }});
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

    let currentIndex = 0;
    let playTimer = null;
    let savedView = null;
    let targetModel = null;
    let binderModel = null;

    legendItems.forEach((item, index) => {{
      const div = document.createElement('div');
      div.className = 'legend-item';
      const role = index === 0 ? 'binder' : 'target';
      div.innerHTML = `<span class="swatch" style="background:${{item.color}}"></span><span class="chain-name">Chain ${{item.chain}}</span><span class="chain-role">${{role}}</span>`;
      legend.appendChild(div);
    }});

    function formatNumber(value) {{
      if (value === null || value === undefined || Number.isNaN(value)) return 'NA';
      return Number(value).toFixed(3);
    }}

    function setPill(element, value) {{
      element.textContent = value ? 'true' : 'false';
      element.className = 'status-pill ' + (value ? 'status-true' : 'status-false');
    }}

    function styleModel(model, chains) {{
      Object.entries(chainColors).forEach(([chain, color]) => {{
        if (chains.includes(chain)) {{
          model.setStyle({{ chain }}, {{ cartoon: {{ color }} }});
        }}
      }});
    }}

    function updateMeta(frame) {{
      stepValue.textContent = frame.step;
      lossValue.textContent = formatNumber(frame.total_loss);
      rankingValue.textContent = formatNumber(frame.ranking_score);
      setPill(acceptedValue, frame.accepted);
      setPill(bestValue, frame.is_global_best);
      stepLabel.textContent = `Step ${{frame.step}}`;
    }}

    function ensureBaseModels() {{
      if (targetModel !== null) return;
      targetModel = viewer.addModel(staticTargetPdb, 'pdb');
      styleModel(targetModel, legendItems.slice(1).map((item) => item.chain));
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
      styleModel(binderModel, [legendItems[0].chain]);
      if (savedView === null) {{
        viewer.zoomTo();
        savedView = viewer.getView();
      }} else {{
        viewer.setView(savedView);
      }}
      viewer.render();
      slider.value = index;
      frameLabel.textContent = `Frame ${{index + 1}} / ${{frames.length}}`;
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
      document.getElementById('play-btn').textContent = 'Playing…';
      playTimer = window.setInterval(() => {{
        renderFrame((currentIndex + 1) % frames.length);
      }}, 1000 / fps);
    }}

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
    return parser.parse_args()


def main():
    args = parse_args()

    steps = discover_steps(args.run_dir, beam=args.beam)
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
