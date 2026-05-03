#!/usr/bin/env python3
"""
快速 3D 结构可视化 — 生成 HTML 文件，在浏览器中查看。

用法:
    python scripts/quick_view_structures.py \
        --checkpoint_dir path/to/checkpoints \
        --output view.html \
        --steps 0 25 49
"""
import argparse
import glob
import os


def find_pdb(checkpoint_dir, step_num):
    """Find PDB file for a given step number."""
    pattern = os.path.join(checkpoint_dir, f"step_{step_num:04d}",
                           f"step_{step_num:04d}_beam_0_complex.pdb")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No PDB found for step {step_num} in {checkpoint_dir}")
    return matches[0]


def generate_html(pdb_contents, step_nums):
    """Generate an HTML page with 3Dmol.js viewers for each step."""

    viewers = []
    for i, (pdb_data, step_num) in enumerate(zip(pdb_contents, step_nums)):
        viewers.append(f"""
        <section class="viewer-card">
            <h3>Step {step_num}</h3>
            <div id="viewer_{i}" class="viewer"></div>
        </section>""")

    viewer_scripts = []
    for i, (pdb_data, step_num) in enumerate(zip(pdb_contents, step_nums)):
        pdb_escaped = pdb_data.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
        viewer_scripts.append(f"""
            var viewer_{i} = $3Dmol.createViewer("viewer_{i}", {{backgroundColor: "white"}});
            viewer_{i}.addModel(`{pdb_escaped}`, "pdb");
            viewer_{i}.setStyle({{chain: "A"}}, {{cartoon: {{color: "spectrum"}}}});
            viewer_{i}.setStyle({{chain: "B"}}, {{cartoon: {{color: "gray"}}}});
            viewer_{i}.zoomTo();
            viewer_{i}.resize();
            viewer_{i}.render();""")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>BindCraft v2 — Structure Viewer</title>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 24px;
            color: #111;
            background: #fafafa;
        }}
        h2 {{
            margin: 0 0 8px;
        }}
        p {{
            margin: 0 0 24px;
        }}
        .viewer-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
            gap: 24px;
            align-items: start;
        }}
        .viewer-card {{
            background: #fff;
            border: 1px solid #d4d4d4;
            padding: 12px;
        }}
        .viewer-card h3 {{
            margin: 0 0 10px;
            text-align: center;
        }}
        .viewer {{
            position: relative;
            width: 100%;
            height: 430px;
            border: 1px solid #ccc;
            overflow: hidden;
            background: #fff;
        }}
        .viewer canvas {{
            position: absolute !important;
            left: 0 !important;
            top: 0 !important;
            width: 100% !important;
            height: 100% !important;
        }}
    </style>
</head>
<body>
    <h2>BindCraft v2 — Hallucination Trajectory Structures</h2>
    <p>Binder (chain A): rainbow spectrum &nbsp;|&nbsp; Target (chain B): gray &nbsp;|&nbsp; 可鼠标旋转/缩放</p>
    <div class="viewer-grid">
        {"".join(viewers)}
    </div>
    <script>
        {"".join(viewer_scripts)}
    </script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Quick 3D structure viewer")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output", default="structure_view.html")
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 25, 49],
                        help="Step numbers to visualize (default: 0 25 49)")
    args = parser.parse_args()

    pdb_contents = []
    valid_steps = []
    for step in args.steps:
        try:
            pdb_path = find_pdb(args.checkpoint_dir, step)
            with open(pdb_path) as f:
                pdb_contents.append(f.read())
            valid_steps.append(step)
            print(f"  Step {step}: {pdb_path}")
        except FileNotFoundError as e:
            print(f"  Warning: {e}")

    html = generate_html(pdb_contents, valid_steps)
    with open(args.output, "w") as f:
        f.write(html)

    print(f"\nSaved: {args.output}")
    print("在浏览器中打开此文件即可查看 3D 结构（支持鼠标旋转/缩放）")


if __name__ == "__main__":
    main()
