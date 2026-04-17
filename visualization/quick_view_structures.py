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
        # Escape PDB for JS embedding
        pdb_escaped = pdb_data.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
        viewers.append(f"""
        <div style="display:inline-block; margin:10px; text-align:center;">
            <h3>Step {step_num}</h3>
            <div id="viewer_{i}" style="width:500px; height:400px; border:1px solid #ccc;"></div>
        </div>""")

    viewer_scripts = []
    for i, (pdb_data, step_num) in enumerate(zip(pdb_contents, step_nums)):
        pdb_escaped = pdb_data.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
        viewer_scripts.append(f"""
            var viewer_{i} = $3Dmol.createViewer(document.getElementById("viewer_{i}"), {{backgroundColor: "white"}});
            viewer_{i}.addModel(`{pdb_escaped}`, "pdb");
            // Binder (chain A) — colored by spectrum
            viewer_{i}.setStyle({{chain: "A"}}, {{cartoon: {{color: "spectrum"}}}});
            // Target (chain B) — gray
            viewer_{i}.setStyle({{chain: "B"}}, {{cartoon: {{color: "gray"}}}});
            viewer_{i}.zoomTo();
            viewer_{i}.render();""")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>BindCraft v2 — Structure Viewer</title>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
</head>
<body style="font-family: Arial, sans-serif; padding: 20px;">
    <h2>BindCraft v2 — Hallucination Trajectory Structures</h2>
    <p>Binder (chain A): rainbow spectrum &nbsp;|&nbsp; Target (chain B): gray &nbsp;|&nbsp; 可鼠标旋转/缩放</p>
    <div>
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
