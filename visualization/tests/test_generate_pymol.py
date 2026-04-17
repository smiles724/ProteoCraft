"""
Tests for scripts/generate_pymol_script.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import tempfile

import pytest

from generate_pymol_script import (
    _discover_pdb_steps,
    _sample_frames,
    _hsv_to_rgb,
    generate_pml,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_pdb(path):
    """Write a minimal valid PDB with one chain-A and one chain-B atom."""
    lines = [
        "REMARK  1  Minimal test PDB\n",
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.50           C\n",
        "ATOM      2  CA  ALA B   1       4.000   5.000   6.000  1.00  0.50           C\n",
        "END\n",
    ]
    with open(path, "w") as f:
        f.writelines(lines)


def _make_minimal_checkpoint(root, steps):
    """Create a minimal checkpoint directory structure.

    Args:
        root:  Root directory path (must already exist).
        steps: List of integer step numbers to create.

    Returns:
        root (str)
    """
    for step_num in steps:
        step_name = f"step_{step_num:04d}"
        step_dir = os.path.join(root, step_name)
        os.makedirs(step_dir, exist_ok=True)

        # Minimal PDB
        pdb_name = f"{step_name}_beam_0_complex.pdb"
        pdb_path = os.path.join(step_dir, pdb_name)
        _make_minimal_pdb(pdb_path)

        # Minimal meta.json
        meta = {
            "step": step_num,
            "beam_id": 0,
            "pdb_file": pdb_name,
        }
        meta_path = os.path.join(step_dir, f"{step_name}_beam_0_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f)

    return root


# ---------------------------------------------------------------------------
# TestDiscoverPdbSteps
# ---------------------------------------------------------------------------

class TestDiscoverPdbSteps:
    def test_finds_all_steps(self, tmp_path):
        _make_minimal_checkpoint(str(tmp_path), [0, 1, 2])
        steps = _discover_pdb_steps(str(tmp_path))
        assert len(steps) == 3

    def test_returns_sorted_tuples(self, tmp_path):
        _make_minimal_checkpoint(str(tmp_path), [5, 0, 3])
        steps = _discover_pdb_steps(str(tmp_path))
        nums = [s[0] for s in steps]
        assert nums == sorted(nums)

    def test_paths_are_absolute(self, tmp_path):
        _make_minimal_checkpoint(str(tmp_path), [0, 1])
        steps = _discover_pdb_steps(str(tmp_path))
        for _, path in steps:
            assert os.path.isabs(path), f"Path is not absolute: {path}"

    def test_empty_dir_returns_empty(self, tmp_path):
        steps = _discover_pdb_steps(str(tmp_path))
        assert steps == []


# ---------------------------------------------------------------------------
# TestSampleFrames
# ---------------------------------------------------------------------------

class TestSampleFrames:
    def _make_steps(self, n):
        return [(i, f"/fake/step_{i:04d}.pdb") for i in range(n)]

    def test_always_includes_first_and_last(self):
        steps = self._make_steps(20)
        sampled = _sample_frames(steps, 5)
        assert sampled[0] == steps[0]
        assert sampled[-1] == steps[-1]

    def test_correct_count(self):
        steps = self._make_steps(20)
        sampled = _sample_frames(steps, 5)
        assert len(sampled) == 5

    def test_n_frames_le_1_returns_first_only(self):
        steps = self._make_steps(10)
        sampled = _sample_frames(steps, 1)
        assert len(sampled) == 1
        assert sampled[0] == steps[0]

    def test_n_frames_zero_returns_first_only(self):
        steps = self._make_steps(10)
        sampled = _sample_frames(steps, 0)
        assert len(sampled) == 1
        assert sampled[0] == steps[0]

    def test_fewer_steps_than_frames_returns_all(self):
        steps = self._make_steps(3)
        sampled = _sample_frames(steps, 10)
        assert len(sampled) == 3

    def test_empty_input_returns_empty(self):
        assert _sample_frames([], 5) == []

    def test_exact_match(self):
        steps = self._make_steps(5)
        sampled = _sample_frames(steps, 5)
        assert len(sampled) == 5


# ---------------------------------------------------------------------------
# TestHsvToRgb
# ---------------------------------------------------------------------------

class TestHsvToRgb:
    def test_red(self):
        r, g, b = _hsv_to_rgb(0.0, 1.0, 1.0)
        assert abs(r - 1.0) < 1e-9
        assert abs(g - 0.0) < 1e-9
        assert abs(b - 0.0) < 1e-9

    def test_blue(self):
        r, g, b = _hsv_to_rgb(240.0 / 360.0, 1.0, 1.0)
        assert abs(r - 0.0) < 1e-9
        assert abs(g - 0.0) < 1e-9
        assert abs(b - 1.0) < 1e-9

    def test_output_in_range(self):
        for h in [0.0, 0.25, 0.5, 0.75, 1.0]:
            r, g, b = _hsv_to_rgb(h, 0.8, 0.9)
            assert 0.0 <= r <= 1.0
            assert 0.0 <= g <= 1.0
            assert 0.0 <= b <= 1.0


# ---------------------------------------------------------------------------
# TestGeneratePml
# ---------------------------------------------------------------------------

class TestGeneratePml:
    def test_output_is_valid_pml(self, tmp_path):
        """Check that the output contains all required PyMOL keywords."""
        _make_minimal_checkpoint(str(tmp_path), [0, 1, 2])
        pml = generate_pml(str(tmp_path), n_frames=3)

        assert "load" in pml, "Expected 'load' command in .pml output"
        assert "png" in pml, "Expected 'png' command in .pml output"
        assert "spectrum" in pml, "Expected 'spectrum' command in .pml output"
        assert "python" in pml, "Expected 'python' block in .pml output"
        assert "python end" in pml, "Expected 'python end' in .pml output"

    def test_n_frames_sampling(self, tmp_path):
        """50 steps, n_frames=5 → exactly 5 load commands for overlay frames
        (plus 5 for per-frame gallery + 1 for the reference = 11 total loads).
        We specifically count overlay_step* loads = 5."""
        _make_minimal_checkpoint(str(tmp_path), list(range(50)))
        pml = generate_pml(str(tmp_path), n_frames=5)

        # Count lines with "load" that reference overlay_step objects
        overlay_loads = [
            line for line in pml.splitlines()
            if line.strip().startswith("load") and "overlay_step" in line
        ]
        assert len(overlay_loads) == 5, (
            f"Expected 5 overlay load commands, got {len(overlay_loads)}"
        )

        # Count per-frame gallery loads
        frame_loads = [
            line for line in pml.splitlines()
            if line.strip().startswith("load") and "frame_step" in line
        ]
        assert len(frame_loads) == 5, (
            f"Expected 5 frame_step load commands, got {len(frame_loads)}"
        )

    def test_absolute_paths(self, tmp_path):
        """All PDB paths in load commands must be absolute."""
        _make_minimal_checkpoint(str(tmp_path), [0, 1, 2])
        pml = generate_pml(str(tmp_path), n_frames=3)

        load_lines = [
            line.strip() for line in pml.splitlines()
            if line.strip().startswith("load") and ".pdb" in line
        ]
        assert load_lines, "No load lines found with .pdb paths"
        for line in load_lines:
            # Format: "load /abs/path/to/file.pdb, object_name"
            parts = line.split(",")
            pdb_path = parts[0].replace("load", "").strip()
            assert os.path.isabs(pdb_path), (
                f"Non-absolute path in load command: {pdb_path!r}"
            )

    def test_quit_at_end(self, tmp_path):
        """Script must end with 'quit'."""
        _make_minimal_checkpoint(str(tmp_path), [0, 1])
        pml = generate_pml(str(tmp_path), n_frames=2)
        non_empty_lines = [l for l in pml.splitlines() if l.strip()]
        assert non_empty_lines[-1].strip() == "quit"

    def test_no_bare_python_statements(self, tmp_path):
        """Python statements must be inside python/python end blocks — not bare."""
        _make_minimal_checkpoint(str(tmp_path), [0, 1])
        pml = generate_pml(str(tmp_path), n_frames=2)

        # Lines that would be invalid bare PyMOL commands (Python syntax outside blocks)
        # We track whether we are inside a python/python end block.
        in_python_block = False
        for line in pml.splitlines():
            stripped = line.strip()
            if stripped == "python":
                in_python_block = True
                continue
            if stripped == "python end":
                in_python_block = False
                continue
            # Outside a python block, lines starting with Python-only syntax are invalid
            if not in_python_block:
                assert not stripped.startswith("os."), (
                    f"Bare Python line found outside python block: {line!r}"
                )
                assert not stripped.startswith("import "), (
                    f"Bare import found outside python block: {line!r}"
                )
                assert not stripped.startswith("stored.view = cmd.get_view()"), (
                    f"Bare stored.view assignment found outside python block: {line!r}"
                )
                assert not stripped.startswith("cmd.set_view("), (
                    f"Bare cmd.set_view found outside python block: {line!r}"
                )

    def test_spectrum_uses_b_factor(self, tmp_path):
        """Per-frame gallery uses 'spectrum b, blue_white_red' with correct range."""
        _make_minimal_checkpoint(str(tmp_path), [0, 1])
        pml = generate_pml(str(tmp_path), n_frames=2)
        assert "spectrum b, blue_white_red" in pml
        assert "minimum=-5, maximum=5" in pml

    def test_orient_present(self, tmp_path):
        """Script must include an 'orient' command for camera setup."""
        _make_minimal_checkpoint(str(tmp_path), [0, 1])
        pml = generate_pml(str(tmp_path), n_frames=2)
        orient_lines = [l for l in pml.splitlines() if l.strip().startswith("orient")]
        assert orient_lines, "Expected at least one 'orient' command"

    def test_empty_checkpoint_raises(self, tmp_path):
        """generate_pml raises ValueError when no PDB steps are found."""
        with pytest.raises(ValueError, match="No PDB steps found"):
            generate_pml(str(tmp_path), n_frames=3)
