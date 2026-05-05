# HYJ Structure Animation Tool

This folder contains the checkpoint-to-HTML structure animation tool received on 2026-05-03.

## Files

- `animate.sh`
  - Convenience wrapper.
  - Usage: `bash animate.sh <run_dir|checkpoints_dir|phase_dir> [fps] [--all-steps]`
- `animate_structure_steps.py`
  - Main Python script.
  - Reads checkpoint `step_*_beam_0_meta.json` and matching PDB files.
  - Aligns every frame by the target chain coordinates.
  - Builds a self-contained 3Dmol HTML animation with play/pause/step slider.
- `1qrn_alpha_53_56_template_animation.example.html`
  - Example output provided by collaborator.

## Typical Usage

Run on a phase directory:

```bash
bash animate.sh /path/to/run/checkpoints/document_puct_refine 4
```

On the HPC, the system `python3` may not have `numpy`. Use the Protenix environment Python explicitly:

```bash
PYTHON_BIN=/hpc2hdd/home/hlin199/.conda/envs/protenix/bin/python \
  bash animate.sh /path/to/run/checkpoints/document_puct_refine 4
```

Run on a full run directory:

```bash
bash animate.sh /path/to/run 4
```

Include rejected steps:

```bash
bash animate.sh /path/to/run/checkpoints/document_puct_refine 4 --all-steps
```

The output HTML is written into the inferred run/phase directory:

```text
<run_or_phase_dir>/<basename>_animation.html
```

## Notes

- By default only accepted steps are included.
- The first chain in the first PDB is treated as the binder.
- All remaining chains are treated as target chains.
- Every frame is aligned by target-chain CA atoms, so the target stays fixed and binder movement is easier to inspect.
- The generated HTML embeds structure frames directly, but it loads 3Dmol from `https://3Dmol.org/build/3Dmol-min.js`.

## Local Test

Tested on:

```text
pd1_final_confirmed_template_check_full_results_20260503/no_hotspot/PD1/checkpoints/document_puct_refine
```

Result:

```text
Discovered steps: 67
Rendered frames: 67
Binder chain: A
Target chains: B
```
