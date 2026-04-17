# benchmark BindCraft Batch Notes

## Purpose

This directory records the local benchmark workflow used to run A6_TCR binder optimization inside:

- `/data/huangyanjie/binder_design/BindCraft-v2`

The goal is to:

- extract chain A from AF3Score PDB files as the initial binder sequence
- treat chain B + chain C as a fixed A6_TCR target
- batch-run BindCraft optimization
- write logs, JSON outputs, and optimization-step checkpoints into the `benchmark` workspace

## Data Flow

Source PDB directory:

- `/data/huangyanjie/binder_design/AF3Score/input`

benchmark workspace:

- input: `/data/huangyanjie/binder_design/BindCraft-v2/benchmark/input/A6_TCR`
- results: `/data/huangyanjie/binder_design/BindCraft-v2/benchmark/results`
- launcher script: `/data/huangyanjie/binder_design/BindCraft-v2/benchmark/script/run_benchmark.sh`
- configuration: `/data/huangyanjie/binder_design/BindCraft-v2/benchmark/yaml/default.yaml`
- documentation: `/data/huangyanjie/binder_design/BindCraft-v2/benchmark/doc`

Generated input files:

- `init_binders.csv`: manifest of all entries and extracted chain A sequences
- `target_chains.fasta`: fixed target chain B and chain C sequences
- `*.fasta`: one FASTA per entry for extracted chain A

## Code Changes

### 1. Support user-specified initial binder sequence

Updated:

- `scripts/run_hallucination.py`
- `bindcraft/hallucination/pipeline.py`

Changes:

- added `--init_binder_seq`
- validated sequence length and amino-acid alphabet
- made the optimization initialize from the provided binder sequence instead of only logging it

### 2. Support local checkpoint directory and bf16

Updated:

- `scripts/run_hallucination.py`

Changes:

- added checkpoint directory resolution:
  - CLI `--checkpoint_dir`
  - env `PROTENIX_CHECKPOINT_DIR`
  - fallback `PROTENIX_ROOT_DIR/checkpoint`
- default model dtype changed to `bf16`
- added explicit CLI `--dtype {bf16,fp32}`

### 3. Support multi-chain target sequence input

Updated:

- `bindcraft/utils.py`

Changes:

- allowed target sequence input as:
  - single string
  - comma-separated string
  - list of chain sequences

### 4. benchmark batch driver

Current entrypoint:

- `benchmark/run_benchmark.py`

Function:

- scans all PDB files under the AF3Score input tree
- extracts chain A/B/C sequences from CA atoms
- verifies chain B and chain C are fixed across all structures
- writes extracted binder FASTA files and manifest
- prepares a minimal `PROTENIX_ROOT_DIR`
- shards jobs by `worker_index / num_workers`
- loads defaults from `benchmark/yaml/default.yaml`
- lets CLI args override YAML values
- saves per-entry result JSON under `benchmark/results`
- saves hybrid optimization-step checkpoints under `benchmark/results/<entry_id>/checkpoints`

### 5. Multi-GPU launcher

Current launcher:

- `benchmark/script/run_benchmark.sh`

Function:

- defines physical GPU IDs inside the script via `GPU_IDS=(...)`
- runs a single preflight pass with `--prepare-only`
- launches one worker process per GPU
- sets `CUDA_VISIBLE_DEVICES=<single_gpu>` for each worker
- passes `--device cuda:0` to keep per-process device mapping stable
- writes combined launcher output to:
  - `/data/huangyanjie/binder_design/BindCraft-v2/benchmark/log/launcher.log`
- each Python worker also writes its own log to:
  - `/data/huangyanjie/binder_design/BindCraft-v2/benchmark/log/worker_<idx>.log`

## Config

Default config file:

- `benchmark/yaml/default.yaml`

Main fields:

- `source_dir`
- `input_dir`
- `results_dir`
- `model_checkpoint_dir`
- `components_cache_dir`
- `protenix_root_dir`
- `model_name`
- `device`
- `mode`
- `n_steps`
- `learning_rate`
- `tau_noise`
- `k_top`
- `b_candidates`
- `t_sample`
- `max_mutations_per_seq`
- `use_annealing`
- `t_init`
- `t_min`
- `use_tau_scheduling`
- `tau_jitter`
- `entry_ids`
- `skip_existing`
- `save_checkpoints`
- `checkpoint_subdir_name`

Note:

- GPU selection is intentionally not in YAML.
- Edit `GPU_IDS=(...)` inside `benchmark/script/run_benchmark.sh` when you want to change the card list.

## Current Launch Method

From project root:

```bash
bash benchmark/script/run_benchmark.sh
```

Common optional arguments:

```bash
bash benchmark/script/run_benchmark --entry-ids alpha_patch_19_23_batch_0
bash benchmark/script/run_benchmark --n-steps 10
bash benchmark/script/run_benchmark --no-skip-existing
```

## Output Layout

Main launcher log:

- `/data/huangyanjie/binder_design/BindCraft-v2/benchmark/log/launcher.log`

Worker logs:

- `/data/huangyanjie/binder_design/BindCraft-v2/benchmark/log/worker_0.log`
- `/data/huangyanjie/binder_design/BindCraft-v2/benchmark/log/worker_1.log`
- ...

Per-entry JSON outputs:

- `/data/huangyanjie/binder_design/BindCraft-v2/benchmark/results/<entry_id>.json`

Hybrid optimization-step checkpoints:

- `/data/huangyanjie/binder_design/BindCraft-v2/benchmark/results/<entry_id>/checkpoints/step_0000/...`
- `/data/huangyanjie/binder_design/BindCraft-v2/benchmark/results/<entry_id>/checkpoints/step_0001/...`
- ...

Each saved step directory contains files written by `bindcraft/checkpoint.py`, including:

- `*_complex.pdb`
- `*_meta.json`
- `*_plddt.json`
- `best.pdb`
- `best.json`
- `best_plddt.json`

## Notes

- This benchmark workflow is sequence-driven.
- The initial binder comes from chain A extracted from each source PDB.
- The fixed target comes from chain B + chain C.
- `n_steps` defaults to `50` in YAML.
- In hybrid mode, step checkpoints include the optimization process, starting from the initialization step.
- If future runs still hit memory issues, the next knobs to consider are:
  - switching the custom wrapper from forced `"torch"` triangle kernels to model-config kernels
  - enabling chunked execution
  - cropping target/interface regions
