# TCR Sequence Prior — Real Model Backends

This document describes the real prior-scorer backends wired behind the
`TCRSequencePriorScorer` interface in
`bindcraft/hallucination/tcr_prior.py`, the config flags, the required
external weights/data, and how to set up **isolated** environments so the
prior dependencies do not pollute the ProteoCraft / base env.

All backends are **frozen** (eval, `no_grad`); we never train or fine-tune
them. By default, if a backend cannot load (missing repo / weights / deps) it
degrades to a **no-op** (prior contributes 0), so the main hallucination / MCTS
loop is never broken. **Strict mode** (`--require_real_tcr_prior true`) turns
these silent no-ops into hard errors — see below.

## Config flags (unchanged public interface)

| flag | meaning |
|------|---------|
| `--use_tcr_prior_gradient` | enable soft (differentiable) prior in the proposal path |
| `--tcr_prior_gradient_weight` | λ for `L_prop_total = L_prop + λ·L_tcr_prior_soft` |
| `--use_tcr_prior_verifier` | enable hard prior in the verifier / tree-value path |
| `--tcr_prior_verifier_weight` | λ for `L_verify_total = L_verify + λ·L_tcr_prior_hard` |
| `--tcr_prior_region` | `all` / `cdr` / `none` (which positions the prior acts on) |
| `--tcr_prior_model_name` | backend selector (see below) |

Additional optional keys (needed only by real backends):

| flag | meaning |
|------|---------|
| `--tcr_prior_checkpoint` | weights path for `tcrlm` / `ligandmpnn` (or env `TCRLM_CKPT` / `LIGANDMPNN_CKPT`) |
| `--tcr_prior_masked` | ESM-C exact masked-marginal PLL (slower); default single-pass |
| `--require_real_tcr_prior` | **strict mode**: raise instead of silently falling back to dummy/no-op |

### Strict mode (`--require_real_tcr_prior true`)

When a real backend (`tcrlm` / `esmc` / `ligandmpnn`) is requested with strict
mode on, `build_tcr_prior_scorer` raises `TCRPriorStrictError` if the backend:

- fails to load (missing repo / package / import error),
- loaded but has **no real weights** (checkpoint missing / load failed),
- loaded but produces **no real per-position output** (dummy/no-op).

For structure-conditioned backends (LigandMPNN) whose output can only be
validated with a backbone, build-time checks confirm load + weights, and the
**runtime** score/gradient path raises if `metadata['pdb_path']` is missing or
the model returns nothing. Each backend also exposes `backend.status()`:
`{loaded, real_weights, real_output, requires_structure, missing_items}`.

### `--tcr_prior_model_name` values

- `dummy` / *(empty)* / `none` / `null` / *any HF id* → original placeholder / HF scorer (unchanged).
- `tcrlm` or `tcrlm:/abs/path/to/tcrLM` → **TCRLM** CDR3 naturalness backend.
- `esmc` or `esmc:esmc_600m` → **ESM-C** whole-TCR naturalness backend.
- `ligandmpnn` or `ligandmpnn:/abs/path/to/LigandMPNN` → **LigandMPNN** conditional CDR prior.

## Math

- **Soft (gradient) path**, for backends that expose per-position AA
  probabilities `p_prior`:
  `L_tcr_prior_soft = mean_i CE(softmax(theta_i), p_prior_i)`
  = `- mean_i Σ_a p_prior(a|context) · log softmax(theta_i)(a)`.
  `p_prior` is detached (no gradient into the prior model).
- **Hard (verifier) path**:
  `L_tcr_prior_hard = - mean_i log p_prior(a_i | context)` (region mean NLL).
  The scorer returns naturalness `= -NLL`; `apply_tcr_verifier_score` adds
  `-naturalness · weight` to `total_loss` and logs `loss_dict['tcr_prior_hard']`.

---

## 1. TCRLM (CDR3 naturalness) — priority 1, hard verifier (+ soft)

- Repo: https://github.com/hliulab/tcrLM — BERT-style masked LM (`FLASHTransformer`,
  `models.tcrLM.Pretrain`) pretrained on >100M CDR3 sequences.
- **Not** a pip/HF package → clone the repo and point the backend at it.
- Scoring: masked-marginal pseudo-log-likelihood. Each region position is
  replaced with `TOKEN_MASK`; the model predicts a 20-way distribution; we
  read the true residue's log-prob. Supports both hard and soft paths.

### Required files
- Repo checkout dir (contains `models/` and `source/`): `TCRLM_HOME` env or
  `tcr_prior_model_name='tcrlm:/abs/path/to/tcrLM'`.
- Pretrained checkpoint (state_dict for `Pretrain`): `--tcr_prior_checkpoint`
  or `TCRLM_CKPT`. **Weights are NOT in the git tree and NOT in the Zenodo zip**
  (both contain only git-LFS pointer files). Fetch the real LFS object
  (`pretrained_model/pretrained_model_tcr.bin`, 158 MB) via the LFS batch API —
  see "Isolated env on Polaris" below for the exact command.
- The distributed checkpoint stores the transformer at the top level
  (`token_emb.*`, `layers.*`) plus `fc_out.*`, while `Pretrain` nests it under
  `self.protflash`; the backend remaps these keys and enforces a ≥95% tensor
  match so a silent partial load (which yields garbage perplexity) is rejected.
- Input is padded/truncated to the model's `seq_max_len=20` (CDR3), and the
  20-way head output is remapped from tcrLM's AA order to bindcraft's canonical
  order.
- `region='cdr'` recommended: pass the CDR mask so only CDR3 positions score.

### Isolated env
```bash
conda create -n tcrlm python=3.10 -y
conda activate tcrlm
git clone https://github.com/hliulab/tcrLM.git $TCRLM_HOME
conda env update -n tcrlm -f $TCRLM_HOME/environments.yaml
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
# deps (einops, etc.) must be importable in the env that runs ProteoCraft,
# OR run the prior in a subprocess (see "Isolation options" below).
```

---

## 2. ESM-C (whole-TCR naturalness) — priority 2, hard + soft

- Package: `esm` (EvolutionaryScale). Weights auto-download from HuggingFace
  hub on first use (`esmc_300m`, `esmc_600m`).
- Scoring: single forward pass → per-position distribution `p(a_i | sequence)`
  mapped to the 20 canonical AAs (fast, differentiable target for the soft
  prior). `--tcr_prior_masked true` switches to exact masked-marginal PLL
  (L forward passes; recommend verifier-only).
- Note: ESM-C logits are 64-dim for TE compatibility; only indices 0–32 map to
  the vocab. We select the 20 canonical-AA token ids via the tokenizer vocab.

### Required files
- None manual — weights download automatically. For offline clusters,
  pre-populate the HF cache (`HF_HOME`) with the ESM-C weights.

### Isolated env
```bash
conda create -n esmc python=3.11 -y
conda activate esmc
pip install esm torch
```

### Alternate loader: `transformers` (no `esm` package / offline-only sites)
Some sites only have `transformers`' native ESM-C support installed (or a
pre-cached HF mirror under a different org, e.g. `biohub/ESMC-6B`), with no
network egress to `huggingface.co` to fetch the EvolutionaryScale-org weights
the `esm` package expects. `ESMCBackend` auto-selects a second loader via
`transformers.models.esmc.modeling_esmc.ESMCForMaskedLM` whenever
`tcr_prior_model_name`'s model id contains a `/` (i.e. looks like a full HF
repo id) instead of a short name:

```yaml
tcr_prior_model_name: esmc:biohub/ESMC-6B   # transformers loader (full repo id)
# vs.
tcr_prior_model_name: esmc:esmc_300m        # esm package loader (short name)
```

To force offline loading from an already-populated `HF_HOME` cache (no
network calls at all), set `HF_HUB_OFFLINE=1` in the environment — without it,
`from_pretrained` still tries a network freshness check first and only falls
back to the local cache after retries/backoff time out. Both loaders share
the same `per_position_logprobs` interface (single-pass and masked-marginal
modes), so `tcr_prior_masked` behaves identically either way.

---

## 3. LigandMPNN (conditional CDR redesign prior) — priority 3, structure-based

- Repo: https://github.com/dauparas/LigandMPNN. Structure-conditioned; scoring
  outputs `out_dict['log_probs']` `[L, 21]` per position (drop the 21st `X`
  column, renormalize over 20 AAs) — usable for both hard and soft priors.
- **Requires a backbone PDB** (pMHC + TCR) per candidate, passed via
  `metadata['pdb_path']`. The ProteoCraft verifier already produces coordinates
  per candidate; dump them to a temp PDB and set `metadata['pdb_path']` to
  enable conditional scoring. Without a structure the backend is a **no-op**.
- `_score_from_pdb` is **fully implemented**: it uses the repo's `parse_PDB`
  + `featurize` + `model.score(use_sequence=True)` path to obtain
  `out_dict['log_probs']` `[L, 21]`, reorders LigandMPNN's alphabetical AA
  columns into bindcraft's `PROTEIN_RESIDUE_NAMES` order, drops `X`, and
  renormalizes over the 20 canonical AAs before selecting the region.

### Required files
- Repo checkout: `LIGANDMPNN_HOME` or `tcr_prior_model_name='ligandmpnn:/path'`.
- Model params: `bash get_model_params.sh ./model_params`; checkpoint via
  `--tcr_prior_checkpoint` or `LIGANDMPNN_CKPT`
  (e.g. `model_params/ligandmpnn_v_32_010_25.pt`).
- ProDy for PDB parsing.

### Isolated env
```bash
conda create -n ligandmpnn_env python=3.11 -y
conda activate ligandmpnn_env
git clone https://github.com/dauparas/LigandMPNN.git $LIGANDMPNN_HOME
pip install -r $LIGANDMPNN_HOME/requirements.txt   # torch, numpy, ProDy
bash $LIGANDMPNN_HOME/get_model_params.sh "$LIGANDMPNN_HOME/model_params"
```

---

## Isolation options

The scorer runs **in-process** inside ProteoCraft, so a backend's Python deps
must be importable in the env that runs the pipeline. Two clean options:

1. **Install prior deps into the ProteoCraft conda env** (`protenix`). This is
   already isolated from `base`. Watch for torch version pins (TCRLM pins
   2.4.1; inference generally tolerates newer torch).
2. **Subprocess isolation** (recommended if versions conflict): run the backend
   model in its own conda env via a small CLI and exchange JSON over stdin/
   stdout. The backend classes are structured so a subprocess adapter can
   implement the same `per_position_logprobs` / `region_nll` contract.

Do **not** install these into the machine `base` env.

## Example runs

```bash
# TCRLM CDR3 hard verifier (region=cdr)
export TCRLM_HOME=/abs/path/to/tcrLM TCRLM_CKPT=/abs/path/to/pretrain.pt
python scripts/run_hallucination.py --mode mcts ... \
  --use_tcr_prior_verifier true --tcr_prior_verifier_weight 0.1 \
  --tcr_prior_region cdr --tcr_prior_model_name tcrlm

# ESM-C whole-TCR naturalness, hard + soft
python scripts/run_hallucination.py --mode hybrid ... \
  --use_tcr_prior_verifier true --tcr_prior_verifier_weight 0.1 \
  --use_tcr_prior_gradient true --tcr_prior_gradient_weight 0.1 \
  --tcr_prior_region all --tcr_prior_model_name esmc:esmc_600m

# LigandMPNN conditional CDR prior (needs per-candidate PDB wiring)
export LIGANDMPNN_HOME=/abs/path/to/LigandMPNN
python scripts/run_hallucination.py --mode mcts ... \
  --use_tcr_prior_verifier true --tcr_prior_verifier_weight 0.1 \
  --tcr_prior_region cdr --tcr_prior_model_name ligandmpnn \
  --tcr_prior_checkpoint $LIGANDMPNN_HOME/model_params/ligandmpnn_v_32_010_25.pt
```

## Tests

`bindcraft/tests/test_tcr_prior.py` covers the interface, the backend
architecture (soft CE pull, hard naturalness ordering, region NLL), backend
dispatch, and graceful no-op fallback when repos/weights/deps are absent.

`bindcraft/__init__.py` and `bindcraft/hallucination/__init__.py` use **lazy
(PEP 562) submodule imports**, so importing `bindcraft.hallucination.tcr_prior`
(or `bindcraft.utils`) no longer drags in the heavy protenix/diffusion stack.
The unit suite therefore runs in any env with just `torch`/`numpy`/`pytest`:

```bash
python -m pytest bindcraft/tests/test_tcr_prior.py -q   # 17 passed, 5 skipped
```

The 5 skipped are opt-in **real-backend** tests. Enable them (with weights
present) via:

```bash
RUN_REAL_TCR_PRIOR=1 \
  TCRLM_HOME=... TCRLM_CKPT=... \
  LIGANDMPNN_HOME=... LIGANDMPNN_CKPT=... TCR_PRIOR_PDB=... \
  python -m pytest bindcraft/tests/test_tcr_prior.py -q
```

They assert real weights load, per-position distributions normalize, TCRLM
discriminates natural vs degenerate CDR3s, and strict mode raises on a missing
checkpoint. Prefer a compute/debug node (not the Polaris login node) since these
load real models.

## Real-backend validation

`scripts/validate_tcr_prior_backends.py` performs **real** validation (not just
interface smoke tests): it loads each backend, checks real weights, runs
`per_position_logprobs` for real output, verifies the soft loss yields non-zero
gradients on the sequence logits, exercises strict mode, and prints a status
table. It stubs `bindcraft.utils` so it can import `tcr_prior` in isolation
(no `protenix`/`scipy`).

```bash
python scripts/validate_tcr_prior_backends.py \
  --esmc-model esmc_300m \
  --tcrlm-repo /path/to/tcrLM --tcrlm-ckpt /path/to/pretrain.pt \
  --ligandmpnn-repo /path/to/LigandMPNN \
  --ligandmpnn-ckpt /path/to/model_params/ligandmpnn_v_32_010_25.pt \
  --pdb /path/to/example.pdb
```

### Validation results (this environment)

All three backends are now **genuinely functional** (validated on a Polaris
compute node, see "Isolated env on Polaris" below):

| backend | real model loaded? | real score? | soft gradient works? | fallback/no-op? | missing items |
| ------- | ------------------ | ----------- | -------------------- | --------------- | ------------- |
| esmc | **yes** | **yes** | **yes** | no | - |
| tcrlm | **yes** | **yes** | **yes** | no | - |
| ligandmpnn | **yes** | **yes** | **yes** | no | - |

Notes / evidence:
- **ESM-C** (`esmc_300m`): loaded real EvolutionaryScale weights, produced
  per-position distributions over the 20 AAs, non-dummy hard score
  (`~ -0.098`), non-zero soft-prior gradients; strict mode passes.
- **TCRLM**: loaded the real pretrained CDR3 masked-LM checkpoint
  (`pretrained_model_tcr.bin`, 158 MB, sha256 `fc53d486…`), per-position
  log-probs sum to 1, hard score `~ -0.186`; strict mode passes. It correctly
  discriminates naturalness — natural human CDR3s score perplexity ~6.6–7.9
  vs 21–576 for degenerate sequences.
- **LigandMPNN** (`ligandmpnn_v_32_010_25.pt`, true `ligand_mpnn` architecture):
  structure-conditioned log-probs on a real backbone PDB, hard score
  `~ -0.281`, non-zero soft gradients; strict mode passes. The backend auto-
  detects `model_type` from the checkpoint (`ligand_mpnn` if it carries an
  `atom_context_num` key, else `protein_mpnn`) and derives `atom_context_num`
  / `k_neighbors` / `parse_all_atoms` / `use_atom_context` accordingly — it no
  longer hardcodes `protein_mpnn` for every checkpoint (an earlier version
  did, and was validated against a mismatched `proteinmpnn_v_48_020.pt`
  checkpoint instead of a real LigandMPNN one; fixed 2026-07-10). Note: the
  ligand-atom context (`other_atoms`) only picks up non-protein/non-water
  HETATM records — for a pMHC candidate whose peptide is encoded as standard
  ATOM residues (the common case), it currently adds nothing beyond the
  backbone-level pMHC+TCR conditioning `protein_mpnn` mode already provides.
  This checkpoint is an off-the-shelf stand-in; the plan is to fine-tune a
  TCR-pMHC-specific MPNN later — the checkpoint path is the only thing that
  will need to change when that lands.

The three backends have **mutually exclusive** heavy dependencies, so they are
not all installable in one production env (see "Dependency conflict" below), but
each is fully validated in the shared isolated env described next.

### Isolated env on Polaris

A single dedicated conda env (kept off `$HOME`/`base`) runs all three backends:

```bash
# Isolated env at /lus/eagle/projects/IMPROVE_Aim1/caom/tcr_prior_backends/env
conda create -y -p /lus/eagle/.../tcr_prior_backends/env python=3.11
ENV=/lus/eagle/.../tcr_prior_backends/env
$ENV/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
$ENV/bin/pip install esm einops prody "numpy<2.3" scipy pytest
```

TCRLM pretrained weights are **not** in the git repo and **not** in the Zenodo
snapshot (both ship only git-LFS *pointer* files). Fetch the real LFS object
directly (public, no auth):

```bash
git clone --depth 1 https://github.com/hliulab/tcrLM.git $TCRLM_HOME
# resolve + download the LFS object for pretrained_model_tcr.bin
curl -X POST https://github.com/hliulab/tcrLM.git/info/lfs/objects/batch \
  -H 'Accept: application/vnd.git-lfs+json' -H 'Content-Type: application/vnd.git-lfs+json' \
  -d '{"operation":"download","transfers":["basic"],"objects":[{"oid":"fc53d486e9b90b49cc5cc5f5a1da8949e16a8dcecc77a8b9d85c6a56fee34717","size":157967498}]}'
# -> download the returned href to $TCRLM_HOME/pretrained_model/pretrained_model_tcr.bin
```

**Run compute on a debug node, not the login node.** Submit via PBS
(`bindcraft` account `IMPROVE_Aim1`); the ESM-C weights auto-download on first
use, so set the ALCF proxy and keep the HF cache on project space:

```bash
#PBS -A IMPROVE_Aim1 -q debug -l select=1:system=polaris \
#    -l filesystems=home:eagle -l walltime=00:20:00
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
export HF_HOME=/lus/eagle/.../tcr_prior_backends/hf_cache   # keep off $HOME
$ENV/bin/python scripts/validate_tcr_prior_backends.py \
  --esmc-model esmc_300m \
  --tcrlm-repo $TCRLM_HOME --tcrlm-ckpt $TCRLM_HOME/pretrained_model/pretrained_model_tcr.bin \
  --ligandmpnn-repo $LIGANDMPNN_HOME --ligandmpnn-ckpt $LIGANDMPNN_HOME/model_params/ligandmpnn_v_32_010_25.pt \
  --pdb /path/to/example.pdb
```

A ready-to-submit job script lives at
`/lus/eagle/.../tcr_prior_backends/validate_job.sh`.

### Dependency conflict: ESM-C vs ProteoCraft's fair-esm

ProteoCraft's `protenix/data/esm/compute_esm.py` uses **fair-esm** (`from esm
import pretrained`, ESM-2), while the ESM-C backend needs the **new**
EvolutionaryScale `esm` package. Both claim the top-level `esm` module and are
mutually exclusive in one interpreter. Consequently, ESM-C cannot run in-process
inside a fair-esm ProteoCraft env — use the dedicated isolated env above, or
run ESM-C via the documented subprocess-isolation option.
