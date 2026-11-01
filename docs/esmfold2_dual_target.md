# Independent two-stage dual-target binder mode

This is a separate opt-in entry point. Existing launch scripts, profiles,
single-target Stage A/B modes, and legacy runtime defaults are preserved. Accelerated execution
requires the pinned optimization environment described in `ESM_OPTIMIZATION_EXPERIMENTS.md`
(Linux Python 3.12, Torch 2.11 / Triton 3.6 and the pinned Biohub forks).
Like the latest `esmfold2_d` search profile, this runner defaults to proposal
`fast`, verifier `fast`, and ESMC `exact`. Explicit `off` values in a custom JSON
remain supported; the low-level library defaults and independent rescoring are
not changed. Set `PROTEOCRAFT_UPLIFT_ROOT` or pass `--optimization-root`.

The optional search setting `verification_order="target"` evaluates each pending
candidate against human first, then the same candidates against mouse. Results
are joined in the original candidate order only after both sides finish; cache
keys, verifier seed, candidate budgets, ranking, and Stage-B preview are unchanged.
The default remains `verification_order="candidate"` for frozen/legacy behavior.
Target grouping can reduce fast-verifier CUDA graph recaptures when target lengths
differ, but requires a numerical/reproducibility check in the actual runtime before
production use. CPU trajectory parity does not establish GPU numerical parity.

The supplied JSON now explicitly runs `strategy=two_stage`, `outer_steps=110`,
and `stagea_steps=70`: 70 Stage-A expansions followed by 40 Stage-B expansions.
The low-level `DualTargetConfig` keeps `stage_a_only` / 70 steps for compatibility
with already frozen JSON/API callers. An old 70-step JSON does not silently
become a different experiment.

```bash
python scripts/run_esmfold2_dual_target.py \
  --config configs/esmfold2_dual_target.json \
  --human-fasta /absolute/path/human_target.fasta \
  --mouse-fasta /absolute/path/mouse_target.fasta \
  --seeds-fasta /absolute/path/cluster_representatives.fasta \
  --output-dir /absolute/path/new_run \
  --validate-only
```

Remove `--validate-only` to run inference/optimization. Target FASTAs must each
have one canonical amino-acid sequence. Seeds must be unique, nonempty canonical
amino-acid sequences; different seed lengths are supported. The runner does not
impose a binder-length category or numerical length cap. Choose lengths according
to the experiment and applicable submission rules, and validate GPU memory and
runtime at the longest planned length before dispatching a campaign. The supplied config explicitly sets
`seed_cysteine_policy=random_replace`: each input Cys is replaced once by a
uniformly sampled non-Cys amino acid before root verification. Other positions
and sequence length are unchanged. Sampling uses `initialization_seed` and a hash
of the original sequence, independent of FASTA order and global RNG state. When
`initialization_seed` is null, it falls back to the search `seed`. The supplied
config fixes `initialization_seed=43`, so searches with seeds 43, 44 and 45 start
from the same transformed sequence. Original
sequences and one-based substitutions are saved in the manifest. Historical KD
labels do not transfer to these initialized sequences. The low-level API defaults
to `reject` for Cys-bearing seeds unless this transformation is explicitly enabled.
Sequence provenance/competition eligibility is not certified by this runner.

`--validate-only` checks inputs and reports requested acceleration settings;
it does not certify the CUDA environment or load models. A real accelerated run
fails before model loading if the source root is missing. No automatic fallback
to stock execution is provided. Acceleration is installed once on the shared
runtime, while human/mouse input features are prepared separately on every call.

Optional `--hotspots-json` accepts separate `human` and `mouse` dictionaries in
the existing runtime `hotspot_params` format. Residue indices must be mapped to
each supplied target sequence; the human indices are never copied to mouse.

## Explicit homotrimer targets

The default target assembly remains `monomer`; a multi-record FASTA or a
pipe-delimited sequence is rejected by that dual-target input gate. To design
against a TNF homotrimer, explicitly add these fields to the JSON `runtime`:

```json
{
  "target_assembly": "homotrimer",
  "target_interface_mode": "contacted",
  "target_interface_contact_cutoff": 5.0,
  "target_interface_min_contacts": 5
}
```

Keep the other frozen runtime/search settings in that JSON. Each human/mouse
target FASTA must contain exactly **three separately named records**, all with
the same canonical protomer sequence for that species. Human and mouse target
sequences must differ. No chain is automatically replicated from a single
record. If `target_chain_roles` is specified, all three roles must be `target`.
The runtime receives three target chains plus one binder chain; original FASTA
IDs, chain order, assembly and the contact-policy settings are recorded in the
manifest and cache identity. `--validate-only` reports the three-chain counts,
total target residue counts and requested interface policy without loading a
model. This validates input plumbing, not whether the predicted trimer adopts
the experimental assembly.

In `contacted` mode, full-verifier interface scoring uses target protomers with
at least five distinct binder–target residue-pair contacts within 5 Å, measured
from predicted heavy-atom coordinates (or the configured thresholds). For each
eligible protomer, it takes the smaller of the two native directional pair-chain
iPTMs; their arithmetic mean is the primary `pair_chain_iptm`. A binder can use
one or multiple protomers, without a requirement to contact all three. No
eligible protomer gives a primary score of zero. Per-chain directional scores,
contact counts and the chosen subset remain available as diagnostics; global
complex iPTM is a separate quantity. Clashes must still be checked against the
**whole** trimer, including protomers excluded from interface scoring.

Soft-gradient contact masks use expected residue contacts derived from the
distogram. When no protomer reaches that proposal threshold, the highest-contact
protomer (including tied protomers) is a recorded proposal-only fallback so optimization can acquire an
interface. This fallback cannot make a no-contact full verification pass.
Differentiable pair-iPTM is a PAE-based surrogate with pair-specific length
normalization, not the native verifier's confidence head output. The outputs
identify whether selection used distogram expectations or coordinate contacts;
these two geometrical evidence sources must not be described as identical.
Expected-contact totals are probabilities summed over residue pairs, not
observed physical contacts; many small probabilities can accumulate for long
sequences. Proposal logs retain the per-chain counts, selected indices, fallback
and distance-bin policy, while verifier summaries retain the full native pair
matrix and physical residue-pair counts.

Only the opt-in mode calibrates distance bins by head: Release 64 bins use
finite-interval midpoint approximations over 2--22 Å, while Experimental 128
bins retain the design Algorithm 12 convention (1.5--54.5 Å midpoints).
Unknown head sizes raise an error. The physical verifier gate uses coordinates
and does not depend on this bin approximation. See the distogram-loss section
in [the official ESMFold2 paper](https://biohub.ai/papers/esm_protein.pdf),
Appendix A.2.8.1; this output head is distinct from confidence geometry embedding.

Stage B receives the same complete target input and interface policy for both
proposals, preview and full verification. A handoff with altered chain IDs,
assembly, roles or contact thresholds is rejected. Legacy monomer handoffs
missing only the four newly introduced opt-in settings resolve them to the old
monomer/all-target defaults; other runtime checks remain strict. Calibrate the
new trimer-aware scores against matched parent/control predictions and run a
real-model smoke before dispatching TNF production work. These changes do not
add pH-aware physics or establish pH selectivity.

## Stage-A search contract

1. Verify each seed against both targets; only complete finite pairs become
   roots. Seed logits are initialized deterministically: each starting residue
   receives the existing sequence anchor of 2.0, other allowed residues start at
   zero, and Cys is masked. This is an initial preference, not a fixed-site mask
   or an ongoing penalty; all allowed residues remain optimizable.
2. Select a shared parent using the existing hierarchical UCB tree, with root
   warmup. `outer_steps` is the total expansion budget across all roots.
3. Run one soft chunk against human and one against mouse, sequentially on the
   same runtime/model weights. Each uses only its own saved logits and target.
4. Each side has one raw argmax slot plus three mutation slots. Mutations are
   generated from the parent using that side's before/after logit deltas. The
   existing positive-score/guarded sampler is reused. No gradient fusion,
   mutation intersection, confidence-snapshot candidates, or cross-side refill.
5. Raw no-ops retain their slot. Duplicate mutations are resampled at most
   `refill_attempts` times, without another soft optimization call. Shortfalls
   are allowed when insufficient distinct proposals exist. Merge identical
   sequences and preserve all origin records.
6. Verify every unique proposal against both targets. A run-local cache keys
   target identity, sequence, target context, runtime config, and verifier seed.
   The same fixed verifier seed is used across candidates for matched scoring.
   Runtime/value failures are logged and excluded; programming errors abort.
7. Rank by `(max(human_loss, mouse_loss), mean_loss, sequence)` ascending.
   Both sides use the same `aligned_common` verifier configuration. Admit the
   best two eligible children by default and backpropagate their worst losses.
   UCB exploration may revisit another parent; this is not greedy hill climbing.
8. A child independently inherits each target's post-update logits, minimally
   aligned to the common child sequence. No averaging or copying across targets.
   The two target states are inherited even when only one proposed the child.
9. Stage-A ranking includes all successfully dual-verified sequences, including
   seeds and candidates not admitted into the tree. A seed may remain best.

For a separate budget per starting sequence and repeat, invoke this runner once
per sequence and search seed, with a distinct output directory for every run.
A FASTA containing multiple starts shares the single `outer_steps` budget;
it does not give each start that many expansions. Keep `initialization_seed`
fixed when comparing search repeats, and use the same `verifier_seed` for matched
scoring. Comparability of human and mouse loss scales still needs evaluation;
the worst-side ranking alone does not establish balanced experimental affinity.

## Stage-B search contract

The native `esmfold2_d` defaults are retained: two handoff roots, 20 soft steps
per proposal chunk, single mutations, a 20-sequence preview pool, up to eight
full-verifier candidates, one preview exploration slot, at most two admitted
children, and `no_group` priors. The pool budget is shared between the two
independent proposal directions (10 human + 10 mouse by default). Stage B uses
the existing action-level PUCT tree and native proposal/prior/preview helpers;
the 4+4 raw-argmax rule belongs to Stage A, not Stage B.

1. Select the best two Stage-A dual-verified sequences by worst loss then mean
   loss. Preserve both node-local soft states, including a state for a best
   verified candidate that was not admitted as a tree child. Save their paired
   logits, target contexts, runtime contract and source hashes in a handoff.
2. Run one soft chunk per target from that target's inherited logits. Generate
   only single mutants. There is no cross-target logit averaging. Inference is
   sequential on the same accelerated runtime, with proposal `fast`, verifier
   `fast` and ESMC `exact` installed by the existing adapters.
3. Preview each unique candidate on **both** targets. Each target's parent and
   candidates share one fixed preview seed; caches are local to the target and
   current step. Rerank by improvement of the worse preview loss. If either
   side's preview fails, the whole pool falls back to gradient priors and records
   that fallback; scores from different sources are never mixed.
4. Use native exploit/explore action selection to choose up to eight candidates.
   Verify every selected candidate on both targets with the same fixed verifier
   seed used in Stage A. Only complete pairs can enter the tree.
5. Preserve the native per-target reward components and hotspot weights. The
   joint tree reward is the smaller of the two native rewards. Each target
   inherits a separate branch quality floor: its handoff pair-iPTM minus 0.05,
   bounded below by `stageb_min_pair_chain_iptm` (default zero). Floors remain
   fixed along descendants; they do not gradually relax. Native policy allows
   one quality-exploration child when guarded slots remain.
6. Final guarded selection checks **both** targets against their respective
   final floors. Final floors follow the native handoff convention separately
   per target: maximum handoff pair-iPTM minus 0.05. Hotspot coverage/distance and
   the worse native reward guide ranking. A result below a floor remains flagged
   as such; a completed run does not certify that a passing candidate exists.

`ranked.json` in two-stage runs ranks the handoff roots plus successfully verified
Stage-B candidates by this guarded rule. `stagea_ranked.json` retains all Stage-A
results. `best_by_worst_loss.json` separately retains the best joint-loss result
across both stages, so the two selection criteria cannot be confused.

## Continue completed Stage A

Use the same search/verifier seeds and target FASTAs as the source run. The
110-step supplied profile is converted to a **40-step Stage-B-only** run:

```bash
python scripts/run_esmfold2_dual_target.py \
  --config configs/esmfold2_dual_target.json \
  --human-fasta /absolute/path/human_target.fasta \
  --mouse-fasta /absolute/path/mouse_target.fasta \
  --stagea-results /absolute/path/completed_stagea_run \
  --output-dir /absolute/path/new_stageb_run \
  --optimization-root /absolute/path/uplifting-source \
  --validate-only
```

New runs save `stagea_handoff.pt` plus its SHA256 receipt. Paired-state handoff
checks targets, seeds and the runtime contract. Legacy 70-step outputs have only
sequences and scores: their top two sequences are explicitly reinitialized with
the normal 2.0 anchor, tagged `sequence_only_reinitialized`, and freshly verified
on both targets. They are not an exact restoration of missing logits/tree state.
Both paths reverify roots in the new run and keep the old directory unchanged.
This resumes at the stage boundary; arbitrary mid-Stage-B resume is not provided.

pH optimization and independent final rescoring remain separate work. Do not
interpret search scores as affinity measurements or pH selectivity.

## Outputs

- `manifest.json`: original/initialized seeds, substitutions, exact configuration
  and running/completed/failed status.
- `initialized_seeds.fasta`: actual sequences sent to root verification.
- `verifications.jsonl`: cached per-target losses, confidence summaries, errors.
- `steps.jsonl`: parent, proposals, mutations, origins, paired scores, admissions.
- `stagea_ranked.json`: all successful Stage-A pairs, ranked by worst/mean loss.
- `stagea_handoff.pt`, `stagea_handoff.json`: paired soft states and hash receipt.
- `stageb_handoff.json`: roots and separate branch/final quality floors.
- `best.json`: current primary selection, guarded during Stage B.
- `best_by_stageb_guarded.json`, `best_by_worst_loss.json`: separate selectors.
- `ranked.json`, `ranked.fasta`: final primary ranking, emitted on completion.
- `optimization_report.json`: requested/applied acceleration, pinned sources and
  runtime counters; finalized on both success and failure. Requested fast alone
  does not prove that a CUDA graph was replayed for these input sizes.

Existing run manifests cannot be overwritten. Interrupted runs retain their
logs; a hard process kill may leave status `running`, never `completed`.

## Validation

`python -m pytest bindcraft/tests/test_esmfold2_dual_target.py bindcraft/tests/test_esmfold2_dual_target_stageb.py -q`

Tests include real runtime proposal plumbing with an injected soft-update
function, full 70+40 budget execution, independent preview/state/floors, source
validation and legacy continuation, without loading models. These tests do not establish GPU quality or
biophysical performance. Run a small real-model smoke evaluation before a large
campaign, then independently rescore selected candidates under a frozen protocol.
