# Optional ESMFold2 runtime optimization

This integration is derived from `smiles724/ProteoCraft` commit
`98aacad9c421718c1925b18d71aa3975969c26ab` (`feature/proteocraft-fast`).
The adapted composition is `proteocraft-uplifting-v2`; it uses selected kernels
from Anthropic's `uplifting-biomolecular-modeling`, not the complete upstream
cookbook. Source: https://github.com/anthropics/uplifting-biomolecular-modeling

All optimization switches default to `off`. The existing `esmfold2_d` profile
continues to use `positive_score_guarded`; CDR/VHH masks, one-step diffusion,
scaffold consistency and AtlasFold cross-evaluation remain separate features.
No Fast profile is recommended until matched GPU and end-to-end validation is
complete. CPU tests do not establish kernel parity, speed or design quality.
The other branch's reported benchmark numbers are not validation of this version.

## Installation

Use a separate Linux x86-64 Python 3.12 environment. Do not upgrade the baseline
ESMFold2 environment in place. Obtain the optimization sources and pin them:

```bash
git clone https://github.com/anthropics/uplifting-biomolecular-modeling.git ../uplifting-biomolecular-modeling
git -C ../uplifting-biomolecular-modeling checkout --detach f4f62fa6592ae4938d49b1757bea0cfeff9f468e
export PROTEOCRAFT_UPLIFT_ROOT="$(cd ../uplifting-biomolecular-modeling && pwd)"
bash environment/esmfold2/install_optimization.sh /absolute/path/to/new/esmopt-env
```

The installer uses torch 2.11.0+cu128, Triton 3.6.0 and the pinned Biohub forks.
This is an environment for the selected Transformers/Triton adapter, not the
complete Biohub SDK or upstream benchmark image. Flash-attn, Transformer Engine,
DockQ and cuEquivariance are not installed by this helper. ESM 3.4.0 declares
DockQ and CUDA-13 cuEquivariance dependencies, so `pip check` reports those missing
packages even though the adapter's tested paths do not use them. Preserve that
diagnostic in experiment receipts; do not treat this environment as validation
of SDK features requiring those packages.
Model revisions are unchanged; download them with the existing
`environment/esmfold2/download_models.py`. Preserve model download receipts.
Use the CCD path and offline settings documented in `environment/esmfold2/README.md`.
Unset `ESMFOLD2_OPT`, `ESMC_OPT`, `EF2_FAST_KIT` and `ESMC_KIT`: implicit autoload
would invalidate the baseline. Keep all upstream licenses and third-party notices.

## Modes and controls

| Component | off | exact | fast | big |
|---|---|---|---|---|
| proposal | stock | skip explicitly unused confidence | fused differentiable kernels and block checkpointing | fast plus chunk size 32 |
| verifier | stock | graph execution and selected caches | exact plus fused backend | fused, chunk size 32, trunk/sampler graphs disabled |
| ESMC | stock | selected QK/rotary kernels with gradient-path guards | unsupported | unsupported |

`fast` and `big` allow numerical changes; `exact` is a parity target, not a
speed guarantee. The adapter rejects fast/big on GPUs outside SM80/SM90.
Different modes must run in separate processes. Cross-input feature/ESMC caches
are disabled because search changes sequences.
Execution binds the requested CUDA device, including graph capture and replay.
Use one optimization device per process: an existing PyTorch capture stream on
another GPU is rejected rather than reused.

Add the following to an existing `scripts/run_hallucination.py` command:

```text
--esmfold2_optimization_root /absolute/path/to/uplifting-biomolecular-modeling
--esmfold2_proposal_optimization fast
--esmfold2_verifier_optimization fast
--esmfold2_esmc_optimization exact
```

The CDR/VHH YAML launchers also forward these optimization settings; use the YAML
boolean `esmfold2_optimization_timing: true` to enable timing there.

Verifier graph token budgets retain upstream defaults unless
`--esmfold2_optimization_graph_max_tokens N` is supplied. `0` explicitly removes
the cap; validate memory on long inputs before doing this. The report contains
requested and effective budgets. An eager fallback is not proof of acceleration.

`optimization_report.json` records source hashes, versions, installed levers and
runtime call/success/error counts, including when timing is off. Handled search
failures also finalize the report; forced process termination cannot guarantee it.
`--esmfold2_optimization_timing` additionally synchronizes CUDA and writes phase
latency to `optimization_report.timing.jsonl`. Enable it consistently across
comparison arms. Nested phase timings must not be added together as wall time;
peak memory in these phase events is process-cumulative.

## Validation sequence

Hold the code, guarded candidate policy, cases, seeds and budgets constant:

1. Baseline environment, all switches off.
2. New environment, all switches off: measure environment migration separately.
3. New environment, individual optimizations and then the combined arm.

The micro runner supports `prior`, `proposal`, `update` and `verifier` phases.
The update phase exercises the complete soft update with one-step interval 1;
this differs from the regular profile cadence. Include short de novo, long
complexes and CDR/VHH cases; also test different sequences with identical lengths
in succession before relying on graph/cache correctness.

```bash
python benchmark/script/run_esm_optimization.py micro \
  --case denovo_PD1 --phase proposal --arm baseline \
  --output outputs/esmopt/micro/baseline --kit-root "$PROTEOCRAFT_UPLIFT_ROOT" --deterministic
python benchmark/script/run_esm_optimization.py micro \
  --case denovo_PD1 --phase proposal --arm proposal_fast \
  --output outputs/esmopt/micro/proposal_fast --reference outputs/esmopt/micro/baseline \
  --kit-root "$PROTEOCRAFT_UPLIFT_ROOT" --deterministic
```

The finite-value comparator checks constrained infinity masks/signs separately
and rejects NaNs. Reported deltas are evidence, not automatic quality acceptance.
The deterministic recipe is applied to both micro arms; bitwise micro parity
alone does not guarantee identical full MCTS trajectories.

## Paired searches and frozen blind rescoring

```bash
python benchmark/script/run_esm_optimization.py prepare \
  --output outputs/esmopt/pilot --kit-root "$PROTEOCRAFT_UPLIFT_ROOT" \
  --arms baseline exact fast big --seeds 43 44 --cases denovo_PD1 tcr_1qrn_no_hotspot
python benchmark/script/run_esm_optimization.py run --manifest outputs/esmopt/pilot/campaign.json
python benchmark/script/run_esm_optimization.py rescore \
  --manifest outputs/esmopt/pilot/campaign.json --output outputs/esmopt/pilot_rescore \
  --top-k 5 --seeds 1043 1044 1045
python benchmark/script/run_esm_optimization.py report \
  --manifest outputs/esmopt/pilot/campaign.json --scores outputs/esmopt/pilot_rescore \
  --output outputs/esmopt/pilot_report.csv
```

Search uses the current profile; blind rescoring defaults to unoptimized
20-loop/200-step ESMFold2. These are distinct protocols. Rescoring requires all
planned searches to finish. Failed searches remain in the report denominator;
only successful searches supply candidates. Rescoring freezes the campaign hash,
search-loss-selected unique sequences, ranks, model, settings and evaluation
seeds in `rescore_spec.json`. Restarting with a changed contract needs a new
output directory. Reports exclude incomplete per-run candidate-by-seed matrices
from quality aggregates and reject duplicate, unexpected or mismatched scores.
The CIF is committed before its score record; both resume and reporting verify
its SHA-256. Interrupted writes without a score are recomputed. Missing/changed
structures or legacy score records without a structure hash are rejected; use
a fresh output directory to rerun legacy scores.
Search failure/missing counts remain visible even before rescoring is available.

Candidate record counts are reported separately from runtime verifier calls.
Exports use `-1` for unavailable trajectory/call counts, rather than deriving
model calls from candidate rows. `export --manifest ... --output ... --top-k 5`
builds the existing independent-evaluator manifest. Search-prediction reference
structures support refolding consistency, not agreement with experimental truth.

Predefine quality/completion thresholds before collecting results. Compare total
wall time including startup and compilation, warmed latency, memory, completion
and blind quality. Multiple seeds or hotspot variants of one source target do
not constitute independent targets. A small pilot cannot establish universal
noninferiority. Historical panel runners and outcome-specific report scripts
from the source branch are intentionally not part of this integration.
