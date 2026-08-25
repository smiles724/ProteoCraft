# Old Protenix + TCR sequence-prior experiment — recovery audit

Audited 2026-08-20 from raw artifacts on `erran`, not from prose summaries.
Recovery scripts: `/root/mingxuan/{recover_old,recompute_naturalness,analyze_old}.py`.
Recovered data: `/root/mingxuan/{old_protenix_recovered,old_naturalness_recomputed}.json`.

## 1. What was actually run

All artifacts survive. **32/32 runs completed** (`status: ok`), 8 arms × 4 cases.

| | |
|---|---|
| Backbone | `protenix_base_20250630_v1.0.0` (identical in all 32) |
| Cases | `9NMW_alpha`, `9NMW_beta`, `1QRN_alpha`, `1QRN_beta` |
| Budget | `n_steps=150`, `k_top=8`, `b_candidates=4`, `mcts_strategy=two_stage` |
| Prior weights | gradient 0.1, verifier 0.1, `region=cdr`, `require_real_tcr_prior=true` |
| Wall clock | ~2.7–3.1 h per run |
| Framework | unchanged in all 32 runs |
| Outputs | `/root/mingxuan/cdr_prior_full_out*/<case>/result.json` |
| Inputs | `/root/mingxuan/CDR_redesign_4cases_minimal_package_20260714/` |

The 8 arms: `_off` (prior off), `cdr_prior_full_out` (TCRLM grad+ver),
`_tcrlm_gradonly`, `_tcrlm_veronly`, `_esmc6b` (grad+ver), `_esmc6b_gradonly`,
`_esmc6b_veronly`, `_ligandmpnn` (verifier only).

## 2. Bugs — status

**TCRLM window bug: fixed, and the fix is still present.**
`TCRLMBackend.SEQ_MAX_LEN = 20`, and `_window_for_region` (tcr_prior.py:596)
centres that window on the requested region instead of truncating to the first
20 residues. The docstring records the original failure. No recovered run shows
the `log(20)`/`0.29957` signature. Re-validated live today: TCRLM loads real
weights, produces real per-position output, gives a non-zero soft gradient, and
does not fall back (`hard_score = -0.1865`).

**LigandMPNN is verifier-only — confirmed from code and config, not just prose.**
`LigandMPNNBackend.requires_structure = True`; without `metadata['pdb_path']` it
returns `None`/`0.0`. The recovered config has
`use_tcr_prior_gradient=false, use_tcr_prior_verifier=true`. So the path was
candidate → Protenix structure → temp PDB → hard score → verifier/MCTS, exactly
as remembered.

## 3. Two problems that change how the old results read

**(a) The baseline in the naturalness table is not the matched `prior_off` arm.**
Recomputing the table from raw `best_seq` values reproduces **24/32 cells to
within 5e-4** — every cell of `tcrlm_fixed`, `esmc6b` and `ligandmpnn`. All 8
`baseline` cells mismatch, and the CDRs differ outright:

| case | `prior_off` best CDR | stored "baseline" CDR |
|---|---|---|
| 9NMW_beta | `AATSVNGDPIAYW` | `VATSMSIYGEGPP` |
| 1QRN_alpha | `AAVTTDSWGILEF` | `CAILTDSWGILYF` |
| 1QRN_beta | `ASSRPGLTDGRPEQFF` | `EASRPGDAGIRFCQYF` |

So the headline comparison was made against some *other* run, not the matched
control sitting next to it. (This also recovers the lost scoring convention: the
old numbers are full-chain context with NLL averaged over CDR positions, not the
CDR scored standalone.)

**(b) No seed control anywhere.** No `seed` key in any yaml, in `params`, or in
any run log. One replicate per cell. The `prior_off` / 9NMW_alpha run peaked at
step 6 of 150, which is what an unlucky draw looks like — so run-to-run variance
is real and unmeasured.

**(c) `best_loss` is not comparable across arms.** `apply_tcr_verifier_score`
adds `-score` into `total_loss`, so every `use_tcr_prior_verifier=true` arm
optimises a different objective than `off`. Only the `gradonly` arms can be
compared to `off` on loss.

## 4. Corrected results (vs the matched `prior_off` arm)

TCRLM NLL (lower = more TCR-like):

| arm | mean Δ vs off | improved |
|---|---:|---:|
| TCRLM prior | **+0.026** | 2/4 |
| ESM-C prior | +0.061 | 2/4 |
| LigandMPNN | +0.162 | 1/4 |

ESM-C NLL (lower = more natural):

| arm | mean Δ vs off | improved |
|---|---:|---:|
| TCRLM prior | +0.170 | 2/4 |
| ESM-C prior | **−0.690** | **4/4** |
| LigandMPNN | −0.410 | 3/4 |

Structure loss, `gradonly` arms only (the comparable ones):

| arm | mean Δ vs off | improved |
|---|---:|---:|
| TCRLM grad-only | −0.172 | 2/4 |
| ESM-C grad-only | −0.539 | 4/4 |

## 5. What holds, what doesn't

**Holds:**
- The ESM-C prior reliably improves ESM-C naturalness — 4/4 cases, ~0.69 nats.
  This survives the baseline correction, and is the only consistent-sign result.
- The TCRLM prior does **not** measurably improve TCRLM naturalness (2/4,
  mean slightly *worse*). "Weak / case-dependent" was, if anything, generous.
- The prior genuinely entered both paths — strict mode was on and no run fell
  back to a dummy scorer.
- Judging LigandMPNN by TCRLM/ESM-C naturalness is not meaningful; it optimises
  neither.

**Does not hold:**
- Any claim that a prior improves final structure quality. n=1, unseeded, and
  the loss is not comparable for half the arms.
- Any claim that the ESM-C prior makes CDR3 "more TCR-like" in general. It
  improves the ESM-C metric while leaving TCRLM slightly worse — largely
  self-scoring, i.e. optimising the metric you then evaluate with.

## 6. Carry-over for the ESMFold2 experiments

1. **Always set and record the seed.** The single biggest defect here.
2. **Never compare a raw `total_loss` across arms** once an arm adds a term to
   it. Report structure loss, naturalness and contact separately (spec 12).
3. **Always cross-evaluate.** A prior improving its own scorer proves little;
   the TCRLM/ESM-C split above is the whole lesson.
4. **Use the arm's own matched control**, generated in the same batch.
5. Keep the `_gradonly` / `_veronly` decomposition — it is what made the loss
   comparability problem visible and separates the two channels cleanly.
6. TCRLM's context window is 20 residues. It is a CDR3-level model; do not
   expect it to score whole-chain naturalness.
