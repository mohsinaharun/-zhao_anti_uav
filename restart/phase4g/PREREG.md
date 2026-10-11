# Phase 4g — Round-count convergence check
Pre-registration, written 2026-10-12 BEFORE any Phase 4g code is written or run.
**Approved by the user 2026-10-12 WITH both proposed additions (+A) no-DP ceiling and (+B) uniform at each R ("Approved with +A and +B, go ahead without stopping").**
Added after external feedback. Purely additive: Phase 4 / 5 files and results are read, never changed or
re-run.

## 1. Question
**Q9.** At a fixed privacy budget (worst-client ε 1.86), does test mAP keep climbing toward the no-DP
ceiling as the number of communication rounds grows, or does it plateau well below it?
Climbing would mean the current DP/no-DP gap is a convergence-speed problem (more rounds would close
it). A plateau would mean a noise ceiling (more rounds would not).

## 2. Setting (Phase 4, unchanged)
`code/fl_env.py` FLEnv (imported read-only): 8 real clients, B 64, 20 local iterations per round, SGD
lr 0.03 momentum 0.9, per-seed calibrated C*, frozen trunk + 65,221-parameter head, frame-level DP,
δ 1e-5, Opacus RDP accountant, worst-client ε. **TEST pool**, 20 paired seeds 200-219, common random
numbers, no attackers.

## 3. Arms
**A2 (tuned exponential), k = 0.5, at R = 12, 25, 50, 100 rounds**, every run at worst-client ε 1.86.
σ_t ∝ exp(−k (t−1)/(R−1)), scaled to the budget exactly as in Phase 4.

**Fixed now: k stays 0.5 at every R.** It is P's tuned value, and the shape is defined on normalised
time (t−1)/(R−1), so it stretches with R. It is not re-tuned per R. Disclosed: Phase 4 tuned k = 0.25
at R 25 (σ 2). A re-tune would cost 35 VAL runs per R and would mix "more rounds" with "a different
schedule".

Fixing ε means **more rounds = more noise per round** (computed with the Phase 4 accountant):

| R | uniform σ at ε 1.86 | A2 σ, first → last round | A2 worst-client ε |
|---|---|---|---|
| 12 | 2.00 | 2.67 → 1.62 | 1.8581 (Phase 4's runs, reused read-only) |
| 25 | 2.71 | 3.58 → 2.17 | 1.8566 |
| 50 | 3.70 | 4.88 → 2.96 | 1.8565 |
| 100 | 5.15 | 6.77 → 4.11 | 1.8553 |

### Proposed additions (NOT in the request; included only if you say yes)
* **(+A) No-DP ceiling at each R** (σ 0, clipping kept, FedAvg, uniform; R 25 / 50 / 100 × 20 seeds =
  60 runs, ~45 min). The 0.250 ceiling is a **12-round** number (Phase 5 no-DP FedAvg). Without DP,
  100 rounds may well score above 0.250, and then "closing the gap" has to be judged against the
  ceiling *at the same R*. **Recommended.** Without it the decision rule uses 0.250 at every R and
  says so.
* **(+B) Uniform at each R** (A1 at ε 1.86, R 25 / 50 / 100 × 20 seeds = 60 runs, ~45 min). This
  separates "more rounds" from "k = 0.5 may not suit long runs". Optional.

## 4. Seeds and cost: 20 seeds, not 10
One 12-round run takes ~27 s with 3 parallel workers (Phase 5c: 380 runs ≈ 1 h), i.e. about
2.2 s per round. The request (A2 at R 25 + 50 + 100) is 175 rounds × 20 seeds = 3,500 rounds,
**≈ 45 min** with 3 workers (≈ 25 min at 10 seeds). With (+A): ≈ 1.5 h; with (+A)+(+B): ≈ 2.3 h.
**Cutting to 10 seeds would save ≈ 20 min and cost √2 wider CIs, so all arms use 20 seeds.**
These timings assume the GPU power cap stays unset (software throttling). R = 12 is reused, not re-run.

## 5. Reproduction check (before the grid; stop on failure)
This code's A2 at R 12 reproduces Phase 4's `eval_P_A2` test mAP exactly on seeds 200-202. If (+A)
is approved, its σ 0 run at R 12 must also reproduce Phase 5's `nodp_noDP_fedavg_none` on seeds
200-202 (verified-only runs; the R 12 numbers reported are the stored ones).

## 6. Decision rules
Family **F11** (Holm, α 0.05): paired differences of A2 between consecutive round counts:
**25 − 12, 50 − 25, 100 − 50.** Each is WIN (significant rise), LOSS (significant fall) or TIE, with
EQUIVALENT if the CI lies within ±0.02 (Phase 4 conventions).
Gap closure: **G(R) = mean mAP_A2(R) / ceiling(R)**, with ceiling(R) = no-DP FedAvg at the same R if
(+A) runs, else 0.250.

**Verdict:**
* **CONVERGENCE-LIMITED** ("more rounds would close it"): the last step (100 − 50) is a WIN, or
  G(100) ≥ 0.8.
* **NOISE CEILING / PLATEAU** ("they would not"): the last step is not a WIN and the best G over all
  R is < 0.5 ("well below" = under half the ceiling).
* **DECLINING** (a plateau sub-case, reported as such): the plateau rule holds and 100 − 50 is a LOSS.
* Otherwise **PARTIAL**: improves, but neither near the ceiling nor clearly flat. The numbers are
  reported as they are.

Reported with every number: ε per R, σ range per R. Table: mAP (mean, sd, 95% CI) vs R, plus
ceiling(R) and G(R). Figure: `plots/phase4g_rounds.png`, mAP vs R (log axis) with ceiling.
Secondary: per-round proxy-val learning curve is **not** logged (cost); only final test mAP.

## 7. Predictions (scored after; not decision rules)
Arithmetic first. Under clipping each round adds a bounded signal step, so total signal grows ∝ R. The
noise std grows ∝ √R·σ, and at fixed ε σ grows roughly like √R (σ 2.0 → 5.15 for 12 → 100 rounds).
So signal-to-noise of the summed updates barely moves: √R/σ = 1.73, 1.85, 1.91, 1.94 for
R 12 / 25 / 50 / 100.
1. **Plateau, not convergence:** 100 − 50 is not a WIN, and no R reaches 0.125 (half of 0.250).
2. A modest rise from 12 to 25 is possible (Phase 4 S1, R 25 at σ 2, scored 0.105 vs 0.059 at R 12, but
   at a looser ε 2.72); expected small at fixed ε.
3. If (+A) runs: the no-DP ceiling rises with R (no-DP at 100 rounds > 0.250), so G(R) falls with R even
   if A2's mAP is flat.

## 8. Outputs
Code `code/phase4g.py` (new; imports `fl_env`, `budget`, `phase4` read-only), launcher
`code/run_phase4g.ps1`, analysis `code/analyse_phase4g.py`, results `results/phase4g/`, report
`phase4g/RESULTS.md` + thesis table, failures in NEGATIVE_RESULTS.md.
