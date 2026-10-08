# Phase 3 — extended noise-pulse test: results (2026-10-08)

Pre-registration: `PREREG.md`. Raw runs (3,120): `restart/results/phase3/`. Analyses:
`analysis_V1/V2/V3.txt|json` (`restart/code/analyse_phase3.py`). Figures:
`restart/figures/phase3_V1/V2/V3`. FL, 8 clients, B=64, 20 local iters, lr 0.03,
mu=0, calibrated fixed C*, pulse = 2x sigma in one round, 20 paired seeds per cell.

## Cell verdicts (pre-registered rule)

| variant | sigma = 1 | sigma = 2 | sigma = 4 |
|---|---|---|---|
| V1: 12 rounds, head only (65k params) | VARIES | VARIES | VARIES |
| V2: 25 rounds, head only | VARIES | VARIES | VARIES |
| V3: 12 rounds, + blocks 7-8 (116k params) | VARIES | NOT IDENTIFIABLE | NOT IDENTIFIABLE |

**No cell is FLAT. Per the decision rule, a_t is NOT flat in general.** 7 of 9 cells
show significant variation (slope and early-vs-late CIs exclude zero); the other 2
are not identifiable because the pulse's pooled harm is not significant — in V3 the
larger model sits near the floor under DP (baseline mAP 0.110 at sigma 2, 0.0065 at
sigma 4), so a pulse has little left to destroy. Even there, slope and shape tests
point the same way (descriptive).

## Shape of a_t
* Always INCREASING: noise in later rounds hurts the final model more.
* 12 rounds (V1, V3): a gradual rise across the run, steepest at the end; the rise
  is significant even excluding the final round (descriptive).
* 25 rounds (V2): flat for rounds 1-20 (slope CI includes 0, descriptive), with the
  sensitivity concentrated in the last ~5 rounds (mean drop 0.003-0.017 vs
  0.021-0.051; final round 0.05-0.135).
* The final-round spike partly reflects no recovery time before evaluation — real for
  a deployed model, but "late rounds matter" is mostly "the last few rounds matter".

## Gradient norm
Falls monotonically in every cell, including over rounds 3..R (all CIs exclude 0):
e.g. V1 sigma 1: 29 -> 12; V3 sigma 1: 43 -> 15. With C* ~ 5.5-6.1 below these
norms, most per-sample gradients are clipped to C* in every round.

## Consequence (pre-registered: report the regime specifically)
The regime where adaptive allocation / DRL has something real to learn: **every
tested configuration with a live model** — 12 and 25 rounds, both model sizes,
sigma 1-4 where the baseline is off the floor. The optimal-allocation theory then
says rho_t ~ sqrt(a_t): spend more budget (less noise) in late rounds, most of all
in the final few. With 25 rounds the target is a "back-loaded" schedule rather than a
smooth ramp.
