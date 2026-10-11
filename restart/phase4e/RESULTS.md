# Phase 4e — Alternative allocation methods: results (A7, Bayesian-optimisation schedule search)
Run 2026-10-11 exactly as pre-registered (`PREREG.md`, approved "go ahead without stopping").
Purely additive: no Phase 4/4b/4c/4d file or result was changed; Phase 4's stored test runs were
only read. Raw outputs: `results/phase4e/`; figure: `plots/phase4e_schedules.png`.

## Verdict (pre-registered rule)
**A7 is NOT a genuine improvement.** At the primary point P it beats uniform but only ties the
best fixed decaying schedule (A3), so it fails the "beat BOTH uniform AND the better of A2/A3"
condition. At S1 and S2 it does not beat uniform at all.

In plain words: a search over complete schedules, with the same compute as the DRL agent,
ends up where a one-line decaying formula already was. It does better than the DRL agent (A5) at
P, but finds nothing beyond "decay the noise".

## Validity check
Reproduction check passed: three stored Phase 4 A1 test runs (seeds 200-202) re-run in Phase 4e
gave **identical** test mAP (e.g. seed 200: 0.028740 both times). The paired comparison against
Phase 4's stored runs is valid.

## Primary: P (R = 12, sigma_uniform = 2, eps = 1.86 worst client, delta = 1e-5), test pool, 20 paired seeds
| arm | test mAP@0.5 | eps |
|---|---|---|
| A1 uniform | 0.0592 | 1.8569 |
| A2 tuned exponential | 0.0731 | 1.8581 |
| A3 theory (rho_t ∝ √a_t) | 0.0736 | 1.8585 |
| A5 TD3+GRU agent | 0.0587 | 1.8569 |
| **A7 BO search** | **0.0747** | 1.8562 |

| comparison | diff [95% CI] | wins/losses | Holm p | result |
|---|---|---|---|---|
| A7 − A1 | +0.0155 [+0.0050, +0.0261] | 15/5 | 0.019 | WIN |
| A7 − A2 | +0.0016 [−0.0054, +0.0087] | 10/10 | 1.000 | TIE (equivalent) |
| A7 − A3 | +0.0011 [−0.0064, +0.0087] | 14/6 | 1.000 | TIE (equivalent) |
| A7 − A5 (secondary, unadjusted) | +0.0160 [+0.0090, +0.0231] | 17/3 | p 0.0001 | — |

The better of A2/A3 at P is A3 (as registered). A7 − A3 CI includes zero → not a genuine improvement.

## Secondary: S2 (R = 12, sigma_uniform = 4, eps = 0.79) and S1 (R = 25, sigma_uniform = 2, eps = 2.72)
| point | A1 | A2 | A3 | A5 | A7 |
|---|---|---|---|---|---|
| S2 (eps 0.79) | 0.0143 | **0.0254** | 0.0213 | 0.0113 | 0.0129 |
| S1 (eps 2.72) | 0.1050 | **0.1105** | 0.0802 | 0.0981 | 0.1060 |

| point | A7 − A1 | A7 − A2 | A7 − A3 | A7 − A5 (unadj.) |
|---|---|---|---|---|
| S2 | −0.0013 [−0.0083, +0.0056] TIE (equiv.) | −0.0125 [−0.0180, −0.0070] **LOSS** (1/19) | −0.0084 [−0.0138, −0.0030] **LOSS** | +0.0016 [−0.0012, +0.0044] |
| S1 | +0.0010 [−0.0063, +0.0083] TIE (equiv.) | −0.0045 [−0.0135, +0.0045] TIE (equiv.) | +0.0258 [+0.0154, +0.0362] WIN | +0.0078 [−0.0036, +0.0193] |

## What the search found (simulator) vs what held on test
| point | chosen trial | simulator gain over uniform (6 fresh selection seeds) | test gain over uniform |
|---|---|---|---|
| P | 58 | +0.051 | +0.016 (WIN) |
| S2 | 75 | **−0.050** | −0.001 |
| S1 | 56 | −0.007 | +0.001 |

* The top search scores (e.g. 0.407 vs uniform 0.323 at P; 0.159 vs 0.032 at S2) were mostly
  noise: the maximum of 100 noisy trials. On fresh seeds the S2 and S1 picks were already no
  better than uniform in the simulator itself — the selection step exposed the over-fit, as
  intended, but the pre-registered rule deploys the best of the 5 regardless.
* Schedule shapes (figure): at P, A7 rises over rounds 1-4 then decays, landing almost on top of
  A2/A3 for rounds 6-12 — hence the same test score. At S2 it has a spurious bump at rounds 8-9;
  at S1 a wave around uniform. Only at P did the search find the "decay" direction.

## Cost
| method | runs per operating point | GPU-hours P / S2 / S1 |
|---|---|---|
| A7 BO search | 330 simulated FL runs (300 search + 30 selection) | 2.2 / 2.2 / 3.1 |
| A5 agent | 336 (300 training + 36 checkpoint eval) | 1.0 / 1.0 / 7.8 (wall-clock, shared GPU) |
| A1-A3 | 0 (A2: 35 tuning runs on VAL) | ~0 |
Wall-clocks are inflated by 3 jobs sharing the GPU under software throttling (no driver power cap).

## Interpretation
1. Same conclusion as Phase 4 from a different method: the gain available from allocation at
   these budgets is what a simple decaying schedule already gets (~+0.014-0.016 mAP at P). A
   schedule search with RL-sized compute does not find more.
2. The simulator's noise (seed-to-seed sd several times the schedule effects) limits ANY
   simulator-trained method; BO handled it better than DRL at P but was fooled at S2 and S1.
3. A7 beats A5 at P (17/3), so A5's poor result is partly the agent, not only the signal.
