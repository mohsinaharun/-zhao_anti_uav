# Phase 2 — results (2026-10-06)

Pre-registration: `PREREG.md`. Raw runs: `restart/results/phase2/`; verbatim analysis:
`restart/results/phase2/analysis.txt` (`restart/code/analyse_phase2.py`). Figure:
`restart/figures/phase2_ablation`. 8 clients, 12 rounds x 1 local epoch, sigma = 2,
n = 10 paired seeds. All numbers on `val`.

## Selections (seed 0, pre-registered rules)
FL lr 0.03 (0.471 vs 0.468 / 0.077). **FedProx mu = 1** (0.491 vs 0.471 at mu=0) — a
one-seed margin of 0.02, about one seed-sd, so weak evidence FedProx helps. FL+DP lr 0.03.

## Gate G2 — PASS (narrowly)
| cell | val mAP@0.5 (mean +- sd, 10 seeds) |
|---|---|
| neither (central, no DP) | 0.557 +- 0.019 |
| FL, no DP | 0.487 +- 0.020 |
| DP, no FL (central) | 0.335 +- 0.034 |
| **FL + DP** | **0.145 +- 0.070** (threshold 0.10) |

## Contrasts (paired, 95% CI, wins out of 10)
| contrast | mean | 95% CI | wins |
|---|---|---|---|
| cost of FL (no DP) | -0.070 | [-0.087, -0.053] | 0/10 |
| cost of DP (no FL) | -0.222 | [-0.250, -0.194] | 0/10 |
| cost of DP inside FL | -0.342 | [-0.396, -0.288] | 0/10 |
| interaction | **-0.120** | [-0.178, -0.062] | 0/10 (p = 0.001) |

**DP hurts far more inside FL than centrally**, although each FL client's epsilon is
LARGER (median 2.98, max 4.55 vs central 1.69): the cells are matched on sigma, not
epsilon. Plausible, untested reasons: per-step noise on the aggregate is ~sigma*C/(256*sqrt 8)
= sigma*C/724 vs sigma*C/1024 centrally; fewer local steps on small clients. The batch
size differs between FL (256) and no-FL (1,024) by design (pre-registered confound).

Per environment, FL+DP: building 0.18, cloud **0.02**, sky 0.26, terrain 0.09 — cloud
is essentially lost under FL+DP.

## Consequences for Phase 3
FL+DP at sigma = 2 sits near the floor (0.145, sd 0.07, high seed variance). Phase 3
must choose its operating sigma off the floor — for FL that means sigma < 2
(e.g. 1-1.5), measured, not assumed.
