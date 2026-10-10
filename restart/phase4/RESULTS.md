# Phase 4 results — learned vs uniform vs fixed-schedule privacy-budget allocation
Run 2026-10-09/10 under PREREG.md + Amendments 1-3. TEST pool (first use), 20 paired seeds
(200-219, common random numbers), frame-level DP, delta = 1e-5, worst-client epsilon.
Raw: `results/phase4/eval_*`, `analysis.json`, `analysis.txt`; figure `plots/phase4_schedules.png`.

## Verdict (pre-registered rule, §7)
**"Learned allocation helps" — NOT supported.** At the primary point P the feedback agent
(A5) TIES uniform (equivalent within ±0.02) and LOSES to both fixed formulas (A2, A3);
the blind agent (A4) LOSES to uniform. A one-line decaying schedule captures the gain.

## Primary family — P (R = 12, eps = 1.86 for every arm), Holm-corrected
| arm | test mAP@0.5 | sd | eps |
|---|---|---|---|
| A1 uniform | 0.0592 | 0.027 | 1.8569 |
| A2 tuned exponential (k = 0.5) | 0.0731 | 0.025 | 1.8581 |
| A3 theory (rho_t ~ sqrt a_t) | 0.0736 | 0.029 | 1.8585 |
| A4 blind DRL | 0.0000 | 0.000 | 1.8569 |
| A5 feedback DRL | 0.0587 | 0.028 | 1.8569 |

| comparison | diff | 95% CI | W/L | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| A5 - A1 | -0.0005 | [-0.0124, +0.0115] | 11/9 | 0.935 | 0.935 | TIE (EQUIVALENT) |
| A5 - A2 | -0.0144 | [-0.0240, -0.0047] | 5/15 | 0.0056 | 0.011 | LOSS |
| A5 - A3 | -0.0149 | [-0.0222, -0.0076] | 4/16 | 0.0004 | 0.0013 | LOSS |
| A4 - A1 | -0.0591 | [-0.0718, -0.0465] | 0/20 | <1e-4 | <1e-4 | LOSS |
| A2 - A1 (secondary) | +0.0139 | [+0.0076, +0.0202] | 16/4 | 0.0002 | – | |
| A3 - A1 (secondary) | +0.0144 | [+0.0026, +0.0262] | 16/4 | 0.019 | – | |

## Secondary operating points (unadjusted)
**S2 (R = 12, eps = 0.79):** A1 0.0143 / A2 **0.0254** / A3 0.0213 / A4 0.0000 / A5 0.0113.
A2-A1 +0.0111 [+0.0036, +0.0187] 16/4; A3-A1 +0.0071 [+0.0027, +0.0114] 16/4;
A5-A1 -0.0029 [-0.0092, +0.0033] 7/13; A5-A2 -0.0141 1/19; A4-A1 -0.0143 0/20.
Near the floor; same ordering as P.

**S1 (R = 25, eps = 2.72):** A1 0.1050 / A2 **0.1105** / A3 0.0802 / A4 0.1078 / A5 0.0981.
A2-A1 +0.0055 [-0.0005, +0.0114] 14/6 (n.s.); A4-A1 +0.0028 [-0.0118, +0.0173] (tie);
A5-A1 -0.0069 [-0.0201, +0.0064] (tie); A3-A1 **-0.0248** [-0.0371, -0.0125] 3/17.
With 25 rounds the gain from decaying shrinks to non-significance, and the theory schedule
LOSES: per-round a_t from Phase 3 V2 is noisy (many rounds at the floor -> sigma capped at
4.25), so the "optimal" schedule inherits the measurement noise.

## What the agents learned
* **A5 at P** hit the bottom of its action range every round (raw actions 0.00-0.10), i.e.
  the most back-loaded schedule the pre-registered 0.25x-4x mapping allows (sigma 3.6 -> 1.2).
  It ignores its feedback signals. In the simulator this scored 0.43 vs uniform ~0.28; on the
  real clients/test pool it is too steep and falls back to uniform's accuracy. Sim-to-real gap.
* **A4 at P and S2 did not learn** (G4b failed: P 0.099 -> 0.0005, S2 0.013 -> 0.000).
  DDPG drifted to front-loading spend, forcing sigma = 20 in late rounds -> 0.000 on all 40
  test runs. At S1 it passed G4b and deployed a smooth decay that ties uniform and A2.
* **Checkpoint caveat:** at S1 BOTH agents' deployed checkpoint is episode 25 (best
  simulator eval at the first window), so the S1 DRL numbers mostly reflect early policies.
  Checkpoint rule applied as registered; reported, not changed.
| agent / op | G4b learned | deployed ckpt | training wall-clock* |
|---|---|---|---|
| A4 P / S2 / S1 | no / no / yes | 250 / 150 / 25 | 6.8 / 5.7 / 5.8 h |
| A5 P / S2 / S1 | yes / yes / yes | 275 / 225 / 25 | 1.0 / 1.0 / 7.8 h |
*300 simulated FL episodes each; wall-clock inflated by up to 4 jobs sharing the GPU under
software throttling (no driver power cap). A1-A3 cost zero training (A2: 35 tuning runs on VAL).

## Context
* Test-pool absolute mAP is much lower than val (uniform P: 0.059 test vs ~0.29 val). Not a
  bug: the frozen session-level split put many tiny-target sessions in test (median box
  110 px^2 vs 621 val; 48% of frames < 100 px^2; 11/28 folders wg2022 IR). Paired
  differences are unaffected; relative gain of A2/A3 at P is ~ +24%.
* The robust finding across P and S2: **less noise in later rounds beats uniform** at equal
  epsilon, consistent with Phase 3's non-flat a_t. Learning the schedule adds nothing over
  a tuned one-parameter formula, at 1-8 GPU-hours per agent per operating point.
