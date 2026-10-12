# Phase 4g — Round-count convergence check: RESULTS
Pre-registration: `phase4g/PREREG.md` (approved 2026-10-12 with +A no-DP ceiling and +B uniform; no
amendments). Grid: 180 new runs (A2 / A1 / no-DP × R 25 / 50 / 100 × 20 seeds), completed
2026-10-12 13:05, no errors. R 12 is reused read-only from Phase 4 (`eval_P_A1`, `eval_P_A2`) and
Phase 5 (`nodp_noDP_fedavg_none`). Analysis: `code/analyse_phase4g.py` → `results/phase4g/analysis.json`,
figure `plots/phase4g_rounds.png`. TEST pool, 20 paired seeds (200-219), common random numbers, no
attackers, worst-client ε at δ = 10⁻⁵.

**Reproduction check passed** before the grid: this code's A2 at R 12 reproduces Phase 4's `eval_P_A2`,
and its σ 0 run reproduces Phase 5's no-DP FedAvg, bit-identically on seeds 200-202.

Run-time note: during roughly the first half of the grid the Phase 4h agents ran alongside it and the
machine ran out of RAM and GPU memory. The 4h agents were stopped (see `phase4h/PREREG.md`
implementation note). Every run is seeded and deterministic, so the slowdown did not affect any number.

## In one paragraph
**More rounds do not close the gap. Past 25 rounds they make it worse.** At a fixed budget of ε 1.86,
A2 scores 0.073 at 12 rounds, 0.086 at 25 (not a significant rise), 0.057 at 50 (a significant fall)
and 0.040 at 100. Over the same range the no-DP ceiling rises from 0.250 to 0.330-0.340. The share
of the ceiling that A2 reaches therefore falls from 29% to 12%. The pre-registered verdict is
**NOISE CEILING / PLATEAU**: the last step (100 − 50) is not a rise, and A2 never reaches half of
the ceiling. It just misses the DECLINING sub-case, because the 100 − 50 fall (−0.017, 6/14)
loses significance after Holm (p_holm 0.056). The overall shape is clearly a rise and then a fall.
A2 at 100 rounds is below A2 at its best R (25) on 18/20 seeds (−0.046). Uniform (A1) follows the same
curve, and A2's advantage over uniform is constant at +0.012 to +0.014 at every R. So the schedule
result from Phase 4 does not depend on the round count, but the answer to "would more rounds help?"
is no.

## Main table
Test mAP@0.5, 20 seeds: mean, sd, 95% CI; worst-client ε; σ first → last round.

| R | A2 (k 0.5) | A1 uniform | No DP (ceiling) | G(R) = A2 / ceiling |
|---|---|---|---|---|
| 12 | 0.0731 ± 0.0250 [0.061, 0.085], ε 1.8581, σ 2.67 → 1.62 | 0.0592 ± 0.0271 [0.047, 0.072], ε 1.8569, σ 2.00 | 0.2498 ± 0.0310 [0.235, 0.264] | 0.292 |
| 25 | **0.0861** ± 0.0404 [0.067, 0.105], ε 1.8566, σ 3.58 → 2.17 | 0.0741 ± 0.0431 [0.054, 0.094], ε 1.8569, σ 2.71 | 0.3212 ± 0.0248 [0.310, 0.333] | 0.268 |
| 50 | 0.0573 ± 0.0317 [0.043, 0.072], ε 1.8565, σ 4.88 → 2.96 | 0.0449 ± 0.0283 [0.032, 0.058], ε 1.8554, σ 3.70 | **0.3399** ± 0.0193 [0.331, 0.349] | 0.169 |
| 100 | 0.0401 ± 0.0206 [0.031, 0.050], ε 1.8553, σ 6.77 → 4.11 | 0.0268 ± 0.0196 [0.018, 0.036], ε 1.8586, σ 5.15 | 0.3297 ± 0.0170 [0.322, 0.338] | 0.122 |

Per-seed A2 / no-DP (gives a CI on G): 0.303 [0.246, 0.359], 0.274 [0.209, 0.339], 0.171 [0.125, 0.218],
0.122 [0.092, 0.153]. Measured against the fixed 12-round ceiling of 0.250, G is 0.29 / 0.34 / 0.23 / 0.16.
That is also below 0.5 everywhere.

## Decision family F11 (Holm over 3): A2 between consecutive round counts
| Step | Diff [95% CI] | W/L | p | p_holm | Verdict |
|---|---|---|---|---|---|
| 25 − 12 | +0.0131 [−0.0028, +0.0289] | 13/7 | 0.101 | 0.101 | TIE |
| 50 − 25 | −0.0288 [−0.0457, −0.0119] | 7/13 | 0.0021 | 0.0062 | **LOSS** |
| 100 − 50 | −0.0172 [−0.0324, −0.0021] | 6/14 | 0.028 | 0.056 | TIE |

**Verdict: NOISE CEILING / PLATEAU** (last step not a WIN; best G 0.292 < 0.5; G(100) 0.122). It is not
labelled DECLINING because the last step is a TIE after Holm. The fall happens between 25 and 50 rounds.

## Secondary (unadjusted, not decision rules)
* **A2 − A1 at each R:** +0.0139 (16/4, p 0.0002), +0.0121 (16/4, p 0.008), +0.0124 (17/3, p 0.0008),
  +0.0133 (15/5, p 0.025). The tuned exponential's edge over uniform is the same size at every round
  count.
* **A1 across R:** 25 − 12 +0.015 (12/8, p 0.13); 50 − 25 −0.029 (4/16, p 0.003); 100 − 50 −0.018 (7/13,
  p 0.016). This is the same rise-then-fall shape as A2.
* **No-DP across R:** 25 − 12 +0.071 (20/0); 50 − 25 +0.019 (17/3); 100 − 50 −0.010 (5/15, p 0.008).
  Without DP, training saturates around 50 rounds, with a small dip at 100.
* **A2 at 100 rounds vs its best R (25):** −0.0460 [−0.0656, −0.0265], 2/18, p 0.0001.
* **Server-side proxy loss** (public session, logged during training): at the final round no-DP reaches
  0.43-0.45 at every R. A2 ends at 0.89 (R 25), 1.11 (R 50) and 1.43 (R 100), and A1 at 0.97, 1.23
  and 1.45. The longer the run at fixed ε, the further from converged the DP model finishes. Per-round
  σ grows faster than the extra rounds can average it away (figure, right panel).

## Why (interpretation, not tested here)
At fixed ε the accountant pays for every extra round with more noise in every round: uniform σ goes
2.00 → 2.71 → 3.70 → 5.15. The pre-registered arithmetic (signal ∝ R under clipping, noise ∝ √R·σ)
predicted a flat ratio √R/σ = 1.73 / 1.85 / 1.91 / 1.94, i.e. a plateau. The data are worse than flat.
A plausible reason is that with a fixed learning rate and momentum, a larger per-round σ leaves the
head bouncing around a noise floor that rises with σ, and the extra rounds do not anneal it. The proxy
loss is consistent with this, but no experiment here isolates it (no learning-rate decay arm). Either
way the pre-registered question gets a clear answer: the DP/no-DP gap at ε 1.86 is a **noise ceiling,
not a convergence-speed problem**.

## Predictions (pre-registered §7)
| # | Prediction | Outcome |
|---|---|---|
| 1 | Plateau: 100 − 50 not a WIN, no R reaches 0.125 | **Held** (best is 0.086 at R 25. The prediction said flat; the data decline.) |
| 2 | Rise from 12 to 25 is small at fixed ε | **Held** (+0.013, TIE) |
| 3a | No-DP at 100 rounds > 0.250 | **Held** (0.330) |
| 3b | G(R) falls with R | **Held** (0.292 → 0.268 → 0.169 → 0.122) |

All four held. The one thing not predicted is the fall after 25 rounds. The prediction was "flat",
not "worse".

## What this means for the thesis
* External feedback asked whether the DP gap is just undertraining. It is not. At ε 1.86, the best
  of 12 / 25 / 50 / 100 rounds reaches 29% of the no-DP model at the same round count (A2, R 12) and
  27% at R 25. Longer runs do worse.
* Phase 4's operating point R = 12 is not the cause of A2's low absolute numbers. R 25 is
  nominally better (+0.013) but not significantly.
* The schedule ranking from Phase 4 (A2 > uniform by about +0.013) is stable across a 8× change in R.
