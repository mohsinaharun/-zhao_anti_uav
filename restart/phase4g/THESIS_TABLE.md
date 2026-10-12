## More rounds do not close the privacy gap

At a fixed privacy budget (worst-client ε 1.86), each extra round must be paid for with more noise in
every round. Accuracy rises slightly from 12 to 25 rounds and then falls. The model without DP keeps
improving up to 50 rounds, so the fraction of it that the private model reaches falls from 29% to
12%. The gap is a noise ceiling, not a training-length problem. The tuned schedule's advantage over
uniform is the same at every round count.

**Table X.** Test mAP@0.5 vs number of communication rounds at fixed worst-client ε. Anti-UAV drone
detector, 8 clients, 20 paired seeds with common random numbers, δ = 10⁻⁵. Mean [95% CI]. G = A2 /
no-DP at the same R. Step = A2 at this R minus A2 at the previous R, wins/losses over 20 seeds,
Holm-adjusted over the three steps (pre-registered family F11).

| Rounds R | σ per round (A2, first → last) | A2 tuned exponential | Uniform | No DP | G | Step (A2) |
|---|---|---|---|---|---|---|
| 12 | 2.67 → 1.62 | 0.073 [0.061, 0.085] | 0.059 [0.047, 0.072] | 0.250 [0.235, 0.264] | 0.29 | — |
| 25 | 3.58 → 2.17 | **0.086** [0.067, 0.105] | 0.074 [0.054, 0.094] | 0.321 [0.310, 0.333] | 0.27 | +0.013, 13/7, tie |
| 50 | 4.88 → 2.96 | 0.057 [0.043, 0.072] | 0.045 [0.032, 0.058] | **0.340** [0.331, 0.349] | 0.17 | **−0.029, 7/13, LOSS** |
| 100 | 6.77 → 4.11 | 0.040 [0.031, 0.050] | 0.027 [0.018, 0.036] | 0.330 [0.322, 0.338] | 0.12 | −0.017, 6/14, tie ᵃ |
| **A2 − uniform** | | +0.014 / +0.012 / +0.012 / +0.013 ᵇ | | | | |

Worst-client ε of every DP run is 1.855-1.859. ᵃ Unadjusted p 0.028, Holm p 0.056. A2 at 100 rounds is
below A2 at 25 rounds on 18/20 seeds (−0.046 [−0.066, −0.027]). ᵇ At R 12 / 25 / 50 / 100: 16/4,
16/4, 17/3, 15/5 wins (secondary, unadjusted, all p < 0.03).

Figure: `plots/phase4g_rounds.png`.
