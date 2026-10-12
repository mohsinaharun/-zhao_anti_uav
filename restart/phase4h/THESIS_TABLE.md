## Constraining the DRL agent around the theory schedule

A5-relative is the same learner as A5 (TD3 + GRU, same state, 300 simulator episodes). The difference
is that it may only move the theory schedule A3 by ±20% per round. Later rounds are re-planned so that
the privacy budget is met exactly. This removes A5's losses but adds no gain: A5-relative ties the
fixed schedules at P and S2 and loses to the tuned exponential at S1. Its adjustments are consistent
and react to feedback, but they do not improve accuracy, and training reward never rose.

**Table Y.** Relative-action agent vs fixed and learned schedules. Anti-UAV drone detector, test
mAP@0.5, 20 paired seeds with common random numbers, worst-client ε at δ = 10⁻⁵. Differences are
A5-relative minus the comparator, wins/losses over 20 seeds, Holm-adjusted over the four comparisons
at each operating point (pre-registered family F10). "Genuine" = beats uniform AND the better of A2/A3.

| | P: R 12, ε 1.86 | S1: R 25, ε 2.72 | S2: R 12, ε 0.79 |
|---|---|---|---|
| A1 uniform | 0.059 | 0.105 | 0.014 |
| A2 tuned exponential | 0.073 | 0.111 | 0.025 |
| A3 theory (anchor) | 0.074 | 0.080 | 0.021 |
| A5 DRL (Phase 4) | 0.059 | 0.098 | 0.011 |
| **A5-relative** | **0.071** | **0.092** | **0.021** |
| − uniform | +0.012, 14/6, tie | −0.013, 4/16, tie | **+0.007, 16/4, WIN** |
| − A2 | −0.002, 9/11, tie ᵉ | **−0.018, 3/17, LOSS** | −0.005, 10/10, tie ᵉ |
| − A3 | −0.003, 11/9, tie ᵉ | **+0.012, 17/3, WIN** | −0.001, 9/11, tie ᵉ |
| − A5 | +0.012, 13/7, tie ᵃ | −0.006, 7/13, tie ᵉ | **+0.009, 14/6, WIN** |
| **Genuine improvement?** | **No** | **No** | **No** |
| Adjustment pattern | structured; −20% in rounds 2-11, final round 1.42× A3 | structured; trims A3's peaks | structured; trims A3's spikes |
| Training reward rose (G4b)? | no | no | no |

ᵉ Equivalent (CI within ±0.02). ᵃ Unadjusted p 0.036, Holm p 0.14. A5-relative's ε equals A3's at
every point (1.8585 / 2.7192 / 0.7932).

Figure: `plots/phase4h_relative_agent.png`.
