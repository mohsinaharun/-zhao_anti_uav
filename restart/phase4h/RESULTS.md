# Phase 4h — Relative-action DRL agent (A5-relative): RESULTS
Pre-registration: `phase4h/PREREG.md` (approved 2026-10-12; no design amendments, one implementation
note). Training: 3 agents × 300 simulator episodes, restarted from scratch 2026-10-12 13:07 after Phase 4g
finished. P and S2 took 2.2 h each and S1 took 3.4 h, the three in parallel. Evaluation: 60 TEST runs (20
paired seeds 200-219 × P / S1 / S2), all completed 16:41, no errors. Comparators A1 / A2 / A3 / A5 are Phase 4's
stored runs (read-only). Analysis: `code/analyse_phase4h.py` → `results/phase4h/analysis.json`, figure
`plots/phase4h_relative_agent.png`. Deployed actors: `results/phase4h/agent_A5r_{P,S1,S2}_best.pt`.

**Gate G4h passed** before training. With zero adjustment the agent reproduces A3 exactly at all three
points. 200 random adjustment sequences meet A3's budget to < 6e-6 relative. The zero policy reproduces
Phase 4's `eval_P_A3` test mAP bit-for-bit on seeds 200-202. **Every evaluation run meets the budget**:
mean worst-client ε is 1.8585 / 2.7192 / 0.7932, identical to A3's.

## In one paragraph
**Constraining the agent to ±20% around the theory schedule does not produce a genuine improvement
at any operating point.** At P, A5-relative scores 0.0708. That ties A3 (−0.003, EQUIVALENT) and A2
(EQUIVALENT), and it does not significantly beat uniform (+0.012, 14/6, Holm p 0.14). At S1 it LOSES to
A2 (−0.018, 3/17) and does not beat uniform (−0.013, 4/16). It does beat A3 there (+0.012, 17/3),
because A3 is weak at S1. At S2 it beats uniform (+0.007, 16/4) but only ties A2 and A3 (both
EQUIVALENT). Against the unconstrained A5 it is better at P (+0.012, 13/7, Holm p 0.14, not
significant) and at S2 (+0.009, 14/6, WIN), and equivalent at S1. The anchor mostly stops the agent
from doing harm. It does not add a gain over the theory schedule it is anchored to.

The learned adjustment is **not degenerate**: STRUCTURED at all three points, consistent across seeds,
and it varies with the feedback signals. At P its main shape is to cut noise below A3 in rounds 2-11
(mostly at the −20% bound). That spends budget early, so the final round absorbs the remainder at
**1.42× A3's σ**. The training reward never improved (G4b fails at all three points). The agent's
pattern is therefore not something it learned to be better; it is where the policy drifted, and on the
test pool it scores the same as A3.

## Main table
Test mAP@0.5, mean over 20 paired seeds; worst-client ε in brackets.

| | A1 uniform | A2 tuned exp. | A3 theory | A5 (Phase 4) | **A5-relative** |
|---|---|---|---|---|---|
| **P** (R 12, σ 2) | 0.0592 (1.8569) | 0.0731 (1.8581) | 0.0736 (1.8585) | 0.0587 (1.8569) | **0.0708** (1.8585) |
| **S1** (R 25, σ 2) | 0.1050 (2.7215) | 0.1105 (2.7222) | 0.0802 (2.7192) | 0.0981 (2.7214) | **0.0921** (2.7192) |
| **S2** (R 12, σ 4) | 0.0143 (0.7935) | 0.0254 (0.7931) | 0.0213 (0.7932) | 0.0113 (0.7935) | **0.0208** (0.7932) |

## Decision family F10 (Holm over 4 per operating point): A5-relative minus …
| | vs A1 | vs A2 | vs A3 | vs A5 | Genuine? |
|---|---|---|---|---|---|
| **P** | +0.0116 [+0.0003, +0.0229], 14/6, p_holm 0.14, TIE | −0.0023 [−0.0090, +0.0044], 9/11, EQUIV | −0.0028 [−0.0115, +0.0059], 11/9, EQUIV | +0.0121 [+0.0009, +0.0233], 13/7, p_holm 0.14, TIE | **No** |
| **S1** | −0.0129 [−0.0252, −0.0006], 4/16, p_holm 0.080, TIE | **−0.0184 [−0.0286, −0.0082], 3/17, LOSS** | **+0.0119 [+0.0064, +0.0174], 17/3, WIN** | −0.0061 [−0.0145, +0.0024], 7/13, EQUIV | **No** |
| **S2** | **+0.0065 [+0.0021, +0.0109], 16/4, WIN** | −0.0046 [−0.0106, +0.0014], 10/10, EQUIV | −0.0005 [−0.0036, +0.0026], 9/11, EQUIV | **+0.0094 [+0.0034, +0.0155], 14/6, WIN** | **No** |

"Genuine" means WIN vs A1 AND WIN vs the better of A2/A3, fixed in advance: A3 at P, A2 at S1 and S2.
**Headline (P): not a genuine improvement.**

## A5 vs A5-relative (constrained vs unconstrained)
* **P:** A5r 0.0708 vs A5 0.0587, +0.012 (13/7). Unadjusted p 0.036, Holm p 0.14, so TIE. The point
  estimate is A3 − A5 (≈ +0.015) almost exactly: the anchor recovers what A5 lost.
* **S1:** 0.0921 vs 0.0981, −0.006 (7/13), EQUIVALENT. A3 is the weakest fixed schedule at S1, and an
  agent anchored to it inherits that.
* **S2:** 0.0208 vs 0.0113, **+0.009 (14/6), WIN**.

## Is the learned adjustment a real pattern? (pre-registered §5)
| | Class | \|δ\| < 0.02 | \|δ\| at bound | Rounds with CI excl. 0 | Between-seed sd | λ range |
|---|---|---|---|---|---|---|
| P | STRUCTURED | 0% | 82% | 91% (consistent) | 0.051 (uses feedback) | 1.00-1.49 |
| S1 | STRUCTURED | 3% | 60% | 71% (consistent) | 0.071 (uses feedback) | 0.99-1.14 |
| S2 | STRUCTURED | 8% | 24% | 73% (consistent) | 0.085 (uses feedback) | 0.92-1.25 |

Shapes (mean δ per round; realised σ / A3 per round in `analysis.json`):
* **P:** δ ≈ −0.20 in rounds 2, 4 and 6-9, and smaller cuts in 3, 5, 10, 11. A few seeds jump to
  +0.19 in rounds 3 and 5, which is most of the between-seed spread. Realised σ / A3 is 0.80-0.98 in
  rounds 1-11, and **1.42 in round 12**, where the saved budget is paid back. P misses SATURATED
  (82% < 90%) only narrowly.
* **S1:** +9% and +14% more noise in rounds 2-3, then −20% in rounds 5-13 (exactly where A3 sits at its
  4.25 peaks), then a drift back to ≈ 0 with wide seed-to-seed scatter in rounds 15-24.
* **S2:** about −10% in rounds 1-4 and −18% in round 5 (A3's spike), then ≈ 0 to +5% in rounds 6-11.
  The shape smooths A3's two spikes, which moves it toward A2's shape. The score lands between A3 and A2.

**What the pattern means.** By the pre-registered tests the pattern is consistent and non-degenerate,
and it responds to feedback. At S1 and S2 it does something interpretable: it trims A3's noise spikes.
At P it shifts noise into the last round instead. None of it beats A3 on the test pool, and none of it
came with a rise in training reward (below), so this is a pattern with no measurable benefit.

## Training
| | G4b: reward first 25 → last 25 episodes | Learned? | Deployed checkpoint (episode) | Hours |
|---|---|---|---|---|
| P | 0.388 → 0.370 | No | 50 | 2.2 |
| S1 | 0.390 → 0.404 | No | 275 | 3.4 |
| S2 | 0.131 → 0.138 | No | 75 | 2.2 |

Unlike A5, A5-relative starts at a good schedule (the warm-up actions are centred on "no adjustment",
i.e. A3). Its training reward is therefore already at A5's eventual level from episode 1 (figure,
right column), and it never improves on that. At P and S2 the deployed checkpoint comes from early in
training. The proxy simulator's evaluation reward is noisy at 3 seeds, the same weakness found in
Phase 4e.

## Predictions (pre-registered §6)
| # | Prediction | Outcome |
|---|---|---|
| 1 | Not genuine at P; ties A3 | **Held** (EQUIVALENT to A3) |
| 2 | Beats A5 at P | **Failed**: +0.012, 13/7, but Holm p 0.14 (TIE) |
| 3a | Pattern not ZERO-COLLAPSE | **Held** (STRUCTURED, 82% at bound) |
| 3b | No use of feedback (between-seed sd < 0.02) | **Failed**: sd 0.051 at P (0.071 S1, 0.085 S2) |
| 3 (shape) | More noise early, less late | **Failed** at P: less noise early, +42% in the final round. At S1, partly (more in rounds 2-3, less in 5-13) |
| 4a | At S1 does not beat A1 | **Held** (−0.013, TIE) |
| 4b | At S2 ties A2 | **Held** (EQUIVALENT) |

## What this means for the thesis
* External feedback suggested that the DRL agent (A5) failed because its problem was too
  unconstrained. Constraining it to ±20% around the theory schedule, with the budget kept exact, fixes
  A5's *harm*: it is at least as good as A5 everywhere and better at S2. It does **not** produce a
  *gain*. At no operating point does it beat both uniform and the best fixed schedule. Its learned
  adjustments are real and non-degenerate but do not improve test accuracy, and its training reward
  never rose.
* This matches Phases 4, 4d and 4e. At these budgets, fixed decaying schedules (A2, A3) are as good as
  any search or learned policy tried, and the remaining differences are a few thousandths of mAP.
