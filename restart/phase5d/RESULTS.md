# Phase 5d — Per-client trust-weight cap on Variant N: RESULTS
Pre-registration: `phase5d/PREREG.md` (approved 2026-10-12 with +A cap-aware attacker and +B m = 1;
no amendments). Grid: 340 runs (m 1.5 / 2 / 1 × 5 conditions, plus the cap-aware attacker at m 1.5 / 2,
× 20 seeds), 17:05-17:58, no errors. Comparators are read-only: Phase 5b (Variant N, standard
attacks), Phase 5c (Variant N and FedAvg under trust-capture), Phase 5 (FedAvg, standard attacks).
Analysis: `code/analyse_phase5d.py` → `results/phase5d/analysis.json`, figure `plots/phase5d_trust_cap.png`.
σ 2 every round, worst-client ε 1.8569 in every run, 20 paired seeds 200-219.

**Reproduction check passed** before the grid. With the cap switched off (κ = 1) the code reproduces
Phase 5b's Variant N (no attack) and Phase 5c's Variant N under trust-capture bit-identically on
seeds 200-202.

## In one paragraph
**The weight cap is not a fix at any margin.** The cap does exactly what it was designed to do: it
pins the attackers' share of the aggregation weight at 2κ in every round, at 50% (m 2), 37.5% (m 1.5)
and 25% (m 1), down from ~73% without a cap. But the model still collapses under the trust-capture
attack:

| Cap | Mean mAP | Seeds at 0.000 |
|---|---|---|
| m 2 | 0.0004 | 19/20 |
| m 1.5 | 0.0000 | 20/20 |
| m 1 | 0.0058 | 14/20 |

Uncapped Variant N scored 0.0005 (18/20 at zero) and FedAvg 0.0137. Test (a1), "collapse prevented",
is not a WIN at any margin. At m 1.5 the capped system is formally worse than FedAvg under the same
attack (a2 LOSS, 0/19). The cap-aware attacker (+A) is no stronger, because the Phase 5c attacker
already saturates the cap. On the other side, the cap costs nothing in the standard conditions:
test (b) has no LOSS anywhere. At m 1 it even **improves** Variant N under ALIE (+0.031, 15/5, WIN).
The conclusion: limiting the attackers' *weight* does not limit their *damage* under DP. Even a
25% share of a coherent, full-norm malicious update overwhelms honest updates that are mostly noise.

## Main table
Test mAP@0.5, mean of 20 seeds, ε 1.86; [number of seeds at 0.000].

| | No attack | ALIE | IPM | Fang | Trust-capture (5c) | Trust-capture, cap-aware (+A) |
|---|---|---|---|---|---|---|
| FedAvg | 0.0592 [0] | 0.0394 [0] | 0.0783 [0] | 0.0442 [0] | 0.0137 [7] | — |
| Variant N, no cap | 0.0711 [0] | 0.0436 [0] | 0.0695 [0] | 0.0704 [0] | 0.0005 [18] | — |
| N + cap 2/K (κ 0.25) | 0.0718 [0] | 0.0514 [0] | 0.0675 [0] | 0.0669 [0] | **0.0004 [19]** | **0.0003 [18]** |
| N + cap 1.5/K (κ 0.1875) | 0.0711 [0] | 0.0657 [0] | 0.0789 [0] | 0.0731 [0] | **0.0000 [20]** | **0.0014 [12]** |
| N + cap 1/K (κ 0.125), +B | 0.0653 [0] | 0.0744 [0] | 0.0807 [0] | 0.0735 [0] | **0.0058 [14]** | — |

## Decision families F12-m (Holm over 6 per margin)
| Test | m 2 | m 1.5 | m 1 (+B) |
|---|---|---|---|
| (a1) capped − uncapped N, trust-capture (must WIN) | −0.0001 [−0.0013, +0.0011], 1/3, EQUIV | −0.0005 [−0.0014, +0.0004], 0/4, EQUIV | +0.0054 [−0.0006, +0.0113], 8/3, EQUIV (Holm p 0.37) |
| (a2) capped − FedAvg, trust-capture (must not LOSE) | −0.0133 [−0.0231, −0.0035], 1/18, TIE (Holm p 0.064) | **−0.0137 [−0.0233, −0.0040], 0/19, LOSS** | −0.0078 [−0.0179, +0.0022], 3/16, EQUIV |
| (b1) − uncapped N, no attack | +0.0007, 10/10, EQUIV | −0.0001, 10/10, EQUIV | −0.0058, 9/11, TIE |
| (b2) − uncapped N, ALIE | +0.0078, 11/9, TIE | +0.0221 [+0.0010, +0.0432], 16/4, TIE (Holm p 0.21) | **+0.0308 [+0.0121, +0.0495], 15/5, WIN** |
| (b3) − uncapped N, IPM | −0.0020, 10/10, EQUIV | +0.0094, 11/9, TIE | +0.0112, 13/7, TIE |
| (b4) − uncapped N, Fang | −0.0035, 10/10, EQUIV | +0.0027, 11/9, TIE | +0.0030, 11/9, EQUIV |
| **Genuine fix?** | **No** (a1 fails) | **No** (a1 fails, a2 LOSS) | **No** (a1 fails) |

"Equivalent" in (a1) means only that the CI is inside ±0.02. Both arms sit at the floor (~0).

**(+A) cap-aware attacker, F12A-m.** At m 1.5: a1 +0.0009 (19/1, Holm p 0.58), a2 −0.0123 (9/11,
TIE). At m 2: a1 −0.0002 (17/2), a2 −0.0134 (5/15, Holm p 0.057, TIE). **Not a fix at either margin.**
The cap-aware attacker chooses c ≈ 0.02-0.06 instead of 0.44-0.47. It is no stronger than the Phase 5c
attacker: +0.0014 at m 1.5 (20/0, p 0.022; marginally *weaker*, since 12 rather than 20 seeds end at
zero) and −0.0001 at m 2. The Phase 5c attacker already takes the full 2κ.

## What the cap did, per round (mean over 20 seeds × 12 rounds)
| | Cap binds | Σw (min) | Positive-trust clients | Attackers' share (uncapped → capped) |
|---|---|---|---|---|
| m 1.5, no attack | 96% | 0.99 (0.75) | 6.7 | — |
| m 1.5, ALIE | 100% | 0.84 (0.38) | 5.1 | 0.075 → 0.060 |
| m 1.5, trust-capture | 100% | 1.00 | 8.0 | **0.733 → 0.375** |
| m 2, no attack | 60% | 1.00 | 6.8 | — |
| m 2, trust-capture | 96% | 1.00 | 8.0 | **0.694 → 0.499** |
| m 1, no attack | 100% | 0.82 (0.50) | 6.6 | — |
| m 1, ALIE | 100% | 0.60 (0.12) | 4.8 | 0.071 → 0.042 |
| m 1, trust-capture | 100% | 1.00 (0.88) | 8.0 | **0.796 → 0.250** |

The pre-run replay of logged trust scores predicted these rates closely (96% / 60% binding without
attack).

## Why the cap fails (interpretation)
Under DP, an honest update of norm N_t is almost all noise: its cosine with the root update averages
0.0065. The trust-capture update has the same norm N_t but is pure, coherent signal pointing against
the honest consensus, in the same direction every round. Averaging cancels the honest noise but adds up
the attacker's push. So a 25% weight on that update moves the model far more than the 75% honest weight
can move it back.

FedAvg itself shows this without any trust weighting. The same crafted update at its natural 17.1% data
share drops FedAvg from 0.059 to 0.014, with 7/20 seeds at zero. A weight cap can at best bring the
attackers down to their headcount share (2/8 = 25%), which is still more than FedAvg gives them. The
cap therefore cannot do better than FedAvg against this attacker, and it does not.

A defence would need to bound what an update can contribute *in direction*, not in weight: for
example, by noticing that an update is far more coherent with the previous rounds' malicious direction
than DP noise allows, or by bounding the aggregate's cosine with any single update. That is not tested
here.

Side effect worth reporting: the cap helps under ALIE. ALIE crafts updates inside the honest spread,
and they occasionally get the largest trust. Capping removes that, and at m 1 Variant N gains +0.031
(15/5, WIN), reaching 0.074. That is higher than FedAvg, FLTrust or either earlier variant scores under ALIE at ε 1.86 in Phases 5-5b, with any schedule.

## Predictions (pre-registered §6)
| # | Prediction | Outcome |
|---|---|---|
| 1 | Cap binds ≥ 90% without attack at m 1.5; ~50-70% at m 2 | **Held** (96%, 60%) |
| 2 | (a1) WIN at both margins; mean > 0.01, < 5/20 seeds at 0 | **Failed**: 0.0000 / 0.0004, 20/20 and 19/20 at 0 |
| 3 | Attackers' share ≈ 2κ, always above 0.171 | **Held** (0.375 / 0.499 / 0.250) |
| 4 | m 2 not a fix, because (a2) is a LOSS | **Failed in the reason**: not a fix, but (a2) is a TIE (Holm p 0.064); a1 is what fails |
| 5 | m 1.5 a narrow genuine fix | **Failed**: a1 fails and a2 is a LOSS |
| 6 | Even where collapse is prevented, mAP < half of N's no-attack | **Held** (trivially: collapse was never prevented) |
| 7 | Cap-aware attacker at most slightly stronger (±0.01) | **Held** (+0.0014 / −0.0001) |

The central prediction, that the cap prevents the collapse, failed. I expected cutting the attackers'
share from ~70% to 25-50% to be enough. The data show that share is not the binding quantity.

## What this means for the thesis
* Phase 5c showed that Variant N fixes FLTrust's DP utility problem but is broken by a rule-aware
  attacker. Phase 5d shows that the obvious repair, a per-client trust-weight cap, does not change that.
  The attackers' weight falls as designed (73% → 25-50%), but the model still collapses (0.000-0.006 mAP).
* The deeper point: **under DP, the size of an attacker's weight is not what limits its damage.**
  Honest updates are mostly noise, so any non-trivial weight on a coherent malicious update dominates.
  FedAvg is badly hurt by the same attack at 17% weight. Weight-based defences (trust caps, and by
  extension weight-limiting robust aggregators) cannot close this gap.
* The cap is free in the standard conditions, and at m 1 it improves Variant N under ALIE. So it is
  a reasonable add-on, but not a defence against an adaptive attacker.
