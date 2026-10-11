# Phase 5 — Security under DP federated training: RESULTS
Pre-registration: `phase5/PREREG.md` (+ Amendment 1, written before any grid run). Grid: 960 runs,
all completed 2026-10-11 19:33, no errors. Analysis: `code/analyse_phase5.py` →
`results/phase5/analysis.json`, figure `plots/phase5_security.png`.
Every number is test mAP@0.5 over 20 paired seeds (200-219), worst-client **ε = 1.86** (δ 1e-5)
unless marked "no-DP". 2 of 8 clients malicious (c3 cloud, c6 terrain; 17% of FedAvg weight).

## In one paragraph
At this privacy level, **FLTrust does not work as a defence: it loses to plain FedAvg with no
attack and under every attack**. The reason is DP itself: each honest update is mostly noise (norm 41 vs the root update's 1.1), so its
cosine with the server's clean update is ~0.008, and FLTrust's rescaling then shrinks every
round's useful step to almost nothing (clean mAP 0.059 → 0.015, −75%). Without DP the same FLTrust
costs nothing (0.253 vs 0.250). Median and Krum also cost accuracy without attack, as predicted.
The privacy schedule does **not** change how much an attack hurts under FLTrust (all 6 tests
equivalent within ±0.02). The learned schedule A5 loses to the theory schedule A3 under every
attack, but it also loses without attack, and its schedule did not react to the attacks at all
(it is a fixed schedule in practice, as Phase 4 found). Under FedAvg, ALIE is the damaging attack
(−0.020, 17/3), and IPM actually *helps* the model under DP (+0.019).

## Check: the new code reproduces Phase 4
FedAvg / no attack for A1, A2, A3, A5: **80 of 80 runs bit-identical** to Phase 4's stored test runs.
ε: A1 1.8569, A2 1.8581, A3 1.8585, A5 1.8569.

## Mean test mAP (20 seeds)
| | none | ALIE | IPM | Fang |
|---|---|---|---|---|
| A1 FedAvg | 0.0592 | 0.0394 | 0.0783 | 0.0442 |
| A2 FedAvg | 0.0731 | 0.0510 | 0.0911 | 0.0574 |
| A3 FedAvg | 0.0736 | 0.0408 | 0.0893 | 0.0423 |
| A5 FedAvg | 0.0587 | 0.0361 | 0.0747 | 0.0447 |
| A1 FLTrust | 0.0151 | 0.0061 | 0.0197 | 0.0196 |
| A2 FLTrust | 0.0142 | 0.0057 | 0.0183 | 0.0184 |
| A3 FLTrust | 0.0124 | 0.0036 | 0.0160 | 0.0162 |
| A5 FLTrust | 0.0089 | 0.0022 | 0.0131 | 0.0122 |
| A1 median | 0.0378 | 0.0042 | 0.0269 | 0.0043 |
| A1 Krum | 0.0085 | 0.0000 | 0.0000 | 0.0133 |
| no-DP FedAvg (ε ∞) | 0.2498 | 0.2568 | 0.2499 | 0.2103 |
| no-DP FLTrust (ε ∞) | 0.2525 | 0.2444 | 0.2408 | 0.2557 |

## Pre-registered verdicts (Holm within each family, α 0.05)
**F1 — does FLTrust defend? (A1, FLTrust − FedAvg)**
| attack | diff [95% CI] | W/L | Holm p | verdict |
|---|---|---|---|---|
| ALIE | −0.0333 [−0.0440, −0.0226] | 0/20 | <0.0001 | LOSS |
| IPM | −0.0586 [−0.0759, −0.0414] | 1/19 | <0.0001 | LOSS |
| Fang | −0.0246 [−0.0371, −0.0121] | 3/17 | 0.0006 | LOSS |

**F2 — cost without attack (A1, defence − FedAvg)**
| defence | diff [95% CI] | W/L | Holm p | verdict |
|---|---|---|---|---|
| FLTrust | −0.0441 [−0.0596, −0.0286] | 2/18 | <0.0001 | LOSS |
| median | −0.0214 [−0.0322, −0.0106] | 4/16 | 0.0006 | LOSS |
| Krum | −0.0507 [−0.0644, −0.0369] | 2/18 | <0.0001 | LOSS |

Secondary: FLTrust − median −0.0227 [−0.0347, −0.0107] 3/17; FLTrust − Krum +0.0066 [−0.0015, +0.0146] 14/6.

* **Headline: "FLTrust is an effective defence at P" — NO** (0 of 3 wins, and a clean LOSS).
* **"FLTrust keeps more of the averaging than median and Krum" — NO.** It is worse than median
  and only level with Krum. The todo's claim fails at ε 1.86.

**F3 — does the schedule change attack damage under FLTrust? (damage difference vs A1)**
All 6 tests TIE (EQUIVALENT): A3 −0.0001 / +0.0010 / +0.0008 and A5 −0.0023 / +0.0004 / +0.0012
(ALIE / IPM / Fang), every CI inside ±0.008, Holm p 1.00. **The schedule does not change robustness.**

**F4 — learned A5 vs theory A3 under FLTrust**
| attack | diff [95% CI] | W/L | Holm p | verdict |
|---|---|---|---|---|
| ALIE | −0.0014 [−0.0025, −0.0003] | 1/18 | 0.025 | LOSS |
| IPM | −0.0029 [−0.0054, −0.0005] | 2/18 | 0.025 | LOSS |
| Fang | −0.0040 [−0.0067, −0.0014] | 2/18 | 0.015 | LOSS |

Read with two caveats: (1) A5 also loses to A3 **without** attack under FLTrust (−0.0036, 0/20), so
this is not an attack effect; (2) **A5's schedule was identical under every attack and aggregator**
(max σ difference 2e-6, also across seeds): the policy is pinned at its minimum action
(Phase 4: σ 3.64 → 1.23) and ignores what it observes. So Q4's motivating question ("does the
learned policy react to the attacker?") has a simple answer: it cannot, because it does not react
to anything. All differences are tiny in absolute terms (≤0.004 mAP).

## Why FLTrust fails under DP (mechanism, measured)
* Honest client update norm under DP: **41.4**; server root update: **1.06**; mean honest cosine
  with the root update: **0.0075**. Without DP: norms 0.74 vs 0.75, cosine 0.086.
* FLTrust rescales each update to the root norm and averages with trust = cosine. The noise
  directions cancel, and what remains along the useful direction is roughly ‖g0‖ × 0.008 per
  round, so the global model barely moves. FedAvg keeps the full (noisy) step and lets the √N
  averaging reduce the noise.
* So the clean cost is DP-specific: no-DP FLTrust − FedAvg = +0.0026 [−0.0170, +0.0222], 12/8.
* A related hint (secondary, unadjusted): under FLTrust the decaying schedules are slightly
  **worse** than uniform without attack (A2 −0.0009, A3 −0.0027 3/17, A5 −0.0062 1/19), the
  reverse of FedAvg (A2 +0.0139, A3 +0.0144). Plausible reason, not tested: with rescaling the
  useful step scales like 1/σ per round rather than the noise like σ², which favours an even σ.

## Predictions (PREREG §8)
1. **Held.** Honest trust under DP 0.0076 (< 0.2); without DP 0.10.
2. **Held.** ALIE's trust is higher than honest clients' (0.022 vs 0.009), most clearly in rounds 1-3 (~0.05 vs 0.01) before it falls below them in later rounds. A DP-specific weakness of FLTrust,
   as predicted; it matters little here because FLTrust's clean cost dominates.
3. **Held.** IPM and Fang attackers get trust 0.000 under DP; neither hurts FLTrust (both slightly
   help: −0.0046 and −0.0045 damage).
4. **Held.** Median (−0.021) and Krum (−0.051) LOSE to FedAvg without attack.
5. **Held as worded (absolute damage), but misleading.** Mean FedAvg damage over the three attacks is
   0.005 at P vs 0.011 without DP. Relative to the clean score, ALIE costs 33% at P
   (0.059 → 0.039) but nothing without DP (−0.007, n.s.). DP noise does not mask ALIE; it makes
   the model more fragile to it. Fang is the attack that hurts without DP (−0.040, 19/1).

## Other secondary results (unadjusted, no verdicts)
* **Attacks on FedAvg at P:** ALIE damage +0.0197 [+0.0080, +0.0315] 17/3; Fang +0.0150
  [−0.0001, +0.0301] 14/6; **IPM −0.0191 [−0.0311, −0.0072] 4/16 (helps).** IPM's attackers send
  −0.5 × the mean of the six honest updates, which carries ~√6 less noise than an honest update;
  replacing two noisy updates with low-noise, shrunk ones reduces the aggregate noise more than
  the reversed direction hurts. Without DP IPM does nothing (−0.0000).
* **The decay advantage survives attacks under FedAvg:** A2 − A1 is +0.012 / +0.013 / +0.013
  under ALIE / IPM / Fang (14/6, 18/2, 15/5). A3's does not under ALIE (+0.001) or Fang (−0.002):
  A3 takes more damage (+0.013 more under ALIE, p 0.054; +0.016 under Fang, p 0.080; not significant).
* **Median** loses to FedAvg under every attack (−0.035 / −0.051 / −0.040, 0/20 each); ALIE and Fang
  are the attacks designed against it and drive it to ~0.004.
* **Krum** picks an attacker's update in 100% of rounds under ALIE and IPM (→ mAP 0.000): the two
  colluders send the same low-noise vector, which looks like the most "central" update when every
  honest update is mostly noise. Under Fang (randomised values) it never picks one (0%); without
  attack it picks c3/c6 in 22% of rounds.
* **Root-set coverage caveat (§3), measured:** honest trust by environment under DP is sky 0.010,
  building 0.007, cloud 0.006, terrain 0.005; without DP sky 0.156, building 0.113, cloud 0.035,
  terrain 0.033. Cloud and terrain clients are trusted ~4× less than sky without DP (terrain
  despite 650 terrain root frames). Under DP all trust scores are too small for this to matter.
* No-DP FLTrust vs FedAvg: tie without attack, ALIE and IPM; **WIN under Fang +0.045
  [+0.024, +0.067] 16/4**. So FLTrust does what it is designed to do, but only without DP.

## What this means for the thesis
* "Privacy budget allocation is a security parameter" is **not supported** at P: under FLTrust the
  schedule leaves attack damage unchanged (equivalent within ±0.008).
* The more useful finding is a **conflict between DP and FLTrust**: at ε 1.86 the server's trust
  signal (cosine) is drowned by DP noise, so FLTrust costs 75% of accuracy and protects nothing.
  The classical robust aggregators lose too. Under DP, plain FedAvg was the best of the four
  aggregators under every attack tested.
* Scope: one operating point (ε 1.86), 2/8 attackers, non-adaptive attacks (not the
  FLTrust-adaptive Fang variant), root set covering 2 of 4 environments.
