# Phase 5b — FLTrust robustness checks: RESULTS
Pre-registration: `phase5b/PREREG.md` (approved 2026-10-11, no amendments). Grid: 1,120 runs, all
completed 2026-10-11 23:39, no errors. Analysis: `code/analyse_phase5b.py` →
`results/phase5b/analysis.json`, figure `plots/phase5b_fltrust.png`. Phase 5's σ 2 and no-DP runs
are reused read-only. 20 paired seeds (200-219), uniform schedule, R 12, 2 of 8 clients malicious.

**Reproduction checks passed** before the grid: FedAvg / no attack at σ 4 is bit-identical to Phase 4's
stored S2 uniform runs (20/20 seeds); this code's FLTrust reproduces Phase 5's (seeds 200-202).
(Analysis-script fix after the first run: a Spearman loop read its own output key; bookkeeping only.)

## In one paragraph
**Part 1:** FLTrust loses to FedAvg at every DP level tested, from ε 30 down to ε 0.79, with no
attack and under every attack (the one exception, Fang at ε 0.79, is a tie at the floor where
every method scores ~0.005). The gap is smallest at the loosest privacy (FLTrust reaches 54% of
FedAvg at ε 30 vs 26% at ε 1.86) and only closes without DP (101%). Both the honest cosine and the
shrink factor track this perfectly (Spearman +1 over five levels). **Part 2:** the DP-aware
**Variant N**, which rescales updates to the norm an honest DP update is known to have instead of
the clean server update's norm, **recovers FLTrust's no-DP behaviour at ε 1.86**: it ties FedAvg
with no attack, under ALIE and under IPM, and **beats FedAvg under Fang** (+0.026, 15/5), and beats
the original FLTrust in all four conditions (17/3 to 20/0). The literal noise-corrected cosine
(**Variant L**) does not help: it still loses to FedAvg in all four conditions. So the failure was the
rescaling step, not the trust scores.

## Part 1 — privacy-level sweep
Mean test mAP@0.5 (20 seeds):
| σ (worst-client ε) | aggregator | none | ALIE | IPM | Fang |
|---|---|---|---|---|---|
| 0 (∞), Phase 5 | FedAvg | 0.250 | 0.257 | 0.250 | 0.210 |
| | FLTrust | 0.253 | 0.244 | 0.241 | 0.256 |
| **0.5 (30.16)** | FedAvg | 0.179 | 0.189 | 0.196 | 0.154 |
| | FLTrust | 0.097 | 0.092 | 0.118 | 0.118 |
| | median | 0.150 | 0.069 | 0.141 | 0.050 |
| | Krum | 0.093 | 0.010 | 0.000 | 0.154 |
| **1 (5.73)** | FedAvg | 0.132 | 0.143 | 0.167 | 0.118 |
| | FLTrust | 0.040 | 0.025 | 0.052 | 0.052 |
| | median | 0.098 | 0.016 | 0.103 | 0.020 |
| | Krum | 0.042 | 0.005 | 0.000 | 0.054 |
| 2 (1.86), Phase 5 | FedAvg | 0.059 | 0.039 | 0.078 | 0.044 |
| | FLTrust | 0.015 | 0.006 | 0.020 | 0.020 |
| | median | 0.038 | 0.004 | 0.027 | 0.004 |
| | Krum | 0.009 | 0.000 | 0.000 | 0.013 |
| **4 (0.79)** | FedAvg | 0.014 | 0.014 | 0.019 | 0.005 |
| | FLTrust | 0.003 | 0.000 | 0.004 | 0.004 |
| | median | 0.008 | 0.000 | 0.003 | 0.000 |
| | Krum | 0.000 | 0.000 | 0.000 | 0.001 |

**FLTrust vs FedAvg at each level (F1 / F2, Holm within each family):**
| σ (ε) | no attack | ALIE | IPM | Fang | effective defence? |
|---|---|---|---|---|---|
| 0.5 (30.16) | −0.082 [−0.100, −0.065] 0/20 LOSS | −0.098, 0/20 LOSS | −0.078, 0/20 LOSS | −0.036 [−0.058, −0.014] 5/15 LOSS | No |
| 1 (5.73) | −0.093 [−0.112, −0.073] 0/20 LOSS | −0.118, 0/20 LOSS | −0.114, 0/20 LOSS | −0.066, 0/20 LOSS | No |
| 2 (1.86), Phase 5 | −0.044, 2/18 LOSS | −0.033, 0/20 LOSS | −0.059, 1/19 LOSS | −0.025, 3/17 LOSS | No |
| 4 (0.79) | −0.011 [−0.017, −0.006] 3/17 LOSS | −0.014, 1/19 LOSS | −0.015, 0/20 LOSS | −0.001 [−0.004, +0.002] 8/12 TIE (equiv.) | No |

* **Direction:** by the pre-registered rule it "changes" only at σ 4, and only under Fang (a tie).
  That tie is at the floor: FedAvg itself scores 0.005 there, so there is nothing to lose. In
  substance FLTrust never stops losing to FedAvg under DP.
* **Median and Krum lose to FedAvg without attack at every DP level** (F2: median −0.029 / −0.034 /
  −0.006, Krum −0.086 / −0.090 / −0.014 at σ 0.5 / 1 / 4, all LOSS) and, secondary, under every attack
  except Krum under Fang at σ 0.5 (tie, 10/10).

**Size of FLTrust's gap (F5: ratio of means FLTrust / FedAvg vs the same ratio at σ 2; Holm over 12):**
| | no attack | ALIE | IPM | Fang |
|---|---|---|---|---|
| σ 2 (reference) | 0.255 | 0.155 | 0.252 | 0.443 |
| σ 0.5 | **0.542** (+0.29, CHANGES) | **0.485** (+0.33, CHANGES) | **0.602** (+0.35, CHANGES) | 0.766 (+0.32, Holm p 0.058) |
| σ 1 | 0.301 (+0.05) | 0.175 (+0.02) | 0.314 (+0.06) | 0.443 (0.00) |
| σ 4 | 0.204 (−0.05) | **0.026** (−0.13, CHANGES) | 0.191 (−0.06) | 0.818 (+0.38, noisy: floor) |
**The size changes** (4 of 12 tests reject): FLTrust's relative deficit is about halved at ε 30, is
unchanged at ε 5.73, and under ALIE grows at ε 0.79.

**Mechanism (no attack):**
| σ | 0 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|
| honest cosine with root update | 0.086 | 0.027 | 0.015 | 0.0075 | 0.0033 |
| shrink factor (root norm / honest norm) | 1.02 | 1/10.9 | 1/20.4 | 1/38.6 | 1/73.7 |
| FLTrust / FedAvg (clean) | 1.01 | 0.54 | 0.30 | 0.26 | 0.20 |
Both measures track performance perfectly (Spearman +1.00 each, one-sided p 0.008). As the
pre-registration noted, Part 1 cannot tell them apart; Part 2 does.

## Part 2 — DP-aware FLTrust at ε 1.86 (σ 2)
| | no attack | ALIE | IPM | Fang |
|---|---|---|---|---|
| FedAvg (Phase 5) | 0.0592 | 0.0394 | 0.0783 | 0.0442 |
| FLTrust (Phase 5) | 0.0151 | 0.0061 | 0.0197 | 0.0196 |
| Variant L (noise-corrected cosine) | 0.0170 | 0.0034 | 0.0201 | 0.0208 |
| **Variant N (noise-aware norm target)** | **0.0711** | **0.0436** | **0.0695** | **0.0704** |

**Variant N — verdict: RECOVERS** (no LOSS vs FedAvg in any of the 4 conditions).
| vs FedAvg (F6-N) | diff [95% CI] | W/L | Holm p | verdict |
|---|---|---|---|---|
| no attack | +0.012 [−0.008, +0.032] | 9/11 | 0.70 | TIE |
| ALIE | +0.004 [−0.016, +0.025] | 10/10 | 0.70 | TIE |
| IPM | −0.009 [−0.025, +0.008] | 6/14 | 0.70 | TIE |
| Fang | **+0.026 [+0.008, +0.045]** | 15/5 | 0.034 | **WIN** |
vs original FLTrust (F7-N): +0.056 / +0.038 / +0.050 / +0.051, 19/1, 17/3, 20/0, 20/0, all WIN.
Caveat: the ties are not "equivalent": the no-attack and ALIE CIs reach +0.032 and +0.025 and the IPM
CI −0.025, beyond ±0.02. "Recovers" means "not shown worse than FedAvg, and better under Fang", the
same pattern FLTrust showed without DP (3 ties, 1 win under Fang).

**Variant L — verdict: PARTIAL by the pre-registered rule, negligible in practice.** It still LOSES
to FedAvg in all 4 conditions (−0.042 / −0.036 / −0.058 / −0.023, 1/19 to 4/16). Against FLTrust it wins
only without attack, by +0.0019 (16/4), and ties under the three attacks. Under ALIE the attackers'
share of trust weight rose to 52% (FLTrust: 34%, recomputed from Phase 5 logs), and its ALIE score is
the lowest of the four (0.003; vs FLTrust −0.0027, n.s.).

Secondary (mean over rounds and seeds): attackers' share of total trust weight under Variant N is 9%
(ALIE), 0.0% (IPM), 0.3% (Fang), vs 17% without attack (their honest share); mean aggregate update
norm 17.8 (no attack) vs 0.45 for Variant L, which keeps FLTrust's rescaling (not logged for FLTrust in Phase 5).

## Predictions (PREREG §5)
1. **Held.** Ratio rises monotonically as σ falls (0.20 < 0.26 < 0.30 < 0.54 < 1.01); both measures track (+1).
2. **Held.** FLTrust LOSES without attack at σ 1 and σ 0.5.
3. **Held.** Median and Krum LOSE without attack at every DP level.
4. **Held** (weakly on the second half). Variant L LOSES without attack; under ALIE it is below
   FLTrust (−0.0027) but not significantly.
5. **Failed, in the good direction.** Predicted "partial" (losses under IPM and ALIE); observed full
   recovery: IPM and ALIE are ties.

## What this means for the thesis
* The Phase 5 conclusion holds across privacy levels: **plain FLTrust is not usable under DP** at
  any ε from 30 to 0.79 in this setting; the damage shrinks as noise falls, and only vanishes without DP.
* The cause is now pinned down. Part 1 shows cosine and shrinkage both track the loss; Part 2
  separates them: correcting the cosine (L) does nothing, correcting the norm target (N) restores
  FLTrust. **FLTrust's norm-equalisation step assumes clean updates; under DP it throws away ~97% of
  every update.** Using the known noise level in that one step fixes it.
* Variant N is the only aggregator tested in Phases 5-5b that is never worse than FedAvg under DP and
  better under one attack. Scope: one operating point (ε 1.86), 2/8 attackers, non-adaptive attacks;
  an attacker who knows the rule could adapt (e.g. align with the root update), untested.
