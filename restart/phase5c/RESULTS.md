# Phase 5c — Variant N follow-up checks: RESULTS
Pre-registration: `phase5c/PREREG.md` (approved 2026-10-12, no amendments). Grid: 380 runs, all
completed 2026-10-12 01:34, no errors. Analysis: `code/analyse_phase5c.py` →
`results/phase5c/analysis.json`, figure `plots/phase5c_variantN.png`. Phase 5 / 5b runs reused
read-only. 20 paired seeds (200-219), uniform schedule, R 12, 2 of 8 clients malicious (c3, c6).

**Reproduction check passed** before the grid: this code's Variant N reproduces Phase 5b's
`variant_s2_fltrust_N_none` bit-identically on seeds 200-202.

## In one paragraph
**Part 1:** Variant N does **not** recover at every privacy level by the pre-registered rule. It
recovers at σ 0 (ε ∞), σ 0.5 (ε 30.16) and σ 2 (ε 1.86): never worse than FedAvg, and better under
Fang. It fails at **σ 1 (ε 5.73)**, losing to FedAvg under IPM (−0.030, 6/14), and at **σ 4 (ε 0.79)**,
losing under ALIE (−0.011, 1/19, at the floor). Everywhere under DP it beats plain FLTrust by a wide
margin. **Part 2:** an attacker that knows Variant N's rule and the server's root update **breaks
it**. With this attack Variant N falls from 0.071 to 0.0005 mAP (18/20 seeds score exactly 0), which
is worse than FedAvg under the same attack (−0.013, 1/18, LOSS) and worse than plain FLTrust (0.0055).
The attack also badly hurts FedAvg (0.059 → 0.014), so it is a strong attack in general. Against
Variant N it is the strongest attack tested.

## Part 1 — Variant N at five privacy levels
Mean test mAP@0.5 (20 seeds); columns none / ALIE / IPM / Fang:
| σ (worst-client ε) | FedAvg | plain FLTrust | **Variant N** |
|---|---|---|---|
| 0 (∞) | 0.250 / 0.257 / 0.250 / 0.210 | 0.253 / 0.244 / 0.241 / 0.256 | 0.253 / 0.244 / 0.241 / 0.256 |
| 0.5 (30.16) | 0.179 / 0.189 / 0.196 / 0.154 | 0.097 / 0.092 / 0.118 / 0.118 | **0.187 / 0.193 / 0.203 / 0.185** |
| 1 (5.73) | 0.132 / 0.143 / 0.167 / 0.118 | 0.040 / 0.025 / 0.052 / 0.052 | **0.147 / 0.143 / 0.137 / 0.137** |
| 2 (1.86), Phase 5b | 0.059 / 0.039 / 0.078 / 0.044 | 0.015 / 0.006 / 0.020 / 0.020 | **0.071 / 0.044 / 0.070 / 0.070** |
| 4 (0.79) | 0.014 / 0.014 / 0.019 / 0.005 | 0.003 / 0.000 / 0.004 / 0.004 | **0.018 / 0.003 / 0.016 / 0.016** |

**Variant N vs FedAvg (F8, Holm within each level, 4 tests):** diff [95% CI], wins/losses
| σ (ε) | none | ALIE | IPM | Fang | recovers? |
|---|---|---|---|---|---|
| 0 (∞) | +0.003 [−0.017, +0.022] 12/8 TIE | −0.012 [−0.043, +0.018] 8/12 TIE | −0.009 [−0.027, +0.009] 9/11 TIE | **+0.045 [+0.024, +0.067] 16/4 WIN** | **yes** |
| 0.5 (30.16) | +0.008 [−0.021, +0.037] 12/8 TIE | +0.004 [−0.017, +0.024] 13/7 TIE | +0.007 [−0.014, +0.028] 11/9 TIE | **+0.031 [+0.009, +0.053] 16/4 WIN** | **yes** |
| 1 (5.73) | +0.014 [−0.016, +0.045] 13/7 TIE | +0.000 [−0.019, +0.019] 9/11 TIE (equiv.) | **−0.030 [−0.052, −0.009] 6/14 LOSS** (Holm p 0.035) | +0.019 [−0.006, +0.044] 17/3 TIE | **no** |
| 2 (1.86), Phase 5b | +0.012 [−0.008, +0.032] 9/11 TIE | +0.004 [−0.016, +0.025] 10/10 TIE | −0.009 [−0.025, +0.008] 6/14 TIE | **+0.026 [+0.008, +0.045] 15/5 WIN** | **yes** |
| 4 (0.79) | +0.004 [−0.003, +0.010] 11/9 TIE (equiv.) | **−0.011 [−0.017, −0.005] 1/19 LOSS** (Holm p 0.007) | −0.003 [−0.010, +0.005] 7/13 TIE (equiv.) | **+0.011 [+0.003, +0.019] 13/7 WIN** | **no** |

**Headline: "Variant N recovers at every privacy level" does NOT hold; it fails at σ 1 and σ 4.**
How to read the two failures:
* **σ 1, IPM.** A real but moderate loss: Variant N reaches 82% of FedAvg under IPM (0.137 vs 0.167).
  IPM is the attack under which FedAvg itself scores highest at every DP level (its small reversed step
  helps FedAvg slightly), so the comparison point is high. At σ 2 the same comparison was −0.009,
  6/14, a tie with the same sign. So IPM is Variant N's weak spot, and at σ 1 the gap is large
  enough to show.
* **σ 4, ALIE.** A loss at the floor (0.003 vs 0.014). It is consistent (1/19) and has a visible
  cause: at ε 0.79 honest updates are almost uncorrelated with the root update, so an ALIE update,
  which sits close to the honest mean, gets more than its share of trust (attackers' weight 21% at
  σ 4 vs 9% at σ 2 and 7% at σ 0.5 and 1; by headcount they are 25%, 2 of 8).
* **Against plain FLTrust**, Variant N wins at every DP level in every condition (secondary,
  unadjusted; 17/3 to 20/0 at σ 0.5-2; 12/8 to 18/2 at σ 4). At σ 0 it is plain FLTrust: identical
  on all 80 seed-condition pairs (prediction 2 held exactly, not just within 1e-4).
* **Fang:** Variant N beats FedAvg at every level (WIN at σ 0, 0.5, 2, 4; tie leaning positive at σ 1,
  17/3), as FLTrust does without DP.

Attackers' share of trust weight under Variant N (mean over rounds and seeds; 2/8 = 25% by headcount):
| | σ 0 | σ 0.5 | σ 1 | σ 2 | σ 4 |
|---|---|---|---|---|---|
| ALIE | 22% | 7% | 7% | 9% | 21% |
| IPM | 1.5% | 0.3% | 0.0% | 0.0% | 0.0% |
| Fang | 3.6% | 0.8% | 0.6% | 0.3% | 0.1% |

## Part 2 — adaptive "trust-capture" attack at ε 1.86 (σ 2)
Attack as pre-registered (§4): both attackers send N_t·(c·ĝ + √(1−c²)·h), with ĝ the root update's
direction, h the honest mean's component orthogonal to ĝ, reversed, and c chosen each round to
maximise the harmful step given the honest trust total S.

| | no attack | ALIE | **adaptive** | damage (none − adaptive) |
|---|---|---|---|---|
| Variant N | 0.0711 | 0.0436 | **0.0005** | +0.071 [+0.055, +0.086], 20/0 |
| FedAvg | 0.0592 | 0.0394 | 0.0137 | +0.046 [+0.031, +0.060], 19/1 |
| plain FLTrust | 0.0151 | 0.0061 | 0.0055 | +0.010 [+0.004, +0.015], 16/4 |

**F9 (Holm over 3):**
| test | diff [95% CI] | W/L | Holm p | verdict |
|---|---|---|---|---|
| 1. damage to Variant N | +0.071 [+0.055, +0.086] | 20/0 | <0.0001 | damage confirmed |
| 2. **Variant N − FedAvg, under the adaptive attack** | **−0.013 [−0.023, −0.003]** | 1/18 | 0.011 | **LOSS** |
| 3. Variant N: adaptive − ALIE | −0.043 [−0.060, −0.026] | 0/20 | 0.0001 | adaptive is stronger |

**Verdict: Variant N is BROKEN by the adaptive attack** (test 2 is a LOSS). 18 of 20 seeds end at
exactly 0.000 mAP (the other two: 0.001, 0.009). Secondary: under this attack Variant N is also worse
than plain FLTrust (−0.005, 1/19), because FLTrust's tiny steps limit how far the attack can push.

**How the attack works in practice (unexpected detail).** The prediction was that the attackers would
take ≥ 85% of the weight in most rounds. They did so only in round 1 (91%). Afterwards their share
settled at about 60% (rounds 4-12), with chosen c ≈ 0.5. The reason: once the model has been
pushed off course, honest clients' updates start pointing the same way as the server's root update
(both try to repair the damage). The honest cosine rose from 0.01 (round 1) to 0.10-0.19, and the
honest trust total S rose from 0.06 to 0.4-1.15. In effect trust partly recovers by itself.
It is not enough: 60% of the weight, every round, on a harmful update of full honest norm is
still enough to destroy the model. Plain FLTrust (same crafted update) keeps S near 0.03, but its
shrunken steps cap the damage. FedAvg gives the attackers only their 25% headcount weight, yet it
is still hurt badly, because the crafted update is aimed against the honest consensus at full
honest norm.

## Predictions (PREREG §5)
1. **Failed.** Predicted recovery at σ 0, 0.5, 1 and ties everywhere at σ 4. Observed: recovers at
   σ 0 and 0.5 (and 2); LOSS under IPM at σ 1; at σ 4 a LOSS under ALIE and a WIN under Fang.
2. **Held**, more strongly than stated: σ 0 Variant N is bit-identical to Phase 5's no-DP FLTrust on
   20/20 seeds in every condition.
3. **Failed.** Attackers took ≥ 85% in only 9% of rounds (round 1 mostly); typical share ~60%,
   because honest trust rises once the model is damaged.
4. **Held** (the main part): Variant N broken, mAP 0.0005 < 0.02, LOSS vs FedAvg. **Second half
   failed:** FedAvg was not "hurt much less": its damage (0.046) is 64% of Variant N's (0.071), and it
   kept only 23% of its clean score.
5. **Failed** by the stated threshold: FLTrust's damage was 0.010 (pre-set "little" bound 0.005). In
   relative terms it kept 36% of a clean score that was already very low (0.015).

## What this means for the thesis
* **Variant N is a big improvement on FLTrust under DP but not a drop-in replacement for FedAvg.**
  It beats plain FLTrust at every DP level and matches FedAvg at most levels, but by the
  pre-registered rule it recovers at 3 of 5 levels (σ 0, 0.5, 2), not all five. The failures are one
  attack per level (IPM at ε 5.73, ALIE at ε 0.79), and the σ 4 one is at the floor. The σ 2 "recovers"
  of Phase 5b was partly level-specific: the same IPM gap is there with the same sign at ε 1.86.
* **Variant N does not survive a rule-aware attacker.** Making all updates the same norm means trust
  alone decides the weights. Under DP the honest cosines are tiny, so an attacker who aligns
  moderately with the root update takes most of the weight. The fix that restores utility (equal
  norms at the honest DP level) is what makes the capture effective. FLTrust's own design avoided this
  by shrinking everything, which is exactly what made it useless under DP. This is the same
  utility-vs-robustness tension seen in Phases 5-5b, now from the other side.
* **No aggregator tested survives this attack**: FedAvg loses 77% of its score and FLTrust is near
  zero. Variant N is the worst of the three under it. A defence would need something the cosine cannot
  provide under DP, e.g. a cap on any one client's trust share (as noted in the PREREG); that is a new
  design, not tested here.
* **Scope / caveats:** the attacker is omniscient (knows the root update g0 exactly and the honest
  updates); an attacker that must estimate g0 is untested and would be weaker. One attack design, one
  operating point (ε 1.86), 2/8 attackers, 12 rounds.
