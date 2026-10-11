# Phase 4f — VisDrone generalization check: results
Run 2026-10-11 as pre-registered (`PREREG.md` + dated note). Purely additive: no Phase 4/4b/4c/4d/4e
file or result changed. Raw runs: `results/phase4f/` (415 runs, ~20 s each, ~2.5 GPU-hours).
Figure: `plots/phase4f_visdrone.png`.

## Verdict
**"Decay beats uniform" REPLICATES on VisDrone (car).** A2 (tuned exponential) beats uniform at
epsilon 1.86 on 17/20 test seeds, Holm p 0.0006. **Classified DRONE-LIKE** by the pre-registered rule
(decay WIN and relative gain >= 10%) — **but only just: +10.5% against the 10% line.** The noise-
sensitivity shape matches the drone almost exactly; the payoff is about half the drone's in relative
terms and small in absolute terms (+0.004 mAP).

## Quality check (the VOC check)
| check | result |
|---|---|
| sigma = 0, VAL mAP@0.5 (seeds 100-102) | **0.238** vs bar 0.104 → **CLEARED** at K 9 / B 64 (no fallback) |
| trivial (median box per tile) | 0.0000 |
| scoring code on ground-truth boxes | 1.000 (no scoring bug) |
| head's structural ceiling (encode → decode) | 0.941 (5.5% of cars share a grid cell) |

VisDrone is not capacity-limited (unlike VOC: 0.058-0.074 under every configuration).

Uniform sweep, VAL (5 seeds, incl. diverged seed 104): sigma 0 0.192 · 0.5 0.157 · 1 0.103 ·
**2 0.045** · 3 0.024 · 4 0.015 · 6 0.004 · 8 0.001. G2 (noise matters) passed. The epsilon 1.86
operating point (sigma 2) loses ~77% of the noise-free score: a high-noise point, like the drone's P.

## Main comparison — TEST (official test-dev, 6,440 tiles), 20 paired seeds, epsilon = 1.8569 (worst client, delta 1e-5)
| arm | test mAP@0.5 | eps |
|---|---|---|
| A1 uniform (sigma 2) | 0.0354 | 1.8569 |
| **A2 exponential, k = 1** (tuned on VAL) | **0.0391** | 1.8555 |
| A3 theory rho_t ∝ √a_t | 0.0361 | 1.8569 |
| A6 Kiani et al. spend-as-you-go | 0.0386 | 1.8569 |

| comparison | diff [95% CI] | W/L | Holm p | result |
|---|---|---|---|---|
| A2 − A1 | +0.0037 [+0.0020, +0.0054] | 17/3 | 0.0006 | **WIN** |
| A3 − A1 | +0.0007 [−0.0012, +0.0026] | 12/8 | 0.459 | TIE (equivalent) |
| A6 − A1 | +0.0032 [+0.0017, +0.0047] | 17/3 | 0.0006 | **WIN** |
| A3 − A2 (secondary) | −0.0030 [−0.0043, −0.0018] | 2/18 | p 0.0001 | — |
| A6 − A2 (secondary) | −0.0005 [−0.0016, +0.0006] | 10/10 | p 0.38 | — |
| A6 − A3 (secondary) | +0.0025 [+0.0009, +0.0041] | 16/4 | p 0.004 | — |

Without the diverged seed 203 (descriptive, per the dated note): A2 − A1 +0.0039 [+0.0022, +0.0057]
17/2; A3 − A1 +0.0007; A6 − A1 +0.0034 17/2. Same conclusions. (Seed 203 scores ~0 in every arm.)

## Pulse test (sensitivity a_t), VAL, 20 seeds, sigma 2 → 4 in one round
Drop per round 1..12: −0.001 0.000 −0.000 0.005 0.004 0.004 0.004 0.006 0.008 0.011 0.018 **0.043**.
Slope +0.0026 [+0.0023, +0.0029], first-minus-last third −0.019 [−0.021, −0.017], pooled +0.0084
[+0.0069, +0.0099] → **VARIES (rising late)**.

## Drone-like or CIFAR-like?
| | drone (P) | **VisDrone** | CIFAR-10 |
|---|---|---|---|
| pulse slope / baseline | +0.039 | **+0.043** | +0.014 |
| pooled drop / baseline | +0.147 | **+0.137** | +0.059 |
| share of drop in last third | 0.70 | **0.79** | 0.67 |
| best decay arm vs uniform (relative) | +24% | **+10.5%** | +1.7% |
| decay arm WINS (Holm) | yes (A2, A3) | **yes (A2)** | no |
| Kiani A6 vs uniform | — | **WIN** | worst arm |

* **Sensitivity profile: drone-like** — relative pulse curves nearly overlap the drone's (figure,
  left), about 3x CIFAR-10's.
* **Effect size: in between**, just over the pre-registered drone-like line. In absolute terms the
  gain is +0.004 mAP on a 0.035 baseline (drone: +0.014 on 0.059).

## Differences from the drone result (reported, not explained away)
* **A3 (theory) does not win here**, and loses to A2 (2/18). Its pulse estimates for rounds 1-3 are
  ≤ 0, so they are floored and those rounds get the maximum noise (sigma 4.19 three times) — a
  noisy-estimate failure, as A3 at S1 in Phase 4. A smooth fit (A2) is more robust.
* **Kiani et al.'s rule wins here** (it was the worst arm on CIFAR-10): its 2-step schedule
  approximates a decay.
* Test-dev is harder than val (uniform 0.035 test vs ~0.056 val), as with the drone test pool.

## Caveats
* Tile-level mAP of one class (car), not the official whole-image VisDrone AP.
* Seeds 104 (tuning) and 203 (test) diverged in the non-private C* calibration (dense-scene loss);
  kept as registered; conclusions unchanged without seed 203.
* g = 10.5% sits just above the 10% classification line; the difference shrinking by just 0.0002 mAP
  would move it to "intermediate" (its CI, +5.6% to +15%, spans both classes). The WIN itself is solid (17/3, CI excludes 0).

## Three-dataset summary (Phase 4 / 4c / 4f, epsilon 1.86 where matched)
| dataset | a_t rises late | decay beats uniform | relative gain |
|---|---|---|---|
| Anti-UAV drone (Phase 4, P) | yes | **yes** (A2, A3) | +24% |
| VisDrone car (Phase 4f) | yes | **yes** (A2) | +10.5% |
| CIFAR-10 (Phase 4c) | yes | no (equivalent) | +1.7% |
| Pascal VOC (Phase 4c) | — | not tested (capacity limit) | — |
