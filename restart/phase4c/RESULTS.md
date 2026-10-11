# Phase 4c results — replication on other data (CIFAR-10, Pascal VOC)
Run 2026-10-11 under phase4c/PREREG.md (approved as written; implementation notes before any run).
Phase 4 / 4b unchanged. Raw: `results/phase4c/{cifar10,voc}/`; figure `plots/phase4c_cifar10.png`.

## CIFAR-10 (top-1 accuracy; clients = drone sizes, so epsilon identical to the drone study)
**Gates:** passed with the original configuration (K = 9, B = 64). Uniform sweep (val, 5 seeds):
sigma 0: 0.564 / 0.5: 0.437 / 1: 0.368 / **2: 0.300** / 3: 0.256 / 4: 0.190 / 6: 0.135 / 8: 0.127
(trivial 0.10). Operating point by the midpoint rule: **sigma 2, R 12, eps 1.857 — the same as
the drone P.**

**Pulse test (20 seeds): a_t VARIES, rising late — the same shape as the drone's.**
Mean drop per pulse round 1..12: -0.006 0.003 0.007 0.016 0.010 0.010 0.018 0.013 0.021 0.020
0.035 0.067. Slope +0.0042/round [0.0035, 0.0050]; first-third minus last-third -0.030
[-0.036, -0.025]; pooled drop 0.018 [0.014, 0.022] (informative). Normalised, the CIFAR-10 and
drone profiles nearly coincide (figure, left): later-round noise hurts more, most of all in the
last two rounds.

**A2 tuning (val, 5 seeds):** k = 0 0.300 / 0.25 0.307 / **0.5 0.309** / 1 0.308 / 1.5 0.295 /
2 0.268 / 3 0.212 -> k = 0.5 (the same k the drone tuning chose at P).

**Test, 20 paired seeds (eps 1.86 every arm):**
| arm | accuracy | sd | eps |
|---|---|---|---|
| A1 uniform | 0.2968 | 0.029 | 1.8569 |
| A2 tuned exponential (k 0.5) | 0.3018 | 0.026 | 1.8581 |
| A3 theory (sqrt a_t) | 0.3013 | 0.026 | 1.8554 |
| A6 Kiani et al. spend-as-you-go | 0.2927 | 0.026 | 1.8569 |

| comparison | diff | 95% CI | W/L | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| A2 - A1 | +0.0050 | [+0.0006, +0.0095] | 16/4 | 0.029 | 0.088 | TIE (EQUIVALENT) |
| A3 - A1 | +0.0045 | [-0.0025, +0.0116] | 12/8 | 0.193 | 0.292 | TIE (EQUIVALENT) |
| A6 - A1 | -0.0041 | [-0.0097, +0.0015] | 9/11 | 0.146 | 0.292 | TIE (EQUIVALENT) |
| A3 - A2 (secondary) | -0.0005 | [-0.0036, +0.0026] | 8/11 | 0.74 | – | |
| A6 - A2 (secondary) | -0.0091 | [-0.0120, -0.0062] | 1/19 | <1e-4 | – | |
| A6 - A3 (secondary) | -0.0086 | [-0.0121, -0.0051] | 2/18 | 1e-4 | – | |

**Verdict (pre-registered): "decay beats uniform" FAILS TO REPLICATE on CIFAR-10** — no arm WINS
after Holm. It is not reversed: the direction is the same (A2 +0.005, 16/4, unadjusted p 0.029),
but the effect is about 1.7% relative (drone: ~24%) and every primary comparison is EQUIVALENT
within the pre-set ±0.02 margin. Kiani et al.'s two-phase rule is the worst arm here: it
loses to both smooth decays on 19/20 and 18/20 seeds (secondary, unadjusted).

Reading: the mechanism transfers (a_t has the same rising shape), but the payoff of exploiting it
is much smaller on CIFAR-10. A non-flat a_t is necessary for a schedule to help, not sufficient
for a large gain.

## Pascal VOC (class-agnostic mAP@0.5) — STOPPED at the gates (pre-registered rule)
Uniform sweep, val, 5 seeds (trivial median-box predictor 0.004; G1 bar = trivial + 0.10 = 0.104):
| config | sigma 0 | 0.5 | 1 | 2 | 3 | 4 | 6 | 8 | G1 | G2 |
|---|---|---|---|---|---|---|---|---|---|---|
| K 9, B 64 | 0.058 | 0.038 | 0.029 | 0.020 | 0.013 | 0.009 | 0.004 | 0.002 | fail | pass |
| K 7, B 64 | 0.074 | 0.052 | 0.037 | 0.016 | 0.009 | 0.006 | 0.004 | 0.002 | fail | pass |
| K 7, B 32 | 0.070 | 0.040 | 0.019 | 0.005 | 0.004 | 0.003 | 0.003 | 0.001 | fail | pass |
(eps at R 12 is higher than the drone's at the same sigma because VOC clients are 0.515x the size:
sigma 1 -> eps 11.5, sigma 2 -> 3.85, sigma 4 -> 1.63.)

**Not a bug:** an oracle prediction built from the targets scores 0.990 mAP through the same
encode / decode / mAP code; the multi-object code matches the drone code exactly on one-box data.
**Cause:** a frozen MobileNet trunk with a 65k-parameter centre-point head and 12 rounds x 20
local steps cannot learn VOC's large, varied, multi-object scenes: even 1,440 non-private central
steps on 1,500 proxy images reach only 0.108 mAP on those same images (0.036 val). The midpoint
operating point would also sit at eps 11-29, far from the drone/CIFAR eps 1.86.
**No replication verdict for VOC.** Any continuation needs a design change (amendment).

**Decision 2026-10-11 (user): VOC is stopped.** No amendment; VOC stays without a replication
verdict and Phase 4c's replication evidence is CIFAR-10 only. VOC code and splits are kept as-is.
