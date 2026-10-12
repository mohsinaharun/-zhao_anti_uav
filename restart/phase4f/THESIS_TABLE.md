## Does "less noise late beats uniform" hold beyond the drone data?

A decaying privacy schedule beats uniform allocation at equal privacy (ε = 1.86) on both drone-view
detection tasks, where noise injected in late rounds is markedly more harmful than early noise, and
it gives no detectable benefit on CIFAR-10, where that late-round sensitivity is about three times
weaker. A strong late-round sensitivity therefore appears necessary for a benefit, but it does not
set its size: VisDrone's sensitivity profile matches the drone's as closely as it can (1.10× its
relative slope), yet its gain is less than half the drone's (+10.5% vs +23.5%). Pascal VOC could
not be tested because the frozen-backbone detector cannot learn it even without noise.

**Table X.** Generalization of the Phase 4 finding. Test-set results, 20 paired seeds with common
random numbers; A2 = exponentially decaying schedule tuned on validation data, A1 = uniform; both
spend the same budget (worst-client ε at δ = 10⁻⁵).

| | Anti-UAV drone (IR) | VisDrone (car) | CIFAR-10 | Pascal VOC |
|---|---|---|---|---|
| **Task** | single-target detection | multi-object detection | 10-class classification | multi-object detection |
| **Learnable (quality bar)** | Passed: 0.487 mAP without DP ᵃ (trivial 0.000) | Passed: 0.238 mAP (bar 0.104) | Passed: 0.562 accuracy (bar 0.200) | **Failed:** 0.058 / 0.074 / 0.070 mAP in 3 configurations (bar 0.104) |
| **ε matched** | Yes, 1.86 | Yes, 1.86 | Yes, 1.86 | No (not reached) |
| **Later rounds more sensitive** | Yes (reference, 1.00×) | Yes, **1.10×** | Yes, **0.36×** | Not tested |
| **A2 beats uniform** | **Yes**: 16/4 seeds, p = 0.0002 ᵇ | **Yes**: 17/3 seeds, p = 0.0002 (Holm 0.0006) | No (equivalent): 16/4 seeds, p = 0.029 (Holm 0.088) | Not tested |
| **Absolute gain [95% CI]** | +0.0139 mAP [+0.0076, +0.0202] | +0.0037 mAP [+0.0020, +0.0054] | +0.0050 accuracy [+0.0006, +0.0095] | — |
| **Relative gain** | **+23.5%** (0.0592 → 0.0731) | **+10.5%** (0.0354 → 0.0391) | **+1.7%** (0.2968 → 0.3018) | — |
| **Caveat** | Theory schedule A3 also wins (+0.0144, 16/4); no learned or searched schedule beats the fixed ones (Table X.2); gain stable from 12 to 100 rounds (Table X.3) | Theory schedule A3 fails (ties uniform; noisy early-round estimates); Kiani et al.'s schedule wins (17/3) | Kiani et al.'s schedule is the worst arm (loses to A2 on 19/20 seeds) | Capacity limit, not a privacy result; scoring verified (oracle 0.990) |

ᵃ Federated training without DP (Phase 2 configuration: one local epoch, batch 256); the other
columns use the Phase 4c protocol's noise-free run (20 local iterations, batch 64, clipping kept).
ᵇ Unadjusted; in the drone study this comparison was secondary (the primary family tested the
learned schedules).
"Later rounds more sensitive": noise-pulse test (σ doubled in one round, 20 seeds); strength = slope
of the harm per round divided by the baseline score, relative to the drone's (drone +0.039,
VisDrone +0.043, CIFAR-10 +0.014). VisDrone scores are tile-level mAP@0.5 for one class.

Sources: `restart/phase4/RESULTS.md`, `restart/phase4c/RESULTS.md`, `restart/phase4f/RESULTS.md`;
raw runs under `restart/results/phase4*/`.

## Learned and searched schedules vs fixed ones (drone)

No learned or searched schedule beats the best fixed decaying schedule at any budget. The unconstrained
DRL agents (A4, A5) at best match uniform and at worst collapse. Bayesian-optimisation search (A7)
and a DRL agent held to ±20% around the theory schedule (A5-relative) avoid that harm, but they only
tie the fixed schedules they are compared with.

**Table X.2.** All allocation methods on the Anti-UAV drone task. Test mAP@0.5, mean over 20 paired
seeds with common random numbers. All methods at an operating point spend the same budget (worst-client
ε at δ = 10⁻⁵, matched to ±0.003). "Genuine improvement" = significantly beats uniform AND the better
of A2/A3, Holm-adjusted within each phase's pre-registered family.

| Method | Type | P: R 12, ε 1.86 | S1: R 25, ε 2.72 | S2: R 12, ε 0.79 | Genuine improvement? |
|---|---|---|---|---|---|
| A1 uniform | fixed | 0.059 | 0.105 | 0.014 | reference |
| A2 exponential decay | fixed, k tuned on validation | 0.073 | **0.111** | **0.025** | beats uniform at P and S2 (16/4); S1 n.s. (14/6) |
| A3 theory, ρ_t ∝ √a_t | fixed, from measured round sensitivity | **0.074** | 0.080 | 0.021 | beats uniform at P and S2 (16/4); **loses** at S1 (3/17) |
| A4 DDPG | learned, absolute σ | 0.000 | 0.108 | 0.000 | No: collapses at P and S2 (0/20) |
| A5 TD3 + GRU | learned, absolute σ | 0.059 | 0.098 | 0.011 | No: loses to A2 and A3 at P (5/15, 4/16) |
| A7 Bayesian optimisation | searched in simulator | 0.075 | 0.106 | 0.013 | No: beats uniform at P but ties A2/A3; loses to A2 at S2 (1/19) |
| A5-relative | learned, ±20% around A3, budget exact | 0.071 | 0.092 | 0.021 | No: ties A2/A3 at P and S2; loses to A2 at S1 (3/17) |

A4's and A5's S1 numbers come from an episode-25 checkpoint (best simulator evaluation). The DRL
agents' training reward never rose for A4 at P/S2 or for A5-relative at any point. Sources:
`restart/phase4/RESULTS.md`, `restart/phase4e/RESULTS.md`, `restart/phase4h/RESULTS.md`.

## More rounds do not close the privacy gap (drone)

At a fixed budget (ε 1.86), every extra round must be paid for with more noise in every round.
Accuracy rises slightly from 12 to 25 rounds and then falls, while the model without DP keeps improving
up to 50 rounds. The DP/no-DP gap is a noise ceiling, not a training-length problem. The decaying
schedule's advantage over uniform is the same at every round count.

**Table X.3.** Test mAP@0.5 vs number of rounds at fixed worst-client ε 1.86 (all DP runs ε 1.855-1.859).
20 paired seeds, mean [95% CI]. G = A2 / no-DP at the same R. Step = change in A2 from the previous R,
wins/losses, Holm over the three steps (pre-registered family F11).

| Rounds R | σ per round (A2, first → last) | A2 | Uniform | No DP | G | Step (A2) |
|---|---|---|---|---|---|---|
| 12 | 2.67 → 1.62 | 0.073 [0.061, 0.085] | 0.059 [0.047, 0.072] | 0.250 [0.235, 0.264] | 0.29 | — |
| 25 | 3.58 → 2.17 | **0.086** [0.067, 0.105] | 0.074 [0.054, 0.094] | 0.321 [0.310, 0.333] | 0.27 | +0.013, 13/7, tie |
| 50 | 4.88 → 2.96 | 0.057 [0.043, 0.072] | 0.045 [0.032, 0.058] | **0.340** [0.331, 0.349] | 0.17 | **−0.029, 7/13, LOSS** |
| 100 | 6.77 → 4.11 | 0.040 [0.031, 0.050] | 0.027 [0.018, 0.036] | 0.330 [0.322, 0.338] | 0.12 | −0.017, 6/14, tie ᵃ |

ᵃ Holm p 0.056. A2 at 100 rounds is below A2 at 25 rounds on 18/20 seeds (−0.046). A2 − uniform is
+0.014 / +0.012 / +0.012 / +0.013 at R 12 / 25 / 50 / 100 (16/4, 16/4, 17/3, 15/5). Source:
`restart/phase4g/RESULTS.md`; figure `plots/phase4g_rounds.png`.
