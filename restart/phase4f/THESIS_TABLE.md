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
| **Caveat** | Theory schedule A3 also wins (+0.0144, 16/4); learned DRL schedules do not beat uniform | Theory schedule A3 fails (ties uniform; noisy early-round estimates); Kiani et al.'s schedule wins (17/3) | Kiani et al.'s schedule is the worst arm (loses to A2 on 19/20 seeds) | Capacity limit, not a privacy result; scoring verified (oracle 0.990) |

ᵃ Federated training without DP (Phase 2 configuration: one local epoch, batch 256); the other
columns use the Phase 4c protocol's noise-free run (20 local iterations, batch 64, clipping kept).
ᵇ Unadjusted; in the drone study this comparison was secondary (the primary family tested the
learned schedules).
"Later rounds more sensitive": noise-pulse test (σ doubled in one round, 20 seeds); strength = slope
of the harm per round divided by the baseline score, relative to the drone's (drone +0.039,
VisDrone +0.043, CIFAR-10 +0.014). VisDrone scores are tile-level mAP@0.5 for one class.

Sources: `restart/phase4/RESULTS.md`, `restart/phase4c/RESULTS.md`, `restart/phase4f/RESULTS.md`;
raw runs under `restart/results/phase4*/`.
