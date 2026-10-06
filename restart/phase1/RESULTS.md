# Phase 1 — results (2026-10-05)

Pre-registration: `PREREG.md` (+ Amendment 1). Raw runs: `restart/results/phase1/`.
Analysis: `restart/code/analyse_phase1.py`. Machine: RTX 5090, 540 W software cap.
Model: MobileNetV3-Small frozen through block 8 (K=9) + 65,221-parameter head.
Train = clients pool, every 5th frame (25,146 frames); gates on val (5,487 frames).

## Gate G1 — PASS
Non-DP SGD, lr 0.03 (chosen from {0.003, 0.01, 0.03} on seed 0), 12 epochs:
val mAP@0.5 = **0.584 +- 0.005** (3 seeds). Trivial median-box predictor: **0.000**.

## Gate G1b — FAILED at B=256, PASSED at B=1,024
C = 45.8 (median per-sample gradient norm at init, 1,024 frames).

| sigma | eps (delta=1e-5) | B=256 mAP | B=1,024 mAP (mean +- sd, 3 seeds) |
|---|---|---|---|
| 0 | inf | 0.588 | **0.576 +- 0.013** |
| 2 | 0.76 / **1.66** | 0.010 | **0.350 +- 0.035** |
| 4 | 0.34 / **0.71** | 0.000 | **0.104 +- 0.076** |
| 8 | 0.16 / 0.33 | 0.000 | 0.001 |
| 16 | 0.12 / 0.16 | 0.000 | 0.000 |
| 32 | 0.11 / 0.12 | 0.000 | 0.000 |

(eps column: B=256 / B=1,024.) B=1,024: Spearman(sigma, mAP) = -1.000, drop 0.576.

## What this fixes for later phases
* Canonical single-client DP config: **K=9, B=1,024, C=45.8, lr=0.03, 12 epochs.**
* The live regime is narrow: **sigma ~ 0-4 (eps >= ~0.7)**; the model is dead from
  sigma = 8. Phase 3's operating sigma must come from this window, and finer steps
  (sigma 1, 3) will be needed there. sigma = 4 is near a cliff (seeds 0.06-0.19).
* Clipping alone costs nothing: sigma=0 (0.576) vs non-DP (0.573).
* The epsilons here are small (strong privacy) — unlike typical "eps ~ tens" settings.
