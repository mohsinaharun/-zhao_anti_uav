# Phase 3 — Noise-pulse test, extended (pre-registered 2026-10-06, BEFORE any run)

User-specified design (thoroughness over time). Question: is a_t — the sensitivity of
final accuracy to noise added in round t — flat, or does it vary by round?

## Common setup
FL, 8 clients (frozen `clients.json`), B = 64, 20 local iterations per round,
lr 0.03, FedProx off (mu = 0), **calibrated fixed clip C*** (per seed: proxy median
after 2 proxy rounds; zero privacy cost), session split unchanged, val pool.

## The pulse
Baseline: sigma_t = sigma in every round. Pulse at round k: sigma_k = 2*sigma (noise
variance x4) in round k only; every other round unchanged. Each pulse run is paired
with the same-seed baseline (same init, same C*, same sampling, the same noise draws —
scaled up in the pulse round). **drop(k, seed) = mAP(baseline) - mAP(pulse at k).**
Epsilon is reported for baseline and pulse runs (a pulse round spends LESS privacy).

## Variants (each: every round k = 1..R, sigma in {1, 2, 4}, 20 paired seeds 0-19)
* **V1** (main): R = 12, K = 9 (backbone frozen through block 8; 65,221 params).
* **V2** (length): R = 25, K = 9.
* **V3** (capacity): R = 12, K = 7 — blocks 7-8 (the 48-channel stride-16 stage)
  also trained, BatchNorm folded into the convolutions (identical function and
  pretrained weights; per-sample DP cannot train BatchNorm). ~115k params. Its own
  feature cache (output of block 6) and its own C* by the same rule.

## Analysis per (variant, sigma) cell
* a_t profile: mean drop(k) over seeds with 95% CI, table + plot.
* **Slope test:** per seed, OLS slope of drop(k) on k; 95% t-CI of the mean slope
  over 20 seeds. Trend if the CI excludes 0.
* **Shape test:** per seed, mean drop over the first third of rounds minus the last
  third; 95% CI. Variation if the CI excludes 0.
* **Informativeness:** mean drop pooled over all rounds must be > 0 (95% CI) — i.e.
  the pulse measurably hurts. If not, the cell is "not identifiable", NOT "flat".
* **Gradient norm:** per round, the clients' mean pre-clip median per-sample norm in
  the BASELINE run; per-seed slope over all rounds and over rounds 3..R (post-
  transient), 95% CIs. Descriptive.
* Cell is **FLAT** iff informative AND slope CI includes 0 AND shape CI includes 0.

## Decision rule (user-specified)
Conclude **a_t is flat in general ONLY IF all 9 cells (V1-V3 x sigma 1/2/4) are FLAT**.
If any cell shows a trend or shape variation, report that configuration specifically
as the regime where adaptive allocation (and DRL) has something real to learn.
No multiple-comparison correction is applied (user rule); p-values are reported so
the reader can apply one.

## Order and reporting
V1 -> report -> V2 -> report -> V3 -> report -> combined verdict. Each reported
separately with a plain-language summary. Then re-runs of Phase 1 / 2 / 4 with C*
(pre-registered separately before they run).

## Implementation notes — 2026-10-06, before any Phase 3 run (no design change)

* Regression check: the modified runner reproduces a stored run exactly
  (calibrated, sigma 1, seed 0: mAP 0.261688, eps 5.7326, identical).
* V3 trainable parameter count is **116,365** (not ~115k): torch's
  `fuse_conv_bn_eval` creates the folded bias with requires_grad=False; it is made
  trainable, since it carries the folded BatchNorm shift.
* V3 per-sample gradients use Opacus **expanded weights** (`ew`), because Opacus hooks
  reject the blocks' input reuse (squeeze-excitation, residual). Verified against
  exact single-sample backprop: relative error 4.8e-7 with TF32 off (2.3e-3 with the
  default TF32, which applies equally to all runs); per-sample norms identical under
  the mean loss. V1/V2 keep the hook engine verified in Phase 1. In-place ops in the
  unfrozen blocks are made out-of-place (identical function: 4.7e-6).
* Phase 3 results are written to `restart/results/phase3/`.
* Measured cost: V1 ~26 s/run, V3 ~70 s/run, V2 ~55 s/run; total ~44 h.
