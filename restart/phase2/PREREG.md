# Phase 2 — pre-registration (written 2026-10-06, BEFORE any Phase 2 run)

Implements todo steps 7-9. Builds on the Phase 1 canonical config (single-frame
input, MobileNetV3-Small frozen through block 8, 65,221-parameter head, C=45.8).
Changes require a dated amendment below, made before the run they affect.

## Clients (step 7) — frozen in `restart/data/clients.json`

8 clients, 2 per environment, from the `clients` pool only; sessions never split.
Target frames: building 9.9k / 9.4k, cloud 7.1k / 6.0k, sky 29.7k / 29.6k,
terrain 13.6k / 13.7k (before the every-5th-frame sampling). Unequal by design
(different sites hold different amounts of data); disclosed.

## Training protocol (identical code path for all four cells)

* One runner, `restart/code/phase2_fl.py`. "No FL" = the same loop with ONE client
  holding the whole pool, so FL and no-FL differ only in the data partition.
* R = **12 rounds x E = 1 local epoch** (= 12 passes over the data, matching
  Phase 1's 12 epochs). Local SGD momentum 0.9, reset each round.
* Aggregation: FedAvg weighted by client frame count, over all trainable params.
* Batch: FL clients B_c = **256**; no-FL B = **1,024** (Phase 1 canonical).
  (Different by necessity: a client holds ~1.3k-6k frames, so B=1,024 would give
  1-6 steps per epoch. Disclosed as a confound of the FL-vs-no-FL contrast.)

## FedProx (step 8)

Proximal term mu/2 ||w - w_global||^2 applied as a decoupled step after each local
optimiser step (data-independent, so it costs no privacy and is not clipped).
* lr for FL chosen first, at mu = 0, from {0.01, 0.03, 0.1}, FL no-DP, seed 0.
* Then **mu chosen from {0, 0.01, 0.1, 1}** at that lr, FL no-DP, seed 0, by val mAP.
  If mu = 0 wins, FedProx is reported as not helping here and the FL cells use mu=0.
* The chosen (lr, mu) are used for both FL cells.

## Privacy (DP cells)

DP-SGD per client (Opacus, per-sample clipping C = 45.8, Poisson sampling), frame-
level unit. **sigma = 2** (Phase 1's live regime). FL-DP lr re-selected at sigma = 0
(clipping only) from {0.01, 0.03, 0.1}, seed 0 (Phase 1 rule). No-FL DP uses the
Phase 1 canonical lr 0.03. Epsilon at delta=1e-5 from the RDP accountant, per client
(q_c = B_c / N_c, steps over all rounds); report max and median over clients and the
no-FL epsilon. The four cells are compared at equal sigma, NOT equal epsilon —
FL clients hold less data, so their epsilon is larger; both are reported.

## The 2x2 ablation (step 9)

Cells: **FL+DP, FL no-DP, DP no-FL, neither**. **n = 10 paired seeds (0-9)**: seed t
shares initialisation (and, within a cell, sampling/noise streams) across cells.
Reported per contrast (mean difference, 95% CI from the paired t-distribution, win
count out of 10):
* cost of FL without DP   = FL no-DP  - neither
* cost of DP without FL   = DP no-FL  - neither
* cost of DP inside FL    = FL+DP     - FL no-DP
* interaction             = (FL+DP - FL no-DP) - (DP no-FL - neither)
Descriptive: final val mAP@0.5 per environment for each cell; per-round curves.

## Gate G2

**PASS** if every cell's mean val mAP@0.5 >= trivial + 0.10 (= 0.10) — i.e. FL+DP
actually trains — and all contrasts above are reported with CIs.
If FL+DP fails: (1) B_c -> 512, (2) R -> 24 rounds of E = 1/2 epoch; each attempt
logged in `restart/NEGATIVE_RESULTS.md`.

`test` is still untouched; all numbers are on `val`.

## Amendment 1 — 2026-10-06, user-directed revision Step 2 (edge batch size), before any run

* Batch size **B = 64** (chosen over 32: an edge board holds 64 frames' activations
  easily, while DP noise on the mean gradient is sigma*C/B, so 32 would double the
  relative noise for no real memory need). DP clipping (C = 45.8, per sample) and
  Gaussian noise are applied per Poisson batch of expected size 64.
* FL: **20 local iterations per round** (fixed number of local DP-SGD steps; top of
  the user's 15-20 range), R = 12 rounds, FedProx mu = 1, lr 0.03 — all else as in
  Phase 2. Epsilon per client from q_c = 64/N_c and 240 steps.
* Central (for an apples-to-apples comparison with Step 1): same as Step 1 but
  B = 64 (12 epochs).
* Sweep: sigma in {0, 0.5, ..., 4.0}, 10 paired seeds, both settings. lr is NOT
  re-selected (the user scheduled learning-rate changes for Step 5).
* Reported: noise level / mAP / epsilon tables for central B=1,024 (Step 1),
  central B=64, FL B=64; whether noise-survivability changes. Descriptive, no gate.

## Amendment 2 — 2026-10-06, user-directed Steps 3-4, before any Step 4 run

* Step 3 (confirmed by the user): FedProx gradient mu*(w - w_global) is now added to
  the already-privatized gradient between Opacus pre_step() (clip, sum, noise, scale)
  and the SGD(momentum) update. Verified on fixed data with an identical noise draw:
  (grad_mu - grad_0) - mu(w - w_g) = 2.4e-7 (float rounding) for the new code, vs a
  100% relative error when the term is put inside the per-sample loss.
* Step 4: mu in {0 (no-FedProx baseline), 0.01, 0.03, 0.1, 0.3, 1}, edge config
  (FL, 8 clients, B=64, 20 local iterations/round, R=12, lr 0.03), at sigma = 0 and
  sigma = 0.5 (the only sigma at B=64 where FL+DP still learns), 10 paired seeds.
  Tags carry `_proxv2` so no pre-Step-3 run is reused. Reported per mu: mean mAP,
  paired difference vs mu=0 with 95% CI and wins, epsilon. Descriptive, no gate.
