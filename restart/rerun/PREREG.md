# Re-runs of Phase 1 / Phase 2 / revision Step 4 with calibrated C* (pre-registered 2026-10-09, before any run)

User-confirmed set (2026-10-09). Every DP run uses the calibrated fixed clip bound
C* (per seed: train "2 rounds' worth" of local steps on the PUBLIC proxy pool without
noise, then C* = proxy median per-sample gradient norm; zero privacy cost). "A round's
worth" = local iterations per round in FL (20 -> 40 steps, unchanged from Phase 3),
and one local epoch in central mode (393 steps at B=64 -> 786 steps). Regression
check: the FL path reproduces a stored run exactly (mAP 0.261688, C* 6.0504).
10 paired seeds (0-9), same session split, val pool, lr 0.03, R = 12, single code path
(`phase2_fl.py`). Epsilon (delta=1e-5) reported with every number. Descriptive (no
new gates); contrasts reported with paired 95% CIs and win counts.

## R1 — Phase 1 re-run: central sigma sweep, B = 64
Central (one client = whole clients pool), B = 64, 12 rounds x 1 epoch, C*, sigma in
{0, 0.5, ..., 4.0}. Compared against the old central B=64 sweep at C = 45.8 and the
Step 1 central B=1,024 sweep at C = 45.8.

## R2 — Phase 2 re-run: FL x DP 2x2 at the edge config
Cells per sigma in {1, 2, 4}: FL+DP (8 clients, B=64, 20 local iters, mu=0, C*),
FL no-DP (no clipping, no noise, same FL config), central DP (R1 runs), central no-DP
(no clipping, no noise, B=64). Contrasts: cost of FL, cost of DP (central), cost of DP
inside FL, interaction. FL+DP runs REUSED from the clipping control
(`results/phase2/*_clipCal_seed*`, identical config and code path).

## R3 — Step 4 re-run: FedProx mu sweep
FL+DP, B=64, 20 local iters, C*, sigma in {1, 2}, mu in {0 (reused), 0.01, 0.03,
0.1, 0.3, 1} (FedProx gradient added after privatization, as confirmed in Step 3).
Per mu: mAP, paired difference vs mu=0 with 95% CI and wins, epsilon.

New runs: R1 90 + R2 20 + R3 100 = 210 (~2 h). Outputs: `restart/results/rerun/`.
