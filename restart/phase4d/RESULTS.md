# Phase 4d results — signal-redesign check (diagnostic; follow-up to Phase 4b)
Run 2026-10-11 under phase4d/PREREG.md (committed before the run). No test data used.
Raw: `results/phase4d/sig_*.json`, `analysis.json`; figure `plots/phase4d_signal.png`.
All 60 replays are identical to Phase 4b's (schedules and every logged observation).

New signal: g_t = server-side median per-sample gradient norm (no noise in the computation) of
the global model on the public proxy session; agent input s_t = its 3-round rolling mean.
Outcome for "final rank" = the final model's VAL mAP@0.5.

| schedule | largest sd / progress (rounds 1-10) | Spearman(s_t, final VAL mAP), rounds 1-8 | Spearman(s_t, s_12), rounds 3-10 | lockstep with own noise dose (raw g_t: mean r, 95% CI) |
|---|---|---|---|---|
| A1 uniform | 6.4% | +0.21 .. +0.44 | +0.79 .. +0.97 | n/a (no dose variation) |
| **A3 theory (decides)** | **6.0%** | **+0.01 .. +0.30** | +0.75 .. +0.99 | **+0.68 [0.51, 0.85] -> LOCKSTEP** |
| A5 learned | 5.6% | -0.16 .. +0.43 | +0.76 .. +0.98 | +0.15 [0.11, 0.19] -> below 0.3 (smooth schedule, little dose variation to test) |

Rolling s_t under A3: r = +0.82 [0.81, 0.83] (averaging does not remove it). Positive control,
old proxy-loss signal under A3: r = +0.88 [0.81, 0.95] (lockstep, as Phase 4b found).

**Decision (pre-registered): (a) FAILS — no round 1-8 reaches |rho| >= 0.5 with the outcome
(best +0.30 under A3; +0.44 under A1). (b) FAILS — the raw gradient norm rises in lockstep with
the round's own noise dose (r = +0.68). -> A5-v2 is NOT trained; Phase 5 uses A5.**

Reading: the gradient norm is a stable property of a run (a seed that starts high stays high,
rho with its own final value 0.75-0.99) but that property says little about how good the final
model is, and its round-to-round movement is driven by the noise just injected. Evaluating a
noise-free quantity does not help, because the model it is evaluated at already carries that
round's noise. Together with Phase 4b: neither server-side signal tested (loss, gradient norm)
gives a feedback agent anything to react to beyond the schedule itself — consistent with A5
converging to a fixed schedule.
