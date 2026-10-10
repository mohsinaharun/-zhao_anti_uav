# Phase 4b results — follow-up investigation (addendum to Phase 4; Phase 4 unchanged)
Run 2026-10-10 under phase4b/PREREG.md (committed before the run).
Raw: `results/phase4b/var_*.json`, `variation_analysis.json`; figure `plots/phase4b_variation.png`.

## Part 1 — variation check (P, R = 12, eps = 1.86; seeds 200-219; no test data used)
Replays reproduce the Phase 4 runs exactly (sigma schedules and C* identical for all 60 runs).
Proxy loss = the agent's main feedback signal (public proxy session 01_2192).

| schedule | largest between-seed sd / training progress (rounds 1-10) | Spearman(loss at t, loss at 12), rounds 1-10 | criterion |
|---|---|---|---|
| A1 uniform | 14.9% (round 2) | -0.07 .. +0.56 | not met |
| **A3 theory (decides)** | **28.6% (round 2)** | **-0.17 .. +0.55** | **not met** |
| A5 learned | 21.9% (round 2) | -0.11 .. +0.75 | met at round 1 only (15.5%, rho 0.53) |

Training progress = mean loss round 0 minus round 12 (about 2.0 in every arm).

**Reading.** There IS large run-to-run spread early (sd 10-29% of total progress in rounds
1-5), but it does not persist: where the spread is large (rounds 1-5) a seed's loss says almost
nothing about where it ends (rho about 0); where it persists (rounds 8-11, rho 0.5-0.75) the spread
is small (4-8%). Under A3 the loss also jumps in rounds 2 and 4 (its sigma = 3.95 rounds),
so the observation mostly reflects THIS round's noise, not the run's state.
An agent reacting to the proxy loss would mostly be reacting to noise.

**Decision (pre-registered): no exploitable variation -> Part 2 (A5-warmstart) NOT run.**
The script (`code/phase4b_agent.py`) is written and kept, unused.
Consequence for Phase 5: A5w does not exist, so the learned schedule used there is **A5**.
