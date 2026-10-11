# Phase 4e — Alternative allocation methods: A7, Bayesian-optimisation schedule search
Pre-registration DRAFT, written 2026-10-11 BEFORE any Phase 4e code is run. Awaiting user
approval. **A new, separate line of investigation: not part of the Phase 1-5 plan and not a
revision of Phase 4/4b/4c/4d**, whose files and results are not touched (Phase 4's stored test
runs are only READ for the paired comparison). Outputs: `results/phase4e/` only.

Question: can a non-RL method find a better complete schedule WITHOUT a round-by-round reactive
signal? (Phases 4b/4d: every server-side signal tracks the round's own injected noise.)

## 1. Method choice: Bayesian optimisation (BO), not CMA-ES
* The objective is expensive (one trial = one full simulated FL run), noisy (seed-to-seed sd of
  the simulator reward is several times the schedule effects seen in Phase 4) and the budget is
  ~100 candidate schedules. BO with a Gaussian-process surrogate is built for exactly this: it
  models the noise explicitly and is sample-efficient below a few hundred evaluations in a few
  dimensions.
* CMA-ES needs a population (~8 for 4 dimensions) and many generations (~12 with this budget);
  with this much noise its ranking-based updates are unreliable at that budget, and it has no
  noise model. It would be the better choice with thousands of cheap evaluations.

## 2. Search space (the schedule as a few parameters)
* log sigma_t is piecewise-linear in t through 5 knots at t = 0, 1/4, 1/2, 3/4, 1 of the
  schedule. Knot 1 is fixed at 0 (overall level is set by the budget anyway); knots 2-5 are the
  4 search parameters, each in [-2, +2] (late/early noise ratio up to e^2 = 7.4x either way).
* Every candidate is scaled by one global factor to the operating point's exact epsilon
  (worst client, Phase 4 `budget.scale_to_budget`, sigma clipped to [0.3, 20]). If a candidate
  cannot meet the budget inside that range, its log-shape is shrunk toward 0 in steps of x0.9
  until it can (deterministic projection).
* Contains A1 (all knots 0) and every A2 exponential exactly (knots on a straight line);
  A5's smooth decays approximately. A3's jagged schedule is NOT representable (disclosed).

## 3. Search procedure (simulator only, zero client privacy cost)
* Environment: Phase 4's proxy simulator (`fl_env.build_data('sim')`, 8 pseudo-clients from
  public proxy data, sampling rates = the real clients'), same reward as A5 (proxy-val
  mAP@0.5 of the final model), same epsilon target per operating point.
* One trial = one candidate schedule run end-to-end on 3 fixed simulator seeds (30000-30002,
  common random numbers across candidates); objective = mean reward.
* 100 candidates: 16 initial (uniform + 15 scrambled-Sobol points, seed 20261012), then 84
  sequential BO steps. GP: Matern-5/2, per-dimension length scales, learned noise variance
  (hyper-parameters by marginal likelihood, scipy). Acquisition: expected improvement over the
  best posterior mean, maximised over 5,000 random + 20 local-refined points.
* Selection (mirrors A5's checkpoint rule): the 5 candidates with the highest GP posterior mean
  are re-run on 6 fresh simulator seeds (900-905); the deployed A7 schedule is the one with the
  highest mean over those 6.
* **Compute budget**: 300 search runs + 30 selection runs = 330 simulated FL runs per operating
  point, vs A5's 300 training episodes + 36 checkpoint-evaluation episodes = 336.
* Gate G4e-a (before any real run): the BO code finds the optimum of a synthetic noisy
  4-dimensional test function (known answer) within the same 100-candidate budget; reported.

## 4. Operating points and evaluation (Phase 4 protocol)
P (R 12, eps 1.86), S1 (R 25, eps 2.72), S2 (R 12, eps 0.79), budgets exactly as Phase 4.
A7's deployed schedule is a FIXED schedule: evaluated on the real clients, TEST pool, 20 paired
seeds 200-219, common random numbers; paired against Phase 4's stored A1 / A2 / A3 (and A5)
test runs for the same seeds. Before that, a reproduction check: 3 stored Phase 4 A1 test runs
are re-run into `results/phase4e/` and must match exactly (shows the pairing is valid).

## 5. Decision rule (fixed now)
* **Primary, at P:** family A7 vs A1, A7 vs A2, A7 vs A3, Holm-corrected (alpha 0.05).
  WIN / LOSS if the Holm-adjusted test rejects (sign of the mean difference), else TIE
  (EQUIVALENT if the 95% CI lies in [-0.02, +0.02]), exactly as Phase 4.
  **A7 is a genuine improvement only if it WINS vs A1 AND vs the better of A2/A3 at P** (better =
  higher mean test mAP at P, as in Phase 4: A3), each with the 95% CI excluding zero.
* **Secondary:** the same family and rule applied separately at S1 and at S2 (Holm within each),
  reported as supporting evidence; A7 vs A5 at each point (unadjusted); the simulator gain of
  the selected schedule over uniform on the selection seeds; A7's schedule plotted against
  A2 / A3 / A5.
* Cost reported: GPU-hours and simulated runs for A7 vs A5 (and 0 for A1-A3).

## 6. Estimated time
Search + selection: ~1 h at P, ~1 h at S2, ~2 h at S1 (simulator ~10-25 s per run); test
evaluation 60 runs ~30 min; code + gate ~1-2 h. Total ~5-6 h.
