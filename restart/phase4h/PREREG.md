# Phase 4h — Relative-action DRL agent (A5-relative)
Pre-registration, written 2026-10-12 BEFORE any Phase 4h code is written or run.
**Approved as written by the user 2026-10-12 (state = A5's exact 5 inputs) ("Approved with +A and +B, go ahead without stopping").**
Added after external feedback. Purely additive: Phase 4 files, agents and results are read, never
changed or re-run.

## 1. Question
**Q10.** Does a constrained, theory-anchored learning problem succeed where the unconstrained one (A5)
failed? A5-relative may only nudge A3's theory schedule by ±20% per round.

## 2. What stays identical to A5 (Phase 4 §4-5, Amendments 1-3)
* **Learner:** TD3 + GRU, same actor/critic sizes (GRU hidden 32, MLP 64), same hyperparameters (Adam
  1e-3, τ 0.01, policy delay 2, target noise 0.1 clipped 0.2, 2R updates per episode, batch of 8
  episodes), same agent seed 7, same training-episode seeds 10000+e.
* **State, per round: A5's exact 5 inputs.** These are round t/R, remaining-budget fraction, proxy
  loss, log proxy gradient norm and change in proxy loss, read through the GRU memory. The proxy
  gradient norm is Phase 4d's signal g_t (server-side, public proxy session 01_2192, zero privacy
  cost). Phase 4d's 3-round rolling mean s_t is a function of the history the GRU already reads, so it is
  not added as a separate input; keeping the state identical isolates the one change being tested
  (the action space). Disclosed: Phase 4d found this signal fails both of its pre-registered tests
  (does not predict the outcome; tracks the round's own noise dose).
* **Training:** 300 episodes, proxy simulator only (Amendment 1 pseudo-clients, sizes equal to the
  real ones), **zero client privacy cost**. Reward is the simulator proxy-val mAP@0.5 of the final
  model. Warm-up 20 episodes with actions ~ N(0.5, 0.3) clipped to [0, 1], which here is centred on
  "no adjustment". Exploration std 0.3 → 0.05.
* **Checkpoint rule:** every 25 episodes, 3 deterministic evaluation episodes (seeds 900-902); the
  deployed checkpoint has the best mean evaluation reward. **G4b gate:** training reward of episodes
  276-300 vs 1-25, Welch 95% CI. Failing it gives "did not learn", and the agent is still evaluated.
* **One agent per operating point:** P (R 12, ε 1.86), S1 (R 25, ε 2.72), S2 (R 12, ε 0.79).

## 3. The one change: the action space
A3 is Phase 4's theory schedule at that operating point, already scaled to the budget:
σ^A3 = (σ^A3_1, …, σ^A3_R).
* **Action → adjustment:** the actor's output a ∈ [0, 1] maps to **δ_t = 0.4·a − 0.2 ∈ [−20%, +20%]**.
  δ_t > 0 means more noise this round than the plan.
* **Plan (keeps the total budget exact):** before round t, the remaining A3 values are rescaled by one
  common factor λ_t. λ_t is chosen so that the rounds already played plus λ_t·(σ^A3_t, …, σ^A3_R) meet
  the target worst-client ε exactly (bisection with the Phase 4 accountant).
  If the agent has not deviated, λ_t = 1 and the plan is A3.
* **Round t noise:** σ_t = (1 + δ_t) · λ_t · σ^A3_t, clipped to Phase 4's [0.3, 20].
  The adjustment is relative to A3's value for that round as re-planned. Any budget saved or overspent
  is spread evenly (in proportion to A3's shape) over all later rounds, not dumped on the last one.
* **Last round:** σ_R solves the exact remainder (= λ_R·σ^A3_R). The agent's last action is recorded but
  has no effect, as in A5. **Total budget met exactly** (checked to < 0.1% for every run).
* **Disclosed consequence:** the realised schedule can drift from A3 by more than 20% in later rounds
  through λ_t, if the agent saves or overspends consistently. The realised σ_t / σ^A3_t per round is
  reported for every run.

**Gate G4h (before training):** (i) an always-δ = 0 policy reproduces A3's schedule to < 1e-4 relative
per round, at all three operating points. (ii) 200 random adjustment sequences all meet the budget to
< 0.1%. (iii) The zero-adjustment policy, run on the TEST pool for seeds 200-202 at P, reproduces
Phase 4's `eval_P_A3` test mAP; any deviation is reported and the cause found before training.

## 4. Evaluation
TEST pool, **20 paired seeds 200-219**, common random numbers, P / S1 / S2. 60 new runs.
Comparators are Phase 4's stored runs (read-only): A1 uniform, A2 tuned exponential, A3 theory, A5.

**Decision rule (Phase 4 §7 conventions), one Holm family of 4 tests per operating point:**
(i) A5r − A1, (ii) A5r − A2, (iii) A5r − A3, (iv) A5r − A5.
WIN / LOSS if Holm rejects, otherwise TIE, with EQUIVALENT if the CI lies within ±0.02.
* **"Better of A2/A3"** is fixed now from Phase 4's test means: **A3 at P** (0.0736 vs 0.0731),
  **A2 at S1** (0.1105 vs 0.0802), **A2 at S2** (0.0254 vs 0.0213).
* **A5-relative is a genuine improvement at an operating point** iff it WINS vs A1 AND WINS vs the
  better of A2/A3 there (95% CI excluding zero, Holm-adjusted).
* **Headline** ("constrained learning succeeds"): genuine improvement at **P**, the primary point as in
  Phase 4. S1 and S2 get their own verdicts with the same rule and are reported alongside.
* **A5 vs A5-relative (test iv)** is reported explicitly at each point. This is the direct
  constrained-vs-unconstrained comparison.

## 5. Is the learned adjustment a real pattern? (fixed now)
From the deployed policy on the 20 test seeds, per operating point, with δ_t taken over the rounds where
the action takes effect (t < R):
* **ZERO-COLLAPSE:** |δ_t| < 0.02 in ≥ 90% of (round, seed) pairs, i.e. "learned to do nothing".
* **SATURATED:** |δ_t| ≥ 0.19 in ≥ 90% of pairs, i.e. bang-bang at the ±20% bounds, as A5 sat at the
  bottom of its range.
* **STRUCTURED** otherwise; this is the non-degenerate case.
* **Consistency:** per-round mean δ_t with 95% CI over the 20 seeds. "Consistent pattern" if at least a
  third of rounds have a CI excluding 0.
* **Feedback use:** mean over rounds of the between-seed sd of δ_t. Below 0.02 means a fixed schedule
  that ignores feedback, as A5 did. Seeds differ only through the feedback signals, so this measures
  whether the agent reacts to them.
* **Also reported:** the realised σ schedule overlaid on A3 / A2 / A5, the drift λ_t, the training curve,
  the selected checkpoint, the G4b result, and GPU-hours.

## 6. Predictions (scored after; not decision rules)
1. **Not a genuine improvement at P:** A5r ties A3 (likely EQUIVALENT); it does not beat both.
2. At P, A5r beats A5 (the anchor alone is worth ~+0.015, A3 − A5 in Phase 4).
3. The adjustment pattern is **SATURATED or STRUCTURED-but-fixed**, not ZERO-COLLAPSE. The likely
   shape is more noise early and less late (the simulator rewarded back-loading A5 to its bound).
   The between-seed sd will be < 0.02, i.e. no use of feedback (Phase 4b/4d: the signal tracks the
   dose).
4. At S1, where A3 itself lost to uniform (−0.025), A5r does not beat A1. At S2, A5r ties A2.

## 7. Cost
Training: 3 agents × 300 simulated episodes. A5 took 1.0 h (P), 1.0 h (S2), 7.8 h (S1) with 4 jobs
sharing the GPU. **Estimate 3-9 h wall-clock**, the three trained in parallel (more if the power cap
stays unset). Evaluation: 60 runs, ~15 min. A1-A3 and A5 are not re-run.

## 8. Outputs
Code `code/phase4h.py` (new; imports `phase4`, `phase4_agents`, `fl_env`, `budget` read-only), launcher
`code/run_phase4h.ps1`, analysis `code/analyse_phase4h.py`, results `results/phase4h/`,
checkpoints `cache/phase4h_ckpt/` (deployed actors copied to `results/phase4h/`), report
`phase4h/RESULTS.md` + thesis table, failures in NEGATIVE_RESULTS.md.
