# Phase 4b — follow-up investigation (addendum; NOT a revision of Phase 4)
Pre-registration, written 2026-10-10 BEFORE the variation check or any 4b training is run.
Phase 4's pre-registered result (phase4/RESULTS.md) stands unchanged and is not re-run.
Outputs: `results/phase4b/` only.

Question: is there run-to-run variation in what a feedback agent observes that it could
exploit, and if so, does A5 started from the theory schedule (A3) beat the fixed formulas?

## Part 1 — Variation check (no test data)
* Real FL setup of Phase 4 at P (R = 12, eps = 1.86), the 20 main-setting seeds 200-219,
  identical init / C* / noise streams. Per round t = 0..12 log the agent's observations:
  proxy loss, proxy gradient norm, change in proxy loss (server-side, public proxy session
  01_2192). Schedules: A1 (uniform), A3 (theory) and A5 (deployed Phase 4 policy).
  The model is NOT evaluated (no test-pool use).
* Report per round: mean, between-seed SD, min-max; one overlaid chart per schedule.
* **"Meaningful variation" (fixed now)**, judged on the A3 schedule (the warm-start's
  starting point). Both must hold at some decision round t in 1..10 (rounds where the
  agent's choice still affects later rounds):
  (a) size: between-seed SD of proxy loss at round t >= 10% of the mean total training
      progress (mean loss at round 0 minus mean loss at round 12);
  (b) persistence: Spearman correlation across seeds between proxy loss at round t and at
      round 12 >= 0.5 (a seed that is behind stays behind, so reacting to it can matter;
      pure measurement noise would not persist).
  If not met: Part 2 is NOT run; reported as "no exploitable variation".

## Part 2 — A5-warmstart (only if Part 1 meets the criterion)
* Identical to A5 (TD3 + GRU, same sizes, seed 7, proxy simulator, 300 episodes, same reward,
  same exploration schedule 0.3 -> 0.05, same update count, same checkpoint rule and G4b),
  except:
  1. **Initial policy = A3's schedule.** A3 at P expressed as actions of the same online
     mapping (share = 0.25 * 16^a); rounds 2 and 4 need share < 0.25 and are clipped to the
     minimum (sigma 3.57 / 3.38 instead of 3.95; all other rounds and total eps identical).
     Actor (and target actor) fitted by behaviour cloning (MSE) to these actions on 5
     simulator roll-outs that follow them (seeds 20000-20004, not stored in the replay buffer).
  2. **Warm-up episodes 1-20 centred on the initial policy** (A3 actions + N(0, 0.3), clipped),
     mirroring A5's warm-up which was centred on its initial (even-share) output.
  3. **Actor updates held until the end of warm-up** (critic trains as in A5 from the start),
     so the initial policy is not overwritten by an untrained critic.
* Evaluation exactly as A5 in Phase 4: TEST pool, seeds 200-219, deployed checkpoint by the
  simulator checkpoint rule; compared with Phase 4's stored A1/A2/A3/A5 runs (same seeds, CRN).
* **Decision rule** (same as Phase 4 §7): family = A5w vs A1, A5w vs A2, A5w vs A3,
  Holm-corrected (alpha 0.05). WIN / LOSS if Holm rejects (sign of mean), else TIE, labelled
  EQUIVALENT if the 95% CI lies inside [-0.02, +0.02]. "Warm-started learning helps" only if
  A5w WINS vs A1 AND vs the better of A2/A3 (A3 at P). Secondary, unadjusted: A5w vs A5.
* **For Phase 5** (learned schedule vs theory): whichever of A5 and A5w has the higher mean
  test mAP@0.5 at P is used, and the choice is stated explicitly there.
