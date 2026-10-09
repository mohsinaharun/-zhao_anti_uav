# Phase 4 — Does learned privacy-budget allocation beat uniform and fixed schedules?
Pre-registration, written 2026-10-09 BEFORE any Phase 4 code or run.

Research question (Phase 0): when does adaptive privacy-budget allocation improve the
accuracy of a DP federated drone detector over uniform and fixed-schedule allocation at
the SAME total privacy budget? Phase 3 found a_t NOT flat — later rounds' noise hurts
more (gradual rise over 12 rounds; concentrated in the last ~5 of 25) — so a
non-uniform schedule has something real to exploit. This phase tests whether it does,
and whether LEARNING the schedule adds anything over a simple formula.

## 1. Fixed setup (inherited, not re-tuned)
FL, 8 clients (`clients.json`), MobileNetV3-Small frozen through block 8 (K = 9),
65,221-parameter head, B = 64, 20 local iterations per round, SGD lr 0.03 momentum 0.9,
FedProx off (mu = 0), calibrated fixed clip bound C* (proxy median after 2 proxy
rounds, per seed; zero privacy cost), frame-level DP, delta = 1e-5, Opacus RDP accountant.
Adaptive clipping (todo step 13) is NOT used: control (a) showed per-round adaptive
clipping LOSES to the calibrated fixed bound at 6/9 sigma levels; C* is the default.

## 2. Equal-budget constraint (all arms)
A schedule assigns each round t a noise multiplier sigma_t. Every arm's schedule is
scaled so the **worst client's epsilon** (delta = 1e-5, RDP accountant over all rounds,
the client's own q_c = 64/N_c, 20 steps per round) equals the uniform arm's, to within
1% (bisection on a global scale factor). sigma_t is clipped to [0.3, 20]; a schedule that
cannot meet the budget inside that range is invalid and reported as such.
Online arms (DRL): the action sets round t's share of the REMAINING budget
(see §4); the last round spends exactly what is left, so the total is met exactly.

## 3. Operating points
* **P (primary): R = 12 rounds, budget = uniform sigma 2** (worst-client eps 1.86).
  Off floor and ceiling (uniform val mAP ~0.29), strongest measured a_t variation.
* **S1 (length): R = 25 rounds, budget = uniform sigma 2** (eps 2.72) — the
  back-loaded a_t regime.
* **S2 (tight budget / "cliff", todo suggestion 5): R = 12, budget = uniform sigma 4**
  (eps 0.79), uniform mAP ~0.09.

## 4. Arms (todo step 12)
* **A1 Uniform**: sigma_t constant.
* **A2 Tuned exponential**: sigma_t proportional to exp(-k (t-1)/(R-1)) (k > 0 = less
  noise late), scaled to budget. k chosen from {0, 0.25, 0.5, 1, 1.5, 2, 3} by mean val
  mAP over 5 TUNING seeds (100-104), separately per operating point.
* **A3 Theory schedule**: rho_t proportional to sqrt(a_t) with a_t = Phase 3 mean drop
  per round at that operating point (V1 for P/S2, V2 for S1; negative estimates floored
  at 0.1 x the median), mapped to sigma_t proportional to 1/sqrt(rho_t), scaled to budget.
  No tuning (uses Phase 3 val measurements only).
* **A4 Blind DRL** (Chen et al.-style): DDPG; state = (t/R, remaining-budget fraction);
  action in [0, 1] -> spend share s_t = 0.25 * 16^action (0.25x-4x) of
  (remaining budget / rounds left) in round t.
* **A5 Feedback DRL** (the new idea): TD3 with a GRU over rounds; state per round adds
  server-side signals computed on PUBLIC proxy data after aggregation (proxy loss, proxy
  median per-sample gradient norm, round-to-round change in proxy loss) — no client data,
  no privacy cost. Same action mapping as A4.
* A4/A5: same episode budget (300 episodes), same actor/critic sizes where shared,
  same exploration schedule. Reward = proxy-val mAP@0.5 of the final model (sparse).

## 5. Where the agents learn — privacy of agent training (suggestion 4)
Agents train ONLY in a simulator built from the PUBLIC proxy pool (11 sessions, all four
environments): the proxy frames are split into 8 pseudo-clients by session/environment,
with a held-out proxy-val split for the reward. **Agent training never touches client,
val or test data, so it costs zero client privacy.** The sim-to-real gap this creates is
reported, not hidden (proxy pool is smaller and differs in mix).
**Checkpoint rule (fixed now; this project has seen checkpoint choice flip a result):**
evaluate the policy on proxy-val every 25 episodes; the deployed policy is the one with
the best mean proxy-val reward over its 25-episode window. Checkpoint sensitivity is
reported on VAL (all checkpoints), never chosen on test.

## 6. Evaluation (todo steps 15-16)
* The **test pool** (12 sessions, untouched so far) is used here for the first time, once
  per pre-registered comparison. Train on `clients`, report test mAP@0.5.
* **20 paired seeds (200-219)**, common random numbers: seed t gives every arm the same
  init, C*, client sampling and noise stream (scaled by each arm's sigma_t).
* Per comparison: mean paired difference, 95% t-CI, win count, epsilon (identical
  across arms by construction; reported).

## 7. Decision rule (todo step 14) — fixed now
Primary family at P, Holm-corrected (alpha = 0.05 family-wise):
  (i) A5 vs A1, (ii) A5 vs A2, (iii) A5 vs A3, (iv) A4 vs A1.
For each: **WIN** if the Holm-adjusted test rejects AND the mean difference > 0;
**LOSS** if it rejects with difference < 0; **TIE** otherwise, further labelled
**EQUIVALENT** if the 95% CI lies inside [-0.02, +0.02] (pre-set equivalence margin).
* "Learned allocation helps" is claimed ONLY if A5 WINS vs A1 AND vs the best of A2/A3.
  Beating only uniform shows a schedule helps, not that learning one does.
* Secondary (not in the family; reported with unadjusted CIs): A2 vs A1, A3 vs A1, all
  pairs at S1 and S2. A "DRL ties a formula" outcome is a valid result and will be
  reported as such, with costs (§8).

## 8. Also reported (todo suggestions)
* **Cost**: GPU-hours and simulated FL episodes to train each agent vs zero for A1-A3.
* **Learned schedule vs theory** (suggestion 3): deployed sigma_t per round for A4/A5
  (mean over test seeds) overlaid on A3's sqrt(a_t) schedule and A2's.
* **Negative results**: every failed or non-replicating attempt in NEGATIVE_RESULTS.md.
* Not in this phase: second dataset (suggestion 8); comparison with Kiani et al.
  (ICLR 2025) — the paper is not in the repo; added once the PDF is provided.

## 9. Gates before evaluation
* G4a: the budget-matching code reproduces the uniform arm's epsilon to < 1% for random
  schedules (unit test), and the uniform arm reproduces a stored Phase 3 baseline exactly.
* G4b: each agent's proxy-val reward must improve over its first-25-episode window
  (95% CI) — otherwise it is reported as "did not learn" and still evaluated.
* No test-pool evaluation runs until G4a passes and this file (plus any dated
  amendments) is committed.

Estimated compute: tuning ~1 h, agent training ~4-6 h (proxy simulator), evaluation
5 arms x 20 seeds x 3 operating points ~ 4-5 h.
