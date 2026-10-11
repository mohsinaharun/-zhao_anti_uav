# Phase 5 — Security (Byzantine robustness) under DP federated training
Pre-registration DRAFT, written 2026-10-11 BEFORE any Phase 5 code is written or run.
**Awaiting the user's review and approval; nothing runs until then.** Todo items 17-19.
Phase 0 §1: security is the SECONDARY question; Phases 4-4f are not changed.

## 1. Questions
* **Q1 (defence; todo 18-19).** Under DP-FL at the drone operating point P (ε 1.86), does FLTrust
  protect the detector against attacks designed to evade detection (ALIE, IPM, Fang), compared with
  plain FedAvg?
* **Q2 (cost of defence; todo 18).** Without any attack, how much accuracy do FLTrust and the
  classical robust aggregators (coordinate-wise median, Krum) cost relative to FedAvg? Todo's
  claim to test: median / Krum discard the averaging that cancels DP noise; FLTrust keeps more.
* **Q3 (allocation as a security parameter).** Does the privacy schedule change how much an attack
  hurts? Theory schedule A3 and learned schedule A5 vs uniform A1, under FLTrust.
* **Q4 (learned vs theory under attack; Phase 4b's hand-over).** Under attack and FLTrust, does
  the learned schedule A5 do better or worse than the theory schedule A3? A5's policy reacts to the
  global model, which the attacker also moves, so its schedule may change under attack.

## 2. Setting (Phase 4, unchanged unless stated)
* Real drone clients (8: 2 per environment), operating point **P**: R 12 rounds, worst-client
  **ε 1.86** (δ 1e-5), B 64, 20 local iterations, SGD lr 0.03 momentum 0.9, μ 0, calibrated fixed
  clip C*, frozen MobileNetV3-Small (K 9) + 65,221-parameter head. Seed formulas and common random
  numbers exactly as Phase 4 (`fl_env.FLEnv`).
* **Evaluation: TEST pool, 20 paired seeds 200-219** — the same seeds as Phase 4.
* **Schedules (same ε):** A1 uniform, A2 tuned exponential (k 0.5), A3 theory ρ_t ∝ √a_t, **A5
  learned** (TD3+GRU, Phase 4's deployed checkpoint at P, run online as in Phase 4: it observes the
  global model on public proxy frames and sets each round's σ; its budget control keeps ε at 1.86
  whatever it observes).

## 3. Root set (todo 17)
FLTrust's server needs a small clean dataset. **The `root` pool fixed in Phase 0 is used**:
session-disjoint from clients, val, proxy and test since 2026-10-05 and never used so far
(2 sessions, 5,578 target frames, cached at the same stride 5 as the other pools). This closes the
earlier leak (finding M-4: the old "clean" set overlapped the test set).
**Disclosed limitation:** the root pool covers only 2 of the 4 environments (sky 4,928 frames,
terrain 650; no cloud or building). FLTrust trusts clients whose updates point like the root update,
so cloud / building clients may be down-weighted. Measured, not assumed: honest clients' trust
scores are logged per environment.

## 4. Threat model and attacks
* **2 of 8 clients malicious (25%)**, fixed: c3 (cloud, 1,300 frames) and c6 (terrain, 2,999),
  one small and one mid-sized client from different environments; they keep their real sizes
  (17% of FedAvg weight). Every round, from round 1.
* **Omniscient and colluding** (the strongest standard model): attackers see the 6 honest clients'
  updates of the round (after DP noise) and send crafted updates in place of their own. Malicious
  clients still compute their own local update first, so the random-number streams of every client
  are identical across attacks (common random numbers hold).
* Attacks, on the update Δ = local model − global model, coordinate-wise over all parameters:
  * **ALIE** (Baruch et al., 2019): Δ_mal = μ − z·s, μ / s = mean / std of the honest updates;
    z from the paper's rule for n 8, m 2: s* = ⌊n/2 + 1⌋ − m = 3, z = Φ⁻¹((n − s*)/n) = 0.319.
  * **IPM** (Xie et al., 2019): Δ_mal = −ε_ipm · μ, **ε_ipm = 0.5**.
  * **Fang** (Fang et al., 2020), full-knowledge trimmed-mean / median attack, applied in update
    space: per coordinate, if the honest mean is positive the attackers send values in
    [w_min/b, w_min] (w_min > 0) or [b·w_min, w_min] (w_min ≤ 0), else in [w_max, b·w_max]
    (w_max > 0) or [w_max/b, w_max] (w_max ≤ 0), w_min / w_max = honest min / max, **b = 2**,
    drawn uniformly from a seeded generator. Not the FLTrust-adaptive variant (disclosed).
* No-attack runs are the reference for every cell.

## 5. Aggregators
* **FedAvg**: size-weighted mean of the 8 updates (as Phase 4).
* **FLTrust** (Cao et al., NDSS 2021): the server trains a copy of the global model on the root
  set with the clients' procedure (20 iterations, B 64, lr 0.03, momentum 0.9) but without DP noise
  or clipping (its own data needs no privacy) → root update g0. Trust TS_i = ReLU(cos(Δ_i, g0));
  each Δ_i rescaled to ‖g0‖; aggregate = Σ TS_i · Δ̂_i / Σ TS_i (global model unchanged if all
  TS_i = 0). DP accounting is unaffected (post-processing).
* **Coordinate-wise median** (unweighted) and **Krum** (f = 2, select one update) as the classical
  references.

## 6. Run grid (all paired over seeds 200-219)
| block | schedules | aggregators | attacks | runs |
|---|---|---|---|---|
| main | A1, A2, A3, A5 | FedAvg, FLTrust | none, ALIE, IPM, Fang | 640 |
| classical reference | A1 | median, Krum | none, ALIE, IPM, Fang | 160 |
| no-DP reference (σ 0, clipping kept) | uniform | FedAvg, FLTrust | none, ALIE, IPM, Fang | 160 |
| **total** | | | | **960** |
The no-DP block shows whether an attack is weak in itself or masked by DP noise. ~20 s per run
(Phase 4) plus the root update: **~6 GPU-hours**.

**Reproduction check (before the grid):** FedAvg / no-attack / A1 for seeds 200-202 must equal
Phase 4's stored test values exactly, else stop. The full FedAvg / no-attack column (A1, A2, A3,
A5) is then compared with Phase 4's stored runs and must match (validates the new code path).

## 7. Decision rules (fixed now; Phase 4 conventions)
Paired over the 20 seeds: mean difference, 95% t-CI, wins/losses, ε. Within each family, Holm
(α 0.05): WIN / LOSS if Holm rejects (sign of the mean), else TIE, labelled EQUIVALENT if the 95% CI
lies within ±0.02 mAP. Damage = mAP(no attack) − mAP(attack), same seed, schedule and aggregator.
* **F1 — defence (Q1), schedule A1:** FLTrust vs FedAvg under ALIE, IPM, Fang (3 tests).
  "FLTrust defends against attack X" iff WIN. **Headline: "FLTrust is an effective defence at P"**
  iff it WINS under at least 2 of the 3 attacks AND is not a LOSS vs FedAvg without attack (F2).
* **F2 — clean cost (Q2), A1, no attack:** FLTrust, median, Krum each vs FedAvg (3 tests).
  Secondary (unadjusted): FLTrust vs median and vs Krum. "FLTrust keeps more of the averaging"
  iff FLTrust is better than both median and Krum (both CIs exclude 0).
* **F3 — allocation as a security parameter (Q3), FLTrust:** damage under A3 vs A1 and under A5
  vs A1, for each attack (6 tests, on the damage difference). "The schedule changes robustness"
  iff any test rejects; the direction is reported.
* **F4 — learned vs theory under attack (Q4), FLTrust:** A5 vs A3 final test mAP under ALIE, IPM,
  Fang (3 tests).
* **Secondary, reported without a verdict:** every FedAvg and A2 cell; median / Krum under attack;
  the no-DP block; A5's realised schedules under attack vs without; FLTrust trust scores (honest
  vs malicious, per environment, per round).

## 8. Predictions (written now, scored after; not decision rules)
1. Under DP, honest updates are dominated by noise, so their cosine with g0 is small: mean honest
   trust score < 0.2 at P (and higher in the no-DP block).
2. ALIE's crafted update is built from the MEAN of 6 noisy honest updates, so it carries ~√6 less
   noise than any single honest update: **ALIE gets a higher mean trust score than honest
   clients** under DP. This would be a DP-specific weakness of FLTrust.
3. IPM points against the honest mean, so its trust is ~0 and FLTrust neutralises it.
4. Median and Krum LOSE to FedAvg without attack (they discard the √N noise averaging).
5. Because P's test mAP is low (uniform 0.059), attack damage is bounded by a floor; absolute
   damage at P will be small even for a successful attack (the no-DP block shows the unmasked size).

## 9. Outputs and cost
Code `code/phase5.py` (new; reuses `fl_env`, `budget`, `phase4_agents` read-only), launcher
`code/run_phase5.ps1`, root-pool features via `code/cache_features.py --pools root`. Results
`results/phase5/`, report `phase5/RESULTS.md`, failures in NEGATIVE_RESULTS.md. ~6 GPU-hours plus
~1 h code and checks. Any deviation: dated amendment here before the affected run.
