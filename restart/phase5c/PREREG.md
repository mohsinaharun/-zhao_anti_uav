# Phase 5c — Variant N follow-up checks
Pre-registration DRAFT, written 2026-10-12 BEFORE any Phase 5c code is written or run.
**Approved as written by the user 2026-10-12 ("Approved as written, go ahead without stopping").**
Purely additive: Phase 5 / 5b files and results are read, never changed or re-run.

Variant N (Phase 5b): FLTrust with every client update rescaled to the norm an honest DP update is
expected to have, N_t = √(‖g0‖² + D·v_t), instead of the root update's norm ‖g0‖; trust
TS_i = ReLU(cos(Δ_i, g0)) and the trust-weighted average are unchanged. At ε 1.86 it tied FedAvg
without attack and under ALIE / IPM, and beat it under Fang.

## 1. Questions
* **Q7 (Part 1).** Does Variant N recover FedAvg-level performance at every privacy level
  (σ 0, 0.5, 1, 2, 4), or only at ε 1.86?
* **Q8 (Part 2).** Does Variant N survive an attacker who knows its rule?

## 2. Setting
Exactly Phase 5b (`code/phase5b.py`, imported read-only): 8 real drone clients, R 12, B 64, 20 local
iterations, SGD lr 0.03 momentum 0.9, per-seed C*, frozen trunk + 65,221-parameter head, TEST pool,
**20 paired seeds 200-219**, common random numbers, uniform schedule, attackers c3 + c6, root set
1,300 frames, root update clipped at C* with noise 0. Worst-client ε at δ 1e-5.

## 3. Part 1 — Variant N at five privacy levels
| σ | ε | FedAvg, FLTrust | Variant N |
|---|---|---|---|
| 0 | ∞ | Phase 5 (read-only) | **new** |
| 0.5 | 30.16 | Phase 5b (read-only) | **new** |
| 1 | 5.73 | Phase 5b (read-only) | **new** |
| 2 | 1.86 | Phase 5 (read-only) | Phase 5b (read-only) |
| 4 | 0.79 | Phase 5b (read-only) | **new** |
New runs: Variant N × 4 levels × none / ALIE / IPM / Fang × 20 seeds = **320 runs**.
At σ 0, v_t = 0, so Variant N is FLTrust up to floating-point rounding of √(‖g0‖²); it is run
anyway rather than assumed, and its agreement with Phase 5's no-DP FLTrust is reported.

**Decision rules (Phase 5 conventions):** paired mean difference, 95% t-CI, wins/losses, ε; WIN / LOSS
if Holm (α 0.05) rejects, else TIE (EQUIVALENT if the CI lies within ±0.02).
* **F8, one family per new level (4 tests each):** Variant N vs FedAvg under none, ALIE, IPM, Fang.
  "Variant N recovers at level L" iff none of the 4 is a LOSS (σ 2 is Phase 5b's F6-N: recovers).
* **Headline:** "Variant N recovers at every privacy level" iff it recovers at all five levels;
  otherwise the levels where it fails are listed.
* Secondary, unadjusted: Variant N vs plain FLTrust at each level; attackers' trust-weight share;
  mean aggregate update norm.
* Floor caveat (as Phase 5b): at σ 4 FedAvg scores 0.005-0.019, so ties there carry little information.

## 4. Part 2 — adaptive attack against Variant N
### Attacker knowledge (the strongest standard model, as in Cao et al.'s own adaptive attack on FLTrust)
The two attackers are omniscient and colluding, as in Phase 5, and in addition know Variant N's
rule, σ_t, C*, the noise formula, and the root update g0 of the round (equivalently, the root data).

### Weakness the attack targets (measured, Phase 5b logs, Variant N, no attack)
Under DP an honest update's cosine with g0 is tiny: mean 0.0065, maximum 0.027; the 6 honest clients'
trust scores sum to only **0.040 per round** (range 0.010-0.101). Variant N gives every update the
same norm N_t, so the weight of an update is decided by its trust alone. An attacker that sends an
update with cosine c ≫ 0.04 therefore takes almost all of the weight, while still looking "aligned"
with the server.

### The attack ("trust-capture"), per round
1. **Trust direction:** ĝ = g0 / ‖g0‖.
2. **Harmful direction:** h = the honest mean μ (of the 6 honest updates this round) made
   orthogonal to ĝ and reversed: h = −(μ − (μ·ĝ)ĝ) / ‖μ − (μ·ĝ)ĝ‖. This pushes against what the
   honest clients agree on, in the part the trust score cannot see.
3. **Cosine choice:** with S = Σ_honest ReLU(cos(Δ_i, g0)) (known to an omniscient attacker), choose
   c ∈ {0.01, 0.02, …, 0.99} maximising the harmful step the aggregate receives,
   f(c) = 2c·√(1 − c²) / (2c + S)   (attackers' weight share × the harmful part of their update).
   For S 0.03-0.10 this gives c ≈ 0.24-0.34 and an attacker weight share of 87-94%.
4. **Update sent by both attackers:** Δ_mal = N_t · (c·ĝ + √(1 − c²)·h); its norm is already N_t,
   so Variant N's rescaling leaves it unchanged.

Why this design: it is the simplest attack that uses all three pieces of the rule (cosine trust,
uniform norm, normalised weights). It does not need to beat a detector, because Variant N has none
beyond the cosine. Not tested (disclosed): attackers who must estimate g0 from μ instead of knowing it,
and attacks on other parts of the pipeline.

### Runs
Adaptive attack at σ 2 (ε 1.86) × **Variant N, FedAvg, plain FLTrust** × 20 seeds = **60 runs**.
FedAvg and FLTrust get the same crafted update, computed with Variant N's N_t and trust rule. They
show whether the attack is harmful in itself and whether Variant N is still at least as good as
doing nothing. No-attack references are Phase 5 / 5b runs (read-only).

**Decision rules (family F9, 3 tests, Holm):**
1. Damage to Variant N: mAP(N, no attack) − mAP(N, adaptive) > 0.
2. **Variant N vs FedAvg under the adaptive attack** (the defence question).
3. Variant N under the adaptive attack vs Variant N under ALIE (its worst Phase 5b attack): is the
   adaptive attack stronger?
* **"Variant N survives the adaptive attack"** iff test 2 is not a LOSS. **"Broken"** iff test 2 is a LOSS.
* Secondary: attackers' realised weight share and chosen c per round; FLTrust and FedAvg damage.

## 5. Predictions (scored after; not decision rules)
1. Variant N recovers at σ 0 (it is FLTrust there), σ 0.5 and σ 1. At σ 4 it ties FedAvg in all four
   conditions (floor). So "recovers at every level" is predicted to hold, with σ 4 uninformative.
2. Variant N at σ 0 matches Phase 5's no-DP FLTrust on ≥ 18/20 seeds per condition to within 1e-4
   (rounding can change later rounds).
3. The adaptive attack captures ≥ 85% of the trust weight in most rounds.
4. **Variant N is broken by the adaptive attack**: large damage (mAP falls to < 0.02) and a LOSS vs
   FedAvg. FedAvg is hurt much less (the attackers have 17% of its weight and their update has the norm
   of an honest one).
5. Plain FLTrust is hurt little (its steps are already ~1/39 of FedAvg's).

If prediction 4 holds, the honest conclusion is that Variant N fixes FLTrust's DP utility problem but
not its security under a rule-aware attacker. A defence would then need something the cosine cannot
give under DP, e.g. a cap on any single update's trust share. That would be a new design, not part
of this phase.

## 6. Outputs and cost
Code `code/phase5c.py` (new; imports `phase5b`, `phase5` read-only), launcher `code/run_phase5c.ps1`,
analysis `code/analyse_phase5c.py`, results `results/phase5c/`, report `phase5c/RESULTS.md` plus a
thesis table, failures in NEGATIVE_RESULTS.md. **380 runs, ~1 h.**
Reproduction check before the grid: this code's Variant N reproduces Phase 5b's
`variant_s2_fltrust_N_none` for seeds 200-202 exactly, else stop. Any deviation: dated amendment
here before the affected run.
