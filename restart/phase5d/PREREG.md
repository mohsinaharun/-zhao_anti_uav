# Phase 5d — Per-client trust-weight cap on Variant N
Pre-registration DRAFT, written 2026-10-12 BEFORE any Phase 5d code is written or run.
**Approved by the user 2026-10-12 WITH both proposed additions (+A) cap-aware attacker and (+B) m = 1 ("Approved with +A and +B, go ahead without stopping").**
New addition, not a revision. Phase 5 / 5b / 5c files and results are read, never changed or re-run.

## 1. Question
**Q11.** Phase 5c Part 2 found that the trust-capture attacker, which knows Variant N's rule and the
root update g0, collapses Variant N at ε 1.86. Mean mAP was 0.0005; 18/20 seeds scored 0.000 to three
decimals (16 exactly 0). The attackers took ~66% of the trust weight. Variant N gives every update the
same norm, so trust alone decides weight. Honest DP updates have tiny cosines with g0, so a moderately
aligned malicious update takes most of the weight.

Does capping any single client's share of the normalised trust weight stop this attacker without
undoing Variant N's accuracy recovery (Phase 5b: ties FedAvg with no attack, under ALIE and IPM, and
beats it under Fang)?

## 2. Setting (Phase 5c Part 2, unchanged)
`code/phase5c.py` / `phase5b.py` / `phase5.py`, imported read-only. 8 real drone clients (K = 8), R 12,
B 64, 20 local iterations, SGD lr 0.03 momentum 0.9, per-seed C*, frozen trunk + 65,221-parameter head.
**σ 2 every round (uniform), worst-client ε 1.86** at δ 1e-5. TEST pool, **20 paired seeds 200-219**,
common random numbers. Attackers c3 + c6 (17.1% of the training data, so 17.1% of FedAvg's weight).
Root set 1,300 frames, root update clipped at C* with noise 0.

## 3. The capping mechanism (for review)
Variant N is unchanged up to the trust weights. Each update Δ_i is rescaled to the expected honest DP
norm N_t = √(‖g0‖² + D·v_t), giving Ũ_i. Its trust is ts_i = ReLU(cos(Δ_i, g0)). Variant N then
aggregates Σ_i w_i Ũ_i with w_i = ts_i / Σ_j ts_j.

**Cap:** κ = m / K, with **m = 1.5 (κ = 0.1875) and m = 2 (κ = 0.25)**, i.e. margins 0.5/K and 1.0/K
as requested. Per round:
1. If Σ ts = 0: no update (as Variant N).
2. w_i = ts_i / Σ_j ts_j.
3. **If max_i w_i ≤ κ, the cap does not bind and the aggregate is computed with Variant N's own
   expression, verbatim.** Results are then bit-identical to Variant N.
4. Otherwise, **water-filling.** Let F (capped clients) start empty. Repeat:
   (i) add to F every client not in F with w_i > κ, and set w_i = κ for all i in F;
   (ii) give the remaining mass r = 1 − κ·|F| to the uncapped clients with positive trust,
   P = {i ∉ F : ts_i > 0}, in proportion to their trust: w_i = r · ts_i / Σ_{j∈P} ts_j.
   Stop when no weight exceeds κ, or P is empty.
5. **Not enough positive-trust clients.** If fewer than 1/κ clients have positive trust (fewer than
   5.33 at m 1.5, fewer than 4 at m 2), every one of them is capped and Σ w_i = κ·n_pos < 1. **The
   unassigned mass is dropped: the round's step shrinks.** It is not handed to zero-trust clients
   (that would undo FLTrust's ReLU), and the weights are not renormalised (that would undo the cap).
6. Aggregate = Σ_i w_i Ũ_i.

Properties: no client exceeds κ; trust order is preserved; zero-trust clients still get 0; the
mechanism is identical to Variant N when the cap does not bind.
Logged per round: the uncapped and capped weights, n_pos, Σ w, whether the cap bound, the attackers'
share, the attacker's c and S.

**Points for review.**
* Step 5 (shrink, not renormalise) is a design choice; alternatives are named above.
* With 2 colluding attackers, a per-client cap still lets the pair take up to 2κ: **37.5% at m 1.5 and
  50% at m 2**. Both are above their 2/8 headcount share and well above their 17.1% FedAvg share.
  A per-client cap cannot push a coalition below |coalition|·κ.

**Offline replay of the logged trust scores (Phase 5b / 5c, before any run).** This is exact only for
round 1, since later rounds depend on the changed model, so it is an estimate only:

| Condition | Cap binds (m 1.5 / m 2) | Mean Σw, minimum (m 1.5) | Attackers' share uncapped → m 1.5 / m 2 |
|---|---|---|---|
| no attack | 96% / 60% of rounds | 0.99, 0.75 | — |
| ALIE | 97% / 89% | 0.84, 0.19 | 0.09 → 0.08 / 0.08 |
| IPM, Fang | 100% / 84% | 0.95, 0.56 | 0.00 → 0.00 / 0.00 |
| trust-capture | 100% / 90% | 1.00, 1.00 | **0.66 → 0.38 / 0.50** |

So **the cap changes Variant N even without an attack.** One honest client usually holds 28-38% of the
trust weight, because the honest cosines are near zero and noisy. Test (b) below is therefore a real
test.

## 4. Runs
Capped Variant N, **m ∈ {1.5, 2} × {no attack, ALIE, IPM, Fang, trust-capture} × 20 seeds = 200 runs.**
The trust-capture attack is Phase 5c's exactly. It picks c as in 5c, i.e. it does **not** know about
the cap (see +A).

Comparators, read-only, same seeds:
* Uncapped Variant N under none / ALIE / IPM / Fang: Phase 5b `variant_s2_fltrust_N_*`.
* Uncapped Variant N under trust-capture: Phase 5c `adaptive_s2_fltrust_N_adaptive`.
* FedAvg under trust-capture: Phase 5c `adaptive_s2_fedavg_adaptive`.
* FedAvg under the four standard conditions: Phase 5 `main_A1_fedavg_*` (secondary).

**Reproduction check (before the grid; stop on failure):** with the cap disabled (κ = 1), this code
reproduces Phase 5b's `variant_s2_fltrust_N_none` and Phase 5c's `adaptive_s2_fltrust_N_adaptive`
bit-identically on seeds 200-202.

### Proposed additions (NOT in the request; included only if you say yes)
* **(+A) Cap-aware attacker** (2 margins × 20 seeds = 40 runs, ~8 min). Same attacker, but it knows the
  cap. Its best reply is the smallest c that still reaches the cap, which leaves a larger harmful
  component √(1 − c²). It searches the same c grid, scoring each c by (attackers' capped weight) ×
  √(1 − c²) under the exact water-filling. Expected to be only slightly stronger (√(1 − c²) goes from
  ~0.95 to ~1.0). It closes the objection "the attacker did not adapt to the defence". **Recommended.**
* **(+B) Tightest cap, m = 1** (κ = 1/K, 5 conditions × 20 seeds = 100 runs, ~20 min). This is the
  limit of any per-client cap: every positive-trust client gets at most 1/8. It shows whether
  tightening further helps. The coalition still gets up to 25% > 17.1%. Optional.

## 5. Decision rules
Phase 5 conventions: paired mean difference, 95% t-CI, wins/losses, ε reported. WIN / LOSS if Holm
(α 0.05) rejects, else TIE (EQUIVALENT if the CI lies within ±0.02).

**Family F12-m, one per margin (Holm over 6 tests):**
* (a1) capped N (trust-capture) − uncapped N (trust-capture). **Must be a WIN.** The collapse is
  prevented and the mAP is clearly above Variant N's 0.0005.
* (a2) capped N (trust-capture) − FedAvg (trust-capture). **Must not be a LOSS.** The defended system
  is no worse than no defence under the same attack.
* (b1-b4) capped N − uncapped N under no attack, ALIE, IPM, Fang. **None may be a LOSS.**

**The cap at margin m is a genuine fix iff (a1) is a WIN AND (a2) is not a LOSS AND none of (b1-b4)
is a LOSS.** Each margin gets its own verdict. If both pass, both are reported. The tighter cap is
preferred only if its (b) tests are also all EQUIVALENT.

Reported alongside, not decision rules:
* "Close to Variant N's undefended performance": capped N under trust-capture as a fraction of uncapped
  N's mean over the other four conditions, and vs capped N under ALIE.
* Seeds at 0.000; the attackers' realised share per round; how often the cap binds; Σw; capped N vs
  FedAvg under none / ALIE / IPM / Fang (does the recovery against FedAvg still hold?).
* Power caveat for (b): a TIE with a wide CI passes (b). The CI and the EQUIVALENT label are always
  shown.
* If (+A) runs: the same 6-test family with the cap-aware attacker in (a1) / (a2), reported as a
  separate verdict ("fix survives a cap-aware attacker"). FedAvg's comparator stays Phase 5c's attack.
  The cap-aware attack is the same as Phase 5c's when there is no cap.

## 6. Predictions (scored after; not decision rules)
1. The cap binds in ≥ 90% of rounds without attack at m 1.5, and in roughly 50-70% at m 2.
2. **(a1) WIN at both margins:** the collapse is prevented (mean mAP > 0.01, fewer than 5/20 seeds at 0.000).
3. The attackers' realised share sits at about 2κ (0.375 / 0.50) in most rounds, always above 0.171.
4. **m 2 is not a genuine fix:** (a2) is a LOSS vs FedAvg, because the attackers get ~3× their FedAvg weight.
5. **m 1.5 is a narrow genuine fix:** (a2) TIE, and no (b) LOSS. The ALIE comparison is negative but a
   TIE, because the step shrinks when few clients have positive trust.
6. Even where the collapse is prevented, capped N under trust-capture stays below half of Variant N's
   no-attack mAP (< 0.036): a mitigation, not a full recovery.
7. If (+A) runs: the cap-aware attacker is at most slightly stronger (difference within ±0.01).

## 7. Outputs and cost
Code `code/phase5d.py` (new; imports `phase5c`, `phase5b`, `phase5` read-only), launcher
`code/run_phase5d.ps1`, analysis `code/analyse_phase5d.py`, results `results/phase5d/`, report
`phase5d/RESULTS.md` + thesis table, failures in NEGATIVE_RESULTS.md.
**200 runs, ~35-40 min** with 3 workers (Phase 5c: 380 runs ≈ 1 h). With (+A): +40 runs (~8 min);
with (+B): +100 runs (~20 min). GPU power cap still unset; the software throttle applies.
