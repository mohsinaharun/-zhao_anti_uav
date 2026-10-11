# Phase 5b — FLTrust robustness checks
Pre-registration DRAFT, written 2026-10-11 BEFORE any Phase 5b code is written or run.
**Approved by the user 2026-10-11 ("Approved, run both σ levels and both variants").**
Purely additive: Phase 5 files and results are read, never changed or re-run.

## 0. What Phase 5 found, and one correction to the mechanism
Phase 5 (ε 1.86, uniform σ 2, R 12, 2 of 8 attackers): FLTrust loses to FedAvg with no attack
(0.015 vs 0.059) and under ALIE, IPM and Fang; honest cosine with the root update 0.0075 under DP
vs 0.086 without.

**Correction (measured on Phase 5 logs, no new runs):** FLTrust does not lose because it gives
honest clients low trust. Trust scores are divided by their sum, so only their *ratios* matter;
uniformly low trust changes nothing. What hurts is the other step, the **rescaling of every
update to the root update's norm**. Under DP an honest update has norm 41.4 (almost all noise),
the root update 1.06, so every update is shrunk ~39× before averaging, and the global model moves
~1/39 of a FedAvg step per round. Without DP the two norms match (0.74 vs 0.75) and nothing is
shrunk. The low cosine and the shrinkage have the same cause (noise norm ∝ σ), so Part 1 cannot
separate them; Part 2 does.

## 1. Questions
* **Q5 (Part 1).** Does FLTrust's performance relative to FedAvg change direction or size when
  privacy is looser or tighter than ε 1.86? Does the honest cosine (and the shrink factor) track it?
* **Q6 (Part 2).** Does a DP-aware FLTrust, which uses the exactly known noise level, recover
  FLTrust's no-DP behaviour (tie or beat FedAvg) at ε 1.86, or does it still lose?

## 2. Setting (Phase 5, unchanged unless stated)
Real drone clients (8), R 12 rounds, B 64, 20 local iterations, SGD lr 0.03 momentum 0.9, calibrated
C* (per seed, independent of σ), frozen MobileNetV3-Small K 9 + 65,221-parameter head, TEST pool,
**20 paired seeds 200-219**, common random numbers as Phase 4/5. Attackers c3, c6; ALIE (z 0.319),
IPM (ε_ipm 0.5), Fang (b 2), omniscient and colluding, every round; root set 1,300 frames; FLTrust root
update clipped at C*, noise 0 (Phase 5 Amendment 1). **Uniform schedule (A1) only.** Worst-client ε
at δ 1e-5 (RDP accountant, as Phase 4).

## 3. Part 1 — privacy-level sweep
| level | σ (uniform) | worst-client ε | vs ε 1.86 | used before in |
|---|---|---|---|---|
| looser-2 | **0.5** | **30.16** | 16× looser | Phase 2 σ sweep |
| looser-1 | **1** | **5.73** | 3.1× looser | Phase 2 σ sweep; Phase 3 pulse test |
| main (Phase 5, read-only) | 2 | 1.86 | — | Phase 4 P, Phase 5 |
| tighter | **4** | **0.79** | 2.3× tighter | Phase 4 S2 |
| no DP (Phase 5, read-only) | 0 | ∞ | — | Phase 5 |

**On "5-10× looser, at a σ already used":** no σ the project has used gives 5-10× (σ 1 → 3.1×,
σ 0.5 → 16×; the only σ in range, 0.75 → ε 10.8, 5.8×, has never been used). **Proposed: run both
σ 1 and σ 0.5**, which bracket the requested range from both sides and both reuse project levels.
(Alternatives for you to choose instead: σ 0.75 only, new level, 5.8×; or σ 1 only.)

Grid per new level: FedAvg, FLTrust, median, Krum × none / ALIE / IPM / Fang × 20 seeds = 320 runs;
**960 runs** for three levels. Phase 5's σ 2 and no-DP cells are reused read-only (σ 2: A1 FedAvg /
FLTrust / median / Krum; σ 0: FedAvg / FLTrust only, median / Krum were not run without DP).

**Reproduction check (before the grid):** FedAvg / no attack at σ 4 must equal Phase 4's stored
S2 uniform test runs (`eval_S2_A1_seed200-219`) exactly for all 20 seeds, else stop.

**Logged per run (as Phase 5):** per-client update norm, cosine with the root update, trust, root
norm, Krum pick, plus the **shrink factor** = root norm / mean honest norm, per round.

### Part 1 decision rules (Phase 5 conventions)
Paired over 20 seeds: mean difference, 95% t-CI, wins/losses, ε. WIN / LOSS if Holm (α 0.05)
rejects, else TIE, EQUIVALENT if the 95% CI lies within ±0.02 mAP.
* **F1-σ and F2-σ, separately at each new level:** exactly Phase 5's F1 (FLTrust vs FedAvg under
  ALIE, IPM, Fang; 3 tests) and F2 (FLTrust, median, Krum vs FedAvg without attack; 3 tests), with
  Phase 5's headline rule ("effective defence" iff ≥2 of 3 WINs and no clean LOSS).
* **Direction:** "FLTrust's direction changes at level L" iff any of its 4 comparisons with FedAvg
  (no attack + 3 attacks) at L is not a LOSS (at ε 1.86 all 4 were LOSS).
* **Size (family F5, 12 tests, Holm):** for each new level and each of the 4 attack conditions, the
  ratio of means R = mean mAP(FLTrust) / mean mAP(FedAvg), compared with the same ratio at σ 2.
  Paired bootstrap over seeds (10,000 resamples, generator seed 0; the same resample indices for both
  levels), two-sided p = 2 × min(share ≤ 0, share ≥ 0) of the ratio difference. A ratio is used
  because absolute gaps shrink mechanically when all scores fall toward 0 at σ 4.
  "The size changes" iff any test rejects; direction reported.
* **Mechanism tracks performance:** over the five levels σ 0, 0.5, 1, 2, 4, Spearman correlation of
  the no-attack ratio R with (a) the mean honest cosine and (b) the mean shrink factor. "Tracks" iff
  ρ = +1 (fully monotone; one-sided p 0.008 for n 5). Both are reported; (a) is the measure you named,
  (b) is the corrected mechanism.
* Floor caveat, fixed now: Phase 4's uniform test mAP at σ 4 is 0.014, so at σ 4 differences are
  small in absolute terms and ratios are noisy; reported as tested, flagged.

## 4. Part 2 — DP-aware FLTrust
Notation per round t: Δ_i = client i's update, g0 = root update, D = 65,221, σ_t known exactly.
**Known noise variance per coordinate of an honest update** (Opacus adds N(0, σ²C*²) to the summed
clipped gradient and divides by B; momentum 0.9 over K 20 steps):
v_t = (lr · σ_t · C* / B)² · Σ_{j=1..K} a_j²,  a_j = Σ_{k=0..K−j} 0.9^k.
Checked on all Phase 5 FLTrust runs at σ 2: observed honest norm / √(D·v_t) = **1.024** (sd 0.012,
range 1.007-1.055), i.e. the formula predicts the norm to within ~2.5% (the rest is signal and
Poisson batch-size variation). At σ 0 both variants below reduce exactly to Phase 5's FLTrust.

### Variant L ("literal": noise-corrected cosine), the version you described
Estimate each update's signal norm by removing the known noise energy, and compute the cosine
with that instead of the raw norm:
ŝ_i = √max(‖Δ_i‖² − D·v_t, 0),  c_i = clip( (Δ_i · g0) / (‖g0‖ · ŝ_i), −1, 1 )  (c_i = sign of the
dot product if ŝ_i = 0),  TS_i = ReLU(c_i). Everything else is FLTrust unchanged (rescale to ‖g0‖,
trust-weighted average).

**What I expect, and why I recommend not relying on it alone** (computed from Phase 5 logs, no runs):
* It cannot fix the shrinkage: updates are still rescaled to ‖g0‖, so the step is still ~1/39.
  Without attack it changes the normalised weights by only 0.09 (total-variation distance).
* It **rewards updates that carry less noise than DP requires**. ALIE's update is built from the
  mean of 6 honest updates, so ‖Δ‖² − D·v_t is small or negative, ŝ ≈ 0 and its corrected cosine
  jumps to ~1. Recomputed on the Phase 5 ALIE runs, the attackers' share of the total weight would
  rise from **34% to 78%**. So the literal version is predicted to be worse than FLTrust under ALIE.

### Variant N (recommended: noise-aware norm target)
One change, at the step that actually causes the failure: FLTrust rescales each update to the norm
a clean client update would have, ‖g0‖. Under DP the norm an honest client's update *should* have
is known: N_t = √(‖g0‖² + D·v_t). Variant N rescales every update to **N_t instead of ‖g0‖**:
Δ̂_i = Δ_i · N_t / ‖Δ_i‖,  aggregate = Σ TS_i Δ̂_i / Σ TS_i,  TS_i = ReLU(cos(Δ_i, g0)) unchanged.
So honest updates keep (almost exactly) their own size, and the global step is FedAvg-sized again;
attackers still cannot win by sending a large update (every update gets the same norm), and the
cosine trust (which excludes IPM and Fang, trust 0.000 in Phase 5) is kept. At σ 0, N_t = ‖g0‖.
Not chosen (disclosed): rescale-only-downward (a norm cap), or a "too clean" detector that rejects
updates whose norm is below the DP floor. Each is a second change; Variant N is the single change
that targets the measured failure.

**Proposed: run both** (L because it is what you asked to test, N as the main variant). Each:
none / ALIE / IPM / Fang × 20 seeds at σ 2 (ε 1.86) = 80 runs; **160 runs**. FedAvg and FLTrust
references are Phase 5's stored runs (identical seeds and common random numbers; read-only).
Reproduction check: the new code with the original FLTrust rule must reproduce Phase 5's stored
FLTrust / no attack runs for seeds 200-202 exactly, else stop.

### Part 2 decision rules
* **F6-N (4 tests, Holm):** Variant N vs FedAvg, no attack + ALIE + IPM + Fang. **"Recovers
  FLTrust's no-DP behaviour" iff none of the 4 is a LOSS** (each WIN or TIE), the pattern FLTrust
  showed without DP (3 ties, 1 win). "Partial" iff it is not a full recovery but beats Phase 5's
  FLTrust in at least one condition (F7-N below). Otherwise "still loses".
* **F7-N (4 tests, Holm):** Variant N vs Phase 5 FLTrust, same 4 conditions.
* **F6-L and F7-L:** the same two families for Variant L, same verdict rule.
* Secondary, no verdicts: attackers' share of total trust weight per round, honest trust by
  environment, step norm per round (aggregate update norm) vs FedAvg's.

## 5. Predictions (scored after; not decision rules)
1. FLTrust/FedAvg no-attack ratio rises monotonically as σ falls: σ 4 < σ 2 (0.26) < σ 1 < σ 0.5
   < σ 0 (1.01); both honest cosine and shrink factor track it (ρ = +1).
2. FLTrust still LOSES to FedAvg without attack at σ 1 and σ 0.5 (noise norm still ≫ root norm:
   predicted shrink ~1/20 and ~1/10). Direction changes only without DP.
3. Median and Krum still LOSE without attack at every DP level.
4. Variant L still LOSES to FedAvg without attack, and is worse than Phase 5 FLTrust under ALIE.
5. Variant N TIES FedAvg without attack and WINS under Fang (attackers get trust 0), but does not
   fully recover: it LOSES under IPM (FedAvg gains from IPM's low-noise updates, Variant N
   discards them) and under ALIE (ALIE's early trust is ~5× an honest client's, and its
   small-norm update is scaled up to N_t). Expected verdict: **partial**.

## 6. Outputs and cost
Code `code/phase5b.py` (new; imports `phase5`, `fl_env`, `budget` read-only), launcher
`code/run_phase5b.ps1` (3 workers, as Phase 5), results `results/phase5b/`, analysis
`code/analyse_phase5b.py`, report `phase5b/RESULTS.md`, plus a thesis table as in Phase 5, failures in
NEGATIVE_RESULTS.md. **1,120 runs, ~2.5-3 h** at Phase 5's pace. Any deviation: dated amendment here
before the affected run.
