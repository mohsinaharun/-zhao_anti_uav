# Paper — Step 1 (analysis) and Step 2 (outline)
Written 2026-10-12 from the `restart/` material only. **No paper text is drafted yet.** The brief
requires stopping here for approval. Open inputs are listed in §0.
Source convention: `pN` = `restart/phaseN/RESULTS.md`, `NEG` = `restart/NEGATIVE_RESULTS.md`,
`P0` = `restart/phase0/PHASE0.md`, `RR` = `restart/results/rerun/analysis.txt`,
`KIANI` = `restart/literature/KIANI2025_COMPARISON.md`.

---

## 0. Inputs still needed from you (blocking)
| # | Needed | Why |
|---|---|---|
| I1 | **Topic paragraph** (the `[one paragraph]` placeholder was not filled) | I can propose one (§1b), but the framing is yours. |
| I2 | **Venue and format** (e.g. IEEE conference, n pages, IEEEtran) | Sets the page budget (§2 assumes IEEE conference, 10 pages + references) and the LaTeX template. |
| I3 | **Audience** | Changes how much DP / FL background goes in §4. |
| I4 | **Authors and affiliations** | Title block. I will not guess these. |
| I5 | **Choice of central claim** (§1b: A, B or C) | The brief says to ask if no single claim exists. Two separate stories exist. |
| I6 | **Privacy unit**: P0 §7 still lists it as PENDING (frame-level used throughout) | Must be stated plainly. Was it ever confirmed? |
| I7 | **Earlier, pre-restart work** (the MNIST/DRL reproduction, P3-H in `p3_only/`) | Brief says "experiments in restart", so I plan to **exclude** these. Confirm. |

---

## 1a. The experiments (purpose → setup → outcome)
Common setting unless noted: Anti-UAV IR drone detection; 89 recording sessions re-split
**session-disjoint** into clients 58 / proxy 11 / test 12 / val 6 / root 2 (`data/split.json`).
8 FL clients (2 per background environment). Frozen MobileNetV3-Small trunk through block 8 + a
65,221-parameter head. DP-SGD (Opacus, Poisson sampling, RDP accountant, δ = 1e-5, frame-level,
worst-client ε). B 64, 20 local iterations, SGD lr 0.03 momentum 0.9. Per-seed clip bound C*
calibrated on the public proxy pool. 20 paired seeds (200-219) with common random numbers (CRN).
Holm within pre-registered families.

| ID | Purpose | Setup | Outcome (exact) | Source |
|---|---|---|---|---|
| E0 | Leakage-free split | Session-level re-split; official split not used | 23 of 89 sessions appear in both official train and val; all pools session-disjoint | P0 §3 |
| E1 | Model is learnable; DP live range | Single client, non-DP and σ sweep, val | Non-DP 0.584 ± 0.005 mAP (trivial 0.000). At B 1,024: σ 2 → 0.350, σ 4 → 0.104, σ ≥ 8 → ~0 | p1 |
| E2 | FL × DP 2×2 (original) | σ 2, C = 45.8, 10 seeds, val | Interaction −0.120 [−0.178, −0.062] ("DP hurts more inside FL") | p2 |
| E2r | **Retraction**: C = 45.8 was miscalibrated | Calibrated C*; re-ran E1/E2 | Interaction **reverses**: +0.135 (σ 1), +0.330 (σ 2), +0.201 (σ 4), 10/10 each | NEG Revision; RR |
| E2c | Adaptive per-round clipping vs calibrated fixed C* | 10 seeds | Adaptive loses at 6 σ levels, e.g. σ 2: −0.112 [−0.169, −0.055], 2/10 | NEG |
| E2f | FedProx μ sweep | σ 1, 2 | No μ reliably helps; μ 1 at σ 2 −0.050, 1/10 | RR R3 |
| E3 | Is round sensitivity a_t flat? (noise-pulse test) | 3 variants × σ 1/2/4, 20 seeds, 3,120 runs | 7/9 cells VARY, 2 not identifiable, 0 flat. a_t always rises late | p3 |
| E4 | **Main**: learned vs fixed vs uniform allocation | A1 uniform, A2 exp. (k tuned on val), A3 theory ρ_t ∝ √a_t, A4 DDPG, A5 TD3+GRU; P (R 12, ε 1.86), S1 (R 25, ε 2.72), S2 (R 12, ε 0.79); test pool | P: A1 0.0592, A2 0.0731, A3 0.0736, A4 0.0000, A5 0.0587. A2−A1 +0.0139 [+0.0076, +0.0202] 16/4. A5−A1 −0.0005 (equivalent). A5 LOSES to A2/A3. A3 LOSES to A1 at S1 (−0.0248) | p4 |
| E4b | Is there run-to-run variation a feedback agent could exploit? | Proxy-loss replays | No; criterion not met → A5-warmstart not run | p4b |
| E4d | Alternative feedback signal (gradient norm) | Replays | Fails both tests (ρ ≤ 0.30; lockstep r = +0.68) | p4d |
| E4e | Bayesian-optimisation schedule search (A7), same compute as A5 | Test | P: beats A1 (+0.0155, Holm 0.019) but ties A2/A3 → not genuine. Loses to A2/A3 at S2 | p4e |
| E4c | Replication on CIFAR-10 (and VOC) | Same clients/ε | CIFAR-10: a_t same shape, A2−A1 +0.0050 (Holm 0.088, equivalent), +1.7%. VOC: stopped at quality gate (0.058-0.074) | p4c |
| E4f | Replication on VisDrone (car) | Same ε | A2−A1 +0.0037 [+0.0020, +0.0054] 17/3, Holm 0.0006, +10.5%. A3 ties. Kiani A6 wins | p4f |
| E4g | Do more rounds close the DP gap? | A2/A1/no-DP at R 12/25/50/100, ε 1.86 | A2 0.073/0.086/0.057/0.040. No-DP 0.250/0.321/0.340/0.330. Noise ceiling; 50−25 LOSS | p4g |
| E4h | DRL constrained to ±20% around A3 (A5r) | Exact budget | Not genuine at P/S1/S2. Ties A2/A3 at P; LOSS vs A2 at S1. G4b fails everywhere | p4h |
| E5 | Do robust aggregators work under DP? | FedAvg / FLTrust / median / Krum; ALIE, IPM, Fang; 2/8 attackers; ε 1.86 | FLTrust LOSES to FedAvg clean (−0.0441, 2/18; −75%) and under every attack. Honest cosine with root 0.0075 | p5 |
| E5b | Across privacy levels; DP-aware variants | σ 0.5/1/4; Variants L and N | FLTrust loses at every DP level. **Variant N** (rescale to √(‖g0‖²+Dv)) recovers at ε 1.86 (3 ties + Fang WIN +0.026). Variant L fails | p5b |
| E5c | Variant N at 5 levels; rule-aware attack | σ 0-4; trust-capture attacker | Recovers at σ 0/0.5/2; LOSES at σ 1 (IPM −0.030) and σ 4 (ALIE −0.011). **Adaptive attack breaks N**: 0.0005, LOSS vs FedAvg −0.013 [−0.023, −0.003], 1/18 | p5c |
| E5d | Per-client trust-weight cap | κ = 2/K, 1.5/K, 1/K; + cap-aware attacker | Cap holds attackers at 50/37.5/25%, but the model still collapses (0.0004/0.0000/0.0058). Not a fix at any margin; free otherwise (κ 1/K WINS under ALIE +0.031) | p5d |
| Eng | Discarded implementation attempts | Agent spend-unit bug; GRU misalignment; simulator redesign; unclipped root update diverged | Logged and fixed by dated amendments before the reported runs | NEG; p4 PREREG; p5 Amend. 1 |

## 1b. Central claim: no single one exists. Three candidates (please choose: I5)
The material has two halves (allocation E3-E4h; security E5-E5d) joined only by the setting.

* **A (allocation only).** *"In DP federated drone detection at strong privacy (ε ≈ 0.8-2.7), all
  the measurable benefit of privacy-budget allocation comes from a fixed schedule that adds less
  noise in later rounds (+24% mAP over uniform at ε 1.86); learned, searched and constrained-learned
  schedules add nothing, and more rounds make it worse."*
  The security half becomes a secondary section, or is dropped.
* **B (security only).** *"Under DP, trust-based robust aggregation fails because it treats noisy
  updates as clean: FLTrust loses to FedAvg at every ε from 30 to 0.79. A noise-aware norm fixes
  utility but not security, because a rule-aware attacker collapses it, and weight caps cannot stop
  that."*
* **C (unified, recommended).** *"At strong privacy, an honest DP update is almost entirely noise
  (cosine with the clean direction ≈ 0.0075 at ε 1.86). This one fact explains both what helps and
  what breaks: allocation gains come only from where noise is placed, with nothing for a feedback
  policy to learn, and trust-based defences either drown the honest signal (FLTrust) or let a
  coherent attacker dominate (Variant N, weight caps)."*
  Risk: C links the halves by a mechanism measured in E5/E4b/E4d, but the allocation half's gain
  (E4) is explained by a_t (E3), not by the cosine. Wording must not overstate the link.

## 1c. Support / contradict / inconclusive (for candidate C)
**Supports**
* E3 (a_t rises late in 7/9 cells), E4 (A2/A3 +0.014 at P and S2), E4f (VisDrone WIN): placement of noise matters.
* E4b, E4d (no exploitable feedback signal; signal tracks the round's own noise dose), E4 (A5 ties
  uniform, ignores its inputs), E4e, E4h: nothing for learning to add.
* E5 / E5b (cosine 0.0075; shrink 1/38.6; FLTrust/FedAvg tracks σ with Spearman +1), E5c, E5d.
* E4g (noise ceiling at fixed ε).

**Contradicts or complicates**
* E4c CIFAR-10: same a_t shape but no significant gain (+1.7%, equivalent). "Placement matters" is
  dataset-dependent; a_t is necessary but not sufficient.
* E4 S1 (R 25): A2 not significant (+0.0055, 14/6); A3 LOSES to uniform (−0.0248).
* E4h: A5r's adjustments use feedback (between-seed sd 0.05-0.085). This contradicts "nothing to
  react to" in its strong form. The reaction just does not help.
* E5c: Variant N fails at σ 1 and σ 4 under standard attacks, so it is not a clean utility fix either.
* E5 IPM *helps* FedAvg under DP (+0.019): an attack-as-denoiser effect. Unexpected, and it fits C
  only partially.

**Inconclusive**
* E4f: classification "drone-like" passes the 10% line by 0.5 points (CI +5.6% to +15%).
* E5d κ = 1/K: (a1) +0.0054 (8/3, Holm 0.37), so not significantly better than no cap.
* E2f FedProx: one-seed selection originally, no reliable effect after calibration.
* E4g last step (100 − 50): −0.017, Holm p 0.056.

## 1d. Gaps a reviewer would attack
1. **Privacy unit is frame-level** (P0 §7 PENDING). Frames in a recording are highly correlated, so
   a recording-level guarantee is much weaker (group privacy). The ε values look strong (1.86) but
   protect single frames.
2. **Hyper-parameter / schedule tuning privacy is not accounted.** A2's k was tuned on val, and A3's
   a_t was measured from 3,120 Phase 3 runs trained on **client data**. The privacy cost of that
   selection is not in the ε reported. A reviewer will cite private hyper-parameter tuning work.
3. **Test-pool reuse.** The same 12-session test pool is used by Phases 4, 4e, 4g, 4h, 5, 5b, 5c, 5d.
   This is mitigated by pre-registration and fixed rules, but there is no global multiplicity
   correction across phases (Holm is within families only).
4. **Low absolute accuracy and a val→test shift.** Uniform at P scores 0.059 on test vs ~0.29 on val.
   The test pool holds many tiny targets (median box 110 px² vs 621 val; 48% of frames < 100 px²;
   p4 Context). Gains are +0.014 mAP in absolute terms.
5. **Small trainable model** (65,221 params, frozen trunk). The DP regime and every effect may be
   specific to small D. E3's V3 (116k) is near the floor.
6. **Single primary dataset for security** (drone only), one attacker set (c3 + c6, 17.1% weight),
   fixed 2/8 fraction. The root set covers 2 of 4 environments (sky, terrain).
7. **Omniscient adaptive attacker** (knows g0). Realistic attackers estimating g0 are untested. Also
   no adaptive attacks on FedAvg-specific defences.
8. **Closest prior method not reproduced on its own benchmark.** Kiani et al.'s rule (A6) was
   implemented in *our* setting (CIFAR-10, VisDrone) via σ rather than client subsampling (KIANI
   caveat), not on MNIST/FMNIST/Adult at client-level DP. **[MISSING]** if the venue requires it.
9. **Simulator for agents** (pseudo-clients from the public proxy pool) and the sim-to-real gap.
   Two agent implementations were discarded for bugs before the reported runs.
10. **A3 depends on noisy a_t estimates** (fails at S1 and on VisDrone). The theory schedule is
    only as good as its measurement.
11. **No DP-native robust aggregators as baselines** (only FedAvg / FLTrust / median / Krum + our variants).
12. **Retraction E2→E2r.** The original FL × DP interaction reversed sign after the clip bound
    was calibrated. It must be reported, which also invites "what else was miscalibrated?".
13. **Phase 2 FedProx μ = 1** was selected on one seed (0.02 margin).
14. **"Public" proxy/root pools** come from the same dataset distribution, a strong assumption.

---

## 2. Outline (assumes IEEE conference, 10 pages + references; to adjust after I2)
Title candidates (for C): *"Mostly Noise: Privacy-Budget Allocation and Trust-Based Robustness in
Differentially Private Federated Drone Detection"* or *"Where to Put the Noise, and Why Trust Fails:
DP Federated Learning at Strong Privacy"*.

| § | Content | Pages | Figures / tables |
|---|---|---|---|
| Abstract | Problem, gap, method, 4 numbers, 1 limitation, takeaway | 0.2 | — |
| 1 Introduction | Drone-detection scenario (shared IR footage across operators); gap: allocation and robust aggregation are studied at weak privacy (ε 10-30) and without paired statistics; contributions C1-C5 with numbers; scope and limitations | 1.0 | Fig. 1: one-panel teaser (honest cosine vs σ, and A2 vs uniform) |
| 2 Related work | DP-SGD and accounting; DP-FL; time-adaptive allocation (Kiani et al.; DRL allocation); Byzantine-robust FL (Krum, median, FLTrust) and attacks (ALIE, IPM, Fang); robustness × DP. What we do NOT contradict: Kiani's direction (save early) | 0.8 | — |
| 3 Preliminaries & theory | System model Δ_i = clip-sum + N(0, σ²C²I); weighted-variance objective V = Σ a_t σ_t² → **Prop. 1** ρ_t ∝ √a_t (Lagrange, zCDP approximation); **Prop. 2** the honest cosine ≈ ‖ḡ‖/√(‖ḡ‖²+Dv) and FLTrust's shrink factor; Variant N target. Assumptions A1-A4; "when it does not apply" (subsampled RDP non-additivity; noisy a_t; non-stationary signal). **Predictions** stated before results | 1.2 | Table I: notation |
| 4 Experimental design | Data and session-level split (leakage check, 23/89 sessions); pools; model (65,221 params); DP config; operating points P/S1/S2; arms; attacks; metrics; stats (paired t-CI, Holm, ±0.02 equivalence); 20 CRN seeds; pre-registration and logging; hardware; forced choices (B, C* calibration, simulator redesign) | 1.3 | Table II: full configuration; Table III: operating points with ε |
| 5 Results — allocation | 5.1 Sanity: DP live range and calibrated C* (incl. retraction). 5.2 a_t is not flat (E3). 5.3 Main comparison (E4). 5.4 Learning adds nothing: A5, A7, A5r, feedback signals (E4b/d/e/h). 5.5 Generalisation: CIFAR-10, VisDrone, VOC gate (E4c/f). 5.6 Rounds at fixed ε (E4g) | 2.0 | Fig. 2: a_t profiles (drone, CIFAR, VisDrone); Table IV: all methods × P/S1/S2; Fig. 3: mAP vs R with ceiling |
| 6 Results — security | 6.1 FLTrust fails under DP at every ε (E5/E5b) + mechanism. 6.2 Variant N repair (E5b) and its limits across ε (E5c). 6.3 Adaptive attacker (E5c) and the weight cap (E5d) | 1.6 | Fig. 4: FLTrust/FedAvg vs σ with cosine and shrink; Table V: Variant N × level; Table VI: trust-capture × cap |
| 6.4 Failed predictions, null results, retractions | All pre-registered predictions that failed (listed in §1c), the E2 retraction, discarded attempts | 0.4 | Table VII: prediction scorecard |
| 7 Discussion | Joint reading; numbered recommendations (each tied to a result); relation to Kiani et al. and FLTrust; trade-offs | 0.6 | — |
| 8 Limitations | Bullets from §1d | 0.3 | — |
| 9 Open problem | Defending against a coherent attacker when honest updates are mostly noise: why weight-based defences cannot work (5d), and why a direction-based one needs a signal DP removes | 0.2 | — |
| 10 Conclusion | One paragraph; practical advice; scientific claim | 0.2 | — |
| References | Verified entries only | 0.5+ | — |

Appendix (if allowed): per-seed values for every headline comparison (n = 20), full Holm tables,
agent training curves, A2 tuning sweep.

**Figures to regenerate (vector, consistent style).** Existing PNGs to adapt:
`plots/phase4g_rounds.png`, `plots/phase4f_visdrone.png`, `plots/phase5b_fltrust.png`,
`plots/phase5c_variantN.png`, `plots/phase5d_trust_cap.png`, `restart/figures/phase3_V1.png`.
New: the Fig. 1 teaser, and the per-seed paired-difference strip plots for the headline comparisons.

## 3. References I can cite now vs need to verify
* **Verified locally:** Kiani et al., ICLR 2025 (PDF in `restart/literature/`; KIANI notes the wrong
  proceedings link to avoid).
* **Named in the PREREGs, need [VERIFY] before BibTeX** (standard papers, but I will check each
  bibliographic detail rather than write from memory):
  - DP-SGD (Abadi et al.), RDP (Mironov), Opacus, FedAvg (McMahan et al.), FedProx;
  - FLTrust (Cao et al., NDSS 2021), Krum (Blanchard et al.), coordinate-wise median (Yin et al.),
    ALIE (Baruch et al.), IPM (Xie et al.), Fang et al. local model poisoning;
  - TD3, DDPG, MobileNetV3;
  - the Anti-UAV dataset (3rd challenge), VisDrone, CIFAR-10, Pascal VOC, pycocotools/COCO;
  - Holm (1979).
  I will verify each with a web lookup when drafting and mark anything I cannot confirm.
