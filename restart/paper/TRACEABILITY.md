# Traceability: every number in `paper.tex` → its source
Paths are relative to `restart/`. `pN` = `phaseN/RESULTS.md`; raw runs are in `results/phaseN/`.
Per-seed checks for Fig. 6 are printed by `paper/make_figures.py`.

## Abstract and Introduction (same numbers in the contributions list)
| Claim | Value | Source |
|---|---|---|
| Worst-client ε range of operating points | 0.79-2.72 at δ 1e-5 | p4 (P 1.86, S1 2.72, S2 0.79) |
| Honest cosine with the root update at ε 1.86 | 0.0075 | p5 "Mechanism"; p5b mechanism table |
| Honest vs root norm at ε 1.86 | 41.4 vs 1.06 | p5 "Why FLTrust fails" |
| Cosine across σ 0 / 0.5 / 1 / 2 / 4 | 0.086 / 0.027 / 0.015 / 0.0075 / 0.0033 | p5b mechanism table |
| A2 − uniform at P | +0.0139 [+0.0076, +0.0202], 16/4 | p4 primary table; Fig. 6 printout |
| Relative gain | 23.5% (0.0592 → 0.0731) | phase4f/THESIS_TABLE.md Table X |
| a_t not flat | 7/9 cells vary, 0 flat | p3 cell verdicts |
| A4 / A5 at P | 0.000 / −0.0005 vs uniform | p4 |
| Rounds at ε 1.86 | A2 0.073 / 0.086 / 0.057 / 0.040; no-DP 0.33-0.34; G 29% → 12% | p4g main table |
| FLTrust clean loss | −0.0441 [−0.0596, −0.0286], 2/18, −75% | p5 F2 |
| FLTrust across ε | loses at 30.16 / 5.73 / 1.86 / 0.79 | p5b F1/F2 table |
| Variant N under trust-capture | 0.0005; vs FedAvg −0.013 [−0.023, −0.003], 1/18 | p5c; phase5c/THESIS_TABLE.md W′ |
| Cap 1/8 | attackers 25%, model 0.0058 | p5d main table and round statistics |
| Uniform test mAP at P | 0.059 | p4 |
| Kiani et al. regime | ε 10-30, three-run averages | literature/KIANI2025_COMPARISON.md |

## Preliminaries and Theory
| Claim | Value | Source |
|---|---|---|
| D | 65,221 | p1 header |
| K, B, lr, momentum | 20, 64, 0.03, 0.9 | code/fl_env.py; p3 header |
| v_t formula | as Eq. (1) | code/phase5b.py `noise_var` |
| Proposition 1 | derivation | this paper (Cauchy-Schwarz); A3 definition in phase4/PREREG.md §4 |
| Post-hoc 2.5% threshold | 1.06 / (41.4 + 1.06) ≈ 0.025 | computed from p5 norms; labelled post hoc |
| H3 √R/σ | 1.73, 1.85, 1.91, 1.94 | phase4g/PREREG.md §7 |
| H4 honest trust < 0.2 | prediction | phase5/PREREG.md §8 prediction 1 |

## Experimental Design
| Claim | Value | Source |
|---|---|---|
| Folders, frames, target frames | 200, 231,557, 216,210 | phase0/PHASE0.md §3 |
| Sessions; official-split overlap | 89; 23 of 89 | phase0/PHASE0.md §3 |
| Pools (sessions) | clients 58, proxy 11, test 12, val 6, root 2 | data/split.json (counted) |
| Root set | 1,300 frames; sky + terrain only | phase5/PREREG.md implementation note; split.json |
| Client sizes | 1,300-6,162 frames | code/fl_env.py REAL_SIZES |
| Attackers' data share | 17.1% | phase5d/PREREG.md §2 (computed from sizes) |
| ALIE z / IPM ε / Krum f | 0.319 / 0.5 / 2 | code/phase5.py |
| C* mean | 5.15-5.50 | NEGATIVE_RESULTS Revision (5.15); p3 (5.50) |
| Adaptive clipping vs C* | −0.112 [−0.169, −0.055], 2/10 at σ 2 | NEGATIVE_RESULTS Revision |
| C = 45.8 clipped 0.3-0.5% | — | NEGATIVE_RESULTS Revision |
| Simulator reward as registered | 0.002-0.004 | phase4/PREREG.md Amendment 1 |
| 80/80 bit-identical | — | p5 "Check" |
| Hardware | RTX 5090, torch 2.7.1+cu128 | `meta` field of results/phase1, phase4, phase4g, phase5d JSONs |
| Run times | ~21-27 s (R 12), 195 s (R 100) | `seconds` in eval_P_A1_seed200, grid_m1_none_seed200, grid_A2_R100_seed200 |

## Results I (allocation)
| Claim | Value | Source |
|---|---|---|
| Single-client non-DP | 0.584 ± 0.005; trivial 0.000 | p1 G1 |
| Spearman σ vs mAP | −1.000 | p1 |
| Clipping cost | 0.576 vs 0.573 | p1 |
| Central σ 0 → σ 2 (C*) | 0.562 → 0.066 | results/rerun/analysis.txt R1 |
| FL+DP σ 2 (C*) | 0.286 | rerun R2 |
| Gradient norm 26.8 → 9.4 | — | results/phase3/analysis_V1.txt σ 2 |
| Pulse at P | drop +0.014 → +0.187; base 0.258; slope +0.0100 [+0.0088, +0.0112] | analysis_V1.txt σ 2 |
| Pulse runs | 3,120 | p3 header |
| Fig. 2 normalised profiles | printed | make_figures.py output |
| A3 − uniform P | +0.0144 [+0.0026, +0.0262], 16/4 | p4 |
| A5 − A2, A5 − A3 | −0.0144 (Holm 0.011), −0.0149 (Holm 0.0013) | p4 |
| S2 A2/A3 − uniform | +0.0111, +0.0071 | p4 secondary |
| S1 A2 − A1, A3 − A1 | +0.0055 [−0.0005, +0.0114]; −0.0248 [−0.0371, −0.0125] | p4 secondary |
| Table IV, all cells | — | phase4f/THESIS_TABLE.md Table X.2 (← p4, p4e, p4h) |
| VisDrone | +0.0037 [+0.0020, +0.0054], 17/3, Holm 0.0006, +10.5%; CI +5.6% to +15%; A6 +0.0032 | p4f |
| CIFAR-10 | +0.0050, Holm 0.088, +1.7%; A6 loses to A2 19/20 | p4c |
| VOC | 0.058-0.074; oracle 0.990 | p4c |
| Proxy-loss spread | 28.6%; 4-8% late; ρ 0.5-0.75 | p4b |
| Gradient-norm signal | best \|ρ\| 0.30; r +0.68 [0.51, 0.85] | p4d |
| A5 schedule invariant | max σ difference 2e-6 | p5 F4 caveat |
| A7 | +0.0155 [+0.0050, +0.0261] Holm 0.019; ties +0.0011 / +0.0016; S2 sim −0.050 | p4e |
| A5r | −0.0028 vs A3; −0.0184 vs A2 at S1, 3/17; sd 0.051 | p4h |
| Rounds | all values in §V-G | p4g |

## Results II (robustness)
| Claim | Value | Source |
|---|---|---|
| FLTrust vs FedAvg under attacks | 0/20, 1/19, 3/17 | p5 F1 |
| Median / Krum clean | −0.0214 (4/16), −0.0507 (2/18) | p5 F2 |
| FLTrust at σ 0.5 / 1 / 4 clean | −0.082 0/20; −0.093 0/20; −0.011 3/17 | p5b |
| Spearman +1.00 | — | p5b mechanism |
| No-DP FLTrust under Fang | +0.045, 16/4 | p5 secondary |
| Variant L | −0.042 / −0.036 / −0.058 / −0.023 | p5b |
| Variant N at ε 1.86 | +0.012 [−0.008, +0.032] 9/11; +0.004; −0.009; Fang +0.026 [+0.008, +0.045] 15/5 | p5b F6-N |
| Variant N per-seed range | −0.026 to +0.131 | make_figures.py printout |
| Variant N across levels | σ 1 IPM −0.030 [−0.052, −0.009] 6/14; σ 4 ALIE −0.011 1/19; 80/80 at σ 0 | p5c |
| Trust-capture | 0.071 → 0.0005; 18/20 at 0.000 (16 exactly 0); FLTrust 0.0055; FedAvg 0.059 → 0.014 | p5c; phase5d/PREREG.md §1 |
| Honest trust sum ~0.04 | — | phase5c/PREREG.md §4 |
| Attackers' share 60-73% | ~60% (p5c), 0.733 (p5d uncapped) | p5c; p5d round statistics |
| Table V (cap), all cells | — | phase5d/THESIS_TABLE.md Table V; p5d |
| Cap 1/8 ALIE gain | +0.031 [+0.012, +0.050], 15/5 | p5d F12 m 1 |

## Failed predictions, nulls, retraction
| Claim | Value | Source |
|---|---|---|
| Original interaction | −0.120 [−0.178, −0.062], 0/10 | p2 |
| Calibrated interaction | +0.135 / +0.330 / +0.201, 10/10 | rerun R2 |
| Scorecard rows | — | p4g, p4h, p5b, p5c, p5d prediction tables |
| FedProx | μ 1 at σ 2: −0.050, 1/10 | rerun R3 |
| F3 schedule vs damage | 6/6 equivalent within ±0.008 | p5 F3 |
| IPM helps FedAvg | +0.019, 16 of 20 seeds better | p5 secondary (damage −0.0191, 4/16) |
| Discarded implementations | — | NEGATIVE_RESULTS Phase 4; phase4/PREREG.md Amend. 2-3; p5 Amend. 1; phase4h/PREREG.md note |

## Discussion, Limitations, Conclusion
All numbers repeat the ones above. Additional figures:
- 97% thrown away: p5b ("throws away ~97%").
- Median box 110 vs 621 px²: p4 Context.
- 1-8 GPU-hours per agent: p4 training table and p4e cost table.
- Plain FLTrust 0.0055 vs Variant N 0.0005 under trust-capture: p5c.
