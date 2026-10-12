## Capping trust weight does not stop the adaptive attack

Phase 5c's rule-aware attacker took ~73% of Variant N's aggregation weight and drove the detector to
zero. Capping any single client's share of the trust weight (κ = m/8) works as designed: the two
attackers are held at exactly 2κ (50%, 37.5%, 25%). But the detector still collapses at every cap.
Under DP, honest updates are almost all noise, so even a 25% weight on a coherent, full-norm malicious
update overwhelms them. Plain FedAvg, at the attackers' 17% data share, is also badly hurt. The cap costs
nothing without this attacker, and at the tightest setting it improves robustness to ALIE.

**Table V.** Per-client trust-weight cap on Variant N. Anti-UAV drone detector, test mAP@0.5, ε = 1.86
(σ 2, 12 rounds, worst-client ε at δ = 10⁻⁵), 20 paired seeds with common random numbers, 2 of 8
clients malicious. Mean [seeds at 0.000]. Bottom rows: Holm-adjusted tests within each margin
(pre-registered family F12-m, 6 tests).

| | Variant N, no cap | Cap 2/K (κ 0.25) | Cap 1.5/K (κ 0.1875) | Cap 1/K (κ 0.125) | FedAvg |
|---|---|---|---|---|---|
| **No attack** | 0.071 | 0.072 | 0.071 | 0.065 | 0.059 |
| **ALIE** | 0.044 | 0.051 | 0.066 | **0.074** ᵇ | 0.039 |
| **IPM** | 0.070 | 0.068 | 0.079 | 0.081 | 0.078 |
| **Fang** | 0.070 | 0.067 | 0.073 | 0.074 | 0.044 |
| **Trust-capture (Phase 5c)** | 0.0005 [18] | 0.0004 [19] | 0.0000 [20] | 0.0058 [14] | 0.0137 [7] |
| **Trust-capture, cap-aware** | — | 0.0003 [18] | 0.0014 [12] | — | — |
| Attackers' weight share | 73% | 50% | 37.5% | 25% | 17.1% (data share) |
| Collapse prevented (vs no cap)? | — | no (tie) | no (tie) | no (tie) | — |
| Under attack vs FedAvg | — | −0.013, 1/18, tie ᵃ | **−0.014, 0/19, LOSS** | −0.008, 3/16, tie ᵉ | — |
| Cost without this attack (any LOSS vs no cap)? | — | none | none | none (ALIE **WIN**) ᵇ | — |
| **Genuine fix?** | — | **No** | **No** | **No** | — |

ᵃ Holm p 0.064. ᵇ +0.031 [+0.012, +0.050] vs Variant N, 15/5. ᵉ Equivalent (CI within ±0.02).
The cap-aware attacker is no stronger than Phase 5c's (+0.0014 and −0.0001), because the latter
already saturates the cap.

Figure: `plots/phase5d_trust_cap.png`.
