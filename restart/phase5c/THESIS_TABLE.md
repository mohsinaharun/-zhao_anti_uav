## How far the DP-aware FLTrust fix goes

Rescaling every update to the norm an honest DP update is expected to have (Variant N) turns FLTrust
from a large loss into roughly FedAvg-level performance across privacy levels. It beats plain FLTrust
at every DP level and beats FedAvg under the Fang attack at almost every level. It does not hold up
everywhere: it loses to FedAvg under one attack at ε 5.73 (IPM) and at ε 0.79 (ALIE, near the floor).
An attacker who knows the rule and the server's root update breaks it completely. Because all updates
then have equal norm, a moderately aligned malicious update takes ~60% of the weight every round.

**Table W.** Variant N vs FedAvg across privacy levels. Anti-UAV drone detector, test mAP@0.5, 20
paired seeds with common random numbers, uniform schedule, worst-client ε at δ = 10⁻⁵, 2 of 8 clients
malicious. Cells: Variant N − FedAvg, wins/losses of 20 seeds, Holm-adjusted verdict within each level
(pre-registered family F8; σ 2 is Phase 5b's F6-N).

| | No DP | σ 0.5 | σ 1 | σ 2 | σ 4 |
|---|---|---|---|---|---|
| **Worst-client ε** | ∞ | 30.16 | 5.73 | 1.86 | 0.79 |
| **FedAvg, no attack** | 0.250 | 0.179 | 0.132 | 0.059 | 0.014 |
| **Variant N, no attack** | 0.253 | 0.187 | 0.147 | 0.071 | 0.018 |
| **No attack** | +0.003, 12/8 tie | +0.008, 12/8 tie | +0.014, 13/7 tie | +0.012, 9/11 tie | +0.004, 11/9 tie ᵉ |
| **ALIE** | −0.012, 8/12 tie | +0.004, 13/7 tie | +0.000, 9/11 tie ᵉ | +0.004, 10/10 tie | **−0.011, 1/19 LOSS** ᵃ |
| **IPM** | −0.009, 9/11 tie | +0.007, 11/9 tie | **−0.030, 6/14 LOSS** | −0.009, 6/14 tie | −0.003, 7/13 tie ᵉ |
| **Fang** | **+0.045, 16/4 WIN** | **+0.031, 16/4 WIN** | +0.019, 17/3 tie | **+0.026, 15/5 WIN** | **+0.011, 13/7 WIN** ᵃ |
| **Recovers (no LOSS)?** | yes ᵇ | **yes** | no | **yes** | no ᵃ |
| **Variant N vs plain FLTrust** | identical ᵇ | 4 × WIN | 4 × WIN | 4 × WIN | 3 × WIN, 1 tie ᶜ |

ᵃ At the floor: FedAvg scores 0.005-0.019 at σ 4. ᵇ Without DP Variant N *is* FLTrust (bit-identical
on 80/80 runs). ᶜ Secondary comparison, unadjusted; ALIE at σ 4 is 12/8. ᵉ Equivalent (CI within ±0.02).

**Table W′.** Adaptive "trust-capture" attack at ε = 1.86 (σ 2). The attackers know Variant N's rule
and the server's root update. Both send an update of honest DP norm, aligned with the root update at
cosine c (≈ 0.5, chosen each round) and otherwise opposed to the honest consensus.

| | No attack | ALIE | Adaptive | Damage (none − adaptive) |
|---|---|---|---|---|
| FedAvg | 0.059 | 0.039 | 0.014 | +0.046 [+0.031, +0.060], 19/1 |
| Plain FLTrust | 0.015 | 0.006 | 0.006 | +0.010 [+0.004, +0.015], 16/4 |
| **Variant N** | **0.071** | **0.044** | **0.0005** | **+0.071 [+0.055, +0.086], 20/0** |
| **Variant N − FedAvg** | +0.012, tie | +0.004, tie | **−0.013 [−0.023, −0.003], 1/18 LOSS** | — |

Pre-registered family F9 (Holm over 3): damage to Variant N confirmed; Variant N LOSES to FedAvg
under the attack, so it is **broken**; the attack is stronger than ALIE (−0.043, 0/20). 18 of 20 Variant N
seeds end at exactly 0. Attackers' share of trust weight: 91% in round 1, ~60% in rounds 4-12. It does
not stay higher because honest updates align with the root update once the model is damaged (honest
cosine 0.01 → 0.10-0.19).
Scope: omniscient attacker (knows g₀ exactly), one attack design, one operating point, 2/8 attackers.

Sources: `restart/phase5c/RESULTS.md`, `restart/phase5b/RESULTS.md`, `restart/phase5/RESULTS.md`;
raw runs under `restart/results/phase5c/`, `restart/results/phase5b/`, `restart/results/phase5/`.
