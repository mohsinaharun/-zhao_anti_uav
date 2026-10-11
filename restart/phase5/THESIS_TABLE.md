## Do Byzantine-robust aggregators protect a DP federated drone detector?

At a meaningful privacy level (ε = 1.86), none of the three robust aggregators protects the
detector: each one scores below plain FedAvg with no attack, and below FedAvg under each of the
three attacks. FLTrust fails because DP noise drowns its trust signal:
honest updates agree with the server's clean update with a cosine of only 0.008, so trust-weighted
aggregation leaves almost no useful step (−75% without attack). The same FLTrust matches FedAvg
without DP and beats it under the Fang attack, so the failure comes from DP, not from FLTrust.

**Table Y.** Robust aggregation under DP. Anti-UAV drone detector, test mAP@0.5, 20 paired seeds
with common random numbers, uniform schedule (A1), worst-client ε = 1.86 at δ = 10⁻⁵, 8 clients of
which 2 are malicious (17% of FedAvg weight). Differences are aggregator − FedAvg under the same
attack and seed; wins/losses out of 20 seeds.

| | FedAvg | FLTrust | Coordinate-wise median | Krum (f = 2) |
|---|---|---|---|---|
| **Designed to stop** | — (no defence) | attacks whose updates point away from a clean server update | outlying values per coordinate | outlying whole updates |
| **No attack** | 0.0592 | 0.0151: **−0.044** [−0.060, −0.029], 2/18 ᵃ | 0.0378: **−0.021** [−0.032, −0.011], 4/16 ᵃ | 0.0085: **−0.051** [−0.064, −0.037], 2/18 ᵃ |
| **ALIE** | 0.0394 | 0.0061: −0.033 [−0.044, −0.023], 0/20 ᵇ | 0.0042: −0.035, 0/20 | 0.0000: −0.039, 0/20 |
| **IPM** | 0.0783 ᶜ | 0.0197: −0.059 [−0.076, −0.041], 1/19 ᵇ | 0.0269: −0.051, 0/20 | 0.0000: −0.078, 0/20 |
| **Fang** | 0.0442 | 0.0196: −0.025 [−0.037, −0.012], 3/17 ᵇ | 0.0043: −0.040, 0/20 | 0.0133: −0.031, 3/17 |
| **Beats FedAvg anywhere?** | — | **No** (0 of 4 conditions) | **No** (0 of 4) | **No** (0 of 4) |
| **Why it fails under DP** | — | Honest trust (cosine with server update) 0.008; the useful step shrinks to ~1% | Discards the √N averaging that cancels DP noise | Discards averaging; picks an attacker's update in 100% of rounds under ALIE and IPM |
| **Without DP (ε = ∞)** | 0.250 | 0.253: tie (+0.003, 12/8); **wins under Fang** (+0.045, 16/4) | not run | not run |

ᵃ Pre-registered family F2 (Holm-adjusted p ≤ 0.0006, all LOSS). ᵇ Pre-registered family F1
(Holm-adjusted p ≤ 0.0006, all LOSS). Other cells: unadjusted p ≤ 0.0001.
ᶜ IPM raises FedAvg's score under DP (+0.019, 16/4): the attackers' update is built from the mean
of six honest updates and carries less noise than the two noisy updates it replaces. Without DP,
IPM has no effect (0.000).

**Table Y′.** Does the privacy schedule change attack damage? FLTrust, ε = 1.86; damage = mAP(no
attack) − mAP(attack); difference vs uniform (A1), 20 paired seeds, pre-registered family F3.

| Schedule | ALIE | IPM | Fang | Verdict |
|---|---|---|---|---|
| A3 theory (ρ_t ∝ √a_t) vs A1 | −0.0001 [−0.003, +0.003] | +0.0010 [−0.001, +0.004] | +0.0008 [−0.002, +0.003] | Equivalent (Holm p 1.00) |
| A5 learned (TD3+GRU) vs A1 | −0.0023 [−0.007, +0.003] | +0.0004 [−0.002, +0.003] | +0.0012 [−0.002, +0.004] | Equivalent (Holm p 1.00) |

The learned A5 also loses to the theory A3 under every attack (F4: −0.001 to −0.004, Holm p ≤
0.025), but by the same amount without attack (−0.004, 0/20). Its schedule was identical in every
condition, so it did not react to the attackers.
Attacks: ALIE (z = 0.319), IPM (ε_ipm = 0.5), Fang trimmed-mean/median attack (b = 2), all
omniscient and colluding, every round. FLTrust root set: 1,300 clean frames from 2 of the 4
environments, session-disjoint from all other pools.

Sources: `restart/phase5/RESULTS.md`, `restart/phase5/PREREG.md`; raw runs under
`restart/results/phase5/`.
