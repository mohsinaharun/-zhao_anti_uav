## Why FLTrust fails under DP, and a one-line fix

Plain FLTrust loses to FedAvg at every privacy level tested, from ε = 30 down to ε = 0.79, and only
matches it without DP. Its deficit shrinks as the noise falls, in step with how much FLTrust's
norm-equalisation shrinks each update (from 1/74 at ε 0.79 to 1/11 at ε 30). Rescaling updates to the
norm an honest DP update is known to have (Variant N), instead of the clean server update's norm,
restores FLTrust's no-DP behaviour at ε = 1.86: it is never worse than FedAvg and beats it under the
Fang attack, while correcting the cosine trust score for noise (Variant L) does not help.

**Table Z.** FLTrust across privacy levels. Anti-UAV drone detector, test mAP@0.5, 20 paired seeds
with common random numbers, uniform schedule, worst-client ε at δ = 10⁻⁵, 2 of 8 clients malicious.
"Ratio" = mean FLTrust / mean FedAvg; differences are FLTrust − FedAvg, wins/losses of 20 seeds.

| | No DP | σ 0.5 | σ 1 | σ 2 | σ 4 |
|---|---|---|---|---|---|
| **Worst-client ε** | ∞ | 30.16 | 5.73 | 1.86 | 0.79 |
| **FedAvg, no attack** | 0.250 | 0.179 | 0.132 | 0.059 | 0.014 |
| **FLTrust, no attack** | 0.253 | 0.097 | 0.040 | 0.015 | 0.003 |
| **Ratio, no attack** | 1.01 | 0.54 | 0.30 | 0.26 | 0.20 |
| **FLTrust vs FedAvg, no attack** | tie (+0.003, 12/8) | LOSS (−0.082, 0/20) | LOSS (−0.093, 0/20) | LOSS (−0.044, 2/18) | LOSS (−0.011, 3/17) |
| **… under ALIE / IPM / Fang** | tie / tie / **WIN** | LOSS / LOSS / LOSS | LOSS / LOSS / LOSS | LOSS / LOSS / LOSS | LOSS / LOSS / tie ᵃ |
| **Honest cosine with server update** | 0.086 | 0.027 | 0.015 | 0.0075 | 0.0033 |
| **Shrink factor (server / honest norm)** | 1.02 | 1/10.9 | 1/20.4 | 1/38.6 | 1/73.7 |
| **Median, Krum vs FedAvg (no attack)** | not run | LOSS, LOSS | LOSS, LOSS | LOSS, LOSS | LOSS, LOSS |

ᵃ At the floor: FedAvg scores 0.005 under Fang at σ 4.
No-DP and σ 2 columns are Phase 5 runs; differences there are unadjusted (no-DP) or Holm (σ 2).
Verdicts at σ 0.5, 1 and 4 are Holm-adjusted within each family, as pre-registered. Both cosine and
shrink factor track the ratio perfectly (Spearman +1 over the five columns).

**Table Z′.** DP-aware FLTrust at ε = 1.86 (σ 2). Difference vs FedAvg [95% CI], wins/losses,
Holm-adjusted verdict (pre-registered families F6).

| | No attack | ALIE | IPM | Fang | Verdict |
|---|---|---|---|---|---|
| FedAvg (mAP) | 0.059 | 0.039 | 0.078 | 0.044 | — |
| FLTrust | −0.044, 2/18 LOSS | −0.033, 0/20 LOSS | −0.059, 1/19 LOSS | −0.025, 3/17 LOSS | fails |
| Variant L: cosine corrected for the known noise energy | −0.042 [−0.058, −0.026], 2/18 LOSS | −0.036, 1/19 LOSS | −0.058, 1/19 LOSS | −0.023, 4/16 LOSS | still loses ᵇ |
| **Variant N: updates rescaled to the expected DP norm √(‖g₀‖² + D·v_σ)** | +0.012 [−0.008, +0.032], 9/11 TIE | +0.004 [−0.016, +0.025], 10/10 TIE | −0.009 [−0.025, +0.008], 6/14 TIE | **+0.026 [+0.008, +0.045], 15/5 WIN** | **recovers** |

ᵇ Pre-registered label "partial": it beats FLTrust only without attack, by +0.002.
v_σ = per-coordinate noise variance of an honest update, known exactly from σ, the clip bound,
learning rate, batch size and momentum (predicts the observed norm within 2.5%); D = 65,221. At σ = 0
Variant N is exactly FLTrust. Variant N's ties are not tight (CIs reach ±0.03).
Scope: one operating point for Part 2, non-adaptive attacks.

Sources: `restart/phase5b/RESULTS.md`, `restart/phase5/RESULTS.md`; raw runs under
`restart/results/phase5/` and `restart/results/phase5b/`.
