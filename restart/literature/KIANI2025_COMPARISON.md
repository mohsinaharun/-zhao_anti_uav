# Kiani et al. (ICLR 2025) vs this thesis — comparison note (2026-10-10)

Kiani, Kulkarni, Dziedzic, Draper, Boenisch. *Differentially Private Federated Learning with
Time-Adaptive Privacy Spending.* ICLR 2025. Author copy:
https://adam-dziedzic.com/static/assets/papers/time_adaptive_dp_fl.pdf
(local copy `kiani2025_time_adaptive_dpfl.pdf`, not committed). Page refs are to that PDF.
NB the ICLR proceedings link 96d00450... is a DIFFERENT paper (AdaDPSyn); do not cite it.

## What they do
* **"Spend-as-you-go"** (Alg. 1, p.5): each client has a "saving" phase (rounds t < T_n) then a
  "spending" phase. In every round sigma is set as if the remaining budget were spent evenly
  (GetNoise). Saving is done by sampling the client LESS (rate q_n < q), which by subsampling
  amplification spends only (q_n/q)^2 of the even share (Eq. 1, p.7). Spending rounds spend the
  even share of what is left, so per-round spend rises then plateaus (Thm 1, Fig. 2).
* Schedule is fixed BEFORE training from budgets only (no data) -> zero extra privacy cost.
* Theory (Thm 2-3): bounds on clipping BIAS; result = clients with stricter budgets (smaller clip
  norms) should save more. It is about matching sampling rates to clients, NOT a derivation of the
  optimal schedule over time. The "early rounds tolerate noise" premise is cited intuition
  (coarse vs fine features), not proved.
* Setup: CLIENT-level DP, 100 clients (Dirichlet 0.1), MNIST/FMNIST/Adult (+CIFAR-10 appendix),
  small CNN/MLP, T = 25, switch at T_n = 13 (mid-training), q = 0.9, q_n in 0.3-0.7, c = 250,
  epsilon 10-30 (strict variants 2/5/10 and 5/5/5). **Results averaged over 3 runs; no CIs, no
  seeds pairing, no significance tests** (p.22).
* Results: +4.7 / +4.4 / +19.9 pp over uniform DP-FedAvg at eps = 10 (FMNIST / MNIST / Adult);
  beats Andrew et al. adaptive clipping (Table 7); robust to the switch round (<2 pp, Table 10);
  CIFAR-10 +0.4 pp only (Table 11). With T = 50 the MNIST gain shrinks to +1.2 pp (Table 5).

## Where we agree
| | Kiani et al. | This thesis (Phase 3/4) |
|---|---|---|
| Direction | save early, spend late beats uniform | A2 (exp. decay) and A3 (theory) beat uniform at P: +0.014 mAP, 16/4, ~+24% rel.; same at S2 |
| Why | intuition: late rounds learn fine features | MEASURED: Phase 3 pulse test, a_t rises over rounds |
| Fixed vs learned | fixed rule, no learning | pre-registered: learned (A5) ties uniform, loses to fixed rules |
| Adaptive clipping | loses to their scheme | lost to calibrated fixed C* (Phase 2 re-run) |
| More rounds | MNIST gain shrinks at T = 50 | gain not significant at R = 25 (S1) |

## Where we differ (what we add)
1. **Rigour of the comparison**: 20 paired seeds with common random numbers, CIs, win counts,
   Holm-corrected pre-registered rule, vs their 3-run averages. Our effect (+0.014) is measured
   with a CI that excludes zero; theirs is not tested.
2. **Mechanism measured, not assumed**: our Phase 3 noise-pulse test measures a_t directly;
   their premise is cited intuition.
3. **Learning the schedule**: they never test a learned schedule; we did (A4/A5) and it adds
   nothing. Their own future-work (p.10-11) asks for tuning the switch point; our A2/A5 results
   say the shape matters less than the direction, and too steep a decay loses.
4. **Regime**: eps 0.79-2.72 (ours) vs 10-30 (theirs); sample(frame)-level vs client-level DP;
   detection mAP on real non-IID drone video vs classification benchmarks.
5. **Agent privacy cost**: our agents train only on public proxy data (zero client privacy).

## Directly comparable as an extra arm
Their per-round spend rule (Eq. 1) is EXACTLY our online action mapping ("share of remaining /
rounds left", `budget.sigma_for_round`) with share = (q_n/q)^2 in saving rounds and 1 after.
With their defaults (q_n = 0.5, q = 0.9 -> share 0.31; switch at mid-training) it lies inside our
agents' action range [0.25, 4]. An arm **A6 "Kiani spend-as-you-go"** = share 0.31 for t < R/2+1,
then 1, on our budget-matching code. Caveat: we would realise the saving with a higher sigma,
not a lower client-sampling rate (our privacy unit is the frame and all 8 clients join every
round); the per-round RDP spend is identical, the training dynamics are not.

## Security note (corrects an earlier remark)
Their per-client clip norms are set so that c_n * sigma_n is equal across clients (Alg. 1 l.12),
so per-client NOISE magnitude is equalised by design. Our P3-H finding (heterogeneous sigma widens
the honest norm band) therefore does not transfer automatically; it would need checking.
