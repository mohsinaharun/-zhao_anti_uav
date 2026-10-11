# Phase 4d — signal-redesign check (follow-up to Phase 4b; diagnostic only)
Pre-registration, written 2026-10-11 BEFORE the check is run. User-specified design and
decision rule; the operational definitions below are fixed now and not revised after results.
Phase 4 / 4b / 4c results are unchanged. Outputs: `results/phase4d/` only.

Context: Phase 4b found A5's feedback (post-round proxy loss) mostly tracks the round's own
noise dose. Question: is a gradient-norm signal, averaged over recent rounds, a cleaner
signal that (a) predicts the run's outcome early and (b) is decoupled from the current
round's noise dose? Only then is a new agent, A5-v2, worth training.

## 1. The new signal (privacy-free operationalisation)
* **g_t** = median per-sample gradient norm (before clipping, no noise added) of the global model
  at the START of round t+1 (i.e. after aggregating round t), computed by the server on the
  public proxy session 01_2192 (256 samples). Zero privacy cost: public data, no client data.
  Note: the gradient computation itself is noise-free, but the model it is evaluated at already
  contains the noise of rounds <= t. Measuring clients' own pre-noise gradients during local
  training would use private client data (extra privacy cost) and is out of scope.
  (This is the quantity A5 already received as `proxy_gnorm`, alongside the proxy loss.)
* **Agent input s_t** = mean of g over the last 3 rounds (g_t, g_t-1, g_t-2; fewer at the start),
  plus round number and remaining-budget fraction (unchanged from A5).

## 2. Diagnostic (Phase 4b method)
Main setting P (R = 12, eps = 1.86), the 20 main-setting seeds 200-219, real clients, schedules
A1 (uniform), A3 (theory), A5 (deployed Phase 4 policy). Replays are identical to Phase 4's runs
(same init, C*, noise streams; checked). Log g_t for t = 0..12 and the final model's **VAL**
mAP@0.5 (the outcome for "final rank"; the test pool is not used).

Reported per schedule, in Phase 4b's format:
1. Largest seed-to-seed spread of s_t (sd / total training progress, progress = s_0 - s_12 mean).
2. Spearman (20 seeds) between s_t and final VAL mAP, t = 1..12; also between s_t and s_12.
3. Lockstep with the schedule's own noise injection (see (b)).

## 3. Decision rule (user-specified; operational definitions fixed now). Judged on A3,
the only schedule with round-to-round noise jumps (sigma 3.95 in rounds 2 and 4); A1 and A5
are reported.
* **(a) early prediction:** Spearman(s_t, final VAL mAP) has the same sign with |rho| >= 0.5 in
  at least 2 of the early-to-mid rounds t = 1..8. (One round alone could be chance: with n = 20,
  |rho| >= 0.5 has p ~ 0.025 per round.)
* **(b) decoupled from noise dose:** per seed, Pearson r over rounds t = 2..11 between the
  detrended RAW signal e_t = g_t - (g_t-1 + g_t+1)/2 and the detrended dose
  d_t = sigma_t - (sigma_t-1 + sigma_t+1)/2 (sigma_t = the noise of round t, which produced g_t).
  **Lockstep** if the mean r over seeds has a 95% CI excluding 0 AND |mean r| >= 0.3.
  (b) holds if NOT lockstep. Decided on the raw g_t, since averaging can only hide a spike;
  the same statistic for s_t is reported. Positive control: the same statistic for the old
  signal (proxy loss), which Phase 4b found tracks the dose.
* **Train A5-v2 only if (a) AND (b).** Otherwise: no new agent; written up as a
  negative/explanatory finding, logged in NEGATIVE_RESULTS.md; Phase 5 uses A5.
* Either way, the result is reported and the user reviews it BEFORE any training.
