# What did not work / what was wrong (restart, kept visible on purpose)

Every failed attempt, bug, or result that did not replicate goes here with its date,
so the reported results can be read against everything that was tried.

## Phase 1

* **2026-10-05 — clipping-norm measurement bug (caught before any DP run).**
  `--measure-C` multiplied Opacus per-sample gradient norms by the batch size,
  assuming `grad_sample` carried the loss-mean's 1/n factor. It does not: checked
  against exact single-sample backprop on 8 frames, the two agree to 0.01. The bad
  value (median 5,861) was deleted unused; the corrected value (~46) is the one
  recorded. No training run used the wrong C.
* **2026-10-05 — Gate G1b FAILED at K=9, B=256, C=45.8, lr=0.03 (3 seeds).**
  val mAP@0.5: sigma 0 -> 0.588, 2 -> 0.010 (eps 0.76), 4..32 -> 0.000. Criteria (1)
  and (2) passed, (3) failed: the model collapses at the weakest noise level, so
  there is no live range. Remediation per pre-registration: B -> 1,024.
  Results: `restart/results/phase1/dp_K9_B256_*`, figure `phase1_noise_sweep_K9_B256`.
* **2026-10-05 — causal 3-frame input (t-10, t-5, t) REJECTED (Amendment 2).**
  Motivation: the single-frame model's top detection lands on static clutter in 24%
  of drone frames (52% in terrain). Result, 3 paired seeds, val mAP@0.5:
  non-DP S 0.584 vs T 0.456 (-0.128, worse in every seed); DP sigma=2 S 0.350 vs
  T 0.096 (-0.255, worse in every seed). Both decision criteria failed -> keep S.
  Likely cause: the FROZEN ImageNet backbone expects the three channels to be one
  scene; three different instants degrade its features, and a tracking camera
  misaligns the frames. This rejects this cheap motion trick with a frozen
  backbone, not motion cues in general (that would need a trainable motion-aware
  stem: more parameters, more DP noise, and out of the thesis's scope).

## Revision (2026-10-06)

* **The fixed clip bound C = 45.8 was badly calibrated for most of training.** It was
  measured (Phase 1 rule) as the median per-sample gradient norm of the UNTRAINED
  model. After the first round, client gradient norms fall to a median of ~8-12, so
  C = 45.8 clipped only 0.3-0.5% of samples while the noise (sigma * C) stayed sized
  for norms ~5x larger. Every fixed-C DP result so far (Phase 1, Phase 2, revision
  Steps 1-2, 4) carries this handicap; they are valid as reported for C = 45.8 but
  understate what DP-FL can do with a calibrated C. Found by the adaptive clipping test.
* **2026-10-06 — Adaptive per-round clipping does NOT beat a calibrated fixed bound
  (control (a), pre-registered).** Against fixed C* (proxy median after 2 proxy rounds,
  set once; mean 5.15 over seeds), adaptive wins at 0 sigma levels and LOSES at 6
  (sigma 1.5-4, CIs entirely below zero, e.g. sigma 2: -0.112 [-0.169, -0.055], 2/10).
  The earlier adaptive-vs-45.8 gain was a calibration effect. Adaptive's per-round bound
  starts at the untrained model's norms (C = 55 in round 1, 13-36 in round 2) — too
  large, so early rounds are over-noised — and only settles at ~5-6 from round 3, where
  it stays flat. No evidence of useful per-round variation in gradient scale.

## Phase 4 (2026-10-09)

* **Pre-registered proxy simulator ("8 pseudo-clients by session") does not train.**
  With one recording per pseudo-client, FedAvg fails completely even without noise:
  reward 0.002-0.004 on the proxy-val sessions, and the model cannot fit its own
  training frames (0.036). The data are fine: the same frames pooled in one client
  give 0.60-0.61, split IID over 8 clients 0.65-0.74, and a model trained on the real
  clients scores 0.68 on those reward sessions. Cause: extreme non-IID (each client =
  one flight) -> client drift. The real clients (2-13 recordings each) are milder and
  train fine. Finding in its own right; simulator redesign pending user decision.
* **Phase 4 agent training attempt 1 discarded (implementation bug).** The online
  action measured "share of remaining budget" in a zCDP-style unit that understates the
  real (subsampled-RDP) cost at low sigma, so high shares burned the budget early and
  forced sigma = 20 later (reward 0). After ~107 episodes A4 had converged to
  "spend everything in round 1" (reward 0); A5 reached 0.16. Fixed by measuring spend in
  exact accountant units (Amendment 2). Logs kept in results/phase4/superseded_attempt1/.
* **Phase 4 A5 attempt 2 discarded (bug).** GRU target misalignment (next-step value computed
  without the first observation). 58 episodes at P; after warm-up the policy drifted to
  overspending (mean reward 0.03). Fixed in Amendment 3; logs in superseded_attempt2_A5/.
* **Phase 4: learned allocation does not help (pre-registered verdict).** At P, A5 ties
  uniform (-0.0005) and loses to A2/A3 (-0.014/-0.015, Holm p 0.011/0.0013); A4 loses to
  uniform (0/20). A4 failed G4b at P and S2 (collapses to sigma = 20 late; 0.000 on test).
  A5 pinned to the minimum action (most back-loaded allowed schedule) - too steep on real
  data. A3 (theory) loses to uniform at S1 (-0.025): noisy per-round a_t estimates.
  Details: phase4/RESULTS.md.
* **Phase 4c CIFAR-10: "decay beats uniform" fails to replicate (pre-registered).** a_t rises late
  exactly like the drone's, but A2 gains only +0.005 (16/4, Holm p 0.088) and every primary
  comparison is equivalent within ±0.02. Kiani et al.'s two-phase rule is the worst arm (loses to
  A2 on 19/20 seeds). Details: phase4c/RESULTS.md.
* **Phase 4c VOC stopped at the gates.** Noise-free FL reaches only 0.058-0.074 mAP under all three
  pre-registered configurations (bar 0.104); oracle check 0.990 rules out a scoring bug; the
  frozen-trunk head cannot learn VOC in 240 local steps. Also: the K = 7 fallback first crashed
  (Opacus hooks in C* calibration; fixed with expanded weights, as phase2_fl).
