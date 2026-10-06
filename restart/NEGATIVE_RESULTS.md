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
