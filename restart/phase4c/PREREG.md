# Phase 4c — Does the Phase 4 finding hold on other data? (CIFAR-10 and Pascal VOC)
Pre-registration DRAFT, written 2026-10-11 BEFORE any Phase 4c code is run. Awaiting user
approval. Todo suggestion 8 (second dataset). Phase 4 / 4b results are not changed.

Finding under test (Phase 4, drone, P): at equal epsilon, schedules with LESS noise in later
rounds (A2 tuned exponential, A3 theory rho_t ~ sqrt(a_t)) beat uniform (+0.014 mAP, 16/4 each).
Question: does "decay beats uniform" hold on (i) an easier classification task and (ii) the
same detection task with large, clear objects? Also tested: Kiani et al. (ICLR 2025)
"spend-as-you-go" as an extra fixed arm.

## 1. Datasets and splits (fixed now; no frames, no sessions -> split by image)
* **CIFAR-10** (classification, top-1 accuracy). Official test set (10,000) = TEST.
  From the 50,000 training images (seed 20261011): PROXY 5,000 (public pool: C* calibration
  only), VAL 5,000 (tuning, operating point, pulse test), CLIENTS 8 x the real drone client
  sizes (1970, 1884, 1500, 1300, 6031, 6162, 2999, 3300; total 25,146). With identical sizes
  the sampling rates, and so the epsilon of any schedule, are identical to the drone study.
  Non-IID: each client's class mix ~ Dirichlet(0.5) (seed 20261011).
* **Pascal VOC** (detection, mAP@0.5, CLASS-AGNOSTIC: every object of the 20 classes is
  "object", mirroring the single-class drone detector). VOC2007 test (4,952) = TEST.
  VOC2007+2012 trainval (16,551) split into PROXY 1,500 / VAL 2,000 / CLIENTS: the drone
  client sizes scaled by 0.515 to fit (1015, 970, 772, 670, 3106, 3173, 1544, 1700; total
  12,950; 101 images unused). Larger
  sampling rates -> higher epsilon at the same sigma; reported. Non-IID: Dirichlet(0.5) over
  each image's dominant class (largest box). VOC is a still-image dataset, so a per-image
  split has no recording leakage; VOC2007 test is disjoint from both trainval sets.

## 2. Model (same as the drone study unless stated)
MobileNetV3-Small, ImageNet weights, frozen through block 8 (K = 9), features cached once.
* CIFAR-10: images upsampled 32 -> 128 px (stride-16 map 8 x 8 x 48); head = the drone head's
  two 3x3 conv + GroupNorm layers, then global average pool + linear (64 -> 10).
  ~65.5k trainable parameters (drone head 65,221) - same D, so the same noise sensitivity.
* VOC: images letterboxed to 640 x 512 (same grid and identical head, 65,221 parameters).
  **Required change (disclosed):** the drone detector assumes at most one target per frame;
  for VOC the targets and loss are extended to several objects per image (an objectness
  positive at every object's centre cell, box regression at each positive). Decode/mAP code
  already handles any number of detections. Verified on the drone data that the extended code
  gives identical loss and mAP when there is one box (regression check before any VOC run).
* Unchanged: DP-SGD (Opacus, Poisson sampling, RDP, delta 1e-5, worst-client epsilon), B = 64,
  20 local iterations per round, SGD lr 0.03 momentum 0.9, FedAvg, mu = 0, calibrated fixed
  clip C* from the PROXY pool (same rule), seeds and common random numbers as Phase 4.

## 3. Gates and operating point (val only), per dataset
* **G1 pipeline**: non-DP FL (R = 12) beats the trivial predictor (CIFAR 0.10; VOC: the
  constant-score predictor) by a clear margin, 3 seeds.
* **G2 noise matters**: uniform sigma sweep {0, 0.5, 1, 2, 3, 4, 6, 8} at R = 12, 5 seeds
  (100-104). Must degrade by >= 30% of the (sigma 0 - trivial) range by sigma 8. If not, in
  order: unfreeze to K = 7 (as Phase 3 V3), then B = 32. Any such change is logged.
* **Operating point P_d**: R = 12 and the sweep sigma whose mean val score is closest to the
  midpoint between sigma 0 and trivial (the drone P sat near the middle of its range).
  Epsilon of P_d reported.

## 4. Pulse test (for A3), per dataset - Phase 3 V1 protocol exactly
At P_d: baseline uniform sigma; pulse at round k = 1..12 sets sigma_k = 2 sigma; 20 paired
seeds (0-19); drop(k) = score(baseline) - score(pulse). Same slope / shape / informativeness
tests as Phase 3. a_t = mean drop(k). If not informative (pooled drop CI includes 0), A3 is
"not identifiable" and dropped from that dataset's family.

## 5. Arms (same total epsilon per dataset; Phase 4 code paths)
* **A1** uniform. **A2** exponential, k tuned on VAL over {0, 0.25, 0.5, 1, 1.5, 2, 3},
  tuning seeds 100-104. **A3** theory rho_t ~ sqrt(a_t) from §4 (floor 0.1 x median, as Phase 4).
* **A6** Kiani et al. spend-as-you-go, mapped exactly onto Phase 4's online rule: share
  (q_n / q)^2 = (0.5 / 0.9)^2 = 0.309 in rounds 1-6 (saving), share 1 in rounds 7-12 (their
  default: switch at mid-training, q = 0.9, q_n = 0.5). Saving is realised as higher sigma, not
  lower client sampling (our unit is the sample and all clients join every round): same
  per-round privacy spend, different dynamics - disclosed.
* No DRL arms (A4/A5): Phase 4/4b showed they add nothing and the feedback is mostly noise;
  cost would be ~10 GPU-hours per dataset.

## 6. Evaluation and decision rule (per dataset)
TEST, 20 paired seeds 200-219, CRN as Phase 4. Mean paired difference, 95% t-CI, win count,
epsilon. **Primary family (Holm, alpha 0.05): A2 vs A1, A3 vs A1, A6 vs A1.** WIN / LOSS /
TIE / EQUIVALENT exactly as Phase 4 (equivalence margin 0.02 in the dataset's metric).
* **"Decay beats uniform" REPLICATES on a dataset** iff A2 or A3 WINS vs A1 there.
  It FAILS to replicate if neither wins; it is REVERSED if either LOSES.
* Secondary (unadjusted): A3 vs A2, A6 vs A2, A6 vs A3; the pulse-test a_t shape compared
  with the drone's (rising late?).
* Overall statement across drone / CIFAR-10 / VOC reported as a 3-row table; no pooling.

## 7. Order, cost, reporting
CIFAR-10 first (cheaper), then VOC. Per dataset: data + cache -> G1 -> G2 + P_d (report) ->
pulse test -> A2 tuning -> test evaluation -> analysis (report). Estimated ~4-6 GPU-hours per
dataset plus ~1 h caching for VOC. Outputs `restart/results/phase4c/{cifar10,voc}/`; failures in
NEGATIVE_RESULTS.md. Any deviation is a dated amendment here before the affected run.
