# Phase 1 — pre-registration (written 2026-10-05, BEFORE any Phase 1 run)

Implements todo steps 4-6. All choices below are fixed before the first run; any
change needs a dated amendment at the bottom of this file, made before the run it
affects.

## Data (from the frozen `restart/data/split.json`)

* **Train ("single client")** = the whole `clients` pool (58 sessions), every
  **5th** frame of each folder (consecutive video frames are near-duplicates).
  Frames without a target are kept. This one-client setting is also the
  "DP without FL" cell of Phase 2's ablation.
* **Selection / gates** use `val` (6 sessions), every **5th** frame.
* `test`, `root`, `proxy` are **not touched** in Phase 1.
* Native 640x512 IR frames, replicated to 3 channels, ImageNet normalisation.
  No augmentation.

## Model

* MobileNetV3-Small, ImageNet weights. Blocks `0..K-1` frozen and their output
  cached; blocks `K..8` + head trainable. **Start K = 9** (everything up to the
  stride-16 output frozen: the "frozen backbone + trainable head" of step 4).
* Head on the 48-channel stride-16 map (32x40 grid): Conv3x3(64)-GroupNorm-ReLU,
  Conv3x3(64)-GroupNorm-ReLU, Conv1x1 -> 5 channels (objectness + box).
  GroupNorm, not BatchNorm (BatchNorm is incompatible with per-sample DP).
* Target encoding: the cell containing the box centre is positive; box = sigmoid
  offsets within the cell + log(w/16), log(h/16). Loss = focal loss on objectness
  (normalised by #positives) + L1 on the positive cell's box.
* Decoding: 3x3 max-pool peak suppression, top-5 cells per frame, score =
  sigmoid(objectness).
* Metric: **mAP@0.5 (pycocotools, single class)** over all val frames, including
  frames without a target.

## Step 4 — non-DP baseline and gate G1

* SGD, momentum 0.9, batch 64, 12 epochs. Learning rate chosen from
  {0.003, 0.01, 0.03} by val mAP@0.5 (seed 0), then **3 seeds** at that lr.
* **Trivial predictor**: one box per frame at the train-set median centre and size,
  score 1. Its val mAP@0.5 is computed and reported.
* **G1 PASSES** if mean mAP@0.5 over the 3 seeds >= **0.30** AND >= trivial + **0.20**.
* If G1 fails: in order, (a) K = 8, (b) K = 7, (c) remove stride: put the head on
  the stride-8 map (block 3 output). Re-run step 4 after each; record every attempt
  in `restart/NEGATIVE_RESULTS.md`.

## Step 5 — noise sweep and gate G1b

* DP-SGD with Opacus: per-sample clipping at C, Gaussian noise multiplier sigma,
  Poisson sampling with expected batch B = 256, 12 epochs, SGD momentum 0.9.
* **C** = median per-sample gradient L2 norm of the model at initialisation over
  1,024 train frames (measured once, recorded, then fixed).
* **lr** chosen at sigma = 0 (clipping only) from {0.01, 0.03, 0.1} by val mAP, seed 0.
* **sigma in {0, 2, 4, 8, 16, 32}, 3 seeds each**, paired (seed t shares init and
  sampling stream across sigma).
* **G1b PASSES** if
  1. mean mAP(sigma=0) - mean mAP(sigma=32) >= **0.10**, AND
  2. Spearman(sigma, mean mAP) <= **-0.80** across the six sigma values, AND
  3. mean mAP(sigma=2) >= trivial + 0.10 (noise does not destroy the model at the
     weakest setting).
* If (1) or (2) fails (noise too weak to matter): lower K by one block (more
  trainable parameters), then halve B; re-sweep after each change.
* If (3) fails (noise too strong): raise B to 1,024, then raise K back.
* Every attempt, passed or failed, is logged in `restart/NEGATIVE_RESULTS.md`.

## Step 6 — epsilon

For every sigma > 0: epsilon at delta = 1e-5 from the Opacus RDP accountant with the
actual sampling rate q = B/N and number of steps. Reported in the same table as
mAP. Privacy unit: one frame (Phase 0 default, disclosed: frames within a session
are correlated, so per-recording protection is weaker than this epsilon suggests).

## Outputs

`restart/results/phase1/*.json`: config, code commit, machine, GPU, seeds, per-epoch
val mAP, final val mAP, C, q, steps, epsilon. Figure: mAP@0.5 vs sigma with seed
spread and epsilon on a secondary label row (one y-axis).

## Amendment 1 — 2026-10-05, before any B=1024 run

G1b failed on criterion (3) at K=9, B=256 (mAP 0.010 at sigma=2). Per the
pre-registered remediation, B is raised to 1,024 (K is already at its maximum, 9).
Interpretation made explicit: for each remediation configuration the DP learning
rate is re-selected at sigma = 0 from {0.01, 0.03, 0.1} by val mAP (seed 0), exactly
as for the first configuration. C stays 45.8 (it is a property of the model at
init, not of B). Gate criteria unchanged.

## Amendment 2 — 2026-10-05, before any temporal-input run: single vs 3-frame input

Motivation (diagnostic, `restart/figures/phase1_error_examples.jpg`): the top-1
detection lands on static clutter (lit windows, bright specks on hillsides) in 24% of
val frames with a drone (52% in terrain); 49.5% of drone-free frames get a
detection with score >= 0.3. Static clutter does not move; a drone does.

* **S (single)**: current input, frame t replicated to 3 channels.
* **T (temporal, causal)**: the 3 input channels are the grayscale frames
  **(t-10, t-5, t)** (clamped at the folder start). Causal, so it stays usable for
  real-time detection. Same frozen MobileNetV3-Small (K=9), same head, same labels
  (frame t's box), same sampled frames t (every 5th). No change to the trainable
  parameter count, so the DP analysis is unaffected.
* Procedure, per variant, paired seeds 0-2:
  1. non-DP: lr chosen from {0.003, 0.01, 0.03} on seed 0, then 3 seeds;
  2. DP at the canonical config (B=1,024, 12 epochs, lr 0.03), sigma = 2, 3 seeds,
     with C re-measured for T by the same rule (median per-sample grad norm at init).
* **Adopt T** if BOTH: (a) non-DP mean val mAP(T) - mAP(S) >= **+0.03**, and
  (b) DP sigma=2 mean val mAP(T) >= mAP(S) - **0.02**. Otherwise keep S.
* If T is adopted: re-run the full G1b noise sweep for T (sigma 0..32, 3 seeds,
  B=1,024) so the canonical config's gate is re-established on the new input.
* Descriptive (not a criterion): the error breakdown (hit / mislocated / wrong place
  / false alarm) for both variants.
