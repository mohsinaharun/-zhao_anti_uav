# Phase 4f — VisDrone generalization check
Pre-registration, written 2026-10-11 BEFORE any Phase 4f code is run, from the user's Phase 4f
instructions (same day). **A new, additive check** in the pattern of Phase 4c (CIFAR-10 / VOC),
using a capacity-matched detection dataset. No Phase 4/4b/4c/4d/4e file or result is changed;
Phase 4c's code is imported read-only. Outputs: `restart/results/phase4f/` only.

Question: does "less noise in later rounds beats uniform at equal epsilon" (Phase 4 drone, P:
A2 / A3 +0.014 mAP over uniform, ~24% relative) hold on a drone-view detection task the frozen
head CAN learn? And does it look like the drone data (large effect) or CIFAR-10 (rising a_t,
tiny effect: +0.005, ~1.7% relative)?

## 1. Dataset and class
* **VisDrone2019-DET** (aerial images from drones; official train 6,471 / val 548 / test-dev
  1,610 images, with public annotations). Downloaded from the Ultralytics mirror of the official
  release; raw data under `cache/ext/visdrone/` (gitignored).
* **Single class: car.** Chosen because it has by far the most training instances: car 144,867
  (in 6,133 of 6,471 images), pedestrian 79,337, motor 29,647, people 27,059, van 24,956 (counted
  from the annotation files, 2026-10-11). Every other class is background (the official per-class
  protocol). VisDrone "ignored regions" (category 0) are COCO 'iscrowd' in evaluation (neither hit
  nor miss) and are not training targets, as in the official toolkit.

## 2. Preprocessing: 2 x 2 tiles (disclosed design choice)
Each image is cut into 4 equal non-overlapping tiles; each tile is letterboxed (top-left anchored,
as VOC) to the drone frame size 640 x 512. Reasons:
* Most images are 1400 x 1050 or 1400 x 788, so a tile (~700 x 525) is near the drone's native
  resolution. Letterboxing whole images would shrink cars ~3x (median car 124 px² at whole-image
  scale; 27% of cars would share a grid cell with another car, which a one-objectness-per-cell head
  cannot separate).
* 6,471 x 4 = 25,884 training tiles, just above the drone clients' total (25,146 frames), so the
  clients can have **exactly the drone client sizes**. Then, with B = 64, every client's sampling
  rate equals the drone's and **uniform sigma = 2 gives exactly epsilon = 1.8569**, the drone P
  budget the user asked for, with no rescaling.
* A car is a training target in a tile if at least 50% of its box area lies in that tile (box
  clipped to the tile); a car cut more than that is an ignore box ('iscrowd') in that tile.
* mAP is computed over tiles (tile-level AP@0.5), not the official whole-image VisDrone AP.

## 3. Splits (fixed now; seed 20261012)
* **TEST** = official test-dev, tiled (6,440 tiles). **VAL** = official val, tiled (2,192 tiles):
  gates, pulse test, A2 tuning. Official splits are scene-disjoint (0 sequence ids shared between
  train and test-dev).
* **Training tiles** (official train): grouped by sequence id (first field of the file name, 208
  sequences). PROXY (public pool, C* calibration only) = whole sequences drawn at random, added
  while the proxy stays at most 738 tiles, stopping at >= 600. CLIENTS: the remaining sequences in
  random order fill clients c0..c7 to exactly the drone sizes (1970, 1884, 1500, 1300, 6031,
  6162, 2999, 3300) in file-name order; a sequence is split only at a client boundary (at most 7
  sequences). Leftover tiles are unused. Non-IID by scene, the analogue of the drone's
  split-by-recording.

## 4. Model and FL (Phase 4c code paths, unchanged)
MobileNetV3-Small, ImageNet weights, frozen through block 8 (K = 9), features cached once; the
drone detection head (65,221 parameters); Phase 4c's multi-object targets / loss / mAP
(`encode_multi`, `loss_multi`, `map50_multi`). DP-SGD (Opacus, Poisson sampling, RDP,
delta 1e-5, worst-client epsilon), 8 clients, B = 64, 20 local iterations, SGD lr 0.03 momentum
0.9, FedAvg, mu = 0, R = 12, calibrated fixed clip C* from the proxy, seeds and common random
numbers exactly as Phase 4c. Only change: decoding keeps the top **100** peaks per tile (COCO's
maxDets; VOC used 20), because a tile can hold up to ~100 cars.

## 5. Quality check (the VOC check), BEFORE any comparison
1. **Noise-free training** (sigma = 0, clipping at C* kept, as the Phase 4c sweep), VAL, seeds
   100-104.
2. **Bar:** mean sigma-0 VAL mAP of seeds 100-102 >= max(0.104, trivial + 0.10), where trivial =
   one median car box per tile (Phase 4c's trivial control). 0.104 is the VOC number the user named;
   the Phase 4c rule is trivial + 0.10. The stricter applies.
3. **Scoring check:** (a) the ground-truth boxes themselves, scored by the same mAP code, must give
   >= 0.99; (b) reported, not a gate: the targets pushed through the head's encode -> decode path
   (the head's structural ceiling: cars sharing a cell or adjacent peaks are lost).
4. Pre-registered fallback chain, exactly as Phase 4c: if the bar is not cleared at K = 9 / B = 64,
   then K = 7 / B = 64, then K = 7 / B = 32 (B = 32 changes the sampling rates; sigma is then
   re-solved for epsilon 1.86). **If none clears the bar, STOP**: report as a capacity-limit
   finding (not a privacy finding), as VOC; no comparison is run.
5. Also reported (not a gate): the Phase 4c uniform sweep sigma in {0, 0.5, 1, 2, 3, 4, 6, 8}
   (5 seeds) and Phase 4c's G2 (noise costs >= 30% of the sigma-0 minus trivial range by sigma 8),
   to show where epsilon 1.86 sits between noise-free and trivial.

## 6. Comparison (only if §5 passes)
* **Budget:** epsilon = 1.8569 worst client (drone P). Uniform sigma solved for it (= 2 at B 64).
* **Pulse test** (Phase 3 V1 / Phase 4c protocol): uniform baseline at that sigma; pulse at round
  k = 1..12 doubles sigma_k; 20 paired seeds 0-19 on VAL; slope / first-minus-last-third / pooled
  tests; verdict VARIES / FLAT / not identifiable.
* **Arms (same epsilon):** A1 uniform; A2 exponential, k tuned on VAL over Phase 4's grid, seeds
  100-104; A3 theory rho_t ∝ √a_t from the pulse test (dropped if not identifiable); A6 Kiani et al.
  spend-as-you-go (Phase 4c mapping).
* **Evaluation:** TEST, 20 paired seeds 200-219, CRN. **Primary family (Holm, alpha 0.05): A2 vs A1,
  A3 vs A1, A6 vs A1**; WIN / LOSS / TIE / EQUIVALENT (margin ±0.02 mAP) exactly as Phase 4c.
  "Decay beats uniform" REPLICATES iff A2 or A3 WINS vs A1; REVERSED if either LOSES; otherwise
  FAILS TO REPLICATE. Secondary (unadjusted): A3 vs A2, A6 vs A2, A6 vs A3.

## 7. Drone-like or CIFAR-like? (fixed now)
* **Effect size** g = (better of A2/A3 test mean − A1 test mean) / A1 test mean.
  Reference: drone +24%, CIFAR-10 +1.7%.
* **DRONE-LIKE** if a decay arm WINS (Holm) and g >= 10%. **CIFAR-LIKE** if the pulse verdict is
  VARIES with a rising slope but no decay arm WINS, or g < 5%. **INTERMEDIATE** otherwise (a WIN
  with 5% <= g < 10%). **NEITHER** if a_t is FLAT or not identifiable.
* Sensitivity profile reported beside drone (Phase 3 V1, sigma 2) and CIFAR-10 (Phase 4c):
  slope / baseline, pooled drop / baseline, and the share of the total drop in the last third.

## 8. Order, cost, reporting
Download (done) -> tiles + features -> scoring check -> sigma-0 + sweep (gate) -> pulse -> A2
tuning -> test -> analysis. ~415 runs; estimated 8-12 GPU-hours. Code `code/phase4f.py`, launcher
`code/run_phase4f.ps1`. Report `phase4f/RESULTS.md` in Phase 4c's format; failures in
NEGATIVE_RESULTS.md. Any deviation: dated amendment here before the affected run.
