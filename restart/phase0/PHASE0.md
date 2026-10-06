# Phase 0 — Foundation (restart, started 2026-10-05)

This restart implements `thesis todo.pdf` phase by phase. It does **not** carry over
conclusions from the earlier work in this repository; only engineering tooling is
reused (GPU guard, resumable/atomic runs, memory cap, detached launcher). Everything
produced by the restart lives under `restart/` on git branch `restart`.

Status legend: **DECIDED** (fixed; change only with a dated note here),
**PENDING** (needs the user), **PROPOSED** (drafted by Claude, awaiting confirmation).

---

## 1. Research question — DECIDED (2026-10-05)

> **When does adaptive privacy-budget allocation improve the accuracy of a
> differentially private federated drone detector, compared with uniform and
> fixed-schedule allocation at the same total privacy budget?**

* Main thesis result = Phase 4 (allocation comparison).
* Security (Phase 5, Byzantine robustness) is **secondary** and starts only after
  Phase 4 is complete.
* A pre-registered tie or loss for DRL is a valid, reportable result.

## 2. Number provenance rule — DECIDED

Every number that appears in the new thesis must trace to a file in
`restart/results/` written by code in `restart/code/`, plus the git commit that
produced it. No number is typed by hand from memory or copied from earlier work.
Each results JSON records: code commit, machine, GPU, seeds, config, timestamp.

(The `thesis todo.pdf` item "list every number in the current thesis" is satisfied
by this rule for the restart; auditing the OLD LaTeX is out of scope unless asked.)

## 3. Split rule — DECIDED

**Split by recording session, never by frame position and never by folder.**
A session is never divided between two pools. Session id from the folder name
(`restart/code/inventory.py` raises on any unknown pattern):

| folder pattern | session id |
|---|---|
| `YYYYMMDD_HHMMSS_a_b` | `YYYYMMDD_HHMMSS` |
| `3700000000002_HHMMSS_n` | `3700000000002_HHMMSS` |
| `NN_VVVV_xxxx-yyyy` | `NN_VVVV` |
| `wg2022_ir_NNN_split_XX` | `wg2022_ir_NNN` |
| `newNN_*` | `newNN` |

**The dataset's official train/validation split is NOT used**: 23 of 89 sessions
appear in both (measured, `restart/data/sessions.csv`). All 200 folders are pooled
and re-split by session.

Census (`restart/code/inventory.py`): 200 folders, **89 sessions**, 231,557 frames,
216,210 with a target. `track1_test` / `track2_test` are excluded (tracking splits,
incomplete labels).

## 4. Environments and clients — PROPOSED (labels await confirmation)

* Labels: **sky** (incl. hazy/vignetted), **cloud**, **building**, **terrain**
  (hills, vegetation, forest). 29 folders labelled by the user (clear sky); 171
  proposed by Claude from one middle frame per folder, 35 flagged low-confidence.
  File: `restart/data/inventory_labelled.csv`; sheets: `restart/phase0/labelling/`.
* A session's environment = the label covering most of its target frames. 64/89
  sessions are single-environment; overall **83.3%** of target frames carry their
  session's label (disclose this impurity).
* Sessions by dominant environment: sky 38, terrain 26, building 15, cloud 10.
* Proposed clients: **8 = 2 per environment** (non-IID by background, one shared
  architecture). Alternative: 5 (sky x2 + one each). PENDING user choice.

## 5. Data pools — PROPOSED (built after §4 is confirmed)

All pools session-disjoint, stratified by environment, assignment seeded and
written once to `restart/data/split.json` (never regenerated after Phase 1 starts):

| pool | purpose | target share of sessions |
|---|---|---|
| **test** | final held-out reporting only; touched once per pre-registered comparison | ~15% |
| **val** | model selection, early stopping, calibration | ~10% |
| **root** | Phase 5 defence root set (FLTrust / server-side checks) | ~5% |
| **proxy** | training the DRL agent, so agent training spends no client privacy (§7) | ~15% |
| **clients** | federated training data | ~55% |

Every environment must appear in test, val and clients.

## 6. Model and metric — DECIDED

* Backbone: **MobileNetV3-Small**, ImageNet weights (torchvision), frozen in Phase 1
  unless the noise gate (§9) forces unfreezing; small trainable detection head.
  Input: IR frames replicated to 3 channels. Resolution and feature stride chosen in
  Phase 1 (median target ~25x25 px in 640x512 frames).
* Optimiser for DP training: SGD (momentum allowed). No adaptive optimiser in DP runs.
* Metric: **mAP@0.5**, single class, computed with one fixed implementation
  (pycocotools) on the **test** pool for reported results, **val** for selection.

## 7. Privacy accounting — DECIDED unless noted

* Mechanism: DP-SGD (Opacus), per-sample clipping, Gaussian noise, Poisson sampling.
* Accountant: **Opacus RDP**. Report **epsilon at delta = 1e-5** for every
  configuration, per client, next to sigma. No result is reported without its epsilon.
* **Privacy unit — PENDING.** Default is frame-level DP. Frames within a recording
  are highly correlated, so frame-level epsilon understates protection of a
  recording. Options: (a) frame-level, disclosed; (b) session-level (group privacy).
* **DRL agent training spends no client privacy**: the agent trains only on the
  **proxy** pool; its training compute is reported (suggestion 2 of the todo).

## 8. Experimental conventions — DECIDED

* Paired trials with common random numbers: every arm of trial t shares seed,
  initialisation and data order. **n >= 10 paired trials** for any comparison.
* Report mean difference, 95% CI and win count; never a single-run number.
* Every phase has a written pre-registration (`restart/phaseN/PREREG.md`) with its
  decision rule, committed **before** its first run.
* A visible log of what did not work / did not replicate: `restart/NEGATIVE_RESULTS.md`.

## 9. Phase gates (go / no-go) — DECIDED

| gate | condition to proceed | if it fails |
|---|---|---|
| G1 (end Phase 1) | single-client non-DP mAP@0.5 clearly above a trivial predictor on val | fix architecture/resolution before any DP |
| G1b (noise sweep) | mAP@0.5 changes monotonically and measurably over sigma in {0,2,4,8,16,32} | unfreeze more layers / lower batch size, re-sweep |
| G2 (end Phase 2) | FL+DP trains in all 4 ablation cells (FL x DP); costs reported | fix FL before Phase 3 |
| G3 (end Phase 3) | noise-pulse weights a_t measured with CIs; operating sigma chosen off floor/ceiling | if a_t is flat, theory predicts no schedule beats uniform: decide in writing whether Phase 4 still runs and as what |
| G4 | Phase 4 pre-registration committed | no Phase 4 runs |

## 10. Open items — PENDING

1. Confirm/correct environment labels (esp. the 35 low-confidence ones).
2. Number of clients: 8 (2 per environment) or 5.
3. Privacy unit: frame-level (disclosed) or session-level.
4. Deadline — sizes the number of DRL training runs and trials in Phase 4.
5. Kiani et al., ICLR 2025 — add the PDF so the comparison uses what it actually says.
