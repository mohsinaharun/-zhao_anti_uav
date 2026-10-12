# Restart track: privacy-budget allocation and robustness in DP federated learning

This folder implements `thesis todo.pdf` phase by phase, on git branch `restart`. It does not carry over
conclusions from earlier work in this repository (see `phase0/PHASE0.md`). Every phase is
pre-registered (`PREREG.md`, approved before any run) and reported in `RESULTS.md`. Failed attempts,
bugs and results that did not replicate are logged in `NEGATIVE_RESULTS.md`.

**Setting.** Anti-UAV drone detection (IR), 8 federated clients split by recording session. Frozen
MobileNetV3-Small trunk with a 65,221-parameter head. DP-SGD (Opacus, Poisson sampling, RDP accountant,
δ = 10⁻⁵, worst-client ε reported with every number), calibrated clip bound C*. Main results use the
TEST pool and 20 paired seeds (200-219) with common random numbers, with Holm correction within each
pre-registered family.

Operating points: **P** = 12 rounds, σ 2, ε 1.86 (primary); **S1** = 25 rounds, ε 2.72; **S2** = 12 rounds,
σ 4, ε 0.79.

## Phases

| Phase | Question | Result |
|---|---|---|
| 0 | Foundation: research question, splits, conventions, gates | Decided 2026-10-05 |
| 1 | Single-client model and DP live range | Non-DP 0.584 mAP; live range σ ≈ 0-4 at B 1,024 |
| 2 | FL × DP ablation | DP hurts far more inside FL than centrally (interaction −0.120) |
| rerun | Phases 1/2 with calibrated C* | Fixed C = 45.8 was miscalibrated; re-run descriptive |
| 3 | Noise-pulse test: is round sensitivity a_t flat? | Not flat (7/9 cells vary): later rounds more sensitive |
| **4** | **Allocation: uniform vs decaying vs theory vs DRL** | A2 (tuned exp.) and A3 (theory) beat uniform at P (+0.014, +24%); DRL agents A4/A5 do not |
| 4b | Run-to-run variation to exploit (A5 warm start) | No exploitable variation; not run |
| 4c | Replication on CIFAR-10 / VOC | CIFAR-10: equivalent (+0.005); VOC stopped at quality gate |
| 4d | Gradient-norm feedback signal | Fails both tests; A5-v2 not trained |
| 4e | Bayesian-optimisation schedule search (A7) | Beats uniform at P but only ties A2/A3: not genuine |
| 4f | Replication on VisDrone | Decay replicates (+10.5%, 17/3); A3 ties uniform |
| **4g** | **Do more rounds close the DP gap? (R 12/25/50/100 at ε 1.86)** | **No: noise ceiling.** A2 0.073 / 0.086 / 0.057 / 0.040 while no-DP rises to 0.33-0.34; share of ceiling 29% → 12%. A2 − uniform ≈ +0.013 at every R |
| **4h** | **DRL agent constrained to ±20% around A3, exact budget (A5-relative)** | **Not genuine at P, S1 or S2.** Ties A2/A3 at P and S2, loses to A2 at S1; removes A5's harm (WIN vs A5 at S2) but adds no gain. Pattern structured and feedback-using, but training reward never rose |
| 5 | Robust aggregation under DP (FLTrust, median, Krum) | FLTrust loses to FedAvg at ε 1.86 (−75% clean); schedule does not change attack damage |
| 5b | FLTrust across DP levels; fixes | Plain FLTrust loses at every DP level; Variant N (DP-aware norm) recovers at ε 1.86 |
| 5c | Variant N at five DP levels; adaptive attack | Recovers at σ 0/0.5/2, loses at σ 1 and 4; broken by an adaptive trust-capture attack |
| **5d** | **Per-client trust-weight cap on Variant N (κ = 2/K, 1.5/K, 1/K)** | **Not a fix at any margin.** The cap holds the attackers at 2κ (50/37.5/25%), yet the model still collapses under trust-capture (0.000-0.006 mAP). Cap-aware attacker no stronger. Free otherwise; at 1/K it improves ALIE (+0.031) |

## Where things are
* `phaseN/PREREG.md`, `phaseN/RESULTS.md`, and for later phases `phaseN/THESIS_TABLE.md`.
* **Main allocation thesis tables:** `phase4f/THESIS_TABLE.md` (Table X: generalization; X.2: all
  allocation methods incl. A7 and A5-relative; X.3: round count, Phase 4g). Security tables:
  `phase5/`, `phase5b/`, `phase5c/`, `phase5d/THESIS_TABLE.md`.
* Code: `code/` (`phaseN.py` + `analyse_phaseN.py`, `run_phaseN.ps1` detached launchers).
* Raw runs and analysis JSON: `results/phaseN/`. Figures: `figures/` (Phases 1-3) and the repo-root
  `plots/` (Phase 4c onward, e.g. `plots/phase4g_rounds.png`, `plots/phase4h_relative_agent.png`).
* Not committed: the dataset, `cache/` (features, training checkpoints), `literature/*.pdf`.

## Reproducing
Python env: `~/Documents/anti_uav/Scripts/python.exe`. Each phase: `python code/phaseN.py run|go` (resumable,
reproduction check first), then `python code/phaseN.py analyse`. **Set the GPU power cap first**
(`nvidia-smi -pl 300` in an admin shell, re-apply after every reboot). See the root `CLAUDE.md`.
